import cv2
import logging
import time
import json
from collections import deque
from pathlib import Path
from typing import Optional
from PIL import Image
import numpy as np

# --- Phase 1: Physics Engine Imports ---
from src.physics_engine.detector import load_detector
from src.physics_engine.tracker import VehicleTracker
from src.physics_engine.homography import CoordinateTransformer
from src.physics_engine.kinematics import KinematicEstimator
from src.physics_engine.zone_manager import ZoneManager, ZoneConfig

# --- Phase 2: Semantic Abstraction Imports ---
from src.semantic_abstractor.set_of_mark import AdaptiveRenderer, RenderContext
from src.semantic_abstractor.vlm_inference import TrafficSemanticAbstractor
from src.semantic_abstractor.entity_extractor import EntityExtractor

# --- Phase 3: Hybrid Database Imports ---
from src.memory_layer.duckdb_client import DuckDBClient
from src.memory_layer.milvus_client import SemanticVectorStore
from src.memory_layer.graph_client import GraphClient

# --- Phase 4: Agentic Orchestrator ---
from src.agentic_orchestrator.sequential_pipeline import agent_app, AGENT_INVOKE_CONFIG

# --- Phase 5: Alert Engine (optional) ---
from src.symbolic_engine.alert_engine import AlertEngine, TrafficAlert

# --- Phase 6: Evaluation Metrics ---
from src.evaluation.metrics import MetricsCollector, time_operation

log = logging.getLogger(__name__)

def process_video(
    video_path: str,
    progress_callback=None,
    alert_callback=None,
    metrics: Optional[MetricsCollector] = None,
):
    """
    Executes the dual-loop Neuro-Symbolic tracking and abstraction pipeline.
    High-frequency loop runs every frame. Low-frequency loop runs at ~3 VLM
    calls/second, derived from the video's actual frame rate.

    Args:
        video_path:         Path to the input video file.
        progress_callback:  Optional callable(frames_done: int, total_frames: int)
                            invoked once per frame so callers can track progress.
        alert_callback:     Optional callable(alert: TrafficAlert) invoked in
                            real-time whenever a kinematic threshold is crossed.
                            When None, the alert engine is disabled entirely.
        metrics:            Optional MetricsCollector that records proxy evaluation
                            metrics (VLM quality, DB latency, alert distribution).
                            Call metrics.log_summary() after this function returns.
    """
    # Reset per-call state stored on the function object
    process_video._prev_window_ptr = None
    log.info("Initializing Neuro-Symbolic Pipeline...")

    # 1. Open Video Stream FIRST to read actual fps and frame count.
    #    fps drives every kinematic calculation — using the wrong value
    #    produces incorrect timestamps, velocities, and accelerations.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Run VLM ~3 times per second regardless of source frame rate.
    semantic_interval = max(1, fps // 3)
    log.info("Video: %d fps | %d frames | VLM every %d frames (~3/sec)",
             fps, total_frames, semantic_interval)

    # 2. Initialize Hybrid Databases
    duckdb_client = DuckDBClient()
    milvus_client = SemanticVectorStore()
    graph_client = GraphClient()

    # 3. Initialize Physics Engine (Micro-Loop) with correct fps
    detector = load_detector("yolov8n.pt", conf=0.3)
    tracker = VehicleTracker(tracker_name="bytetrack")
    transformer = CoordinateTransformer("calibration.yaml")
    kinematics = KinematicEstimator(fps=float(fps))

    # --- Multi-frame VLM buffer -------------------------------------------
    # Holds the last SOM_BUFFER_SIZE SoM PIL images.  Passed to the VLM as a
    # native video clip giving genuine temporal context (MRoPE encoding).
    # Covers ~2 s of traffic at 3 VLM samples/sec.
    SOM_BUFFER_SIZE = 6
    _som_buffer: deque = deque(maxlen=SOM_BUFFER_SIZE)

    # --- Motion-energy gating --------------------------------------------
    # Skip the VLM call on frames where the scene is effectively static
    # (mean absolute frame difference below threshold).  Reduces wasted
    # VLM compute by 40–60% on typical traffic video.
    # Set to 0.0 to disable gating.
    MOTION_SKIP_THRESHOLD = 2.0    # mean absolute pixel difference (0–255)
    _prev_gray: np.ndarray | None = None
    _motion_score: float = 999.0

    # --- Alert-triggered forced VLM call ---------------------------------
    # When the AlertEngine fires a COLLISION_SUSPECTED or HARD_BRAKING alert
    # the exact critical frame is always sent to the VLM — regardless of
    # the fixed-interval schedule or motion-energy gate.
    _flags: dict = {"force_vlm": False}

    def _alert_handler(alert) -> None:
        if alert.alert_type in ("COLLISION_SUSPECTED", "HARD_BRAKING"):
            _flags["force_vlm"] = True
        # Fix 4: persist every alert to DuckDB immediately (direct INSERT, no buffer).
        duckdb_client.insert_alert(alert)
        if alert_callback:
            alert_callback(alert)
        if metrics is not None:
            metrics.record_alert_fired(alert.alert_type)

    alert_engine = AlertEngine(on_alert=_alert_handler) if alert_callback else None
    if alert_engine:
        log.info("Alert engine active — real-time kinematic alerts + forced VLM on critical events.")

    # --- Entity profile tracking -----------------------------------------
    # Accumulates first_seen timestamp per vehicle for entity_profiles.
    _vehicle_first_seen: dict = {}

    # Zone manager is optional — only active when zone_config.json exists.
    # Draw zones at /zone-ui before running the pipeline.
    zone_manager = None
    if Path("zone_config.json").exists():
        zone_config = ZoneConfig.from_json("zone_config.json")
        zone_manager = ZoneManager(zone_config)
        log.info("Zone '%s' active — gates: %s",
                 zone_config.zone_id, [g.name for g in zone_config.gates])

    # 4. Initialize Semantic Abstractor (Macro-Loop)
    renderer = AdaptiveRenderer()
    vlm = TrafficSemanticAbstractor(model_id="Qwen/Qwen2.5-VL-3B-Instruct")
    extractor = EntityExtractor(model_name="qwen2.5:72b")

    frame_id = 0
    start_time = time.time()
    if metrics is not None:
        metrics.begin()

    log.info("--- Starting Video Processing ---")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_id / fps

        # ==========================================
        # THE MICRO-LOOP (High-Frequency Physics)
        # ==========================================
        
        # 1. Detect & Track
        raw_dets = detector.predict(frame)
        tracked_boxes = tracker.update(raw_dets[0], frame)
        
        # 2. Map to Real-World 3D Space
        real_coords = transformer.get_real_world_coords(tracked_boxes)
        
        # 3. Extract Kinematic State Vectors
        # Adjusted to match your KinematicEstimator signature
        state_vectors = kinematics.update(real_coords) 
        
        # 4. Stream to Analytical Database
        duckdb_client.insert_state_vectors(timestamp, frame_id, state_vectors)

        # 5. Real-time alerts (skipped when no alert_callback was provided).
        # Fix 5: only pass warm tracks to the alert engine — cold tracks (< 15 frames)
        # have zero acceleration from finite differences and would produce false alerts.
        if alert_engine is not None:
            warm_sv = {k: v for k, v in state_vectors.items() if k in kinematics.warm_tracks}
            alert_engine.check(warm_sv, real_coords, timestamp, frame_id)

        # 6. Motion-energy score for VLM gating (cheap, runs every frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if _prev_gray is not None:
            _motion_score = float(
                np.mean(np.abs(gray.astype(np.float32) - _prev_gray.astype(np.float32)))
            )
        _prev_gray = gray

        # Track first appearance per vehicle for entity profiles
        for tid in state_vectors:
            if tid not in _vehicle_first_seen:
                _vehicle_first_seen[tid] = timestamp

        # 7. Zone crossing detection (skipped if no zone_config.json)
        if zone_manager is not None:
            crossing_events = zone_manager.update(
                tracked_boxes, real_coords, timestamp, frame_id
            )
            for event in crossing_events:
                duckdb_client.insert_crossing_event(event)
                log.info("[%.1fs] Vehicle %d → %s via %s",
                         timestamp, event.track_id, event.direction, event.gate_name)

        # ==========================================
        # THE MACRO-LOOP (Low-Frequency Semantics)
        # ==========================================

        # Detect interval frames that get skipped by the motion gate (for metrics).
        _is_interval_frame = (
            frame_id > 0
            and len(tracked_boxes) > 0
            and frame_id % semantic_interval == 0
        )
        if (metrics is not None
                and _is_interval_frame
                and _motion_score < MOTION_SKIP_THRESHOLD
                and not _flags["force_vlm"]):
            metrics.record_motion_skip()

        # MACRO-LOOP condition:
        #   - Fixed interval: every semantic_interval frames (~3/sec)
        #   - OR forced: when AlertEngine flagged a critical event this frame
        # Motion-energy gate: skip static scenes (saves ~40-60% VLM calls)
        # unless a critical alert forced this tick.
        _run_macro = (
            frame_id > 0
            and len(tracked_boxes) > 0
            and (
                (frame_id % semantic_interval == 0
                 and (_motion_score >= MOTION_SKIP_THRESHOLD or _flags["force_vlm"]))
                or _flags["force_vlm"]
            )
        )

        if _run_macro:
            _force_was_set = _flags["force_vlm"]
            _flags["force_vlm"] = False
            if metrics is not None and _force_was_set:
                metrics.record_force_vlm()
            log.debug("[%.1fs] Running Semantic Abstraction (motion=%.1f, frames=%d)...",
                      timestamp, _motion_score, len(_som_buffer) + 1)

            HISTORY_WINDOW_SECS = 5.0
            chunk_start = max(0.0, timestamp - HISTORY_WINDOW_SECS)
            time_window_ptr = f"{chunk_start:.1f}-{timestamp:.1f}"

            # 1. Visual Grounding (Set-of-Mark) — render current frame
            som_frame = frame.copy()
            render_ctx = RenderContext()
            render_ctx.update(tracked_boxes, timestamp)
            renderer.render(som_frame, render_ctx)
            som_pil = Image.fromarray(cv2.cvtColor(som_frame, cv2.COLOR_BGR2RGB))

            # Append to rolling buffer (deque handles maxlen eviction)
            _som_buffer.append(som_pil)

            # 2. Behaviour history from DuckDB (change-only, last 5s)
            active_ids = [int(t[4]) for t in tracked_boxes]
            behavior_summary = duckdb_client.get_behavior_summary(active_ids, timestamp)

            # 3. VLM Inference — pass frame buffer for temporal context.
            # Multi-frame path: Qwen2.5-VL receives the buffer as a native
            # video clip (frame-list format + fps).  The model uses MRoPE
            # temporal position encoding to understand motion across frames.
            # Single-frame fallback if buffer only has 1 entry (early in video).
            with time_operation() as _vlm_timer:
                vlm_triples = vlm.generate_scene_graph_triples(
                    list(_som_buffer),
                    timestamp,
                    state_vectors,
                    kinematics.warm_tracks,
                    behavior_summary=behavior_summary,
                    fps=3.0,  # VLM sample rate: semantic_interval ≈ fps/3
                )
            if metrics is not None:
                metrics.record_vlm_call(
                    latency_ms=_vlm_timer.elapsed_ms,
                    parse_success=bool(vlm_triples),
                    triple_count=len(vlm_triples) if vlm_triples else 0,
                )

            if vlm_triples:
                # NL for Milvus (better embeddings than JSON syntax)
                nl_description = " ".join(
                    f"{t['subject']} {t['predicate']} {t['object']}."
                    for t in vlm_triples
                )
                with time_operation() as _milvus_timer:
                    milvus_client.insert_event_chunk(
                        nl_description, chunk_start, timestamp, frame_id
                    )
                if metrics is not None:
                    metrics.record_milvus_insert(_milvus_timer.elapsed_ms)

                # JSON for EntityExtractor (qwen2.5:72b needs SPO keys)
                scene_description = json.dumps(vlm_triples)
                # Fix 3: pass active track IDs so hallucinated vehicle IDs are filtered.
                validated_triples = extractor.extract_triples(scene_description, timestamp, set(active_ids))

                if validated_triples:
                    # Previous window for PRECEDES edge (None on first tick)
                    prev_window_ptr = getattr(process_video, "_prev_window_ptr", None)

                    with time_operation() as _graph_timer:
                        graph_client.insert_vlm_triples(validated_triples, time_window_ptr)
                    if metrics is not None:
                        metrics.record_graph_insert(_graph_timer.elapsed_ms)

                    # Insert PRECEDES temporal edges linking this window to the last
                    if prev_window_ptr is not None:
                        gap_s = float(time_window_ptr.split("-")[0]) - float(prev_window_ptr.split("-")[0])
                        graph_client.insert_temporal_edges(
                            active_ids, prev_window_ptr, time_window_ptr, gap_s
                        )

                    process_video._prev_window_ptr = time_window_ptr

            # 4. Entity profile update (longitudinal per-vehicle memory)
            # Parse behavior_summary lines to get per-vehicle narratives
            # and upsert into Milvus entity_profiles collection.
            if behavior_summary:
                for line in behavior_summary.strip().splitlines():
                    line = line.strip()
                    if not line.startswith("Vehicle"):
                        continue
                    try:
                        # Extract vehicle ID from "Vehicle N: ..."
                        parts = line.split(":", 1)
                        vid = int(parts[0].replace("Vehicle", "").strip())
                        narrative = line  # full line as the profile text
                        milvus_client.upsert_entity_profile(
                            track_id=vid,
                            summary=narrative,
                            first_seen=_vehicle_first_seen.get(vid, timestamp),
                            last_seen=timestamp,
                        )
                    except (ValueError, IndexError):
                        pass

        if progress_callback is not None:
            progress_callback(frame_id, total_frames)

        if metrics is not None:
            metrics.record_frame()

        frame_id += 1

    cap.release()
    log.info("--- Video Processing Complete in %.2fs ---", time.time() - start_time)

    # Safely close database connections
    duckdb_client.close()
    milvus_client.close()
    graph_client.close()

    if metrics is not None:
        metrics.end()
        metrics.log_summary()

def interactive_agent_loop():
    """Boots up the LangGraph agent to query the processed hybrid databases."""
    print("\n=============================================")
    print("Neuro-Symbolic Agentic Brain Initialized")
    print("=============================================")
    print("Ask questions about the traffic event (e.g., 'Did Vehicle 4 brake too hard?').")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("User >> ")
        if query.lower() in ['exit', 'quit']:
            break

        initial_state = {"query": query}
        final_state = agent_app.invoke(initial_state, AGENT_INVOKE_CONFIG)

        print(f"\nAgent >> {final_state.get('final_summary', 'No summary generated.')}\n")

if __name__ == "__main__":
    SAMPLE_VIDEO = "data/raw_videos/traffic_sample.mp4"
    
    if not Path("calibration.yaml").exists():
        print("Error: calibration.yaml not found!")
        print("Please open the web calibration tool at /calibrate-ui and complete calibration first.")
        exit(1)
        
    process_video(SAMPLE_VIDEO, progress_callback=lambda done, total: print(f"  [{done}/{total}] frames processed") if done % 300 == 0 else None)
    interactive_agent_loop()