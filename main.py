import cv2
import logging
import time
import json
import threading
import queue as queue_module
import math
from collections import deque
from pathlib import Path
from PIL import Image
import numpy as np

# --- Phase 1: Physics Engine Imports ---
from src.physics_engine.detector import load_pose_detector
from src.physics_engine.tracker import VehicleTracker
from src.physics_engine.homography import CoordinateTransformer
from src.physics_engine.kinematics import KinematicEstimator
from src.physics_engine.zone_manager import ZoneManager, ZoneConfig
from src.physics_engine.zone_blur import build_entry_buffer, apply_detection_blur
from src.physics_engine.lane_classifier import LaneClassifier

# --- Phase 3: Hybrid Database Imports ---
from src.memory_layer.duckdb_client import DuckDBClient

# --- Phase 4: Agentic Orchestrator ---
from src.agentic_orchestrator.sequential_pipeline import agent_app, AGENT_INVOKE_CONFIG

log = logging.getLogger(__name__)

SOM_BUFFER_SIZE = 6
TARGET_PHYSICS_FPS = 20.0


def _vlm_worker(
    task_queue: queue_module.Queue,
    stop_event: threading.Event,
    reference_name: str = "Origin",
    nearest_landmark_fn=None,
) -> None:
    """
    Background thread: drains semantic tasks enqueued by the physics loop.

    Each task is an 8-tuple:
        (frame, tracked_boxes, state_vectors, warm_tracks,
         timestamp, frame_id, vehicle_first_seen, behavior_summary)
    A sentinel value of None signals the worker to shut down cleanly.
    """
    from src.semantic_abstractor.set_of_mark import VehicleIdRenderer, RenderContext, draw_zone_overlay
    from src.semantic_abstractor.vlm_inference import TrafficSemanticAbstractor
    from src.semantic_abstractor.entity_extractor import EntityExtractor
    from src.memory_layer.milvus_client import SemanticVectorStore
    from src.memory_layer.graph_client import GraphClient

    renderer = VehicleIdRenderer()
    vlm = TrafficSemanticAbstractor(model_id="gemma4:e2b")
    extractor = EntityExtractor(model_name="gemma4:e2b")
    milvus_client = SemanticVectorStore()
    graph_client = GraphClient()

    _som_buffer: deque = deque(maxlen=SOM_BUFFER_SIZE)
    # parallel: (timestamp, frozenset of IDs) per buffered frame
    _id_buffer:  deque = deque(maxlen=SOM_BUFFER_SIZE)
    _prev_window_ptr = None

    while True:
        try:
            task = task_queue.get(timeout=1.0)
        except queue_module.Empty:
            if stop_event.is_set():
                break
            continue

        if task is None:  # sentinel — shut down cleanly
            task_queue.task_done()
            break

        # Unpack task — zone_occupants and zone_cfg are optional (backward compat).
        task_fields = task
        (frame, tracked_boxes, state_vectors, warm_tracks,
         timestamp, frame_id,
         vehicle_first_seen, behavior_summary) = task_fields[:8]
        zone_occupants = task_fields[8] if len(task_fields) > 8 else frozenset()
        zone_cfg       = task_fields[9] if len(task_fields) > 9 else None

        HISTORY_WINDOW_SECS = 5.0
        chunk_start = max(0.0, timestamp - HISTORY_WINDOW_SECS)
        time_window_ptr = f"{chunk_start:.1f}-{timestamp:.1f}"

        # 1. Set-of-Mark visual grounding
        #    When a zone is active:
        #      a) Apply spotlight overlay (dims outside, draws zone border/gates)
        #         BEFORE badge rendering so badges appear on top at full brightness.
        #      b) Pass zone_occupant_ids to the render context so only in-zone
        #         vehicles receive SoM badges — out-of-zone vehicles are invisible
        #         to the VLM (no ID = no hallucination about them).
        som_frame = frame.copy()
        if zone_cfg is not None and zone_cfg.polygon:
            draw_zone_overlay(
                som_frame,
                zone_cfg.polygon,
                gates=zone_cfg.gates,
                zone_id=zone_cfg.zone_id,
            )
        render_ctx = RenderContext()
        render_ctx.update(tracked_boxes, timestamp)
        if zone_occupants:
            render_ctx.zone_occupant_ids = zone_occupants
        renderer.render(som_frame, render_ctx)
        som_pil = Image.fromarray(cv2.cvtColor(som_frame, cv2.COLOR_BGR2RGB))

        # Append frame + its (timestamp, IDs) to parallel buffers.
        # When zone is active, only record IDs for in-zone vehicles so the
        # per-frame timeline injected into the VLM prompt stays zone-scoped.
        frame_ids_this = frozenset(
            int(t[4]) for t in tracked_boxes
            if (not zone_occupants or int(t[4]) in zone_occupants)
        )
        _som_buffer.append(som_pil)
        _id_buffer.append((timestamp, frame_ids_this))

        # Build per-frame ID timeline from the buffer for the VLM prompt.
        frame_id_timeline = [(ts, sorted(ids)) for ts, ids in _id_buffer]
        all_active_ids = sorted(set().union(*(ids for _, ids in _id_buffer)))

        # Filter state_vectors to zone occupants so VLM physics context is focused.
        vlm_state_vectors = (
            {k: v for k, v in state_vectors.items() if k in zone_occupants}
            if zone_occupants else state_vectors
        )

        # Build zone_context dict for VLM prompt injection.
        zone_context = None
        if zone_cfg is not None and zone_occupants:
            zone_context = {
                "zone_id": zone_cfg.zone_id,
                "occupant_ids": sorted(zone_occupants),
            }

        # 2. VLM inference
        vlm_triples = vlm.generate_scene_graph_triples(
            list(_som_buffer),
            timestamp,
            vlm_state_vectors,
            warm_tracks,
            behavior_summary=behavior_summary,
            fps=3.0,
            tracked_boxes=tracked_boxes,
            all_active_ids=all_active_ids,
            frame_id_timeline=frame_id_timeline,
            reference_name=reference_name,
            nearest_landmark_fn=nearest_landmark_fn,
            zone_context=zone_context,
        )

        if vlm_triples:
            nl_description = " ".join(
                f"{t['subject']} {t['predicate']} {t['object']}."
                for t in vlm_triples
            )
            milvus_client.insert_event_chunk(
                nl_description, chunk_start, timestamp, frame_id
            )

            scene_description = json.dumps(vlm_triples)
            active_ids = [int(t[4]) for t in tracked_boxes]
            validated_triples = extractor.extract_triples(
                scene_description, timestamp, set(active_ids)
            )

            if validated_triples:
                graph_client.insert_vlm_triples(validated_triples, time_window_ptr)

                if _prev_window_ptr is not None:
                    gap_s = (float(time_window_ptr.split("-")[0])
                             - float(_prev_window_ptr.split("-")[0]))
                    graph_client.insert_temporal_edges(
                        active_ids, _prev_window_ptr, time_window_ptr, gap_s
                    )

                _prev_window_ptr = time_window_ptr

        # 3. Entity profile upsert
        if behavior_summary:
            for line in behavior_summary.strip().splitlines():
                line = line.strip()
                if not line.startswith("Vehicle"):
                    continue
                try:
                    parts = line.split(":", 1)
                    vid = int(parts[0].replace("Vehicle", "").strip())
                    milvus_client.upsert_entity_profile(
                        track_id=vid,
                        summary=line,
                        first_seen=vehicle_first_seen.get(vid, timestamp),
                        last_seen=timestamp,
                    )
                except (ValueError, IndexError):
                    pass

        task_queue.task_done()

    milvus_client.close()
    graph_client.close()
    log.info("VLM worker shut down.")


def process_video(
    video_path: str,
    progress_callback=None,
    run_physics: bool = True,
    run_vlm: bool = True,
    model_path: str = "models/yolo_pose-4/weights/best.pt",
):
    """
    Executes the dual-loop Neuro-Symbolic tracking and abstraction pipeline.
    High-frequency loop runs every frame. Low-frequency loop runs at ~3 VLM
    calls/second, derived from the video's actual frame rate.

    Args:
        video_path:         Path to the input video file.
        progress_callback:  Optional callable(frames_done: int, total_frames: int)
                            invoked once per frame so callers can track progress.
    """
    log.info("Initializing Neuro-Symbolic Pipeline...")

    # 1. Open Video Stream FIRST to read actual fps and frame count.
    #    fps drives every kinematic calculation — using the wrong value
    #    produces incorrect timestamps, velocities, and accelerations.
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Recorded-video mode: keep timestamps in source-video time but subsample
    # aggressively enough that metre-scale projection jitter does not explode
    # into unrealistic speeds. For 100 FPS input this yields FRAME_STEP=5
    # (about 20 Hz physics), which is a better match for traffic motion.
    FRAME_STEP: int = max(1, int(math.ceil(fps / TARGET_PHYSICS_FPS)))
    physics_fps: float = fps / FRAME_STEP

    # Run VLM ~3 times per second of video, counted in processed frames.
    semantic_interval = max(1, int(physics_fps) // 3)
    log.info(
        "Video: %.2f fps | %d frames | FRAME_STEP=%d → physics at %.2f fps"
        " | VLM every %d processed frames (~3/sec)",
        fps, total_frames, FRAME_STEP, physics_fps, semantic_interval,
    )

    # 2. Initialize Databases (DuckDB always; Milvus/Graph handled in VLM worker thread)
    duckdb_client = DuckDBClient()

    # 3. Initialize Physics Engine (Micro-Loop) with correct fps
    detector = load_pose_detector(model_path, conf=0.3)
    # Use the class names baked into the model checkpoint — avoids the COCO
    # mismatch when a custom-trained pose model has non-COCO class indices.
    _class_names: dict = detector.model.names
    log.info("Model class map: %s", _class_names)
    tracker = VehicleTracker(tracker_name="bytetrack")
    if run_physics:
        transformer = CoordinateTransformer("calibration.yaml")

        # Read ByteTrack's track_buffer so KinematicEstimator's grace period
        # matches exactly — a track the estimator has forgotten but ByteTrack
        # can still re-associate would otherwise restart a 15-frame SG warm-up.
        _bt_missed = 30  # ByteTrack default track_buffer
        try:
            from boxmot.utils import TRACKER_CONFIGS
            import yaml as _yaml
            _bt_cfg = _yaml.safe_load(
                (Path(TRACKER_CONFIGS) / "bytetrack.yaml").read_text()
            )
            _bt_missed = _bt_cfg.get("track_buffer", {}).get("default", 30)
        except Exception:
            pass  # fall back to 30
        log.info("KinematicEstimator max_missed_frames aligned to ByteTrack track_buffer=%d", _bt_missed)

        kinematics = KinematicEstimator(
            fps=physics_fps,
            max_missed_frames=_bt_missed,
            frame_step=FRAME_STEP,
            speed_window_s=0.25,
            accel_window_s=0.5,
            min_trusted_samples=5,
            max_speed_kmh=140.0,
            max_pos_accel_kmhps=25.0,
            max_brake_kmhps=35.0,
        )
        duckdb_client.set_named_points(transformer.named_points, transformer.reference_name)
        log.info(
            "Reference point: '%s' | Named landmarks: %s",
            transformer.reference_name,
            [p["name"] for p in transformer.named_points],
        )

    # --- Motion-energy gating --------------------------------------------
    # Skip the VLM call on frames where the scene is effectively static
    # (mean absolute frame difference below threshold).  Reduces wasted
    # VLM compute by 40–60% on typical traffic video.
    # Set to 0.0 to disable gating.
    MOTION_SKIP_THRESHOLD = 2.0    # mean absolute pixel difference (0–255)
    _prev_gray: np.ndarray | None = None
    _motion_score: float = 999.0

    # --- Entity profile tracking -----------------------------------------
    # Accumulates first_seen timestamp per vehicle for entity_profiles.
    _vehicle_first_seen: dict = {}

    # Zone manager is optional — only active when zone_config.json exists.
    # Draw zones at /zone-ui before running the pipeline.
    zone_manager = None
    zone_config  = None
    _entry_buffer: np.ndarray | None = None
    if Path("zone_config.json").exists():
        zone_config = ZoneConfig.from_json("zone_config.json")
        zone_manager = ZoneManager(zone_config)
        log.info("Zone '%s' active — gates: %s",
                 zone_config.zone_id, [g.name for g in zone_config.gates])
        # Pre-compute the 5 m entry buffer in pixel space once at startup.
        # Used every frame to extend the unblurred region upstream of E1.
        if run_physics:
            _entry_buffer = build_entry_buffer(
                transformer.H,
                transformer.named_points,
                extension_m=5.0,
            )
            if _entry_buffer is not None:
                log.info("Entry buffer built: 5 m upstream of C0/C6 gate")

    # Lane classifier — loaded from lane_config.json when available.
    # Run the setup script (or call LaneClassifier.from_frame()) once to
    # generate the config.  When absent, lane index is not stored.
    lane_classifier: LaneClassifier | None = None
    if Path("lane_config.json").exists():
        lane_classifier = LaneClassifier.load("lane_config.json")
        log.info(
            "LaneClassifier loaded: %d lanes, boundaries=%s",
            lane_classifier.n_lanes(), lane_classifier.boundaries,
        )

    # 4. Start VLM worker thread (Macro-Loop runs in background)
    if run_vlm:
        # Bounded queue: if VLM is slower than physics, drop frames rather
        # than accumulating unbounded memory.
        _vlm_queue: queue_module.Queue = queue_module.Queue(maxsize=4)
        _vlm_stop = threading.Event()
        _ref_name = transformer.reference_name if run_physics else "Origin"
        _nearest_fn = transformer.nearest_landmark if run_physics else None
        _vlm_thread = threading.Thread(
            target=_vlm_worker,
            args=(_vlm_queue, _vlm_stop, _ref_name, _nearest_fn),
            name="vlm-worker",
            daemon=True,
        )
        _vlm_thread.start()
        log.info("VLM worker thread started — macro-loop running in background.")

    frame_id = 0          # actual video frame index (increments by FRAME_STEP)
    processed = 0         # count of frames sent through YOLO
    start_time = time.time()
    log.info("--- Starting Video Processing ---")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip intermediate frames cheaply — no decode, just advance the
        # demuxer. frame_id stays aligned to the actual video timeline so
        # timestamps remain correct for DuckDB and the kinematic estimator.
        # Track how many grabs succeeded so frame_id is exact at end-of-file.
        grabbed = 0
        for _ in range(FRAME_STEP - 1):
            if not cap.grab():
                break
            grabbed += 1

        timestamp = frame_id / fps   # video wall-clock time in seconds

        # ==========================================
        # THE MICRO-LOOP (High-Frequency Physics)
        # ==========================================

        # 1. Detect & Track (always — tracker feeds both physics and VLM)
        # Blur everything outside the zone + 5 m entry buffer so YOLO
        # focuses on the measurement corridor and incoming vehicles.
        # The original frame is kept for VLM and video output.
        detect_frame = (
            apply_detection_blur(
                frame,
                zone_config.polygon if zone_config is not None else None,
                _entry_buffer,
            )
            if zone_manager is not None
            else frame
        )
        pose_result, keypoints_np = detector.predict_with_pose(detect_frame)
        tracked_boxes = tracker.update(pose_result, frame)

        # 2-7. Physics sub-loop — kinematics, DuckDB, alerts, zones
        if run_physics:
            # Pass raw pose bboxes + keypoints so the transformer can use kp31
            # (rear wheel bottom, z=0) instead of bbox bottom-centre.
            pose_boxes_np = (
                pose_result.boxes.data.cpu().numpy()
                if pose_result.boxes is not None and len(pose_result.boxes) > 0
                else np.empty((0, 6), dtype=np.float32)
            )
            real_coords, _ = transformer.get_real_world_coords(
                tracked_boxes,
                pose_detections=(pose_boxes_np, keypoints_np),
            )
            measurement_quality = dict(transformer.last_measurement_quality)
            state_vectors, synthetic_rows = kinematics.update(
                real_coords, timestamp, frame_id, measurement_quality
            )
            state_quality = dict(kinematics.last_state_quality)
            transformer.update_velocity_cache(state_vectors)
            class_labels = {
                int(t[4]): _class_names.get(int(t[6]), "unknown")
                for t in tracked_boxes
            }

            # Lane classification — appended to class_label as
            # "<vehicle_type>:lane<N>" so DuckDB queries can filter by lane.
            if lane_classifier is not None:
                for tid, (_, world_y) in real_coords.items():
                    lane_idx = lane_classifier.classify(world_y)
                    class_labels[tid] = (
                        f"{class_labels.get(tid, 'unknown')}:lane{lane_idx}"
                    )
            # Column 5 of tracked_boxes is the YOLO detection confidence (0–1).
            # Stored in DuckDB so data quality reports can compute mean confidence
            # per vehicle/window and flag low-confidence analyses.
            detection_confidences = {
                int(t[4]): float(t[5])
                for t in tracked_boxes
            }
            duckdb_client.insert_state_vectors(
                timestamp, frame_id, state_vectors, class_labels,
                detection_confidences, state_quality,
            )
            # Write gap-filled synthetic rows (interpolated=True) so DuckDB has
            # a continuous time-series with no holes from missed detections.
            if synthetic_rows:
                duckdb_client.insert_interpolated_rows(synthetic_rows, class_labels)
                log.debug(
                    "[%.1fs] Persisted %d interpolated rows for gap-filled tracks",
                    timestamp, len(synthetic_rows),
                )

            for tid in state_vectors:
                if tid not in _vehicle_first_seen:
                    _vehicle_first_seen[tid] = timestamp

            if zone_manager is not None:
                crossing_events = zone_manager.update(
                    tracked_boxes, real_coords, timestamp, frame_id
                )
                for event in crossing_events:
                    duckdb_client.insert_crossing_event(event)
                    log.info("[%.1fs] Vehicle %d → %s via %s",
                             timestamp, event.track_id, event.direction, event.gate_name)
        else:
            real_coords = {}
            state_vectors = {}
            state_quality = {}

        # Motion-energy score (cheap, runs every frame — used by VLM gate)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if _prev_gray is not None:
            _motion_score = float(
                np.mean(np.abs(gray.astype(np.float32) - _prev_gray.astype(np.float32)))
            )
        _prev_gray = gray

        # ==========================================
        # THE MACRO-LOOP (Low-Frequency Semantics)
        # ==========================================

        # MACRO-LOOP condition:
        #   - Fixed interval: every semantic_interval frames (~3/sec)
        #   - Motion-energy gate: skip static scenes (saves ~40-60% VLM calls)
        _run_macro = (
            processed > 0
            and len(tracked_boxes) > 0
            and processed % semantic_interval == 0
            and _motion_score >= MOTION_SKIP_THRESHOLD
        )

        if run_vlm and _run_macro:
            # Identify which vehicles are currently inside the zone (if any).
            # This drives both behavior_summary filtering and VLM visual focus.
            zone_occupants: frozenset = frozenset()
            if zone_manager is not None:
                zone_occupants = zone_manager.get_zone_occupants()

            # Pre-compute behavior_summary on the main thread (fast DuckDB read).
            # When a zone is active and occupied, only request kinematics for
            # zone vehicles — the VLM doesn't need out-of-zone narrative.
            active_ids = [int(t[4]) for t in tracked_boxes]
            if zone_occupants:
                summary_ids = [i for i in active_ids if i in zone_occupants]
            else:
                summary_ids = active_ids
            behavior_summary = duckdb_client.get_behavior_summary(summary_ids, timestamp)

            task = (
                frame.copy(),
                list(tracked_boxes),
                dict(state_vectors),
                frozenset(
                    tid for tid, q in state_quality.items()
                    if q.get("track_warm") and q.get("trusted_for_rules")
                ) if run_physics else frozenset(),
                timestamp,
                frame_id,
                dict(_vehicle_first_seen),
                behavior_summary,
                zone_occupants,           # NEW: frozenset of in-zone track IDs
                zone_config if zone_manager is not None else None,  # NEW: ZoneConfig or None
            )
            try:
                _vlm_queue.put_nowait(task)
                log.debug("[%.1fs] Enqueued semantic task (queue depth=%d)",
                          timestamp, _vlm_queue.qsize())
            except queue_module.Full:
                # VLM worker is still busy — drop this frame rather than block.
                log.debug("[%.1fs] VLM queue full — dropping frame %d",
                          timestamp, frame_id)

        processed += 1
        if progress_callback is not None:
            progress_callback(frame_id, total_frames)

        frame_id += 1 + grabbed

    cap.release()
    log.info("--- Video Processing Complete in %.2fs ---", time.time() - start_time)

    # Wait for VLM worker to drain and shut down gracefully
    if run_vlm:
        log.info("Waiting for VLM worker to finish remaining tasks...")
        _vlm_queue.put(None)  # sentinel
        _vlm_thread.join()

    # Safely close database connections
    duckdb_client.close()

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
