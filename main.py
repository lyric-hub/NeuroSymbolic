import cv2
import time
import json
from pathlib import Path
from PIL import Image

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
from src.agentic_orchestrator.sequential_pipeline import agent_app

def process_video(video_path: str, progress_callback=None):
    """
    Executes the dual-loop Neuro-Symbolic tracking and abstraction pipeline.
    High-frequency loop runs every frame. Low-frequency loop runs at ~3 VLM
    calls/second, derived from the video's actual frame rate.

    Args:
        video_path:         Path to the input video file.
        progress_callback:  Optional callable(frames_done: int, total_frames: int)
                            invoked once per frame so callers can track progress.
    """
    print("Initializing Neuro-Symbolic Pipeline...")

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
    print(f"Video: {fps} fps | {total_frames} frames | VLM every {semantic_interval} frames (~3/sec)")

    # 2. Initialize Hybrid Databases
    duckdb_client = DuckDBClient()
    milvus_client = SemanticVectorStore()
    graph_client = GraphClient()

    # 3. Initialize Physics Engine (Micro-Loop) with correct fps
    detector = load_detector("yolov8n.pt", conf=0.3)
    tracker = VehicleTracker(tracker_name="bytetrack")
    transformer = CoordinateTransformer("calibration.yaml")
    kinematics = KinematicEstimator(fps=float(fps))

    # Zone manager is optional — only active when zone_config.json exists.
    # Draw zones at /zone-ui before running the pipeline.
    zone_manager = None
    if Path("zone_config.json").exists():
        zone_config = ZoneConfig.from_json("zone_config.json")
        zone_manager = ZoneManager(zone_config)
        print(f"Zone '{zone_config.zone_id}' active — gates: {[g.name for g in zone_config.gates]}")

    # 4. Initialize Semantic Abstractor (Macro-Loop)
    renderer = AdaptiveRenderer()
    vlm = TrafficSemanticAbstractor(model_id="Qwen/Qwen2.5-VL-3B-Instruct")
    extractor = EntityExtractor(model_name="qwen2.5:72b")

    frame_id = 0
    start_time = time.time()

    print("\n--- Starting Video Processing ---")
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

        # 5. Zone crossing detection (skipped if no zone_config.json)
        if zone_manager is not None:
            crossing_events = zone_manager.update(
                tracked_boxes, real_coords, timestamp, frame_id
            )
            for event in crossing_events:
                duckdb_client.insert_crossing_event(event)
                print(f"[{timestamp:.1f}s] Vehicle {event.track_id} → {event.direction} via {event.gate_name}")

        # ==========================================
        # THE MACRO-LOOP (Low-Frequency Semantics)
        # ==========================================
        
        if frame_id % semantic_interval == 0 and len(tracked_boxes) > 0:
            print(f"[{timestamp:.1f}s] Running Semantic Abstraction...")
            
            # Define chunking window
            chunk_start = max(0.0, timestamp - (semantic_interval/fps) - 2.0)
            time_window_ptr = f"{chunk_start:.1f}-{timestamp:.1f}"
            
            # 1. Visual Grounding (Set-of-Mark)
            som_frame = frame.copy()
            render_ctx = RenderContext()
            render_ctx.update(tracked_boxes, timestamp)
            renderer.render(som_frame, render_ctx)
            
            # Convert to PIL Image for the VLM
            som_pil = Image.fromarray(cv2.cvtColor(som_frame, cv2.COLOR_BGR2RGB))
            
            # 2. VLM Inference
            # state_vectors passed here so the VLM receives real-world position,
            # speed, and acceleration — giving it spatial and motion awareness
            # that cannot be derived from a single frozen frame.
            vlm_triples = vlm.generate_scene_graph_triples(
                som_pil, timestamp, state_vectors, kinematics.warm_tracks
            )
            
            if vlm_triples:
                # Convert triples to natural language for semantically meaningful
                # embeddings. JSON syntax degrades vector quality; readable sentences
                # align with the natural language queries issued at retrieval time.
                nl_description = " ".join(
                    f"{t['subject']} {t['predicate']} {t['object']}."
                    for t in vlm_triples
                )
                milvus_client.insert_event_chunk(
                    nl_description, chunk_start, timestamp, frame_id
                )

                # Keep the original structured JSON for the EntityExtractor —
                # qwen2.5:72b needs the SPO keys to populate the SPOTriple schema.
                scene_description = json.dumps(vlm_triples)

                # 3. Entity Extraction & Strict Validation
                # Pass through the local LLM to guarantee schema before Graph DB insertion
                validated_triples = extractor.extract_triples(scene_description, timestamp)
                
                if validated_triples:
                    graph_client.insert_vlm_triples(validated_triples, time_window_ptr)

        if progress_callback is not None:
            progress_callback(frame_id, total_frames)

        frame_id += 1

    cap.release()
    print(f"--- Video Processing Complete in {time.time() - start_time:.2f}s ---")
    
    # Safely close database connections
    duckdb_client.close()
    milvus_client.close()
    graph_client.close()

def interactive_agent_loop():
    """Boots up the LangGraph agent to query the processed hybrid databases."""
    print("\n=============================================")
    print("🧠 Neuro-Symbolic Agentic Brain Initialized")
    print("=============================================")
    print("Ask questions about the traffic event (e.g., 'Did Vehicle 4 brake too hard?').")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("User >> ")
        if query.lower() in ['exit', 'quit']:
            break
            
        initial_state = {"query": query}
        final_state = agent_app.invoke(initial_state)
        
        print(f"\nAgent >> {final_state.get('final_summary', 'No summary generated.')}\n")

if __name__ == "__main__":
    SAMPLE_VIDEO = "data/raw_videos/traffic_sample.mp4"
    
    if not Path("calibration.yaml").exists():
        print("Error: calibration.yaml not found!")
        print("Please open the web calibration tool at /calibrate-ui and complete calibration first.")
        exit(1)
        
    process_video(SAMPLE_VIDEO, progress_callback=lambda done, total: print(f"  [{done}/{total}] frames processed") if done % 300 == 0 else None)
    interactive_agent_loop()