"""
Quick SoM rendering test — bypasses load_detector() to avoid auto-loading
the TensorRT engine when TensorRT is not installed.

Usage:
    python test_som.py
Output:
    /tmp/som_test.mp4  (or som_test.mp4 in project root if /tmp write fails)
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Force the .pt model directly — skip engine sibling check
MODEL_PT   = "models/best.pt"
VIDEO_IN   = "data/raw_videos/20230426_124420_tp00024.mp4"
VIDEO_OUT  = "/tmp/som_test.mp4"
CONF       = 0.3
MAX_FRAMES = 300   # ~10 s at 30 fps — enough to see the overlay; remove for full video

sys.path.insert(0, str(Path(__file__).parent))

from src.physics_engine.tracker import VehicleTracker
from src.semantic_abstractor.set_of_mark import AdaptiveRenderer, RenderContext


def main():
    print(f"Loading YOLO from {MODEL_PT} (PT, no TensorRT) ...")
    model = YOLO(MODEL_PT)

    print("Loading tracker (bytetrack) ...")
    tracker = VehicleTracker(tracker_name="bytetrack")

    renderer = AdaptiveRenderer()
    ctx      = RenderContext()

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {VIDEO_IN}")

    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit  = min(MAX_FRAMES, total) if MAX_FRAMES else total

    print(f"Video: {width}x{height} @ {fps} fps — writing first {limit} frames to {VIDEO_OUT}")

    out = cv2.VideoWriter(
        VIDEO_OUT,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    for frame_id in range(limit):
        ret, frame = cap.read()
        if not ret:
            break

        # Detection (direct YOLO call, not via factory)
        results = model.predict(
            source=frame,
            conf=CONF,
            iou=0.7,
            verbose=False,
            device=0,   # cuda:0
        )

        # Tracking
        tracked_boxes = tracker.update(results[0], frame)

        # SoM rendering
        timestamp = frame_id / fps
        ctx.update(tracked_boxes, timestamp)
        renderer.render(frame, ctx)

        out.write(frame)

        if frame_id % 30 == 0:
            n_tracks = len(tracked_boxes)
            print(f"  frame {frame_id:4d}/{limit}  tracks={n_tracks}")

    cap.release()
    out.release()
    print(f"\nDone. Output saved to: {VIDEO_OUT}")
    print("Open it in VS Code or any video player.")


if __name__ == "__main__":
    main()
