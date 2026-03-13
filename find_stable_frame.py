"""
Scan the first N frames of a traffic video and find the first stable frame
where camera jitter has settled.

Method: compute mean absolute pixel difference between consecutive grayscale
frames (same motion-energy metric used in main.py for VLM gating).
A frame is considered "stable" when the motion score drops below STABLE_THRESHOLD
and stays below it for STABLE_WINDOW consecutive frames.
"""

import sys
import cv2
import numpy as np


STABLE_THRESHOLD = 8.0   # mean absolute pixel diff; tune if needed
STABLE_WINDOW    = 10    # consecutive frames below threshold = confirmed stable
MAX_SCAN_FRAMES  = 500   # scan at most this many frames from the start


def find_stable_start(video_path: str) -> int:
    """
    Returns the 0-based frame index of the first stable frame.
    If no jitter is detected the function returns 0.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n{'='*60}")
    print(f"Video : {video_path}")
    print(f"FPS   : {fps:.1f}   Total frames: {total_frames}")
    print(f"{'='*60}")

    prev_gray      = None
    scores         = []
    stable_start   = 0          # best guess: start from 0
    consecutive    = 0
    candidate      = None

    for frame_idx in range(min(MAX_SCAN_FRAMES, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if prev_gray is not None:
            diff  = np.mean(np.abs(gray - prev_gray))
            scores.append((frame_idx, diff))

            if diff < STABLE_THRESHOLD:
                if candidate is None:
                    candidate = frame_idx          # tentative start of stable zone
                consecutive += 1
                if consecutive >= STABLE_WINDOW:
                    stable_start = candidate
                    print(f"  Stable zone found at frame {stable_start} "
                          f"({stable_start / fps:.2f}s)")
                    break
            else:
                consecutive = 0
                candidate   = None

        prev_gray = gray

    cap.release()

    # Print motion-score summary for first 60 measured frames
    print(f"\n  Motion scores (frame → mean-abs-diff):")
    for idx, score in scores[:60]:
        bar   = "#" * int(score / 2)
        flag  = "  ← JITTER" if score >= STABLE_THRESHOLD else ""
        print(f"    frame {idx:4d}: {score:6.2f}  {bar}{flag}")
    if len(scores) > 60:
        print(f"    ... ({len(scores) - 60} more frames not shown)")

    print(f"\n  → Recommended start_frame: {stable_start}  "
          f"({stable_start / fps:.2f}s into video)\n")
    return stable_start


if __name__ == "__main__":
    videos = [
        "data/ulloor/20250523_162245_tp00026.mp4",
        "data/ulloor/20250523_154828_tp00025.mp4",
        "data/ulloor/20250523_165701_tp00027.mp4",
    ]

    if len(sys.argv) > 1:
        videos = sys.argv[1:]

    results = {}
    for v in videos:
        try:
            sf = find_stable_start(v)
            results[v] = sf
        except Exception as e:
            print(f"ERROR processing {v}: {e}")

    print("\n" + "="*60)
    print("SUMMARY — recommended start_frame per video:")
    for v, sf in results.items():
        short = v.split("/")[-1]
        print(f"  {short:40s}  frame {sf}")
    print("="*60)
