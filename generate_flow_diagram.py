"""
Methodology flow diagram for the Neuro-Symbolic Traffic Safety Analysis pipeline.
Run:  python generate_flow_diagram.py
Output: docs/methodology_flow.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs("docs", exist_ok=True)

FIG_W, FIG_H = 24, 18
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor="#0d0d1a")
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C = {
    "bg":       "#0d0d1a",
    "input":    "#c0392b",
    "physics":  "#1a6bb5",
    "queue":    "#e67e22",
    "semantic": "#27ae60",
    "memory":   "#8e44ad",
    "agent":    "#d4801a",
    "output":   "#2c7a7b",
    "alert":    "#e74c3c",
    "gate":     "#f39c12",
    "text":     "#f0f0f0",
    "dim":      "#888899",
    "thread_a": "#1a3a5c",
    "thread_b": "#1a3a28",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def box(ax, x, y, w, h, label, color, sublabel=None,
        fontsize=8.5, radius=0.22, alpha=0.93):
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.06,rounding_size={radius}",
        linewidth=1.3, edgecolor="#ffffff33",
        facecolor=color, alpha=alpha, zorder=4,
    )
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + 0.14, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=C["text"], zorder=5)
        ax.text(x, y - 0.16, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.8, color="#ffffffaa", zorder=5, style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=C["text"], zorder=5)
    return (x, y)


def diamond(ax, x, y, w, h, label, color, fontsize=7.5):
    dx, dy = w/2, h/2
    xs = [x, x+dx, x, x-dx, x]
    ys = [y+dy, y, y-dy, y, y+dy]
    ax.fill(xs, ys, color=color, alpha=0.92, zorder=4)
    ax.plot(xs, ys, color="#ffffff33", linewidth=1.0, zorder=5)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=C["text"], zorder=6)


def arr(ax, x1, y1, x2, y2, color="#ffffff66", lw=1.3, ls="-",
        rad=0.0, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}"), zorder=6)


def section_bg(ax, x, y, w, h, color, label, label_side="left"):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1,rounding_size=0.3",
        linewidth=1.5, edgecolor=color + "55",
        facecolor=color + "18", alpha=1.0, zorder=1,
    )
    ax.add_patch(rect)
    lx = x + 0.25 if label_side == "left" else x + w - 0.25
    ax.text(lx, y + h/2, label, ha="center", va="center",
            fontsize=7, fontweight="bold", color=color,
            rotation=90, alpha=0.9, zorder=2)


# ===========================================================================
# TITLE
# ===========================================================================
ax.text(FIG_W/2, 17.6,
        "A Neuro-Symbolic Approach to Real-Time Traffic Safety Analysis",
        ha="center", va="center",
        fontsize=15, fontweight="bold", color=C["text"], zorder=8)
ax.text(FIG_W/2, 17.2,
        "Methodology Flow  ·  Physics Micro-Loop (main thread, per frame)  +  Semantic Macro-Loop (VLM worker thread, ~3 Hz)",
        ha="center", va="center",
        fontsize=9, color=C["dim"], zorder=8)

# ===========================================================================
# SECTION BACKGROUNDS
# ===========================================================================
section_bg(ax, 0.4, 14.5,  10.2, 8.5,  C["physics"],  "PHYSICS MICRO-LOOP\n(Main Thread — every frame)")
section_bg(ax, 12.0, 10.2, 11.2, 4.8,  C["semantic"], "SEMANTIC MACRO-LOOP\n(VLM Worker Thread — ~3 Hz)")
section_bg(ax, 0.4,  3.8,  22.8, 3.2,  C["memory"],   "HYBRID MEMORY LAYER")
section_bg(ax, 0.4,  0.3,  22.8, 3.2,  C["agent"],    "AGENTIC ORCHESTRATOR")

# ===========================================================================
# ROW 0 — INPUT
# ===========================================================================
box(ax, 12, 16.8, 3.2, 0.65, "Raw Traffic Video", C["input"],
    sublabel="MP4 / AVI / MOV  ·  OpenCV VideoCapture", fontsize=9)

arr(ax, 12, 16.47, 12, 16.07)

# ===========================================================================
# ROW 1 — DETECT & TRACK  (shared by both loops)
# ===========================================================================
box(ax,  8.5, 15.8, 3.0, 0.60, "YOLO Detection",    C["physics"],
    sublabel="best.pt  ·  conf=0.3  ·  lru_cache")
box(ax, 15.5, 15.8, 3.0, 0.60, "ByteTrack Tracker", C["physics"],
    sublabel="IoU-based MOT  ·  persistent track_id")

ax.plot([8.5, 15.5], [16.07, 16.07], color="#ffffff55", lw=1.3, zorder=5)
arr(ax, 8.5,  16.07, 8.5,  16.1)
arr(ax, 15.5, 16.07, 15.5, 16.1)
arr(ax, 8.5,  15.5,  15.5, 15.5)    # YOLO → Tracker
arr(ax, 8.5,  15.5,  8.5,  15.1)    # YOLO → Homography
arr(ax, 15.5, 15.5, 15.5, 15.1)     # Tracker → Homography

ax.plot([8.5, 15.5], [15.1, 15.1], color="#ffffff55", lw=1.3, zorder=5)
arr(ax, 12, 15.1, 12, 14.77)

ax.text(4.5, 15.8, "tracked_boxes\n(x1,y1,x2,y2,\ntrack_id,conf,cls)",
        fontsize=6.5, color=C["dim"], ha="center", va="center")

# ===========================================================================
# ROW 2 — HOMOGRAPHY
# ===========================================================================
box(ax, 12, 14.5, 4.0, 0.60, "Homography Transform", C["physics"],
    sublabel="Pixel centroid → Real-world metres  ·  calibration.yaml")

arr(ax, 12, 14.2, 12, 13.82)

# ===========================================================================
# ROW 3 — KINEMATICS
# ===========================================================================
box(ax, 12, 13.55, 4.2, 0.60, "Kinematic Estimator", C["physics"],
    sublabel="Savitzky-Golay (window=15, poly=3)  →  [x, y, vx, vy, ax, ay]")

# fan out
arr(ax, 12,   13.25, 12,   12.82)                        # → DuckDB
arr(ax, 12,   13.25,  5.5, 12.52, rad=0.15)              # → Alert Engine
arr(ax, 12,   13.25, 18.8, 13.25, rad=0.0)               # → Zone Manager

# ===========================================================================
# ROW 4 — DuckDB  |  Alert Engine  |  Zone Manager
# ===========================================================================
box(ax, 12,  12.55, 3.2, 0.60, "DuckDB Insert",      C["memory"],
    sublabel="vehicle_trajectories  ·  flush / 100 frames")
box(ax,  5.5, 12.55, 3.0, 0.60, "Alert Engine",      C["alert"],
    sublabel="5 alert types  ·  warm_tracks only  ·  3 s cooldown")
box(ax, 19.0, 12.55, 2.8, 0.60, "Zone Manager",      C["physics"],
    sublabel="Gate crossings → DuckDB zone_crossings")

# Alert → DuckDB direct
arr(ax, 5.5, 12.25, 5.5, 11.5, rad=0.0, color=C["alert"])
ax.text(4.4, 11.85, "traffic_alerts\n(immediate)", fontsize=6.2,
        color=C["alert"], ha="center")

# ===========================================================================
# DIVIDER + QUEUE
# ===========================================================================
ax.plot([0.5, 23.5], [10.5, 10.5], color="#ffffff22", lw=1.2, ls="--", zorder=3)
ax.text(12, 10.38, "── Macro-loop trigger:  every  semantic_interval  frames  OR  force_vlm ──",
        ha="center", va="center", fontsize=7.5, color=C["dim"], zorder=7)

# Queue box
box(ax, 12, 11.1, 4.0, 0.60, "Bounded Task Queue", C["queue"],
    sublabel="queue.Queue(maxsize=4)  ·  put_nowait  ·  drop if full")

# DuckDB → behavior_summary → queue
arr(ax, 12, 12.25, 12, 11.4)
ax.text(13.3, 11.82, "behavior_summary\n(DuckDB read,\nmain thread)", fontsize=6,
        color=C["dim"], ha="center")

# force_vlm → queue
arr(ax, 5.5, 12.25, 10.0, 11.2, rad=-0.2, color=C["alert"], ls="--")
ax.text(7.2, 11.9, "force_vlm\n(COLLISION /\nHARD_BRAKING)",
        fontsize=6, color=C["alert"], ha="center")

# queue → VLM worker
arr(ax, 14.0, 11.1, 14.0, 10.85, color=C["queue"])

# ===========================================================================
# VLM WORKER THREAD
# ===========================================================================

# --- _id_buffer + _som_buffer ---
box(ax, 14.0, 10.55, 5.5, 0.60, "Frame Buffers (deque maxlen=6)", C["semantic"],
    sublabel="_som_buffer: SoM PIL frames  ·  _id_buffer: (timestamp, frozenset[IDs])")

arr(ax, 14.0, 10.25, 14.0, 9.85)

# --- SoM Renderer ---
box(ax, 14.0, 9.55, 4.8, 0.60, "Set-of-Mark Renderer", C["semantic"],
    sublabel="AdaptiveRenderer  ·  coloured ID badges drawn on frame")

arr(ax, 14.0, 9.25, 14.0, 8.85)

# --- id_constraint block ---
box(ax, 19.5, 9.55, 3.8, 0.85, "ID Constraint Builder",  C["semantic"],
    sublabel="all_active_ids = union(_id_buffer)\nframe_id_timeline per frame\nid_constraint → VLM prompt",
    fontsize=7.5)
arr(ax, 14.0, 10.25, 19.5, 9.97, rad=-0.2, color=C["semantic"], ls="--")
arr(ax, 19.5,  9.12, 19.5, 8.12, rad=0.0,  color=C["semantic"], ls="--")

# --- VLM ---
box(ax, 14.0, 8.55, 5.0, 0.65, "VLM Inference", C["semantic"],
    sublabel="Qwen2.5-VL-3B-Instruct  ·  6-frame video clip  ·  MRoPE fps=3\nCoT 2-step prompt  ·  id_constraint + physics_block + timeline")

arr(ax, 14.0, 8.22, 14.0, 7.85)

# id_constraint feeds VLM prompt
arr(ax, 19.5, 8.12, 16.5, 8.4, rad=0.1, color=C["semantic"], ls="--")

# --- Entity Extractor ---
box(ax, 14.0, 7.55, 5.0, 0.60, "Entity Extractor", C["semantic"],
    sublabel="qwen2.5:72b via Ollama  ·  Pydantic SPOTriple  ·  ID hallucination filter")

# fan to memory
arr(ax, 14.0, 7.25, 7.5,  4.95, rad=0.1)
arr(ax, 14.0, 7.25, 12.0, 4.95, rad=0.0)
arr(ax, 14.0, 7.25, 18.5, 4.95, rad=-0.1)

# ===========================================================================
# MEMORY LAYER
# ===========================================================================
box(ax,  4.5, 4.45, 4.0, 0.80, "DuckDB",               C["memory"],
    sublabel="vehicle_trajectories\ntraffic_alerts · zone_crossings")
box(ax, 10.0, 4.45, 3.8, 0.80, "Milvus — Events",      C["memory"],
    sublabel="NL event descriptions\n384-dim ANN vector store")
box(ax, 15.5, 4.45, 3.5, 0.80, "Kùzu Graph",           C["memory"],
    sublabel="SPO triples\nINTERACTS_WITH + PRECEDES")
box(ax, 21.0, 4.45, 3.0, 0.80, "Milvus — Profiles",   C["memory"],
    sublabel="Per-vehicle\nlongitudinal summaries")

# Alert Engine → DuckDB (memory)
arr(ax, 5.5, 11.2, 4.5, 4.85, rad=0.25, color=C["alert"])

# DuckDB (micro) → DuckDB (memory) label
ax.text(3.0, 8.8, "Physics data\nstored in\nDuckDB", fontsize=6.5,
        color=C["memory"], ha="center", va="center")

# all memory → agent
for mx in [4.5, 10.0, 15.5, 21.0]:
    arr(ax, mx, 4.05, mx, 3.48, color=C["memory"])

ax.plot([4.5, 21.0], [3.48, 3.48], color=C["memory"]+"88", lw=1.2, zorder=5)
arr(ax, 12.0, 3.48, 12.0, 3.28, color=C["memory"])

# ===========================================================================
# AGENTIC LAYER
# ===========================================================================
box(ax, 12.0, 3.0, 3.0, 0.50, "User Query", C["output"], fontsize=8.5)

arr(ax, 12.0, 2.75, 12.0, 2.35)

diamond(ax, 12.0, 2.1, 3.8, 0.55,
        "Hierarchical Router\ncosine-sim  ·  14+10 prototypes", C["agent"])

# full_analysis path
arr(ax, 10.1, 2.1, 7.5, 1.6, rad=0.15, color="#ffffff88")
ax.text(8.3, 2.05, "full_analysis\n(6 tools)", fontsize=7, color=C["agent"], ha="center")
box(ax, 6.0, 1.3, 4.8, 0.60, "LangGraph  — 6 Tools", C["agent"],
    sublabel="semantic_events · entity_profiles · graph\nphysics · rules · zone_flow",
    fontsize=7.5)

# semantic_lookup path
arr(ax, 13.9, 2.1, 16.5, 1.6, rad=-0.15, color="#ffffff88")
ax.text(16.0, 2.05, "semantic_lookup\n(2 tools)", fontsize=7, color=C["agent"], ha="center")
box(ax, 18.0, 1.3, 4.0, 0.60, "LangGraph  — 2 Tools", C["agent"],
    sublabel="semantic_events\nentity_profiles",
    fontsize=7.5)

# both → final answer
arr(ax,  6.0, 1.0,  6.0, 0.72)
arr(ax, 18.0, 1.0, 18.0, 0.72)
ax.plot([6.0, 18.0], [0.72, 0.72], color="#ffffff55", lw=1.3, zorder=5)
arr(ax, 12.0, 0.72, 12.0, 0.52)

box(ax, 12.0, 0.38, 5.0, 0.35,
    "Final Answer  (FastAPI SSE  /  CLI)", C["output"],
    fontsize=8.5, radius=0.15)

# ===========================================================================
# LEGEND
# ===========================================================================
items = [
    mpatches.Patch(color=C["input"],    label="Video Input"),
    mpatches.Patch(color=C["physics"],  label="Physics Micro-Loop (main thread)"),
    mpatches.Patch(color=C["alert"],    label="Alert Engine"),
    mpatches.Patch(color=C["queue"],    label="Task Queue (thread boundary)"),
    mpatches.Patch(color=C["semantic"], label="Semantic Macro-Loop (VLM worker thread)"),
    mpatches.Patch(color=C["memory"],   label="Hybrid Memory Layer"),
    mpatches.Patch(color=C["agent"],    label="Agentic Orchestrator (LangGraph)"),
    mpatches.Patch(color=C["output"],   label="User I/O"),
]
ax.legend(handles=items, loc="lower left", bbox_to_anchor=(0.01, 0.01),
          fontsize=7.5, framealpha=0.3, facecolor="#1a1a2e",
          edgecolor="#ffffff22", labelcolor=C["text"],
          ncol=2, handlelength=1.2)

# ===========================================================================
# SAVE
# ===========================================================================
out = "docs/methodology_flow.png"
plt.tight_layout(pad=0.2)
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out}")
