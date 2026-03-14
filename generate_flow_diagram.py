"""
Flow diagram generator for the TrafficAgent Neuro-Symbolic pipeline.
Run:  python generate_flow_diagram.py
Output: docs/methodology_flow.png  (3200×2000 px, 200 dpi)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

os.makedirs("docs", exist_ok=True)

# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 20, 14
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor="#0f0f1a")
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C = {
    "bg":       "#0f0f1a",
    "input":    "#c0392b",   # red     — video input
    "physics":  "#1a6bb5",   # blue    — micro-loop
    "semantic": "#27ae60",   # green   — macro-loop
    "memory":   "#8e44ad",   # purple  — memory layer
    "agent":    "#d4801a",   # orange  — agent
    "output":   "#2c7a7b",   # teal    — final answer
    "text":     "#f0f0f0",
    "dim":      "#a0a0b0",
    "border":   "#2a2a3a",
    "gate":     "#e67e22",   # gate/decision diamond colour
    "alert":    "#e74c3c",
}

# ---------------------------------------------------------------------------
# Helper: draw a rounded box and return its centre
# ---------------------------------------------------------------------------
def box(ax, x, y, w, h, label, color, sublabel=None,
        fontsize=8.5, radius=0.25, alpha=0.92, text_color="#f0f0f0"):
    fc = color + "cc"   # slight transparency via hex alpha isn't reliable;
                        # use alpha= parameter instead
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.08,rounding_size={radius}",
        linewidth=1.2,
        edgecolor="#ffffff44",
        facecolor=color,
        alpha=alpha,
        zorder=3,
    )
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + 0.13, label,
                ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                color=text_color, zorder=4)
        ax.text(x, y - 0.17, sublabel,
                ha="center", va="center",
                fontsize=fontsize - 1.5,
                color="#ffffffaa", zorder=4, style="italic")
    else:
        ax.text(x, y, label,
                ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                color=text_color, zorder=4)
    return (x, y)


# ---------------------------------------------------------------------------
# Helper: draw a diamond (decision / gate)
# ---------------------------------------------------------------------------
def diamond(ax, x, y, w, h, label, color, fontsize=7.5):
    dx, dy = w / 2, h / 2
    xs = [x,      x + dx, x,      x - dx, x]
    ys = [y + dy, y,      y - dy, y,      y + dy]
    ax.fill(xs, ys, color=color, alpha=0.9, zorder=3)
    ax.plot(xs, ys, color="#ffffff44", linewidth=1.0, zorder=4)
    ax.text(x, y, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color=C["text"], zorder=5)
    return (x, y)


# ---------------------------------------------------------------------------
# Helper: arrow
# ---------------------------------------------------------------------------
def arrow(ax, x1, y1, x2, y2, color="#ffffff55", lw=1.2,
          style="->", connectionstyle="arc3,rad=0.0"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            connectionstyle=connectionstyle,
        ),
        zorder=5,
    )


# ---------------------------------------------------------------------------
# Helper: section label (left-side band annotation)
# ---------------------------------------------------------------------------
def band_label(ax, y_center, label, color):
    ax.text(0.22, y_center, label,
            ha="center", va="center",
            fontsize=7.5, fontweight="bold",
            color=color, rotation=90, alpha=0.85, zorder=6)
    ax.plot([0.42, 0.42], [y_center - 0.9, y_center + 0.9],
            color=color, linewidth=2.5, alpha=0.5, zorder=3)


# ===========================================================================
# TITLE
# ===========================================================================
ax.text(FIG_W / 2, 13.55,
        "TrafficAgent — Neuro-Symbolic Pipeline: Methodology Flow",
        ha="center", va="center",
        fontsize=14, fontweight="bold", color=C["text"], zorder=6)

ax.text(FIG_W / 2, 13.15,
        "Dual-Loop Architecture  |  Physics Micro-Loop (per frame)  +  Semantic Macro-Loop (~3 Hz)",
        ha="center", va="center",
        fontsize=9, color=C["dim"], zorder=6)


# ===========================================================================
# ROW 0 — INPUT
# ===========================================================================
box(ax, 10, 12.55, 2.8, 0.65, "Raw Traffic Video", C["input"],
    sublabel="OpenCV VideoCapture", fontsize=9.5)

band_label(ax, 12.55, "INPUT", C["input"])

arrow(ax, 10, 12.22, 10, 11.82)

# ===========================================================================
# ROW 1 — DETECT & TRACK  (micro-loop starts)
# ===========================================================================
band_label(ax, 11.35, "PHYSICS\nMICRO-LOOP\n(per frame)", C["physics"])

# YOLO + Tracker side by side
box(ax,  7.5, 11.35, 2.6, 0.65, "YOLO Detection",       C["physics"],
    sublabel="yolov8n.pt  conf=0.3")
box(ax, 12.5, 11.35, 2.6, 0.65, "ByteTrack Tracker",    C["physics"],
    sublabel="Appearance-free IoU MOT")

# horizontal join line at top, then arrow down from video to YOLO
arrow(ax, 10, 11.82, 7.5, 11.82, connectionstyle="arc3,rad=0.0")
arrow(ax, 10, 11.82, 12.5, 11.82, connectionstyle="arc3,rad=0.0")
arrow(ax,  7.5, 11.82,  7.5, 11.68)
arrow(ax, 12.5, 11.82, 12.5, 11.68)
# YOLO → Tracker
arrow(ax,  8.8, 11.35, 11.2, 11.35)

# both converge down to Homography
arrow(ax,  7.5, 11.02,  7.5, 10.62, connectionstyle="arc3,rad=0.0")
arrow(ax, 12.5, 11.02, 12.5, 10.62, connectionstyle="arc3,rad=0.0")
arrow(ax,  7.5, 10.62,  10, 10.62, connectionstyle="arc3,rad=0.0")
arrow(ax, 12.5, 10.62,  10, 10.62, connectionstyle="arc3,rad=0.0")

# ===========================================================================
# ROW 2 — HOMOGRAPHY
# ===========================================================================
box(ax, 10, 10.35, 3.4, 0.65, "Homography Transform",  C["physics"],
    sublabel="Pixel bottom-centre → Real-world metres (calibration.yaml)")

arrow(ax, 10, 10.02, 10, 9.62)

# ===========================================================================
# ROW 3 — KINEMATICS
# ===========================================================================
box(ax, 10, 9.35, 3.4, 0.65, "Kinematic Estimator",    C["physics"],
    sublabel="Savitzky-Golay (window=15, poly=3)  →  [x,y,vx,vy,ax,ay]")

# fan out to three targets
arrow(ax, 10, 9.02, 10, 8.62)                    # centre → DuckDB
arrow(ax, 10, 9.02,  5.2, 8.12, connectionstyle="arc3,rad=0.15")  # left → Alert
arrow(ax, 10, 9.02, 15.4, 8.62, connectionstyle="arc3,rad=-0.1")  # right → Zone

# ===========================================================================
# ROW 4a — DuckDB  |  Alert Engine  |  Zone Manager
# ===========================================================================
box(ax, 10, 8.35, 3.0, 0.60, "DuckDB Insert",          C["memory"],
    sublabel="vehicle_trajectories  (flush/100 frames)")

box(ax,  5.2, 8.35, 2.8, 0.60, "Alert Engine",         C["alert"],
    sublabel="5 types  |  3 s cooldown  |  warm tracks only")

box(ax, 15.4, 8.35, 2.8, 0.60, "Zone Manager",         C["physics"],
    sublabel="Gate crossings → DuckDB zone_crossings")

# Alert → force-VLM flag (dashed, labeled)
ax.annotate("", xy=(13.5, 7.55), xytext=(6.4, 8.05),
            arrowprops=dict(arrowstyle="->", color=C["alert"],
                            lw=1.3, linestyle="dashed",
                            connectionstyle="arc3,rad=-0.25"), zorder=5)
ax.text(9.5, 7.60, "force_vlm = True\n(COLLISION / HARD_BRAKING)",
        fontsize=6.5, color=C["alert"], ha="center", zorder=6)

# DuckDB → down to motion gate
arrow(ax, 10, 8.05, 10, 7.72)

# ===========================================================================
# DIVIDER — micro → macro
# ===========================================================================
ax.plot([0.5, 19.5], [7.55, 7.55],
        color="#ffffff22", linewidth=1.2, linestyle="--", zorder=2)
ax.text(10, 7.45, "── Macro-loop trigger: every  semantic_interval  frames  OR  force_vlm ──",
        ha="center", va="center", fontsize=7, color=C["dim"], zorder=6)

band_label(ax, 7.0, "SEMANTIC\nMACRO-LOOP\n(~3 Hz)", C["semantic"])

# ===========================================================================
# ROW 5 — Motion Gate (diamond)
# ===========================================================================
diamond(ax, 10, 7.15, 2.8, 0.55,
        "Motion Energy Gate\nmean|Δframe| ≥ threshold ?", C["gate"])

# YES → down
arrow(ax, 10, 6.88, 10, 6.52)
ax.text(10.18, 6.70, "yes", fontsize=7, color=C["gate"])

# NO → skip label (right)
ax.annotate("", xy=(17.5, 7.15), xytext=(11.4, 7.15),
            arrowprops=dict(arrowstyle="->", color="#ffffff44",
                            lw=1.0, linestyle="dotted",
                            connectionstyle="arc3,rad=0.0"), zorder=5)
ax.text(14.5, 7.28, "no → skip (static scene)", fontsize=7,
        color=C["dim"], ha="center")

# ===========================================================================
# ROW 6 — SoM Renderer
# ===========================================================================
box(ax, 10, 6.25, 3.4, 0.60, "Set-of-Mark Renderer",   C["semantic"],
    sublabel="AdaptiveRenderer  → coloured ID badges on frame  → deque(maxlen=6)")

arrow(ax, 10, 5.95, 10, 5.58)

# ===========================================================================
# ROW 7 — VLM
# ===========================================================================
box(ax, 10, 5.30, 4.0, 0.60, "VLM Inference",           C["semantic"],
    sublabel="Qwen2.5-VL-3B  |  6-frame clip  |  MRoPE fps=3  |  CoT 2-step prompt")

arrow(ax, 10, 5.00, 10, 4.62)

# ===========================================================================
# ROW 8 — Entity Extractor
# ===========================================================================
box(ax, 10, 4.35, 3.8, 0.60, "Entity Extractor",        C["semantic"],
    sublabel="qwen2.5:72b via Ollama  |  Pydantic SPOTriple  |  ID hallucination filter")

# fan out to Milvus-events, Kùzu, Milvus-profiles
arrow(ax, 10, 4.05,  5.5, 3.62, connectionstyle="arc3,rad=0.12")
arrow(ax, 10, 4.05, 10.0, 3.62)
arrow(ax, 10, 4.05, 15.2, 3.62, connectionstyle="arc3,rad=-0.12")

# ===========================================================================
# ROW 9 — Memory Layer
# ===========================================================================
band_label(ax, 3.35, "MEMORY\nLAYER", C["memory"])

box(ax,  5.5, 3.35, 3.2, 0.60, "Milvus — traffic_events",  C["memory"],
    sublabel="384-dim ANN  |  NL event descriptions")

box(ax, 10.0, 3.35, 3.2, 0.60, "Kùzu Graph",               C["memory"],
    sublabel="INTERACTS_WITH + PRECEDES\n(SPO triples, typed nodes)")

box(ax, 15.2, 3.35, 3.2, 0.60, "Milvus — entity_profiles", C["memory"],
    sublabel="384-dim ANN  |  per-vehicle\nlongitudinal summaries")

# Alert storage
arrow(ax,  5.2, 8.05,  5.5, 3.65, connectionstyle="arc3,rad=0.25")
ax.text(3.0, 6.1, "traffic_alerts\n→ DuckDB\n(immediate)", fontsize=6,
        color=C["alert"], ha="center")

# all three memory → converge to agent
arrow(ax,  5.5, 3.05,  5.5, 2.62, connectionstyle="arc3,rad=0.0")
arrow(ax, 10.0, 3.05, 10.0, 2.62)
arrow(ax, 15.2, 3.05, 15.2, 2.62, connectionstyle="arc3,rad=0.0")

arrow(ax,  5.5, 2.62,  8.8, 2.62, connectionstyle="arc3,rad=0.0")
arrow(ax, 15.2, 2.62, 11.2, 2.62, connectionstyle="arc3,rad=0.0")
arrow(ax, 10.0, 2.62, 10.0, 2.38)

# DuckDB (already in memory layer) arrow too
arrow(ax, 10, 8.05, 10.0, 3.65, connectionstyle="arc3,rad=0.3")
ax.text(12.6, 6.35, "DuckDB\n(trajectories\n+ alerts)", fontsize=6,
        color=C["memory"], ha="center")


# ===========================================================================
# ROW 10 — Hierarchical Router
# ===========================================================================
band_label(ax, 2.0, "AGENTIC\nLAYER", C["agent"])

box(ax, 10, 2.62, 2.6, 0.50, "User Query", C["output"],
    sublabel=None, fontsize=8.5, radius=0.20)
# (arrow already drawn above from memory to query box)

arrow(ax, 10, 2.37, 10, 2.02)

diamond(ax, 10, 1.78, 3.4, 0.55,
        "Hierarchical Router\ncosine-sim vs 14+10 prototypes", C["agent"])

# full_analysis → left
ax.annotate("", xy=(5.2, 1.30), xytext=(8.3, 1.78),
            arrowprops=dict(arrowstyle="->", color="#ffffff77", lw=1.2,
                            connectionstyle="arc3,rad=0.2"), zorder=5)
ax.text(6.0, 1.72, "full_analysis", fontsize=7, color=C["agent"])

# semantic_lookup → right
ax.annotate("", xy=(14.8, 1.30), xytext=(11.7, 1.78),
            arrowprops=dict(arrowstyle="->", color="#ffffff77", lw=1.2,
                            connectionstyle="arc3,rad=-0.2"), zorder=5)
ax.text(13.7, 1.72, "semantic_lookup", fontsize=7, color=C["agent"])

# ===========================================================================
# ROW 11 — LangGraph  (two tool-set variants)
# ===========================================================================
box(ax,  5.2, 1.05, 4.4, 0.55,
    "LangGraph  — 6 tools",        C["agent"],
    sublabel="semantic_events · entity_profiles · graph\nphysics · rules · zone_flow",
    fontsize=7.5)

box(ax, 14.8, 1.05, 4.0, 0.55,
    "LangGraph  — 2 tools",        C["agent"],
    sublabel="semantic_events · entity_profiles",
    fontsize=7.5)

# both → converge to final answer
arrow(ax,  5.2, 0.78,  5.2, 0.42, connectionstyle="arc3,rad=0.0")
arrow(ax, 14.8, 0.78, 14.8, 0.42, connectionstyle="arc3,rad=0.0")
arrow(ax,  5.2, 0.42, 10.0, 0.42, connectionstyle="arc3,rad=0.0")
arrow(ax, 14.8, 0.42, 10.0, 0.42, connectionstyle="arc3,rad=0.0")
arrow(ax, 10.0, 0.42, 10.0, 0.22)

# ===========================================================================
# FINAL OUTPUT
# ===========================================================================
box(ax, 10, 0.12, 4.0, 0.30, "Final Answer (via FastAPI SSE / CLI)",
    C["output"], fontsize=8, radius=0.15)

# ===========================================================================
# LEGEND
# ===========================================================================
legend_items = [
    mpatches.Patch(color=C["input"],    label="Video Input"),
    mpatches.Patch(color=C["physics"],  label="Physics Micro-loop"),
    mpatches.Patch(color=C["alert"],    label="Alert Engine"),
    mpatches.Patch(color=C["semantic"], label="Semantic Macro-loop"),
    mpatches.Patch(color=C["memory"],   label="Memory Layer"),
    mpatches.Patch(color=C["agent"],    label="Agentic Orchestrator"),
    mpatches.Patch(color=C["gate"],     label="Decision / Gate"),
    mpatches.Patch(color=C["output"],   label="Input / Output"),
]
ax.legend(
    handles=legend_items,
    loc="lower left",
    bbox_to_anchor=(0.01, 0.01),
    fontsize=7.5,
    framealpha=0.25,
    facecolor="#1a1a2a",
    edgecolor="#ffffff33",
    labelcolor=C["text"],
    ncol=2,
    handlelength=1.2,
)

# ===========================================================================
# SAVE
# ===========================================================================
out_path = "docs/methodology_flow.png"
plt.tight_layout(pad=0.3)
plt.savefig(out_path, dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"Saved → {out_path}")
