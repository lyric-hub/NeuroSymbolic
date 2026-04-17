"""
VLM inference module for semantic scene abstraction.

Replaced Qwen/Qwen2.5-VL-3B-Instruct (HuggingFace) with Gemma 4 E2B
served through the local Ollama instance — the same backend already
used by the agentic orchestrator (sequential_pipeline.py).

Architectural benefits of the Ollama path:
1. Single model server: Ollama manages GPU memory for both the agent LLM
   and the VLM.  No competing HuggingFace process on the same GPU.
2. No torch/transformers in this process: cleaner separation of concerns.
3. Multimodal input via base64 image encoding — Ollama's native format.
4. Multi-frame temporal context conveyed via per-frame timeline text block
   (replaces Qwen's MRoPE native video format).

Chain-of-Thought + enriched SPO schema are preserved unchanged.

References:
  - Gemma 4 Technical Report (google/gemma-4-E2B-it)
  - TrafficVLM temporal phase modelling (arXiv 2404.09275)
  - DriveVLM dual-system architecture (arXiv 2402.12289)
"""

import base64
import io
import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

log = logging.getLogger(__name__)

_REQUIRED_KEYS = {"subject", "predicate", "object"}

# Valid motion state labels injected into VLM instructions.
_MOTION_STATES = {"APPROACHING", "DIVERGING", "PARALLEL", "STATIONARY"}

# Valid interaction phase labels.
_PHASES = {"approach", "conflict", "resolution", "normal"}

# Default Ollama model tag — matches the agent LLM in sequential_pipeline.py.
_DEFAULT_MODEL = "gemma4:e2b"


def _pil_to_base64(image: Image.Image) -> str:
    """
    Encode a PIL Image as a base64 JPEG string for Ollama multimodal input.

    JPEG at quality=85 balances visual fidelity with token efficiency.
    SoM badges remain legible at this quality.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class TrafficSemanticAbstractor:
    """
    Wraps Gemma 4 E2B (via local Ollama) to perform low-frequency semantic
    scene abstraction.

    Converts Set-of-Mark (SoM) overlaid frame sequences into structured
    Subject-Predicate-Object triples enriched with motion state and
    interaction phase labels.

    Multi-frame mode (preferred):
        Pass a list of 2–6 consecutive SoM PIL Images.  All frames are
        encoded as base64 and sent in a single multimodal message so the
        model can reason across temporal context.

    Single-frame mode (fallback):
        Pass a list with a single PIL Image.  Behaviour is identical.
    """

    def __init__(self, model_id: str = _DEFAULT_MODEL) -> None:
        log.info("Connecting to Ollama VLM: %s", model_id)
        self._llm = ChatOllama(model=model_id, temperature=0.0)

    # ------------------------------------------------------------------
    # Physics block helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_physics_block(
        state_vectors: Dict[int, List[float]],
        warm_tracks: Set[int],
        reference_name: str = "Origin",
        nearest_landmark_fn=None,
    ) -> str:
        """
        Converts kinematics state_vectors to a readable text block.

        When nearest_landmark_fn is provided (from CoordinateTransformer),
        positions are described as "X.Xm from <closest landmark name>".
        Otherwise falls back to cardinal-direction format relative to origin.

        Tracks not yet in warm_tracks have unreliable velocity/acceleration
        estimates (fewer than window_length samples). They are labelled
        "(initialising)" to prevent the VLM from treating a cold-start
        zero velocity as a genuine stop.
        """
        def _pos_str(x: float, y: float) -> str:
            if nearest_landmark_fn is not None:
                lm_name, lm_dist = nearest_landmark_fn(x, y)
                return f"{lm_dist:.1f}m from {lm_name}"
            x_dir = "E" if x >= 0 else "W"
            y_dir = "N" if y >= 0 else "S"
            return (
                f"{abs(x):.1f}m {x_dir}, {abs(y):.1f}m {y_dir} "
                f"of {reference_name}"
            )

        lines = []
        for track_id, sv in state_vectors.items():
            x, y, vx, vy, ax, ay = sv
            pos = _pos_str(x, y)
            if track_id not in warm_tracks:
                lines.append(
                    f"  Vehicle {track_id}: "
                    f"position=({pos}), "
                    f"speed=(initialising — not enough frames yet), "
                    f"acceleration=(initialising)"
                )
                continue
            speed = (vx ** 2 + vy ** 2) ** 0.5
            accel = (ax ** 2 + ay ** 2) ** 0.5
            dot = vx * ax + vy * ay
            signed_accel = -accel if dot < 0 else accel
            lines.append(
                f"  Vehicle {track_id}: "
                f"position=({pos}), "
                f"speed={speed:.1f} m/s, "
                f"acceleration={signed_accel:+.1f} m/s²"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Main inference entry point
    # ------------------------------------------------------------------

    def generate_scene_graph_triples(
        self,
        frame_buffer: List[Image.Image],
        timestamp: float,
        state_vectors: Optional[Dict[int, List[float]]] = None,
        warm_tracks: Optional[Set[int]] = None,
        behavior_summary: Optional[str] = None,
        fps: float = 3.0,
        tracked_boxes=None,
        all_active_ids: Optional[List[int]] = None,
        frame_id_timeline: Optional[List[tuple]] = None,
        reference_name: str = "Origin",
        nearest_landmark_fn=None,
        zone_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generates enriched SPO triples from a sequence of SoM frames.

        Args:
            frame_buffer:     List of 1–6 consecutive SoM PIL Images
                              (most recent last).  All frames are base64
                              encoded and sent together so the model sees
                              temporal evolution across the clip.
            timestamp:        Video timestamp of the most recent frame (s).
            state_vectors:    Optional {track_id: [x,y,vx,vy,ax,ay]} from
                              KinematicEstimator.
            warm_tracks:      Set of track IDs with full Savitzky-Golay
                              coverage (unreliable IDs labelled initialising).
            behavior_summary: Change-only motion narrative from DuckDB
                              (last 5 s).  Preferred over state_vectors
                              snapshot when available.
            fps:              Frame rate of the buffer (retained for API
                              compatibility — not used by Ollama path).
            zone_context:     Optional dict with 'zone_id' and 'occupant_ids'
                              for zone-focused analysis.

        Returns:
            List of enriched triple dicts:
            ``[{"subject": ..., "predicate": ..., "object": ...,
               "motion_state": ..., "phase": ..., "timestamp": ...}]``
        """
        if not frame_buffer:
            return []

        # --- Physics context -------------------------------------------------
        physics_block = ""
        if behavior_summary:
            physics_block = (
                "\nVerified vehicle behaviour history (last 5 s) from the "
                "tracking engine (use this to ground your analysis — do not "
                "contradict it):\n"
                + behavior_summary
                + "\n"
            )
        elif state_vectors:
            physics_block = (
                "\nVerified physics data from the tracking engine "
                "(use this to ground your analysis — do not contradict it):\n"
                + self._build_physics_block(
                    state_vectors, warm_tracks or set(),
                    reference_name, nearest_landmark_fn,
                )
                + "\n"
            )

        # --- Zone context ----------------------------------------------------
        zone_block = ""
        if zone_context:
            zid = zone_context.get("zone_id", "")
            occupants = sorted(zone_context.get("occupant_ids", []))
            n = len(occupants)
            zone_block = (
                f"\n## Zone of Interest: '{zid}'\n"
                f"The frames show a highlighted zone (bright border, dimmed exterior). "
                f"There {'is' if n == 1 else 'are'} currently {n} "
                f"vehicle{'s' if n != 1 else ''} inside: {occupants}.\n"
                "Describe ONLY interactions within or at the boundary of this zone. "
                "Ignore vehicles outside the highlighted area.\n"
            )

        # --- ID constraint ---------------------------------------------------
        if all_active_ids is not None:
            active_ids = sorted(all_active_ids)
        elif tracked_boxes is not None:
            active_ids = sorted({int(t[4]) for t in tracked_boxes})
        elif state_vectors:
            active_ids = sorted(state_vectors.keys())
        else:
            active_ids = []

        # Per-frame presence timeline — prevents the VLM from describing
        # interactions between vehicles that were never in the same frame.
        timeline_block = ""
        if frame_id_timeline:
            lines = ["Vehicle presence per frame in this clip:"]
            for i, (ts, ids) in enumerate(frame_id_timeline, 1):
                marker = " <- current" if i == len(frame_id_timeline) else ""
                lines.append(
                    f"  Frame {i} (t={ts:.2f}s): Vehicles {ids}{marker}"
                )
            timeline_block = "\n".join(lines) + "\n"

        id_constraint = (
            f"The ONLY valid vehicle IDs across this clip are: {active_ids}. "
            "Use ONLY these exact IDs. "
            + timeline_block
        ) if active_ids else ""

        # --- System prompt: CoT + richer schema ------------------------------
        system_prompt = (
            "You are an expert autonomous driving and traffic safety analyst. "
            "Analyze the provided traffic camera frames. "
            "Vehicles are marked with numerical IDs.\n"
            + zone_block
            + id_constraint
            + physics_block
            + "Step 1 — Think briefly (1–2 sentences) about the most "
            "safety-critical interactions.\n"
            "Step 2 — Output your analysis STRICTLY as a JSON list of "
            "enriched SPO triples. Each triple must include:\n"
            "  'subject': acting entity (e.g. 'Vehicle 4')\n"
            "  'predicate': action or spatial relationship "
            "(e.g. 'tailgating', 'collided_with')\n"
            "  'object': receiving entity or environment "
            "(e.g. 'Vehicle 9', 'intersection')\n"
            "  'motion_state': one of APPROACHING / DIVERGING / "
            "PARALLEL / STATIONARY\n"
            "  'phase': one of approach / conflict / resolution / normal\n"
            "Do not include markdown or conversational text outside the "
            "JSON array.\n"
            "Example: [{\"subject\":\"Vehicle 4\",\"predicate\":\"tailgating\","
            "\"object\":\"Vehicle 9\","
            "\"motion_state\":\"APPROACHING\",\"phase\":\"conflict\"}]"
        )

        # --- Build multimodal message -----------------------------------------
        # Each frame becomes an image_url content block (base64 JPEG).
        # All frames are sent in a single HumanMessage so the model sees
        # the full temporal clip before generating triples.
        content_blocks: List[Dict[str, Any]] = []
        for frame in frame_buffer:
            b64 = _pil_to_base64(frame)
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        # Brief user-turn instruction to anchor what the model should do
        # with the images (system prompt carries the full schema instruction).
        n_frames = len(frame_buffer)
        frame_label = (
            f"{n_frames} consecutive traffic frames (oldest to newest)"
            if n_frames > 1
            else "1 traffic frame"
        )
        content_blocks.append({
            "type": "text",
            "text": (
                f"Analyze the {frame_label} above and output the "
                "enriched SPO triple list as instructed."
            ),
        })

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content_blocks),
        ]

        # --- Invoke Ollama ---------------------------------------------------
        try:
            response = self._llm.invoke(messages)
            raw_output = response.content
        except Exception as exc:
            log.error(
                "[VLM] Ollama invocation failed at t=%.1fs: %s", timestamp, exc
            )
            return []

        return self._parse_json_triples([raw_output], timestamp)

    # ------------------------------------------------------------------
    # JSON parsing
    # ------------------------------------------------------------------

    def _parse_json_triples(
        self,
        text: List[str],
        timestamp: float,
    ) -> List[Dict[str, Any]]:
        """
        Parses VLM raw output into validated enriched SPO triple dicts.

        Robustness layers:
        1. Strip markdown code fences.
        2. Extract first [...] array (tolerates CoT preamble prose).
        3. Drop triples missing required keys (subject / predicate / object).
        4. Drop triples with empty string values.
        5. Normalise motion_state and phase to known enums; default to
           APPROACHING and normal if absent or unrecognised.
        """
        raw = text[0] if text else ""

        # Layer 1: strip markdown fences
        clean = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

        # Layer 2: extract the first [...] array, tolerating CoT preamble
        match = re.search(r"\[.*\]", clean, re.DOTALL)
        if not match:
            log.warning(
                "[VLM] No JSON array found at t=%.1fs. Raw: %r",
                timestamp, raw,
            )
            return []

        try:
            triples = json.loads(match.group())
        except json.JSONDecodeError as exc:
            log.warning(
                "[VLM] JSON parse error at t=%.1fs: %s. Raw: %r",
                timestamp, exc, raw,
            )
            return []

        if not isinstance(triples, list):
            log.warning(
                "[VLM] Expected list, got %s at t=%.1fs.",
                type(triples), timestamp,
            )
            return []

        valid = []
        for triple in triples:
            if not isinstance(triple, dict):
                continue
            # Layer 3: required key presence
            if not _REQUIRED_KEYS.issubset(triple.keys()):
                continue
            # Layer 4: no empty string values for required keys
            if not all(str(triple[k]).strip() for k in _REQUIRED_KEYS):
                continue

            # Layer 5: normalise optional enrichment fields
            ms = str(triple.get("motion_state", "")).upper().strip()
            triple["motion_state"] = (
                ms if ms in _MOTION_STATES else "APPROACHING"
            )

            ph = str(triple.get("phase", "")).lower().strip()
            triple["phase"] = ph if ph in _PHASES else "normal"

            triple["timestamp"] = timestamp
            valid.append(triple)

        dropped = len(triples) - len(valid)
        if dropped:
            log.warning(
                "[VLM] Dropped %d/%d malformed triples at t=%.1fs.",
                dropped, len(triples), timestamp,
            )

        return valid
