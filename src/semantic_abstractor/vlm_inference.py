"""
VLM inference module for semantic scene abstraction.

Key improvements over single-frame approach (based on 2024-2025 research):

1. Multi-frame temporal input (Qwen2.5-VL native video format).
   Passes the last N SoM frames as a video clip using Qwen2.5-VL's
   MRoPE temporal position encoding. The model sees actual motion,
   not just a snapshot — capturing approach, conflict, and resolution
   phases that single-frame sampling misses.

2. Richer output schema.
   Each triple now includes a `motion_state` field (APPROACHING /
   DIVERGING / PARALLEL / STATIONARY) and a `phase` field
   (approach / conflict / resolution / normal). These are validated
   by EntityExtractor downstream.

3. Chain-of-Thought before JSON.
   A two-step instruction ("think briefly, then output JSON") reduces
   hallucination and improves triple quality on dense scenes.

4. max_new_tokens raised to 512.
   256 was insufficient for complex multi-vehicle scenes where the
   model needed to describe 5+ interactions.

References:
  - Qwen2.5-VL Technical Report (arXiv 2502.13923)
  - TrafficVLM temporal phase modelling (arXiv 2404.09275)
  - DriveVLM dual-system architecture (arXiv 2402.12289)
"""

import json
import re
import torch
from PIL import Image
from typing import List, Dict, Any, Set
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

_REQUIRED_KEYS = {"subject", "predicate", "object"}

# Valid motion state labels injected into VLM instructions.
_MOTION_STATES = {"APPROACHING", "DIVERGING", "PARALLEL", "STATIONARY"}

# Valid interaction phase labels.
_PHASES = {"approach", "conflict", "resolution", "normal"}


class TrafficSemanticAbstractor:
    """
    Wraps Qwen2.5-VL-3B to perform low-frequency semantic scene abstraction.

    Converts Set-of-Mark (SoM) overlaid frame sequences into structured
    Subject-Predicate-Object triples enriched with motion state and
    interaction phase labels.

    Multi-frame mode (preferred):
        Pass a list of 2–8 consecutive SoM PIL images. The model uses its
        native video understanding (MRoPE temporal position encoding) to
        reason about motion trajectories across frames.

    Single-frame mode (fallback):
        Pass a list with a single PIL image. Behaviour is identical to the
        previous single-frame implementation.
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        print(f"Loading VLM: {model_id}...")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = next(self.model.parameters()).device

    # ------------------------------------------------------------------
    # Physics block helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_physics_block(
        state_vectors: Dict[int, List[float]],
        warm_tracks: Set[int],
    ) -> str:
        """
        Converts kinematics state_vectors to a readable text block.

        Tracks not yet in warm_tracks have unreliable velocity/acceleration
        estimates (fewer than window_length samples). They are labelled
        "(initialising)" to prevent the VLM from treating a cold-start
        zero velocity as a genuine stop.
        """
        lines = []
        for track_id, sv in state_vectors.items():
            x, y, vx, vy, ax, ay = sv
            if track_id not in warm_tracks:
                lines.append(
                    f"  Vehicle {track_id}: "
                    f"position=({x:.1f}m, {y:.1f}m), "
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
                f"position=({x:.1f}m, {y:.1f}m), "
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
        state_vectors: Dict[int, List[float]] = None,
        warm_tracks: Set[int] = None,
        behavior_summary: str = None,
        fps: float = 3.0,
    ) -> List[Dict[str, Any]]:
        """
        Generates enriched SPO triples from a sequence of SoM frames.

        Args:
            frame_buffer:     List of 1–8 consecutive SoM PIL Images
                              (most recent last).  When >1 frame is
                              provided the model receives them as a
                              native video input via Qwen2.5-VL's frame-
                              list format, giving genuine temporal context.
            timestamp:        Video timestamp of the most recent frame (s).
            state_vectors:    Optional {track_id: [x,y,vx,vy,ax,ay]} from
                              KinematicEstimator.
            warm_tracks:      Set of track IDs with full Savitzky-Golay
                              coverage (unreliable IDs labelled initialising).
            behavior_summary: Change-only motion narrative from DuckDB
                              (last 5 s).  Preferred over state_vectors
                              snapshot when available.
            fps:              Frame rate of the buffer (VLM sample rate,
                              typically fps // semantic_interval ≈ 3.0).
                              Used by Qwen2.5-VL's MRoPE temporal encoding.

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
                + self._build_physics_block(state_vectors, warm_tracks or set())
                + "\n"
            )

        # --- ID constraint ---------------------------------------------------
        active_ids = sorted(state_vectors.keys()) if state_vectors else []
        id_constraint = (
            f"The ONLY valid vehicle IDs in this frame are: {active_ids}. "
            "Use ONLY these exact IDs when referring to vehicles "
            "(e.g. 'Vehicle 14', not 'Vehicle 1'). "
        ) if active_ids else ""

        # --- System prompt: CoT + richer schema ------------------------------
        # Two-step instruction: brief internal reasoning → structured JSON.
        # This reduces hallucination on dense multi-vehicle scenes
        # (validated by DriveVLM / DriveLM research).
        system_prompt = (
            "You are an expert autonomous driving and traffic safety analyst. "
            "Analyze the provided traffic camera footage. Vehicles are marked with numerical IDs. "
            + id_constraint
            + physics_block
            + "Step 1 — Think briefly (1–2 sentences) about the most safety-critical interactions. "
            "Step 2 — Output your analysis STRICTLY as a JSON list of enriched SPO triples. "
            "Each triple must include:\n"
            "  'subject': acting entity (e.g. 'Vehicle 4')\n"
            "  'predicate': action or spatial relationship (e.g. 'tailgating', 'collided_with')\n"
            "  'object': receiving entity or environment (e.g. 'Vehicle 9', 'intersection')\n"
            "  'motion_state': one of APPROACHING / DIVERGING / PARALLEL / STATIONARY\n"
            "  'phase': one of approach / conflict / resolution / normal\n"
            "Do not include markdown or conversational text outside the JSON array. "
            "Example: [{'subject':'Vehicle 4','predicate':'tailgating','object':'Vehicle 9',"
            "'motion_state':'APPROACHING','phase':'conflict'}]"
        )

        # --- Build message for Qwen2.5-VL ------------------------------------
        # Multi-frame path: use native video frame-list format.
        # Single-frame path: use image format (same behaviour as before).
        if len(frame_buffer) > 1:
            visual_content = {
                "type": "video",
                "video": frame_buffer,
                "fps": fps,
                # Per-frame pixel budget keeps memory predictable.
                # 256*28*28 ≈ 200k pixels/frame — sufficient for SoM badges.
                "min_pixels": 16 * 28 * 28,
                "max_pixels": 256 * 28 * 28,
            }
        else:
            visual_content = {"type": "image", "image": frame_buffer[0]}

        messages = [{
            "role": "user",
            "content": [
                visual_content,
                {"type": "text", "text": system_prompt},
            ],
        }]

        # --- Tokenise and run inference --------------------------------------
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,   # raised from 256: dense scenes need more tokens
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for out_ids, in_ids in zip(generated_ids, inputs.input_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return self._parse_json_triples(output_text, timestamp)

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
            print(f"[VLM] No JSON array found at t={timestamp:.1f}s. Raw: {raw!r}")
            return []

        try:
            triples = json.loads(match.group())
        except json.JSONDecodeError as exc:
            print(f"[VLM] JSON parse error at t={timestamp:.1f}s: {exc}. Raw: {raw!r}")
            return []

        if not isinstance(triples, list):
            print(f"[VLM] Expected list, got {type(triples)} at t={timestamp:.1f}s.")
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
            triple["motion_state"] = ms if ms in _MOTION_STATES else "APPROACHING"

            ph = str(triple.get("phase", "")).lower().strip()
            triple["phase"] = ph if ph in _PHASES else "normal"

            triple["timestamp"] = timestamp
            valid.append(triple)

        dropped = len(triples) - len(valid)
        if dropped:
            print(f"[VLM] Dropped {dropped}/{len(triples)} malformed triples at t={timestamp:.1f}s.")

        return valid
