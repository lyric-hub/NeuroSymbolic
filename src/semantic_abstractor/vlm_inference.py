import json
import re
import torch
from PIL import Image
from typing import List, Dict, Any, Set
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

_REQUIRED_KEYS = {"subject", "predicate", "object"}

class TrafficSemanticAbstractor:
    """
    Wraps the Qwen2.5-VL-3B model to perform low-frequency semantic scene abstraction.
    Converts Set-of-Mark (SoM) overlaid frames into structured Subject-Predicate-Object triples.
    """
    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        print(f"Loading VLM: {model_id}...")

        # Load the model with automatic mixed precision; device_map="auto" handles
        # placement across available GPUs/CPU transparently.
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto"
        )

        # Load the processor for handling image/text interleaved inputs
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Derive the input device from the model's first parameter so inputs are
        # always moved to the correct device regardless of hardware configuration.
        self.device = next(self.model.parameters()).device

    @staticmethod
    def _build_physics_block(
        state_vectors: Dict[int, List[float]],
        warm_tracks: Set[int],
    ) -> str:
        """
        Converts the kinematics state_vectors dict into a readable text block
        for injection into the VLM prompt.

        Tracks whose IDs are not in ``warm_tracks`` have fewer than
        ``window_length`` samples and their velocity / acceleration estimates
        are provisional (finite-difference only).  These are labelled
        "(initialising)" so the VLM does not treat a cold-start zero as a
        genuine stop.

        Args:
            state_vectors: Mapping of ``{track_id: [x, y, vx, vy, ax, ay]}``.
            warm_tracks: Set of track IDs with full Savitzky-Golay coverage.
        """
        lines = []
        for track_id, sv in state_vectors.items():
            x, y, vx, vy, ax, ay = sv
            if track_id not in warm_tracks:
                # Too few frames — do not report a misleading "speed=0 m/s".
                lines.append(
                    f"  Vehicle {track_id}: "
                    f"position=({x:.1f}m, {y:.1f}m), "
                    f"speed=(initialising — not enough frames yet), "
                    f"acceleration=(initialising)"
                )
                continue
            speed = (vx ** 2 + vy ** 2) ** 0.5
            accel = (ax ** 2 + ay ** 2) ** 0.5
            # Negative projection of acceleration onto velocity direction = braking
            dot = vx * ax + vy * ay
            signed_accel = -accel if dot < 0 else accel
            lines.append(
                f"  Vehicle {track_id}: "
                f"position=({x:.1f}m, {y:.1f}m), "
                f"speed={speed:.1f} m/s, "
                f"acceleration={signed_accel:+.1f} m/s²"
            )
        return "\n".join(lines)

    def generate_scene_graph_triples(
        self,
        som_image: Image.Image,
        timestamp: float,
        state_vectors: Dict[int, List[float]] = None,
        warm_tracks: Set[int] = None,
        behavior_summary: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Takes an image with tracking ID overlays and prompts the VLM to extract interactions.

        Args:
            som_image: PIL Image containing the Set-of-Mark overlays from the physics engine.
            timestamp: The current video timestamp.
            state_vectors: Optional kinematic state from the physics engine.
                           Format: {track_id: [x, y, vx, vy, ax, ay]}
                           When provided, real-world position, speed, and acceleration
                           are injected into the prompt giving the VLM spatial and
                           motion awareness it cannot derive from a single frame.
            warm_tracks: Set of track IDs with full Savitzky-Golay coverage
                         (sourced from ``KinematicEstimator.warm_tracks``).
                         Tracks absent from this set are labelled "(initialising)"
                         in the prompt so the VLM does not confuse a cold-start
                         zero with a genuine stop.

        Returns:
            A list of dictionaries representing SPO triples (e.g., Subject, Predicate, Object).
        """
        # Build physics context block.
        # If a pre-computed behavior_summary is provided, use that — it contains
        # change-only narratives from DuckDB history (much richer than a snapshot).
        # Fall back to the current-frame snapshot when no history is available.
        physics_block = ""
        if behavior_summary:
            physics_block = (
                "\nVerified vehicle behaviour history (last 5 s) from the tracking engine "
                "(use this to ground your analysis — do not contradict it):\n"
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

        # Build active track ID constraint so VLM only references real tracked IDs
        active_ids = sorted(state_vectors.keys()) if state_vectors else []
        id_constraint = (
            f"The ONLY valid vehicle IDs in this frame are: {active_ids}. "
            "Use ONLY these exact IDs when referring to vehicles (e.g. 'Vehicle 14', not 'Vehicle 1'). "
        ) if active_ids else ""

        # The prompt forces the VLM to act as a structured data extractor
        system_prompt = (
            "You are an expert autonomous driving and traffic safety analyst. "
            "Analyze the provided traffic camera image. Vehicles have been marked with numerical IDs. "
            + id_constraint
            + physics_block +
            "Identify all safety-critical interactions and spatial relationships between "
            "the marked vehicles, pedestrians, and the environment. "
            "You must output your analysis STRICTLY as a JSON list of Subject-Predicate-Object (SPO) triples. "
            "Do not include markdown formatting or conversational text. "
            "Example format: [{'subject': 'Vehicle 4', 'predicate': 'tailgating', 'object': 'Vehicle 9'}, "
            "{'subject': 'Vehicle 2', 'predicate': 'waiting_at', 'object': 'red_light'}]"
        )

        # Format the input message for Qwen2.5-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": som_image},
                    {"type": "text", "text": system_prompt},
                ],
            }
        ]

        # Process the inputs using the chat template
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

        # Generate the response
        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)
            
        # Trim the input tokens from the output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for out_ids, in_ids in zip(generated_ids, inputs.input_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Parse the JSON output
        return self._parse_json_triples(output_text, timestamp)

    def _parse_json_triples(self, text: List[str], timestamp: float) -> List[Dict[str, Any]]:
        """
        Parses the VLM's raw string output into validated SPO triple dicts.

        Robustness layers applied in order:
        1. Strip markdown code fences the VLM may stubbornly add.
        2. Use regex to extract the JSON array even when wrapped in prose.
        3. Gate: silently drop any triple missing subject/predicate/object keys.
        4. Gate: silently drop any triple whose values are empty strings.
        """
        raw = text[0] if text else ""

        # Layer 1: strip markdown fences
        clean = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

        # Layer 2: extract the first [...] array, tolerating surrounding prose
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
            print(f"[VLM] Expected a list but got {type(triples)} at t={timestamp:.1f}s.")
            return []

        # Layer 3 & 4: key presence gate + empty value gate
        valid = []
        for triple in triples:
            if not isinstance(triple, dict):
                continue
            if not _REQUIRED_KEYS.issubset(triple.keys()):
                continue
            if not all(str(triple[k]).strip() for k in _REQUIRED_KEYS):
                continue
            triple["timestamp"] = timestamp
            valid.append(triple)

        dropped = len(triples) - len(valid)
        if dropped:
            print(f"[VLM] Dropped {dropped}/{len(triples)} malformed triples at t={timestamp:.1f}s.")

        return valid