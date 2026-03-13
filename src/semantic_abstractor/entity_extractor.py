"""
Entity Extraction — Validation Layer
=====================================
Takes raw VLM JSON output and forces it into the strict Kùzu-graph-ready
SPO schema using a local LLM (qwen2.5:72b via Ollama).

This is the bridge between the neural VLM perception stage and the symbolic
graph storage stage.  Every triple written to Kùzu passes through here.
"""

import json
import logging
from typing import List, Literal, Optional, Set

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity type description — shared between schema docstring and LLM prompt.
# ---------------------------------------------------------------------------
_TYPE_DESCRIPTION = (
    "'Vehicle' for any motorised road user (car, truck, motorcycle, bus); "
    "'Pedestrian' for people or cyclists on foot; "
    "'Infrastructure' for static road features "
    "(traffic_light, stop_sign, intersection, lane, road)"
)

_MOTION_STATES = frozenset({"APPROACHING", "DIVERGING", "PARALLEL", "STATIONARY"})
_PHASES = frozenset({"approach", "conflict", "resolution", "normal"})


# ---------------------------------------------------------------------------
# Pydantic schema — every triple that enters the graph must match this.
# ---------------------------------------------------------------------------

class SPOTriple(BaseModel):
    """A single Subject-Predicate-Object interaction extracted from the scene."""

    subject: str = Field(
        description="Acting entity using its tracked ID label (e.g. 'Vehicle 4')"
    )
    subject_type: Literal["Vehicle", "Pedestrian", "Infrastructure"] = Field(
        description=f"Entity type of the subject: {_TYPE_DESCRIPTION}"
    )
    predicate: str = Field(
        description="Action or spatial relationship (e.g. 'tailgating', 'turning_left')"
    )
    object: str = Field(
        description="Receiving entity or environment feature (e.g. 'Vehicle 9', 'intersection')"
    )
    object_type: Literal["Vehicle", "Pedestrian", "Infrastructure"] = Field(
        description=f"Entity type of the object: {_TYPE_DESCRIPTION}"
    )
    timestamp: float = Field(
        description="Exact time of the event in seconds"
    )
    motion_state: Literal["APPROACHING", "DIVERGING", "PARALLEL", "STATIONARY"] = Field(
        default="APPROACHING",
        description="Relative motion state between subject and object"
    )
    phase: Literal["approach", "conflict", "resolution", "normal"] = Field(
        default="normal",
        description="Interaction phase at the time of the event"
    )


class SceneGraphOutput(BaseModel):
    """Container for all SPO triples extracted from one VLM inference."""

    triples: List[SPOTriple] = Field(
        description="All subject-predicate-object interactions observed in the scene"
    )


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class EntityExtractor:
    """
    Validation and extraction node in the Neuro-Symbolic pipeline.

    Receives raw VLM text (JSON or free-form) and returns a list of
    validated SPOTriple dicts ready for insertion into the Kùzu graph.
    Uses a local LLM (ChatOllama) at temperature=0 for deterministic parsing.

    Args:
        model_name: Ollama model tag. Defaults to 'qwen2.5:72b'.
    """

    _SYSTEM = (
        "You are a strict data extraction parser for a traffic analysis system. "
        "Extract ALL traffic entity interactions from the VLM description below and "
        "convert them into Subject-Predicate-Object triples.\n\n"
        "Rules:\n"
        "- Use exact tracked IDs for vehicle subjects/objects (e.g. 'Vehicle 4').\n"
        "- subject_type and object_type must be one of: Vehicle, Pedestrian, Infrastructure.\n"
        "- motion_state must be one of: APPROACHING, DIVERGING, PARALLEL, STATIONARY.\n"
        "- phase must be one of: approach, conflict, resolution, normal.\n"
        "- If a field is uncertain, use the stated default.\n\n"
        "{format_instructions}"
    )

    def __init__(self, model_name: str = "qwen2.5:72b") -> None:
        self._llm = ChatOllama(model=model_name, temperature=0.0)
        self._parser = JsonOutputParser(pydantic_object=SceneGraphOutput)
        self._system_text = self._SYSTEM.format(
            format_instructions=self._parser.get_format_instructions()
        )

    def extract_triples(
        self,
        raw_vlm_text: str,
        current_time: float,
        active_track_ids: Optional[Set[int]] = None,
    ) -> List[dict]:
        """
        Parse raw VLM output into validated SPO dicts for Kùzu insertion.

        Args:
            raw_vlm_text:      JSON string or free-text from vlm_inference.py.
            current_time:      Video timestamp in seconds (used only for logging).
            active_track_ids:  Set of integer track IDs visible in the current frame.
                               When provided, triples referencing Vehicle IDs not in
                               this set are dropped — preventing hallucinated ghost
                               nodes from entering the graph.

        Returns:
            List of validated triple dicts (may be empty on parse failure).
        """
        try:
            log.debug("Extracting entities at t=%.2fs", current_time)
            messages = [
                SystemMessage(content=self._system_text),
                HumanMessage(
                    content=(
                        f"VLM Description: {raw_vlm_text}\n"
                        f"Timestamp of event: {current_time}"
                    )
                ),
            ]
            result = self._parser.parse(
                self._llm.invoke(messages).content
            )
            triples = result.get("triples", [])

            # Fix 3: filter out triples where the VLM hallucinated a Vehicle ID
            # that does not exist in the current tracked frame.
            if active_track_ids is not None:
                before = len(triples)
                triples = [
                    t for t in triples
                    if self._entity_id_valid(t.get("subject", ""), t.get("subject_type", ""), active_track_ids)
                    and self._entity_id_valid(t.get("object", ""), t.get("object_type", ""), active_track_ids)
                ]
                dropped = before - len(triples)
                if dropped:
                    log.warning(
                        "Dropped %d triple(s) with hallucinated Vehicle IDs at t=%.2fs",
                        dropped, current_time,
                    )

            log.debug("Extracted %d triples at t=%.2fs", len(triples), current_time)
            return triples

        except Exception:
            log.exception(
                "Entity extraction failed at t=%.2fs — returning empty list",
                current_time,
            )
            return []

    @staticmethod
    def _entity_id_valid(name: str, entity_type: str, active_ids: Set[int]) -> bool:
        """
        Returns True if the entity is valid for the current frame.
        Non-vehicle entities (Infrastructure, Pedestrian) always pass.
        Vehicle entities must have a numeric ID present in active_ids.
        """
        if entity_type != "Vehicle":
            return True
        parts = name.strip().split()
        if len(parts) < 2:
            return False
        try:
            return int(parts[-1]) in active_ids
        except ValueError:
            return False


# ---------------------------------------------------------------------------
# Quick smoke-test (run with: python -m src.semantic_abstractor.entity_extractor)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    extractor = EntityExtractor()
    mock_vlm = json.dumps([{
        "subject": "Vehicle 4",
        "predicate": "tailgating",
        "object": "Vehicle 9",
        "motion_state": "APPROACHING",
        "phase": "conflict",
        "timestamp": 12.5,
    }])
    structured = extractor.extract_triples(mock_vlm, current_time=12.5)
    print(json.dumps(structured, indent=2))
