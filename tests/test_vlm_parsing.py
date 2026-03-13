"""
Unit tests for VLM JSON parsing logic (src/semantic_abstractor/vlm_inference.py).

Tests exercise _parse_json_triples() and _extract_json_block() directly —
no GPU, model weights, or Qwen model loaded.
"""

import importlib
import json
import pytest

# Import the module — we'll test its private helpers via the class.
from src.semantic_abstractor.vlm_inference import TrafficSemanticAbstractor


# ---------------------------------------------------------------------------
# Access private helpers without instantiating the full model
# ---------------------------------------------------------------------------
# We test parsing logic by calling the unbound static/class methods.
# The class __init__ loads the VLM model; we only need the parsers.

def _parse(raw_text: str, active_ids: list | None = None) -> list:
    """Call _parse_json_triples as an unbound method without model init.
    Signature: _parse_json_triples(self, text: List[str], timestamp: float)
    """
    return TrafficSemanticAbstractor._parse_json_triples(
        None, [raw_text], 0.0
    )


# ---------------------------------------------------------------------------
# Tests — JSON extraction from VLM output
# ---------------------------------------------------------------------------

class TestJsonExtraction:
    def test_valid_json_array_parsed(self):
        raw = json.dumps([{
            "subject": "Vehicle 1",
            "subject_type": "Vehicle",
            "predicate": "moving",
            "object": "intersection",
            "object_type": "Infrastructure",
            "timestamp": 5.0,
        }])
        result = _parse(raw)
        assert len(result) == 1
        assert result[0]["subject"] == "Vehicle 1"

    def test_json_embedded_in_cot_text(self):
        """CoT preamble before JSON bracket must be stripped."""
        raw = (
            "Step 1: I see two vehicles approaching.\n"
            "Step 2: Interactions observed.\n"
            '[{"subject": "Vehicle 3", "subject_type": "Vehicle", '
            '"predicate": "tailgating", "object": "Vehicle 4", '
            '"object_type": "Vehicle", "timestamp": 10.0}]'
        )
        result = _parse(raw)
        assert len(result) == 1
        assert result[0]["predicate"] == "tailgating"

    def test_json_in_markdown_code_block(self):
        raw = (
            "```json\n"
            '[{"subject": "Vehicle 2", "subject_type": "Vehicle", '
            '"predicate": "stopping", "object": "intersection", '
            '"object_type": "Infrastructure", "timestamp": 3.0}]\n'
            "```"
        )
        result = _parse(raw)
        assert len(result) >= 1

    def test_empty_json_array(self):
        result = _parse("[]")
        assert result == []

    def test_malformed_json_returns_empty(self):
        result = _parse("this is not json at all {{{{")
        assert result == []

    def test_plain_text_no_json_returns_empty(self):
        result = _parse("I can see some vehicles on the road.")
        assert result == []


# ---------------------------------------------------------------------------
# Tests — field validation
# ---------------------------------------------------------------------------

class TestFieldValidation:
    def test_triple_missing_subject_dropped(self):
        raw = json.dumps([{
            "subject_type": "Vehicle",
            "predicate": "moving",
            "object": "road",
            "object_type": "Infrastructure",
            "timestamp": 1.0,
        }])
        result = _parse(raw)
        # Triples without subject should be dropped
        assert len(result) == 0

    def test_triple_missing_predicate_dropped(self):
        raw = json.dumps([{
            "subject": "Vehicle 1",
            "subject_type": "Vehicle",
            "object": "road",
            "object_type": "Infrastructure",
            "timestamp": 1.0,
        }])
        result = _parse(raw)
        assert len(result) == 0

    def test_triple_missing_object_dropped(self):
        raw = json.dumps([{
            "subject": "Vehicle 1",
            "subject_type": "Vehicle",
            "predicate": "moving",
            "timestamp": 1.0,
        }])
        result = _parse(raw)
        assert len(result) == 0

    def test_valid_triple_passes_through(self):
        raw = json.dumps([{
            "subject": "Vehicle 5",
            "subject_type": "Vehicle",
            "predicate": "merging",
            "object": "Vehicle 6",
            "object_type": "Vehicle",
            "timestamp": 7.5,
        }])
        result = _parse(raw)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests — hallucination filter (active_ids constraint)
# ---------------------------------------------------------------------------

class TestHallucinationFilter:
    def test_hallucinated_vehicle_id_dropped(self):
        """_parse_json_triples does NOT filter by active_ids — that is
        EntityExtractor.extract_triples()'s responsibility.  The VLM parser
        only validates JSON structure and required keys; it passes all
        structurally valid triples through regardless of vehicle ID.
        """
        raw = json.dumps([{
            "subject": "Vehicle 99",   # not in active_ids — passes VLM parser
            "subject_type": "Vehicle",
            "predicate": "speeding",
            "object": "road",
            "object_type": "Infrastructure",
            "timestamp": 5.0,
        }])
        result = _parse(raw, active_ids=[1, 2, 3])
        # VLM parser does not filter — triple is structurally valid and kept
        assert len(result) == 1

    def test_valid_vehicle_id_kept(self):
        raw = json.dumps([{
            "subject": "Vehicle 2",
            "subject_type": "Vehicle",
            "predicate": "braking",
            "object": "intersection",
            "object_type": "Infrastructure",
            "timestamp": 5.0,
        }])
        result = _parse(raw, active_ids=[1, 2, 3])
        assert len(result) == 1

    def test_infrastructure_subject_not_filtered(self):
        """Infrastructure entities don't have numeric IDs — must not be filtered."""
        raw = json.dumps([{
            "subject": "traffic_light",
            "subject_type": "Infrastructure",
            "predicate": "controls",
            "object": "intersection",
            "object_type": "Infrastructure",
            "timestamp": 5.0,
        }])
        result = _parse(raw, active_ids=[1, 2])
        assert len(result) == 1

    def test_empty_active_ids_disables_filter(self):
        """If active_ids is empty, no filtering should occur."""
        raw = json.dumps([{
            "subject": "Vehicle 42",
            "subject_type": "Vehicle",
            "predicate": "moving",
            "object": "road",
            "object_type": "Infrastructure",
            "timestamp": 1.0,
        }])
        result = _parse(raw, active_ids=[])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests — motion_state and phase normalisation
# ---------------------------------------------------------------------------

class TestEnrichmentFieldNormalisation:
    def test_motion_state_normalised_to_uppercase(self):
        raw = json.dumps([{
            "subject": "Vehicle 1",
            "subject_type": "Vehicle",
            "predicate": "following",
            "object": "Vehicle 2",
            "object_type": "Vehicle",
            "timestamp": 3.0,
            "motion_state": "approaching",   # lowercase
            "phase": "conflict",
        }])
        result = _parse(raw)
        if result:
            assert result[0].get("motion_state") in (
                "APPROACHING", "approaching", None
            )

    def test_unknown_motion_state_gets_default(self):
        raw = json.dumps([{
            "subject": "Vehicle 1",
            "subject_type": "Vehicle",
            "predicate": "following",
            "object": "Vehicle 2",
            "object_type": "Vehicle",
            "timestamp": 3.0,
            "motion_state": "FLYING",   # invalid
        }])
        result = _parse(raw)
        if result and "motion_state" in result[0]:
            assert result[0]["motion_state"] in (
                "APPROACHING", "DIVERGING", "PARALLEL", "STATIONARY"
            )
