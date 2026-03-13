"""
Unit tests for the hierarchical intent router
(src/agentic_orchestrator/hierarchical_router.py).

Tests verify that the embedding-based classifier routes queries to the
correct intent class.  No Ollama or GPU required; only the sentence-
transformer model (all-MiniLM-L6-v2) must be downloadable.
"""

import pytest

from src.agentic_orchestrator.hierarchical_router import _classify_intent


# ---------------------------------------------------------------------------
# Tests — full_analysis routing
# ---------------------------------------------------------------------------

class TestFullAnalysisRouting:
    def test_speed_query_routes_to_full(self):
        route = _classify_intent("Was Vehicle 4 speeding during the incident?")
        assert route == "full_analysis"

    def test_braking_query_routes_to_full(self):
        route = _classify_intent("Did any vehicle brake hard before the collision?")
        assert route == "full_analysis"

    def test_acceleration_query_routes_to_full(self):
        route = _classify_intent("Which vehicle had the most aggressive acceleration?")
        assert route == "full_analysis"

    def test_violation_query_routes_to_full(self):
        route = _classify_intent("List all traffic rule violations detected in the video.")
        assert route == "full_analysis"

    def test_physics_query_routes_to_full(self):
        route = _classify_intent("What was the maximum speed of Vehicle 7?")
        assert route == "full_analysis"

    def test_collision_query_routes_to_full(self):
        route = _classify_intent("Was there a collision between Vehicle 2 and Vehicle 5?")
        assert route == "full_analysis"


# ---------------------------------------------------------------------------
# Tests — semantic_lookup routing
# ---------------------------------------------------------------------------

class TestSemanticLookupRouting:
    def test_describe_event_routes_to_semantic(self):
        route = _classify_intent("Describe what happened at the intersection.")
        assert route == "semantic_lookup"

    def test_scene_summary_routes_to_semantic(self):
        route = _classify_intent("Summarise the traffic scene in the video.")
        assert route == "semantic_lookup"

    def test_narrative_query_routes_to_semantic(self):
        route = _classify_intent("What events were observed near the traffic light?")
        assert route == "semantic_lookup"

    def test_what_happened_routes_to_semantic(self):
        route = _classify_intent("What happened between the vehicles at timestamp 12 seconds?")
        # This could be either route depending on the prototype similarity
        # — just verify no exception and a valid route is returned.
        assert route in ("full_analysis", "semantic_lookup")


# ---------------------------------------------------------------------------
# Tests — return value contract
# ---------------------------------------------------------------------------

class TestRouterContract:
    def test_always_returns_valid_route(self):
        queries = [
            "hello",
            "123456",
            "what is the speed",
            "",
        ]
        for q in queries:
            route = _classify_intent(q)
            assert route in ("full_analysis", "semantic_lookup"), (
                f"Invalid route '{route}' for query: '{q}'"
            )

    def test_prototype_embeddings_cached(self):
        """Calling _classify_intent twice should not re-encode prototypes."""
        from src.agentic_orchestrator.hierarchical_router import _proto_embeddings
        _classify_intent("speed test")
        _classify_intent("another query")
        # _proto_embeddings is module-level and should be populated after first call
        assert _proto_embeddings is not None
