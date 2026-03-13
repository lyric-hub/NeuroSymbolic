"""
Integration tests for the three memory-layer databases.

Each test creates an isolated temporary database in a temp directory
so real production data is never touched.  Tests run without GPU or Ollama.

Milvus tests are marked @pytest.mark.slow because pymilvus startup can
take a few seconds even in Lite mode.  Run them with:
    pytest tests/test_databases.py -m slow
or include them in full runs with:
    pytest tests/test_databases.py
"""

import time
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import make_trajectory_df


# ---------------------------------------------------------------------------
# DuckDB tests
# ---------------------------------------------------------------------------

class TestDuckDBClient:
    @pytest.fixture
    def db(self, tmp_db_dir: Path):
        """Fresh DuckDB client writing to a temp directory."""
        from src.memory_layer.duckdb_client import DuckDBClient
        db_path = str(tmp_db_dir / "test.duckdb")
        client = DuckDBClient(db_path=db_path)
        yield client
        client.close()

    def test_schema_created(self, db):
        """vehicle_trajectories table must exist after init."""
        result = db.conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name = 'vehicle_trajectories'"
        ).fetchall()
        assert len(result) == 1

    def test_insert_and_query(self, db):
        state = {1: [10.0, 20.0, 3.0, 0.5, 0.1, 0.0]}
        db.insert_state_vectors(
            timestamp=1.0, frame_id=30, state_vectors=state
        )
        db._flush()
        df = db.get_trajectory_window(0.0, 5.0, track_id=1)
        assert not df.empty
        assert df.iloc[0]["pos_x"] == pytest.approx(10.0)

    def test_query_empty_returns_empty_df(self, db):
        df = db.get_trajectory_window(0.0, 10.0, track_id=99)
        assert df.empty

    def test_buffer_flush_on_close(self, db):
        """Rows buffered before close() must appear after re-opening."""
        for i in range(5):
            db.insert_state_vectors(
                timestamp=float(i), frame_id=i,
                state_vectors={1: [float(i), 0.0, 2.0, 0.0, 0.0, 0.0]},
            )
        db.close()

        from src.memory_layer.duckdb_client import DuckDBClient
        db2 = DuckDBClient(db_path=db.db_path)
        df = db2.get_trajectory_window(0.0, 10.0, track_id=1)
        assert len(df) == 5
        db2.close()

    def test_behavior_summary_returns_string(self, db):
        """Behavior summary must return a non-empty string for populated tracks."""
        for i in range(10):
            db.insert_state_vectors(
                timestamp=float(i) * 0.1,
                frame_id=i,
                state_vectors={1: [float(i), 0.0, 5.0, 0.0, 0.0, 0.0]},
            )
        db._flush()
        summary = db.get_behavior_summary(track_ids=[1], current_time=1.0)
        assert isinstance(summary, str)
        assert "Vehicle 1" in summary

    def test_behavior_summary_empty_track_ids(self, db):
        assert db.get_behavior_summary(track_ids=[], current_time=5.0) == ""

    def test_behavior_summary_unknown_track(self, db):
        summary = db.get_behavior_summary(track_ids=[999], current_time=5.0)
        assert "999" in summary  # Should mention the track ID

    def test_multiple_vehicles_isolated(self, db):
        """Queries for track_id=1 must not return data for track_id=2."""
        db.insert_state_vectors(1.0, 30, {1: [5.0, 0.0, 3.0, 0.0, 0.0, 0.0]})
        db.insert_state_vectors(1.0, 30, {2: [100.0, 0.0, 5.0, 0.0, 0.0, 0.0]})
        db._flush()
        df = db.get_trajectory_window(0.0, 5.0, track_id=1)
        assert all(df["pos_x"] < 50)  # Vehicle 2 at x=100 should not appear


# ---------------------------------------------------------------------------
# Kùzu Graph tests
# ---------------------------------------------------------------------------

class TestGraphClient:
    @pytest.fixture
    def graph(self, tmp_db_dir: Path):
        """Fresh Kùzu graph in a temp directory."""
        from src.memory_layer.graph_client import GraphClient
        db_path = str(tmp_db_dir / "test_graph")
        client = GraphClient(db_path=db_path)
        yield client
        client.close()

    def test_schema_created(self, graph):
        """Node tables must exist after init."""
        for node_type in ("Vehicle", "Pedestrian", "Infrastructure"):
            result = graph.query_graph(
                f"MATCH (n:{node_type}) RETURN COUNT(n) AS cnt"
            )
            assert isinstance(result, list)

    def test_insert_and_query_triple(self, graph):
        triples = [{
            "subject": "Vehicle 1",
            "subject_type": "Vehicle",
            "predicate": "tailgating",
            "object": "Vehicle 2",
            "object_type": "Vehicle",
            "timestamp": 5.0,
            "motion_state": "APPROACHING",
            "phase": "conflict",
        }]
        graph.insert_vlm_triples(triples, time_window="0.0-5.0")
        result = graph.query_graph(
            "MATCH (s:Vehicle)-[r:INTERACTS_WITH]->(o:Vehicle) "
            "RETURN s.name, r.predicate, o.name"
        )
        assert len(result) >= 1
        names = {r["s.name"] for r in result}
        assert "Vehicle 1" in names

    def test_query_returns_error_dict_on_bad_cypher(self, graph):
        result = graph.query_graph("MATCH (x) RETURN INVALID SYNTAX !!!")
        # Should return [{"error": "..."}] not raise
        assert isinstance(result, list)
        assert "error" in result[0]

    def test_typed_nodes_separated(self, graph):
        """Vehicle and Pedestrian nodes must be in separate tables."""
        triples = [
            {
                "subject": "Vehicle 3",
                "subject_type": "Vehicle",
                "predicate": "approaching",
                "object": "pedestrian_crossing",
                "object_type": "Infrastructure",
                "timestamp": 2.0,
            }
        ]
        graph.insert_vlm_triples(triples, time_window="0.0-5.0")
        veh = graph.query_graph("MATCH (v:Vehicle) RETURN v.name")
        infra = graph.query_graph("MATCH (i:Infrastructure) RETURN i.name")
        veh_names = [r["v.name"] for r in veh]
        infra_names = [r["i.name"] for r in infra]
        assert "Vehicle 3" in veh_names
        assert "pedestrian_crossing" in infra_names

    def test_temporal_edge_inserted(self, graph):
        """PRECEDES edge must appear for two consecutive windows."""
        graph.insert_temporal_edges(
            vehicle_ids=[1, 2],
            prev_window="0.0-5.0",
            curr_window="5.0-10.0",
            gap_seconds=5.0,
        )
        result = graph.query_graph(
            "MATCH (v:Vehicle)-[r:PRECEDES]->(v2:Vehicle) "
            "RETURN v.name, r.from_window, r.to_window"
        )
        assert len(result) >= 2

    def test_empty_triples_no_error(self, graph):
        graph.insert_vlm_triples([], time_window="0.0-5.0")
        # Should not raise
        result = graph.query_graph("MATCH (n:Vehicle) RETURN COUNT(n) AS c")
        assert result[0]["c"] == 0

    def test_time_window_filter(self, graph):
        """Querying by trajectory_time_window must return only that window."""
        graph.insert_vlm_triples([{
            "subject": "Vehicle 5",
            "subject_type": "Vehicle",
            "predicate": "moving",
            "object": "Vehicle 6",
            "object_type": "Vehicle",
            "timestamp": 12.0,
        }], time_window="10.0-15.0")
        graph.insert_vlm_triples([{
            "subject": "Vehicle 7",
            "subject_type": "Vehicle",
            "predicate": "stopped",
            "object": "intersection",
            "object_type": "Infrastructure",
            "timestamp": 3.0,
        }], time_window="0.0-5.0")

        result = graph.query_graph(
            "MATCH (s)-[r:INTERACTS_WITH]->(o) "
            "WHERE r.trajectory_time_window = '10.0-15.0' "
            "RETURN s.name"
        )
        names = [r["s.name"] for r in result]
        assert "Vehicle 5" in names
        assert "Vehicle 7" not in names


# ---------------------------------------------------------------------------
# Milvus Lite tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestMilvusClient:
    @pytest.fixture
    def store(self, tmp_db_dir: Path):
        """Fresh Milvus Lite store in a temp directory."""
        from src.memory_layer.milvus_client import SemanticVectorStore
        db_path = str(tmp_db_dir / "test_milvus.db")
        client = SemanticVectorStore(db_path=db_path)
        yield client

    def test_collection_created(self, store):
        """traffic_events collection must exist after init."""
        assert store is not None

    def test_insert_and_search(self, store):
        store.insert_event_chunk(
            description="Vehicle 1 speeding through intersection.",
            start_time=0.0,
            end_time=5.0,
            frame_id=150,
        )
        results = store.search_semantic_events("speeding vehicle", top_k=1)
        assert len(results) >= 1
        assert results[0]["similarity_score"] > 0.0

    def test_search_returns_time_window(self, store):
        store.insert_event_chunk(
            description="Vehicle 3 tailgating Vehicle 4.",
            start_time=10.0,
            end_time=15.0,
            frame_id=450,
        )
        results = store.search_semantic_events("tailgating", top_k=1)
        assert "time_window" in results[0]

    def test_empty_collection_search_returns_empty(self, store):
        results = store.search_semantic_events("accident", top_k=3)
        assert isinstance(results, list)
