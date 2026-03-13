"""
Pipeline Metrics — Proxy Evaluation Without Ground Truth
==========================================================

Since TrafficAgent has no labelled ground-truth dataset, this module
collects *intrinsic* and *proxy* metrics that can be computed purely from
the system's own outputs.

Metric categories
-----------------
1. Pipeline throughput   — FPS, VLM call rate, latency percentiles
2. VLM quality (proxy)  — JSON parse rate, triple completeness, hallucination rate
3. Alert quality         — Alert rate, distribution, cooldown effectiveness
4. Database performance  — Write throughput, query latency (P50/P95)
5. Agent quality         — Tool call distribution, response latency, empty-response rate

Usage
-----
Attach a ``MetricsCollector`` to the pipeline and query it after processing::

    from src.evaluation.metrics import MetricsCollector

    mc = MetricsCollector()
    process_video(video_path, metrics=mc)

    report = mc.report()
    print(json.dumps(report, indent=2))

Or call ``mc.log_summary()`` to print a formatted summary to the logger.
"""

import logging
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PipelineMetrics:
    """
    Snapshot of all collected pipeline metrics.

    All latency values are in milliseconds.
    All rates are per-second (or per-call where noted).
    """

    # --- Throughput ---
    total_frames_processed: int = 0
    total_processing_time_s: float = 0.0
    micro_loop_fps: float = 0.0           # frames / total_processing_time

    # --- VLM ---
    vlm_calls_total: int = 0
    vlm_json_parse_successes: int = 0
    vlm_json_parse_failures: int = 0
    vlm_json_parse_rate: float = 0.0      # successes / total calls
    vlm_total_triples: int = 0
    vlm_avg_triples_per_call: float = 0.0
    vlm_hallucinated_id_count: int = 0
    vlm_hallucination_rate: float = 0.0   # hallucinated triples / total triples
    vlm_incomplete_triple_count: int = 0  # triples missing required SPO keys
    vlm_completeness_rate: float = 0.0    # complete triples / total triples
    vlm_latency_ms_p50: float = 0.0
    vlm_latency_ms_p95: float = 0.0
    vlm_motion_skip_count: int = 0        # frames skipped by motion-energy gate
    vlm_force_vlm_count: int = 0          # frames where alert forced VLM

    # --- Alerts ---
    alerts_total: int = 0
    alerts_suppressed_by_cooldown: int = 0
    alert_cooldown_effectiveness: float = 0.0  # suppressed / (suppressed + fired)
    alert_type_distribution: Dict[str, int] = field(default_factory=dict)

    # --- Database (latencies in ms) ---
    duckdb_flush_latency_ms_p50: float = 0.0
    duckdb_flush_latency_ms_p95: float = 0.0
    duckdb_total_rows_written: int = 0
    milvus_insert_latency_ms_p50: float = 0.0
    milvus_insert_latency_ms_p95: float = 0.0
    milvus_search_latency_ms_p50: float = 0.0
    milvus_search_latency_ms_p95: float = 0.0
    milvus_avg_similarity_score: float = 0.0   # higher = more relevant retrieval
    graph_insert_latency_ms_p50: float = 0.0
    graph_insert_latency_ms_p95: float = 0.0
    graph_query_latency_ms_p50: float = 0.0
    graph_query_latency_ms_p95: float = 0.0
    graph_node_count: int = 0
    graph_edge_count: int = 0

    # --- Agent ---
    agent_queries_total: int = 0
    agent_response_latency_ms_p50: float = 0.0
    agent_response_latency_ms_p95: float = 0.0
    agent_avg_tool_calls_per_query: float = 0.0
    agent_tool_call_distribution: Dict[str, int] = field(default_factory=dict)
    agent_empty_response_count: int = 0
    agent_empty_response_rate: float = 0.0    # empty responses / total queries
    agent_route_distribution: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Lightweight, thread-safe-enough metrics collector for the TrafficAgent pipeline.

    Attach to the pipeline at construction time and call the ``record_*``
    methods at the appropriate hook points.  Retrieve a ``PipelineMetrics``
    snapshot at any time by calling ``snapshot()``.

    Thread safety note: append() on Python lists is GIL-protected, making
    it safe to call record_* from the background pipeline thread while
    snapshot() is called from the main thread.
    """

    def __init__(self) -> None:
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

        # Throughput
        self._frames_processed: int = 0

        # VLM
        self._vlm_parse_results: List[bool] = []         # True = success
        self._vlm_triple_counts: List[int] = []
        self._vlm_hallucinated: List[int] = []           # count per call
        self._vlm_incomplete: List[int] = []             # count per call
        self._vlm_latencies_ms: List[float] = []
        self._motion_skip_count: int = 0
        self._force_vlm_count: int = 0

        # Alerts
        self._alerts_fired: int = 0
        self._alerts_suppressed: int = 0
        self._alert_types: List[str] = []

        # DuckDB
        self._duckdb_flush_ms: List[float] = []
        self._duckdb_rows_written: int = 0

        # Milvus
        self._milvus_insert_ms: List[float] = []
        self._milvus_search_ms: List[float] = []
        self._milvus_similarity_scores: List[float] = []

        # Kùzu
        self._graph_insert_ms: List[float] = []
        self._graph_query_ms: List[float] = []

        # Agent
        self._agent_latencies_ms: List[float] = []
        self._agent_tool_calls: List[List[str]] = []     # per-query list of tool names
        self._agent_empty_responses: int = 0
        self._agent_routes: List[str] = []

    # -------------------------------------------------------------------
    # Pipeline lifecycle
    # -------------------------------------------------------------------

    def begin(self) -> None:
        """Call at the start of process_video()."""
        self._start_time = time.perf_counter()

    def end(self) -> None:
        """Call at the end of process_video()."""
        self._end_time = time.perf_counter()

    # -------------------------------------------------------------------
    # VLM hooks
    # -------------------------------------------------------------------

    def record_vlm_call(
        self,
        latency_ms: float,
        parse_success: bool,
        triple_count: int,
        hallucinated_count: int = 0,
        incomplete_count: int = 0,
    ) -> None:
        """
        Record one VLM inference call.

        Args:
            latency_ms:         Wall-clock time for the VLM forward pass (ms).
            parse_success:      True if JSON was successfully extracted.
            triple_count:       Number of valid SPO triples returned.
            hallucinated_count: Triples whose vehicle ID was not in active_ids.
            incomplete_count:   Triples missing required SPO fields.
        """
        self._vlm_latencies_ms.append(latency_ms)
        self._vlm_parse_results.append(parse_success)
        if parse_success:
            self._vlm_triple_counts.append(triple_count)
            self._vlm_hallucinated.append(hallucinated_count)
            self._vlm_incomplete.append(incomplete_count)

    def record_motion_skip(self) -> None:
        """Call when a VLM invocation is skipped due to low motion energy."""
        self._motion_skip_count += 1

    def record_force_vlm(self) -> None:
        """Call when an alert forces an out-of-schedule VLM invocation."""
        self._force_vlm_count += 1

    def record_frame(self) -> None:
        """Call once per processed frame in the micro-loop."""
        self._frames_processed += 1

    # -------------------------------------------------------------------
    # Alert hooks
    # -------------------------------------------------------------------

    def record_alert_fired(self, alert_type: str) -> None:
        """Record an alert that passed the cooldown gate and was emitted."""
        self._alerts_fired += 1
        self._alert_types.append(alert_type)

    def record_alert_suppressed(self) -> None:
        """Record an alert that was suppressed by the cooldown gate."""
        self._alerts_suppressed += 1

    # -------------------------------------------------------------------
    # Database hooks
    # -------------------------------------------------------------------

    def record_duckdb_flush(self, latency_ms: float, rows: int) -> None:
        """Record one DuckDB buffer flush operation."""
        self._duckdb_flush_ms.append(latency_ms)
        self._duckdb_rows_written += rows

    def record_milvus_insert(self, latency_ms: float) -> None:
        self._milvus_insert_ms.append(latency_ms)

    def record_milvus_search(self, latency_ms: float, top_similarity: float) -> None:
        self._milvus_search_ms.append(latency_ms)
        self._milvus_similarity_scores.append(top_similarity)

    def record_graph_insert(self, latency_ms: float) -> None:
        self._graph_insert_ms.append(latency_ms)

    def record_graph_query(self, latency_ms: float) -> None:
        self._graph_query_ms.append(latency_ms)

    # -------------------------------------------------------------------
    # Agent hooks
    # -------------------------------------------------------------------

    def record_agent_query(
        self,
        latency_ms: float,
        tool_calls: List[str],
        final_summary: str,
        route: str,
    ) -> None:
        """
        Record one completed agent query.

        Args:
            latency_ms:    End-to-end latency for the full agent loop (ms).
            tool_calls:    Ordered list of tool names called during this query.
            final_summary: The text in state['final_summary'].
            route:         'full_analysis' or 'semantic_lookup'.
        """
        self._agent_latencies_ms.append(latency_ms)
        self._agent_tool_calls.append(tool_calls)
        self._agent_routes.append(route)
        if not final_summary or len(final_summary.strip()) < 10:
            self._agent_empty_responses += 1

    # -------------------------------------------------------------------
    # Snapshot computation
    # -------------------------------------------------------------------

    @staticmethod
    def _percentile(data: List[float], pct: int) -> float:
        """Return the Nth percentile of a list, or 0.0 if list is empty."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = max(0, int(len(sorted_data) * pct / 100) - 1)
        return round(sorted_data[idx], 2)

    def snapshot(self) -> PipelineMetrics:
        """
        Compute and return a ``PipelineMetrics`` snapshot from all
        recorded observations.  Safe to call multiple times.
        """
        m = PipelineMetrics()

        # --- Throughput ---
        if self._start_time and self._end_time:
            elapsed = self._end_time - self._start_time
            m.total_processing_time_s = round(elapsed, 2)
            m.total_frames_processed = self._frames_processed
            m.micro_loop_fps = (
                round(self._frames_processed / elapsed, 2) if elapsed > 0 else 0.0
            )

        # --- VLM ---
        total_calls = len(self._vlm_parse_results)
        successes = sum(self._vlm_parse_results)
        m.vlm_calls_total = total_calls
        m.vlm_json_parse_successes = successes
        m.vlm_json_parse_failures = total_calls - successes
        m.vlm_json_parse_rate = round(successes / total_calls, 3) if total_calls else 0.0

        total_triples = sum(self._vlm_triple_counts)
        m.vlm_total_triples = total_triples
        m.vlm_avg_triples_per_call = (
            round(total_triples / successes, 2) if successes else 0.0
        )

        total_hall = sum(self._vlm_hallucinated)
        m.vlm_hallucinated_id_count = total_hall
        m.vlm_hallucination_rate = (
            round(total_hall / total_triples, 3) if total_triples else 0.0
        )

        total_incomplete = sum(self._vlm_incomplete)
        m.vlm_incomplete_triple_count = total_incomplete
        complete = total_triples - total_incomplete
        m.vlm_completeness_rate = (
            round(complete / total_triples, 3) if total_triples else 0.0
        )

        m.vlm_latency_ms_p50 = self._percentile(self._vlm_latencies_ms, 50)
        m.vlm_latency_ms_p95 = self._percentile(self._vlm_latencies_ms, 95)
        m.vlm_motion_skip_count = self._motion_skip_count
        m.vlm_force_vlm_count = self._force_vlm_count

        # --- Alerts ---
        m.alerts_total = self._alerts_fired
        m.alerts_suppressed_by_cooldown = self._alerts_suppressed
        total_alert_attempts = self._alerts_fired + self._alerts_suppressed
        m.alert_cooldown_effectiveness = (
            round(self._alerts_suppressed / total_alert_attempts, 3)
            if total_alert_attempts else 0.0
        )
        type_counts: Dict[str, int] = defaultdict(int)
        for t in self._alert_types:
            type_counts[t] += 1
        m.alert_type_distribution = dict(type_counts)

        # --- DuckDB ---
        m.duckdb_flush_latency_ms_p50 = self._percentile(self._duckdb_flush_ms, 50)
        m.duckdb_flush_latency_ms_p95 = self._percentile(self._duckdb_flush_ms, 95)
        m.duckdb_total_rows_written = self._duckdb_rows_written

        # --- Milvus ---
        m.milvus_insert_latency_ms_p50 = self._percentile(self._milvus_insert_ms, 50)
        m.milvus_insert_latency_ms_p95 = self._percentile(self._milvus_insert_ms, 95)
        m.milvus_search_latency_ms_p50 = self._percentile(self._milvus_search_ms, 50)
        m.milvus_search_latency_ms_p95 = self._percentile(self._milvus_search_ms, 95)
        m.milvus_avg_similarity_score = (
            round(statistics.mean(self._milvus_similarity_scores), 4)
            if self._milvus_similarity_scores else 0.0
        )

        # --- Kùzu ---
        m.graph_insert_latency_ms_p50 = self._percentile(self._graph_insert_ms, 50)
        m.graph_insert_latency_ms_p95 = self._percentile(self._graph_insert_ms, 95)
        m.graph_query_latency_ms_p50 = self._percentile(self._graph_query_ms, 50)
        m.graph_query_latency_ms_p95 = self._percentile(self._graph_query_ms, 95)

        # --- Agent ---
        m.agent_queries_total = len(self._agent_latencies_ms)
        m.agent_response_latency_ms_p50 = self._percentile(self._agent_latencies_ms, 50)
        m.agent_response_latency_ms_p95 = self._percentile(self._agent_latencies_ms, 95)

        all_tool_calls = [t for calls in self._agent_tool_calls for t in calls]
        m.agent_avg_tool_calls_per_query = (
            round(len(all_tool_calls) / len(self._agent_tool_calls), 2)
            if self._agent_tool_calls else 0.0
        )
        tool_counts: Dict[str, int] = defaultdict(int)
        for t in all_tool_calls:
            tool_counts[t] += 1
        m.agent_tool_call_distribution = dict(tool_counts)

        m.agent_empty_response_count = self._agent_empty_responses
        m.agent_empty_response_rate = (
            round(self._agent_empty_responses / len(self._agent_latencies_ms), 3)
            if self._agent_latencies_ms else 0.0
        )

        route_counts: Dict[str, int] = defaultdict(int)
        for r in self._agent_routes:
            route_counts[r] += 1
        m.agent_route_distribution = dict(route_counts)

        return m

    def report(self) -> Dict[str, Any]:
        """Return the metrics snapshot as a plain JSON-serialisable dict."""
        return asdict(self.snapshot())

    def log_summary(self) -> None:
        """Log a human-readable metrics summary at INFO level."""
        m = self.snapshot()
        log.info("=" * 60)
        log.info("TRAFFICAGENT PIPELINE METRICS")
        log.info("=" * 60)
        log.info(
            "THROUGHPUT  frames=%d  time=%.1fs  fps=%.1f",
            m.total_frames_processed, m.total_processing_time_s, m.micro_loop_fps,
        )
        log.info(
            "VLM         calls=%d  parse_rate=%.1f%%  avg_triples=%.1f  "
            "hallucination_rate=%.1f%%  completeness=%.1f%%",
            m.vlm_calls_total,
            m.vlm_json_parse_rate * 100,
            m.vlm_avg_triples_per_call,
            m.vlm_hallucination_rate * 100,
            m.vlm_completeness_rate * 100,
        )
        log.info(
            "VLM LATENCY p50=%.0fms  p95=%.0fms  motion_skips=%d  force_vlm=%d",
            m.vlm_latency_ms_p50, m.vlm_latency_ms_p95,
            m.vlm_motion_skip_count, m.vlm_force_vlm_count,
        )
        log.info(
            "ALERTS      fired=%d  suppressed=%d  cooldown_eff=%.1f%%  dist=%s",
            m.alerts_total, m.alerts_suppressed_by_cooldown,
            m.alert_cooldown_effectiveness * 100,
            m.alert_type_distribution,
        )
        log.info(
            "DUCKDB      rows=%d  flush_p50=%.0fms  flush_p95=%.0fms",
            m.duckdb_total_rows_written,
            m.duckdb_flush_latency_ms_p50, m.duckdb_flush_latency_ms_p95,
        )
        log.info(
            "MILVUS      insert_p95=%.0fms  search_p95=%.0fms  avg_sim=%.3f",
            m.milvus_insert_latency_ms_p95,
            m.milvus_search_latency_ms_p95,
            m.milvus_avg_similarity_score,
        )
        log.info(
            "GRAPH       insert_p95=%.0fms  query_p95=%.0fms",
            m.graph_insert_latency_ms_p95, m.graph_query_latency_ms_p95,
        )
        log.info(
            "AGENT       queries=%d  latency_p95=%.0fms  avg_tools=%.1f  "
            "empty_rate=%.1f%%  routes=%s",
            m.agent_queries_total,
            m.agent_response_latency_ms_p95,
            m.agent_avg_tool_calls_per_query,
            m.agent_empty_response_rate * 100,
            m.agent_route_distribution,
        )
        log.info("=" * 60)


# ---------------------------------------------------------------------------
# Convenience context manager for timing individual operations
# ---------------------------------------------------------------------------

class _Timer:
    """Context manager that records elapsed time in milliseconds."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "_Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0


def time_operation() -> _Timer:
    """
    Measure the wall-clock time of a code block in milliseconds.

    Usage::

        with time_operation() as t:
            result = some_db_call()
        metrics.record_milvus_search(t.elapsed_ms, top_similarity=0.92)
    """
    return _Timer()
