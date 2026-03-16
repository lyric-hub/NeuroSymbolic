# End-to-End Pipeline Test Plan (30-minute raw video)

## Step 0 — Fix api.py Bug ✅ (already done)
Removed `alert_callback=_on_pipeline_alert` from `_run_job()` in `api.py`.
`process_video()` has no such parameter — would have caused `TypeError` at runtime.

---

## Step 1 — Prerequisites Checklist

Run these before starting:

```bash
ls calibration.yaml          # MUST exist — pipeline exits without it
ls zone_config.json          # Optional — zones disabled if missing
ollama list                  # Must show qwen2.5:72b
ls yolov8n.pt                # Or custom .pt
nvidia-smi                   # GPU must be available for VLM
ls ~/.cache/huggingface/hub/ # Must have Qwen2.5-VL-3B-Instruct cached
```

If `calibration.yaml` missing → go to Step 2 first.

---

## Step 2 — Calibration (if calibration.yaml missing)

```bash
uvicorn api:app --reload
# Open: http://localhost:8000/calibrate-ui
```
1. Select video → scrub to a clear frame with road markings
2. Place ≥ 4 points, enter real-world metres, **name each point** (e.g. "Stop Line", "North Kerb")
3. Compute & Save → verify RMSE < 0.3m (green)

---

## Step 3 — Run the Pipeline

### Option A — API (recommended)
```bash
# Terminal 1
uvicorn api:app --reload

# Terminal 2
curl -X POST http://localhost:8000/run_physics/ \
  -H "Content-Type: application/json" \
  -d '{"video_path": "data/raw_videos/<your_video>.mp4", "run_physics": true, "run_vlm": true}'

# Poll status
curl http://localhost:8000/job/<job_id>
```

### Option B — CLI (direct)
```bash
python -c "from main import process_video; process_video('data/raw_videos/<your_video>.mp4')"
```

**Expected runtime:** 30–60 minutes for a 30-minute video on DGX Spark.

---

## Step 4 — Validate Database Outputs

```python
# DuckDB
import duckdb
conn = duckdb.connect("data/duckdb_storage/physics_vectors.duckdb")
print(conn.execute("SELECT COUNT(*) FROM vehicle_trajectories").fetchone())
print(conn.execute("SELECT COUNT(DISTINCT track_id) FROM vehicle_trajectories").fetchone())
print(conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM vehicle_trajectories").fetchone())
# Expect: thousands of rows, multiple tracks, range 0.0 → ~1800.0

# Milvus
from src.memory_layer.milvus_client import SemanticVectorStore
store = SemanticVectorStore()
print(store.search_semantic_events("vehicle speeding", top_k=3))
print(store.search_entity_profiles("aggressive driver", top_k=3))

# Kùzu
from src.memory_layer.graph_client import GraphClient
g = GraphClient()
print(g.query_graph("MATCH (v:Vehicle) RETURN COUNT(v) AS cnt"))
print(g.query_graph("MATCH (s)-[r:INTERACTS_WITH]->(o) RETURN COUNT(r) AS cnt"))
print(g.query_graph("MATCH (v)-[r:PRECEDES]->(v2) RETURN COUNT(r) AS cnt"))
```

---

## Step 5 — Agent Query Tests

```bash
curl -X POST http://localhost:8000/chat/ -H "Content-Type: application/json" \
  -d '{"query": "What happened in this video?"}'

curl -X POST http://localhost:8000/chat/ -H "Content-Type: application/json" \
  -d '{"query": "Which vehicle was driving most aggressively?"}'

curl -X POST http://localhost:8000/chat/ -H "Content-Type: application/json" \
  -d '{"query": "Were there any accidents or near-misses?"}'

curl -X POST http://localhost:8000/chat/ -H "Content-Type: application/json" \
  -d '{"query": "How many vehicles entered from the North gate?"}'
```

---

## Step 6 — Metrics (optional, for presentation)

```python
from main import process_video
from src.evaluation.metrics_collector import MetricsCollector
mc = MetricsCollector()
process_video("data/raw_videos/<your_video>.mp4", metrics=mc)
mc.report()
```

---

## Pass Criteria

- [ ] `/run_physics/` returns 202 (no TypeError)
- [ ] Pipeline completes without crash
- [ ] DuckDB: rows spanning 0–1800s, multiple track_ids
- [ ] Milvus: results for "vehicle", "speeding"
- [ ] Kùzu: INTERACTS_WITH + PRECEDES edges > 0
- [ ] Agent: grounded answers for all 4 test queries
