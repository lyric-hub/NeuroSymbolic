# TrafficAgent — Neuro-Symbolic Traffic Analytics

A dual-loop pipeline that fuses real-time physics tracking with VLM semantic understanding
to support intersection analysis, OD studies, and AI-powered traffic safety queries.

---

## What Can Be Measured

### Physical / Kinematic Parameters (per vehicle, per frame)
| Parameter | Unit | Source |
|-----------|------|--------|
| Position (x, y) | metres (real-world) | Homography → DuckDB |
| Velocity (vx, vy) | m/s | Savitzky-Golay filter → DuckDB |
| Speed magnitude | m/s | √(vx² + vy²), queryable via agent |
| Acceleration (ax, ay) | m/s² | SG derivative → DuckDB |
| Hard braking | boolean | ax or ay < −4 m/s², flagged by agent |
| Track continuity | frames | Gap-interpolated up to 5 missed frames |

> All kinematic values are derived from homography-mapped real-world coordinates and
> smoothed with a Savitzky-Golay filter (window=15 frames, poly=3).
> The agent marks estimates as `(initialising)` for vehicles with fewer than 15 frames of history.

### Traffic Flow Parameters (zone-based)
| Parameter | Description | Source |
|-----------|-------------|--------|
| Gate entry count | Vehicles crossing a named gate inward | DuckDB `zone_crossings` |
| Gate exit count | Vehicles crossing a named gate outward | DuckDB `zone_crossings` |
| Total zone volume | Sum of all gate entries | `query_zone_flow` tool |
| Origin-Destination pairs | (entry_gate → exit_gate) per vehicle | DuckDB CTE join |
| Dwell time | Seconds spent inside the zone | exit_time − entry_time |
| OD confidence | `confirmed` (both gates crossed) / `estimated` (inferred) | Per-event confidence flag |
| Flow by time window | Filter any metric to a time range | start_time / end_time args |

### Semantic / Behavioural Parameters (VLM-extracted)
| Parameter | Example Query | Source |
|-----------|--------------|--------|
| Vehicle interactions | "which vehicles were tailgating?" | Milvus + Kùzu |
| Near-miss / collision | "was there a crash near the intersection?" | Milvus semantic search |
| Infrastructure violations | "did any vehicle run the red light?" | Kùzu graph |
| Scene description | "describe what happened at t=30s" | Milvus vector search |
| Entity relationships | "what did Vehicle 4 do?" | Kùzu Cypher |

---

## Architecture

```
Video frames
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  MICRO-LOOP (every frame)                           │
│  YOLOv8 → ByteTrack → Homography → Savitzky-Golay  │
│      │                                   │          │
│  ZoneManager (crossing events)       DuckDB         │
└─────────────────────────────────────────────────────┘
    │ every 30 frames
    ▼
┌─────────────────────────────────────────────────────┐
│  MACRO-LOOP (semantic abstraction)                  │
│  Set-of-Mark render → Qwen2.5-VL-3B → EntityExtract │
│      │                      │               │       │
│   Milvus (vectors)    Kùzu (graph)     DuckDB zones │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│  LangGraph ReAct Agent       │
│  qwen2.5:72b via Ollama      │
│  Tools: semantic / graph /   │
│         physics / zone-flow  │
└──────────────────────────────┘
```

---

## Prerequisites

### Hardware
- NVIDIA GPU recommended (tested on DGX Spark GB10, 128 GB unified memory)
- CPU-only works but is slower for the VLM

### Software
- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/) running locally with `qwen2.5:72b` pulled

### Install dependencies
```bash
uv sync
```

### Pull the Ollama model
```bash
ollama pull qwen2.5:72b
```

---

## Quick Start — Live Detection & Tracking Stream

### 1. Start Ollama (if not already running)
```bash
ollama serve &
```

### 2. Start the API server
```bash
cd /workspace
uvicorn api:app --host 0.0.0.0 --port 8000 &
```

### 3. Forward port 8000 (DevContainer only)
In VSCode: `Ctrl+Shift+P` → **Forward a Port** → `8000`

### 4. Open the live stream in your browser
```
http://localhost:8000/stream?video=data/ulloor/20250523_162245_tp00026.mp4
```

Switch videos or trackers via query params:
```
http://localhost:8000/stream?video=data/ulloor/20250523_154828_tp00025.mp4&tracker=ocsort
http://localhost:8000/stream?video=data/ulloor/20250523_165701_tp00027.mp4&tracker=botsort&conf=0.4
```

| Param | Default | Options |
|-------|---------|---------|
| `video` | required | any path from `GET /videos` |
| `tracker` | `bytetrack` | `bytetrack`, `ocsort`, `sfsort`, `botsort`, `deepocsort`, `hybridsort`, `boosttrack`, `strongsort` |
| `conf` | `0.3` | 0.0 – 1.0 |

List all available videos: `http://localhost:8000/videos`

### 5. Stop the server
```bash
kill -9 $(pgrep -f "uvicorn api:app")
```

---

## End-to-End Testing Guide

### Step 1 — Place a test video

```
data/raw_videos/traffic_sample.mp4
```

Any short intersection clip works (30–120 seconds, ≥720p recommended).

---

### Step 2 — Start the API server

```bash
uvicorn api:app --reload
```

Open `http://localhost:8000` in your browser.
The nav bar has links to **Calibration** and **Zone Drawing**.

---

### Step 3 — Camera Calibration

> Required once per camera position. Produces `calibration.yaml`.

1. Go to `http://localhost:8000/calibrate-ui`
2. Select your video from the dropdown
3. Scrub to a frame where lane markings are clearly visible
4. Click to place **≥4 image points** at known real-world locations
   (e.g., lane corners, kerb markers)
5. Enter the corresponding real-world (X, Y) coordinates in metres for each point
6. Click **Compute Homography**
7. Verify RMSE is low (< 0.5 m is good for an intersection)
8. Click **Save** — writes `calibration.yaml` to the project root

---

### Step 4 — Draw the Zone (optional but recommended for OD study)

> Defines the intersection boundary and entry/exit gates.

1. Go to `http://localhost:8000/zone-ui`
2. Select the same video and frame
3. **Click** to draw the zone polygon around the intersection area
4. **Double-click** to close the polygon
5. Click **Add Gate**, then click two points to draw a gate line at each entry/exit arm
   - Name each gate (e.g., `North`, `South`, `East`, `West`)
   - The arrow shows the inward direction; if wrong, redraw with endpoints swapped
6. Set a **Zone ID** (e.g., `intersection_01`)
7. Click **Save Zone** — writes `zone_config.json` to the project root

---

### Step 5 — Run the Pipeline

**Option A — via API (recommended)**
```bash
# Upload video and start processing
curl -X POST http://localhost:8000/upload_video/ \
     -F "file=@data/raw_videos/traffic_sample.mp4"
# Returns: {"job_id": "abc123", "status": "processing"}

# Poll until done
curl http://localhost:8000/job/abc123
# Returns: {"job_id": "abc123", "status": "done"}
```

**Option B — direct script**
```bash
python main.py
```
> Requires `calibration.yaml` to exist. Zone manager activates automatically if `zone_config.json` exists.

**Expected terminal output:**
```
Zone 'intersection_01' active — gates: ['North', 'South', 'East', 'West']
[1.0s] Running Semantic Abstraction...
[3.3s] Vehicle 4 → enter via North
[5.1s] Vehicle 7 → exit via South
...
--- Video Processing Complete in 142.3s ---
```

---

### Step 6 — Query the Agent

**Via API:**
```bash
curl -X POST http://localhost:8000/chat/ \
     -H "Content-Type: application/json" \
     -d '{"query": "How many vehicles entered from the North gate?"}'
```

**Via interactive CLI** (after `main.py` finishes):
```
User >> How many vehicles entered from the North gate?
User >> What is the OD matrix for the intersection?
User >> Did any vehicle brake hard near the intersection?
User >> Was there a near-miss or tailgating event?
User >> What did Vehicle 4 do between t=10s and t=20s?
```

---

## Example Agent Queries by Category

### Zone / OD Flow
```
How many vehicles passed through the zone?
What is the origin-destination breakdown?
How many vehicles entered via the East gate in the first 30 seconds?
Which vehicles had the longest dwell time?
```

### Physics / Kinematics
```
Did Vehicle 4 brake hard?
What was the maximum speed of Vehicle 7?
Which vehicle had the highest acceleration?
```

### Semantic / Behavioural
```
Was there a collision or near-miss?
Describe the traffic events around t=45s.
Which vehicles were tailgating?
Did any vehicle run the red light?
```

### Combined
```
Did the vehicle that entered from the North gate brake hard before exiting South?
Which vehicle interacted with the pedestrian and how fast was it going?
```

---

## Data Storage Locations

| Store | Path | Contents |
|-------|------|----------|
| DuckDB | `data/duckdb_storage/physics_vectors.duckdb` | Trajectories + zone crossings |
| Milvus | `data/milvus_storage/semantic_memory.db` | VLM event embeddings |
| Kùzu graph | `data/graph_storage/traffic_graph` | SPO triples, entity nodes |
| Calibration | `calibration.yaml` | Homography matrix + metadata |
| Zone config | `zone_config.json` | Polygon + gate definitions |

---

## Resetting Between Test Runs

```bash
rm -rf data/duckdb_storage/ data/milvus_storage/ data/graph_storage/
```

Databases are re-created automatically on the next run.
Do **not** delete `calibration.yaml` or `zone_config.json` unless reconfiguring.

---

## Known Limitations

- **Cold-start kinematics**: Vehicles with < 15 frames of history report `(initialising)` for speed/acceleration; full estimates require ~0.5s at 30 fps.
- **Track ID reassignment**: ByteTrack may assign a new ID to a reappearing vehicle after a long occlusion (> 5 frames). The old ID's OD record will be incomplete.
- **Gate confidence**: Crossings detected via polygon membership change (not direct line intersection) are marked `estimated`. Confirmed rate improves with tighter gate placement at the actual kerb lines.
- **Milvus lock**: If the server crashes, delete `data/milvus_storage/.semantic_memory.db.lock` before restarting.
- **Ollama required**: The entity extractor and agent both require `qwen2.5:72b` to be running via Ollama. Start with `ollama serve` before launching the pipeline.
