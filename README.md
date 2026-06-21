# TrafficAgent — Neuro-Symbolic Traffic Analytics

TrafficAgent processes road video into three linked forms of memory:

- metric vehicle trajectories and traffic measurements in DuckDB;
- semantic event and vehicle-profile embeddings in Milvus Lite;
- temporal entity relationships in Kùzu.

A LangGraph agent routes natural-language questions to deterministic traffic
analysis tools, semantic retrieval, graph queries, or a combination of them.

## Capabilities

- YOLO or Grounding DINO object detection
- YOLO-Pose ground-contact estimation
- ByteTrack and other BoxMOT trackers
- camera-to-world homography calibration
- position, velocity, acceleration, and data-quality estimation
- short detector-gap interpolation with explicit quality flags
- lane classification from world coordinates
- zone, gate, entry/exit, dwell-time, and OD analysis
- Set-of-Mark visual grounding for VLM scene interpretation
- semantic event search and longitudinal vehicle profiles
- temporal interaction and causal graph queries
- deterministic traffic-rule evaluation
- traffic volume, queue, turning-movement, V85, proximity, and TTC analysis
- auditable agent tool traces and neural-symbolic contradiction reporting

## Architecture

```text
Recorded traffic video
        |
        v
YOLO-Pose detector -> BoxMOT tracker
        |
        +--------------------- physics loop ----------------------+
        |                                                        |
        v                                                        v
ground-point projection -> homography -> kinematic estimator -> DuckDB
        |                                      |                 |
        |                                      +-> quality flags |
        |                                      +-> lane labels   |
        +-> ZoneManager -> gate crossings / OD events ------------+
        |
        +-------------------- semantic loop ----------------------+
        |     sampled at about 3 Hz and skipped for static scenes |
        v                                                        |
zone-focused Set-of-Mark frames -> Gemma VLM -> validated triples|
        |                                                        |
        +-> Milvus Lite: event chunks and vehicle profiles       |
        +-> Kùzu: interactions, temporal edges, violations, causes

Natural-language query
        |
        v
embedding router -> symbolic plan selector -> plan-scoped ReAct agent
        |
        +-> DuckDB analysis tools
        +-> Milvus semantic tools
        +-> Kùzu graph tools
        +-> deterministic rule engine
        |
        v
answer + route + tool trace + contradiction warnings
```

The physics loop is the numerical source of truth. VLM output provides semantic
context, but vehicle IDs are validated against active tracker IDs before graph
insertion.

## Source Layout

```text
api.py                              FastAPI server and frontend routes
main.py                             video-processing pipeline and CLI agent
frontend/                           dashboard, calibration, and zone interfaces
src/
  physics_engine/
    detector.py                     YOLO, YOLO-Pose, Grounding DINO adapters
    tracker.py                      BoxMOT tracker integration
    homography.py                   ground-point filtering and world projection
    kinematics.py                   motion estimation and quality gating
    lane_classifier.py              bird's-eye lane detection/classification
    zone_manager.py                 zone membership, gates, and OD events
    calibration_router.py           calibration API
    zone_router.py                  zone configuration API
  semantic_abstractor/
    set_of_mark.py                  tracked-ID and zone-focused rendering
    vlm_inference.py                Ollama VLM scene interpretation
    entity_extractor.py             triple validation and normalization
  memory_layer/
    duckdb_client.py                trajectories, crossings, alerts, traces
    milvus_client.py                semantic events and entity profiles
    graph_client.py                 temporal and causal relationship graph
  symbolic_engine/
    rule_engine.py                  deterministic traffic rules
    threshold_registry.py           threshold definitions and explanations
    alert_engine.py                 reusable frame-level alert detector
  agentic_orchestrator/
    hierarchical_router.py          embedding-based query routing
    sequential_pipeline.py          planning and LangGraph ReAct workflow
    tools.py                        traffic-analysis tool implementations
```

## Physics Pipeline

### Detection and tracking

The main processing path loads a pose detector and uses keypoint 31, the rear
wheel ground-contact point, as the preferred homography reference. If pose data
is unavailable, the coordinate transformer can fall back to the bounding-box
bottom centre.

Pose detections are associated with tracked boxes using IoU. A per-track Kalman
filter smooths pixel-space ground points and bridges short keypoint dropouts.
Projection outside the calibrated world region is marked as lower quality.

The tracker defaults to ByteTrack. The reusable tracker adapter also supports
the BoxMOT tracker configurations installed in the environment.

### Kinematics

High-frame-rate video is subsampled to a target physics rate of approximately
20 Hz while timestamps remain aligned with source-video time.

For each track, the estimator derives:

| Field | Meaning |
|---|---|
| `pos_x`, `pos_y` | homography-projected position in metres |
| `vel_x`, `vel_y` | median velocity over a trailing 0.25-second window |
| `speed_ms` | velocity magnitude |
| `accel_x`, `accel_y` | velocity change over a trailing 0.5-second window |
| `interpolated` | row was generated to fill a detector gap |
| `measurement_used` | a measured ground point contributed to the state |
| `predicted_only` | state relied on a prediction rather than a measurement |
| `outlier_rejected` | implausible motion was rejected |
| `track_warm` | sufficient history exists for stable estimates |
| `trusted_for_rules` | row is eligible for symbolic rule evaluation |
| `association_iou` | pose-to-track association quality |
| `inside_calibration_core` | position lies inside the calibrated region |

Default plausibility limits are 140 km/h maximum speed, 25 km/h/s maximum
positive acceleration, and 35 km/h/s maximum braking magnitude.

### Zones, gates, and lanes

`zone_config.json` defines:

- a zone polygon;
- named entry, exit, or bidirectional gates;
- a zone speed limit;
- an optional expected traffic-flow direction.

Gate-line intersections produce `confirmed` crossing events. Polygon membership
changes that occur without a detected line intersection produce `estimated`
events assigned to the nearest gate.

If `lane_config.json` exists, each world-Y position is assigned a zero-based
lane index. The stored class label becomes values such as `car:lane0`.

## Semantic Pipeline

The semantic worker runs in a bounded background queue so slow VLM inference
does not block detection and tracking. Tasks are dropped when the queue is full
instead of allowing unbounded memory growth.

Approximately three semantic samples are attempted per second. Static scenes
are skipped using frame-difference motion gating.

Each semantic task:

1. dims areas outside the configured zone;
2. draws zone and gate context;
3. adds Set-of-Mark vehicle IDs;
4. restricts visible IDs and physics context to zone occupants when possible;
5. sends a short buffered frame sequence to Ollama `gemma4:e2b`;
6. validates extracted entities against active track IDs;
7. writes event descriptions to Milvus and relationships to Kùzu.

Milvus contains two collections:

- `traffic_events`: time-windowed VLM event descriptions;
- `entity_profiles`: longitudinal behavioral summaries per tracked vehicle.

Kùzu stores typed `Vehicle`, `Pedestrian`, and `Infrastructure` nodes with:

- `INTERACTS_WITH` relationships;
- `PRECEDES` temporal relationships;
- `HAS_VIOLATION` symbolic feedback relationships;
- `CAUSES` agent-attributed causal relationships.

## Agent and Analysis Tools

The query router compares the query embedding with prototype sets and selects:

- `semantic_lookup`: Milvus event/profile retrieval only;
- `full_analysis`: symbolic planning plus a focused set of analysis tools.

For full analysis, a deterministic keyword planner selects a plan such as
vehicle type, incident, conflict, flow, volume, queue, turning movement, speed
compliance, or relational analysis. Only the tools relevant to that plan are
bound to the LLM.

Available analysis includes:

- semantic event and entity-profile search;
- Kùzu relationship queries;
- trajectory and kinematic verification;
- vehicle-type lookup;
- deterministic rule evaluation;
- zone flow and OD analysis;
- vehicle proximity and TTC;
- multi-vehicle kinematic comparison;
- interval snapshots;
- V85 and speed statistics;
- traffic volume reports;
- queue detection;
- turning-movement counts;
- data-quality reports;
- threshold explanations;
- causal links and reasoning-trace retrieval.

Every query receives a session ID. Tool calls and output excerpts are persisted
to DuckDB. A final contradiction check warns when semantic descriptions indicate
normal traffic while deterministic rules find violations.

## Symbolic Traffic Rules

The rule engine evaluates trusted trajectory rows using explicit thresholds:

| Rule | Default condition |
|---|---|
| `SPEEDING` | sustained speed above the configured zone limit, default 50 km/h |
| `HARD_BRAKING` | signed acceleration below -3.0 m/s² |
| `AGGRESSIVE_ACCELERATION` | signed acceleration above 3.0 m/s² |
| `WRONG_WAY` | heading differs from configured flow by more than 90° |
| `STATIONARY_IN_LANE` | below 0.5 m/s for more than 10 seconds |
| `TAILGATING` | time headway below 1.5 seconds for more than 3 seconds |

Results include severity, severity score, exact evidence, and any assumptions
used. Threshold provenance can be queried through the agent.

## Installation

Requirements:

- Python 3.12 or later
- OpenCV-compatible video support
- Ollama for semantic video processing
- an NVIDIA GPU is recommended but not required

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pull the default local semantic models:

```bash
ollama pull gemma4:e2b
ollama pull all-minilm
```

## Agent Configuration

Create a local `.env` file. It is ignored by Git.

Local Ollama:

```dotenv
AGENT_LLM_PROVIDER=ollama
AGENT_MODEL=gemma4:e2b
AGENT_EMBED_MODEL=all-minilm
```

OpenAI-compatible provider:

```dotenv
AGENT_LLM_PROVIDER=openai
AGENT_API_KEY=your-key
AGENT_API_BASE_URL=https://api.groq.com/openai/v1
AGENT_MODEL=llama-3.3-70b-versatile
```

When the provider is not `ollama`, intent routing and Milvus use the local
`all-MiniLM-L6-v2` sentence-transformer. The video semantic worker currently
uses Ollama `gemma4:e2b` independently of the agent provider.

## Quick Start

Start Ollama when using semantic processing or a local agent:

```bash
ollama serve
```

Start the API:

```bash
./start.sh
```

Alternatively:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Open:

- dashboard: <http://localhost:8000/>
- calibration: <http://localhost:8000/calibrate-ui>
- zone editor: <http://localhost:8000/zone-ui>
- API documentation: <http://localhost:8000/docs>

## Processing a Video

### 1. Upload

```bash
curl -X POST http://localhost:8000/upload_video/ \
  -F "file=@traffic_sample.mp4"
```

The file is saved under `data/raw_videos/`. Uploading does not start processing.

### 2. Calibrate

Open `/calibrate-ui`, select the video, and provide at least four image-to-world
point pairs. Save the result as `calibration.yaml`.

The calibration API also supports named landmarks and KML point import.

### 3. Configure a zone

Open `/zone-ui` to draw the polygon and gates. Save the configuration as
`zone_config.json`. This step is optional, but zone flow, OD, turning movements,
zone-scoped semantics, and per-zone limits depend on it.

### 4. Run

```bash
curl -X POST http://localhost:8000/run_physics/ \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "traffic_sample.mp4",
    "run_physics": true,
    "run_vlm": true
  }'
```

The response contains a job ID:

```json
{
  "job_id": "3f16...",
  "status": "pending",
  "filename": "traffic_sample.mp4"
}
```

Poll progress:

```bash
curl http://localhost:8000/job/3f16...
```

Only one processing job can run in the current single-process job registry.

Set `run_vlm` to `false` for physics-only processing. `run_physics=false` is
available for semantic experiments, but normal traffic analysis requires
physics and a valid calibration.

### 5. Query

```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"query":"What was the 85th percentile speed?"}'
```

The response includes the answer, selected route, session ID, reasoning steps,
and contradiction warnings.

Example questions:

```text
How many vehicles entered through the North gate?
What is the origin-destination matrix?
Which vehicle had the highest speed?
Were any motorcycles speeding?
Was there a queue, and how long did it last?
How close did Vehicle 4 get to Vehicle 9?
What was their minimum time-to-collision?
Which vehicles were tailgating?
Explain the threshold used for hard braking.
Show the reasoning trace for session <session-id>.
```

## Useful API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health/` | service health |
| `GET` | `/videos` | list available MP4 files |
| `POST` | `/upload_video/` | save a video |
| `POST` | `/run_physics/` | start a processing job |
| `GET` | `/job/{job_id}` | poll processing status |
| `POST` | `/chat/` | query the agent |
| `GET` | `/stats/` | memory-layer counts |
| `GET` | `/models/` | list uploaded model weights |
| `POST` | `/upload_model/` | upload `.pt` weights |
| `GET` | `/stream` | MJPEG detection/tracking preview |
| `GET` | `/calibrate/status` | calibration status |
| `POST` | `/calibrate/compute` | compute and save homography |
| `GET` | `/zone/status` | zone status |
| `POST` | `/zone/save` | save zone configuration |

Preview a video with tracked boxes:

```text
http://localhost:8000/stream?video=data/raw_videos/traffic_sample.mp4
```

Set-of-Mark preview:

```text
http://localhost:8000/stream?video=data/raw_videos/traffic_sample.mp4&som=true
```

The preview endpoint accepts `tracker` and `conf` query parameters.

## Data Storage

| Store | Default path | Main contents |
|---|---|---|
| DuckDB | `data/duckdb_storage/physics_vectors.duckdb` | trajectories, quality flags, crossings, alerts, reasoning traces, calibration metadata |
| Milvus Lite | `data/milvus_storage/semantic_memory.db` | event embeddings and vehicle profiles |
| Kùzu | `data/graph_storage/traffic_graph_db` | entities, interactions, temporal edges, violations, causal edges |
| Calibration | `calibration.yaml` | homography and named world points |
| Zone | `zone_config.json` | polygon, gates, speed limit, flow direction |
| Lanes | `lane_config.json` | world-Y lane boundaries |

Generated databases, videos, model weights, result files, PDFs, and notebooks
are excluded from Git by the project `.gitignore`.

To reset generated memory:

```bash
rm -rf data/duckdb_storage data/milvus_storage data/graph_storage
```

Keep `calibration.yaml`, `zone_config.json`, and `lane_config.json` if the camera
setup has not changed.

## Testing

Compile the Python source:

```bash
python -m compileall -q api.py main.py src tests
```

Run the tool-selection evaluation:

```bash
python tests/eval_tool_selection.py
```

The router evaluation requires either the configured Ollama embedding model or
the local sentence-transformer backend. Full agent evaluation additionally
requires a working LLM provider:

```bash
python tests/eval_tool_selection.py --full
```

`tests/eval_tool_selection.py` is an evaluation program rather than a collection
of pytest test functions, so invoking `pytest` alone does not run this suite.

## Current Limitations

- Homography accuracy is strongest inside the convex hull of calibration points.
- Kinematics remain untrusted until enough clean measurements have accumulated.
- Long occlusions can cause tracker ID reassignment and incomplete OD histories.
- Lane classification assumes lanes are separable by world-Y boundaries.
- VLM output is probabilistic even though entity IDs and rule results are validated.
- The API job registry and alert history are process-local and are not suitable
  for multi-worker deployment without an external store.
- The reusable `AlertEngine` and SSE alert endpoints exist, but the current
  `process_video` path does not instantiate the alert engine.
- Semantic processing currently requires the Ollama `gemma4:e2b` model.
