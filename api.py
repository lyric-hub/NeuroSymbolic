import asyncio
import json
import shutil
import uuid
import cv2
import numpy as np
from enum import Enum
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pathlib import Path
from typing import List

# Import your existing pipeline and agent
from main import process_video
from src.agentic_orchestrator.sequential_pipeline import agent_app, AGENT_INVOKE_CONFIG
from src.physics_engine.calibration_router import router as calibration_router
from src.physics_engine.zone_router import router as zone_router
from src.symbolic_engine.alert_engine import TrafficAlert

# ---------------------------------------------------------------------------
# Alert broadcast — one asyncio.Queue per connected SSE client.
# Alerts are produced in a background thread (_run_job) and pushed to every
# subscribed client via the main event loop using call_soon_threadsafe.
# ---------------------------------------------------------------------------
_alert_subscribers: List[asyncio.Queue] = []
_main_loop: asyncio.AbstractEventLoop | None = None

# Rolling in-memory buffer so late-connecting clients can fetch recent alerts.
_ALERT_HISTORY_SIZE = 200
from collections import deque
_alert_history: deque = deque(maxlen=_ALERT_HISTORY_SIZE)


def _on_pipeline_alert(alert: TrafficAlert) -> None:
    """
    Called from the background processing thread.
    Buffers the alert in _alert_history and broadcasts to all SSE subscribers.
    Thread-safe: uses call_soon_threadsafe to cross the thread boundary.
    """
    if _main_loop is None:
        return
    payload = alert.to_dict()
    # Append to history (deque is not thread-safe but GIL makes single append safe)
    _alert_history.append(payload)
    for q in list(_alert_subscribers):
        _main_loop.call_soon_threadsafe(q.put_nowait, payload)

app = FastAPI(
    title="Neuro-Symbolic Traffic Agent API",
    description="API for video analytics and agentic reasoning",
    version="1.0.0"
)


@app.on_event("startup")
async def _startup() -> None:
    """Captures the event loop reference used by the alert broadcaster."""
    global _main_loop
    _main_loop = asyncio.get_event_loop()

app.include_router(calibration_router)
app.include_router(zone_router)

# Allow frontend frameworks (React, Vue, Streamlit) to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, change this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Job Registry ---
class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"

# In-memory store: job_id -> {status, filename, error}
# Sufficient for single-process deployments; replace with Redis for multi-worker setups.
job_registry: dict[str, dict] = {}

def _run_job(job_id: str, video_path: str):
    """Wrapper that updates job_registry around the blocking process_video call."""
    job_registry[job_id]["status"] = JobStatus.PROCESSING

    def on_progress(frames_done: int, total_frames: int) -> None:
        job_registry[job_id]["frames_processed"] = frames_done
        job_registry[job_id]["total_frames"] = total_frames

    try:
        process_video(
            video_path,
            progress_callback=on_progress,
            alert_callback=_on_pipeline_alert,
        )
        job_registry[job_id]["status"] = JobStatus.DONE
    except Exception as e:
        job_registry[job_id]["status"] = JobStatus.FAILED
        job_registry[job_id]["error"] = str(e)

# --- Data Models ---
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    query: str
    summary: str

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    filename: str
    frames_processed: int | None = None
    total_frames: int | None = None
    error: str | None = None

# --- Endpoints ---

@app.post("/upload_video/", response_model=JobResponse, status_code=202)
async def upload_and_process_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Uploads a video and starts background processing.
    Returns a job_id that can be polled via GET /job/{job_id}.
    """
    if not file.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    # Prevent concurrent jobs — all three databases are single-writer.
    active = [j for j in job_registry.values() if j["status"] == JobStatus.PROCESSING]
    if active:
        raise HTTPException(
            status_code=409,
            detail="A processing job is already running. Wait for it to finish before uploading another video.",
        )

    save_path = Path(f"data/raw_videos/{file.filename}")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job_id = str(uuid.uuid4())
    job_registry[job_id] = {
        "status": JobStatus.PENDING,
        "filename": file.filename,
        "frames_processed": None,
        "total_frames": None,
        "error": None,
    }

    background_tasks.add_task(_run_job, job_id, str(save_path))

    return JobResponse(job_id=job_id, status=JobStatus.PENDING, filename=file.filename)


@app.get("/job/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """
    Polls the status of a video processing job.
    Frontend should poll this until status is 'done' or 'failed' before enabling the chat UI.
    """
    job = job_registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobResponse(job_id=job_id, **job)

@app.post("/chat/", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest):
    """
    Endpoint to talk to the LangGraph Agent.
    Passes the natural language query to the Hierarchical Router.
    """
    # Querying while the video is still being ingested returns incomplete results.
    active = [j for j in job_registry.values() if j["status"] == JobStatus.PROCESSING]
    if active:
        raise HTTPException(
            status_code=409,
            detail="Video is still being processed. Wait for the job to finish before querying.",
        )

    import logging as _log
    _log.getLogger(__name__).info("Chat query: %s", request.query)

    try:
        # agent_app.invoke() is synchronous (runs Ollama LLM calls).
        # Running it directly in an async function blocks the event loop and
        # freezes every other request.  asyncio.to_thread() moves it to a
        # thread-pool worker, keeping the event loop free.
        initial_state = {"query": request.query}
        final_state = await asyncio.to_thread(
            agent_app.invoke, initial_state, AGENT_INVOKE_CONFIG
        )

        summary = final_state.get("final_summary", "No summary generated.")
        return ChatResponse(query=request.query, summary=summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts/stream")
async def stream_alerts():
    """
    Server-Sent Events endpoint for real-time traffic alerts.

    Each connected client gets its own queue; alerts are broadcast to all
    active connections simultaneously.  A heartbeat comment is sent every
    15 s to keep the connection alive through proxies and load balancers.

    Connect from JavaScript::

        const es = new EventSource("/alerts/stream");
        es.onmessage = (e) => {
            const alert = JSON.parse(e.data);
            console.log(alert.alert_type, alert.message);
        };

    Alert payload fields:
        alert_type, severity, track_id, involved_ids,
        timestamp, frame_id, message, evidence
    """
    q: asyncio.Queue = asyncio.Queue()
    _alert_subscribers.append(q)

    async def _generator():
        try:
            while True:
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {json.dumps(payload)}\n\n"
                except asyncio.TimeoutError:
                    # SSE comment — invisible to onmessage, keeps TCP alive
                    yield ": heartbeat\n\n"
        finally:
            _alert_subscribers.remove(q)

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # Disable nginx output buffering
        },
    )


@app.get("/alerts/history")
async def get_alert_history(limit: int = Query(50, ge=1, le=500)):
    """
    Returns the most recent alerts buffered in-process.

    Useful for a frontend that connects after alerts have already fired
    (e.g. page reload mid-processing).  Only keeps the last
    ``_ALERT_HISTORY_SIZE`` alerts in memory.
    """
    return {"alerts": list(_alert_history)[-limit:]}


@app.get("/stream")
async def stream_video(
    video: str = Query(..., description="Path to video file"),
    tracker: str = Query("bytetrack", description="Tracker name"),
    conf: float = Query(0.3, description="Detection confidence threshold"),
    som: bool = Query(False, description="Use Set-of-Mark rendering instead of bounding boxes"),
):
    """
    Streams a video with live detection and tracking overlaid as MJPEG.

    Normal mode:  http://localhost:8000/stream?video=data/ulloor/your_file.mp4
    SoM mode:     http://localhost:8000/stream?video=data/ulloor/your_file.mp4&som=true
    """
    from src.physics_engine.detector import load_detector
    from src.physics_engine.tracker import VehicleTracker
    from src.semantic_abstractor.set_of_mark import AdaptiveRenderer, RenderContext

    video_path = Path(video)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video}")

    rng = np.random.default_rng(42)
    palette = rng.integers(80, 255, size=(256, 3)).tolist()

    def _generate_frames():
        detector = load_detector("models/best.pt", conf=conf)
        vt       = VehicleTracker(tracker_name=tracker)
        renderer = AdaptiveRenderer() if som else None
        ctx      = RenderContext() if som else None
        cap      = cv2.VideoCapture(str(video_path))
        fps      = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_id = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                raw_dets      = detector.predict(frame)
                tracked_boxes = vt.update(raw_dets[0], frame)

                if som:
                    ctx.update(tracked_boxes, frame_id / fps)
                    renderer.render(frame, ctx)
                else:
                    for box in tracked_boxes:
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                        track_id = int(box[4])
                        cls_id   = int(box[6]) if len(box) > 6 else 0
                        conf_val = float(box[5]) if len(box) > 5 else 0.0

                        colour   = palette[track_id % 256]
                        cls_name = detector.model.names.get(cls_id, str(cls_id))
                        label    = f"#{track_id} {cls_name} {conf_val:.2f}"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
                        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + jpeg.tobytes() + b"\r\n")
                frame_id += 1
        finally:
            cap.release()

    return StreamingResponse(
        _generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/videos")
async def list_videos():
    """Lists available videos in data/."""
    video_dir = Path("data")
    videos = sorted(str(p) for p in video_dir.rglob("*.mp4"))
    return {"videos": videos}


@app.get("/health/")
async def health_check():
    """Simple check to see if the API is alive."""
    return {"status": "healthy", "system": "Neuro-Symbolic Agentic Brain"}

# --- Serve Frontend ---
# Must be AFTER all API routes to avoid catching API paths
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def serve_frontend():
    """Serves the frontend dashboard."""
    return FileResponse("frontend/index.html")

@app.get("/calibrate-ui")
async def serve_calibration_ui():
    """Serves the web-based camera calibration tool."""
    return FileResponse("frontend/calibration.html")

@app.get("/zone-ui")
async def serve_zone_ui():
    """Serves the interactive zone and gate drawing tool."""
    return FileResponse("frontend/zone.html")
