#!/usr/bin/env bash
# ============================================================
#  TrafficAgent — Server Launcher
#  Kills any stale process on PORT before starting, and
#  releases the port cleanly on Ctrl+C or SIGTERM.
# ============================================================

PORT=8000

# ---- helpers ----
free_port() {
    local pids
    pids=$(lsof -t -i TCP:"$PORT" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "[start.sh] Killing existing process(es) on port $PORT: $pids"
        kill -9 $pids 2>/dev/null
        sleep 0.5
    fi
}

cleanup() {
    echo ""
    echo "[start.sh] Shutting down..."
    # Kill the uvicorn process group we started
    if [ -n "$UVICORN_PID" ] && kill -0 "$UVICORN_PID" 2>/dev/null; then
        kill -TERM "$UVICORN_PID" 2>/dev/null
        wait "$UVICORN_PID" 2>/dev/null
    fi
    free_port
    echo "[start.sh] Port $PORT released. Bye."
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# ---- clear the port before starting ----
free_port

# ---- activate venv if present ----
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "[start.sh] Starting TrafficAgent API on http://localhost:$PORT"
echo "[start.sh] Press Ctrl+C to stop."
echo ""

# Run uvicorn in the foreground; capture PID for cleanup
uvicorn api:app --reload --host 0.0.0.0 --port "$PORT" &
UVICORN_PID=$!

# Wait for uvicorn to exit (normal or crash)
wait "$UVICORN_PID"
UVICORN_PID=""
