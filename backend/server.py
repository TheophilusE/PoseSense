import asyncio
import os
import logging
import uvicorn
from typing import Optional, Set
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

try:
    from .processor import PoseProcessor, select_best_camera
except ImportError:  # Allows running this file directly from backend/.
    from processor import PoseProcessor, select_best_camera

logger = logging.getLogger(__name__)

clients: Set[WebSocket] = set()
pose: Optional[PoseProcessor] = None
pose_init_error: Optional[str] = None


def init_pose_processor() -> Optional[PoseProcessor]:
    global pose_init_error
    cap = select_best_camera()
    if cap is None:
        pose_init_error = "No camera source available."
        logger.warning("No camera source available; backend will keep retrying.")
        return None

    try:
        processor = PoseProcessor(source=cap, fps=30.0)
        pose_init_error = None
        return processor
    except Exception as exc:
        pose_init_error = str(exc)
        logger.exception("Failed to initialize pose processor: %s", exc)
        try:
            cap.release()
        except Exception:
            pass
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pose
    # Startup
    pose = init_pose_processor()
    task = asyncio.create_task(producer())
    yield
    # Shutdown
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

app = FastAPI(lifespan=lifespan)

@app.get("/")
def index():
    return HTMLResponse("""
<!doctype html>
<html>
  <body>
    <h3>Pose WebSocket</h3>
    <pre id="log"></pre>
    <script>
      const log = document.getElementById('log');
      const ws = new WebSocket(`ws://${location.host}/ws`);
      ws.onmessage = (e) => {
        log.textContent = e.data + "\\n" + log.textContent;
      };
    </script>
  </body>
</html>
""")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "clients": len(clients),
        "pose_ready": pose is not None,
        "pose_error": pose_init_error,
    }

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            # Keep alive ping if desired
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        clients.discard(ws)

async def producer():
    global pose
    frame_idx = 0
    while True:
        if not clients:
            await asyncio.sleep(0.01)
            continue

        if pose is None:
            pose = init_pose_processor()
            await asyncio.sleep(0.5)
            continue

        payload = pose.process(frame_idx)
        frame_idx += 1
        if payload:
            msg = json_dumps(payload)
            # broadcast
            await asyncio.gather(*[
                c.send_text(msg) for c in list(clients)
            ], return_exceptions=True)
        await asyncio.sleep(0)  # yield to event loop

# Compact and fast JSON dumps
try:
    from msgspec.json import encode as _encode
    def json_dumps(obj): return _encode(obj).decode("utf-8")
except Exception:
    import json
    def json_dumps(obj): return json.dumps(obj, separators=(",", ":"))

if __name__ == "__main__":
    import importlib.util

    app_target = "server:app"
    try:
        if importlib.util.find_spec("backend.server") is not None:
            app_target = "backend.server:app"
    except ModuleNotFoundError:
        app_target = "server:app"

    uvicorn.run(
        app_target,
        host=os.getenv("POSESENSE_HOST", "127.0.0.1"),
        port=int(os.getenv("POSESENSE_PORT", "8000")),
        reload=os.getenv("POSESENSE_RELOAD", "0") == "1",
    )
