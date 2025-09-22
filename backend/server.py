import asyncio
import uvicorn
from typing import Set
from contextlib import asynccontextmanager
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from processor import PoseProcessor, select_best_camera

clients: Set[WebSocket] = set()
pose: PoseProcessor     = PoseProcessor(source=select_best_camera(), fps=30.0)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
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
    frame_idx = 0
    while True:
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
    uvicorn.run("server:app", host="localhost", port=8000, reload=False)
