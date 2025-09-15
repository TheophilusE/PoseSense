from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from .posestream import mmpose_live_stream

app = FastAPI(title="PoseSense Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/pose")
async def posestream(websocket: WebSocket):
    await websocket.accept()
    try:
        async for frame in mmpose_live_stream():
            await websocket.send_json(frame)
    except WebSocketDisconnect:
        print("Client disconnected")
