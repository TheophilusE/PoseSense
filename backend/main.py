from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from .posestream import mediapipe_live_stream

app = FastAPI(title="PoseSense MediaPipe Backend", version="1.0.0")

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
        async for frame in mediapipe_live_stream():
            await websocket.send_json(frame)
    except WebSocketDisconnect:
        print("Client disconnected")
