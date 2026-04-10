# Backend

FastAPI pose streaming service using MediaPipe Tasks and OpenCV.

## Install

Run from repository root:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r backend/requirements.txt
```

The backend uses MediaPipe Tasks Pose Landmarker. On first startup it tries to
download a default model to `backend/models/pose_landmarker_full.task`.
If your environment has no internet access, set `POSESENSE_POSE_MODEL_PATH`
to a local `.task` model file before starting the server.

## Run

```bash
uvicorn backend.server:app --reload --host 127.0.0.1 --port 8000
```

## Endpoints

- `GET /`: simple websocket test page
- `GET /health`: backend health and camera readiness
- `GET /camera.mjpeg`: live camera MJPEG stream
- `WS /ws`: live pose stream (`poseFrame` payloads)

## Notes

- The backend continuously retries camera initialization if no capture device is immediately available.
- Start from repository root so module imports resolve as a package (`backend.server`).
