# Backend

FastAPI pose streaming service using MediaPipe and OpenCV.

## Install

Run from repository root:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r backend/requirements.txt
```

## Run

```bash
uvicorn backend.server:app --reload --host 127.0.0.1 --port 8000
```

## Endpoints

- `GET /`: simple websocket test page
- `GET /health`: backend health and camera readiness
- `WS /ws`: live pose stream (`poseFrame` payloads)

## Notes

- The backend continuously retries camera initialization if no capture device is immediately available.
- Start from repository root so module imports resolve as a package (`backend.server`).
