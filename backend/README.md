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

## Pose Payload Notes

`poseFrame.meta` now includes intermediate bridge-rig data for retargeting:

- `intermediate_targets`: solved humanoid target points (pelvis/chest/neck/head, limbs, feet, hands)
- `intermediate_targets_raw`: raw targets directly from MediaPipe landmarks
- `intermediate_targets_normalized`: pelvis-centered and scale-normalized targets
- `segment_directions`: normalized segment direction vectors (upper/lower arms and legs, spine, feet, hands)
- `landmark_schema`: currently `mediapipe_pose_33`

The backend solves arm and leg chains with two-bone IK per frame before
producing canonical joint rotations, which keeps chain lengths stable and
reduces jitter from raw landmarks.

## Notes

- The backend continuously retries camera initialization if no capture device is immediately available.
- Start from repository root so module imports resolve as a package (`backend.server`).
