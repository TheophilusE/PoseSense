# PoseSense

PoseSense is split into two independently runnable applications:

- `backend/`: FastAPI + MediaPipe + OpenCV pose processing service
- `frontend/`: Three.js + TypeScript + Vite visualization client

## Project Structure

```text
PoseSense/
	backend/
		server.py
		processor.py
		canonical.py
		posemath.py
		requirements.txt
		...
	frontend/
		index.html
		package.json
		vite.config.js
		src/
		public/
		...
	LICENSE
	README.md
```

## Backend Setup

Backend pose inference uses MediaPipe Tasks Pose Landmarker.

From the repository root:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r backend/requirements.txt
uvicorn backend.server:app --reload --host 127.0.0.1 --port 8000
```

On first run, the backend attempts to download the default model to
`backend/models/pose_landmarker_full.task`.
You can provide your own model path with `POSESENSE_POSE_MODEL_PATH`.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Live camera stream:

```bash
http://127.0.0.1:8000/camera.mjpeg
```

## Frontend Setup

From the repository root:

```bash
cd frontend
npm install
npm run dev
```

## Convenience Scripts (Repo Root)

You can run both apps from the repository root using npm scripts.

1. Install workspace script dependencies:

```bash
npm install
```

2. Ensure frontend dependencies are installed:

```bash
npm --prefix frontend install
```

3. Activate your backend virtual environment and start both services:

```bash
source .venv/Scripts/activate
npm run dev
```

Additional scripts:

- `npm run backend:dev` starts only backend.
- `npm run frontend:dev` starts only frontend.
- `npm run build` runs frontend production build.

Production build:

```bash
cd frontend
npm run build
npm run preview
```

## Dev Workflow

1. Start backend on port `8000`.
2. Start frontend (Vite defaults to `5173`).
3. Open the frontend URL shown by Vite.
4. Use the Render panel toggles for `Camera Feed` and `ML Pose Overlay`.

The frontend connects to `ws://<hostname>:8000/ws` for live pose frames.

## Retargeting Pipeline

Runtime retargeting now follows a bridge-rig flow:

1. MediaPipe world landmarks are converted into an intermediate humanoid target set.
2. Backend solves two-bone IK for arms and legs to keep chain lengths stable.
3. Canonical joint rotations are streamed over websocket.
4. Frontend retargets canonical joints onto the Mixamo skeleton (including feet/hands/head where available).