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

From the repository root:

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r backend/requirements.txt
uvicorn backend.server:app --reload --host 127.0.0.1 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
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

The frontend connects to `ws://<hostname>:8000/ws` for live pose frames.