# Frontend

Three.js + TypeScript + Vite client for visualizing and retargeting live pose data.

## Install

Run from repository root:

```bash
cd frontend
npm install
```

## Run

```bash
cd frontend
npm run dev
```

The app expects backend websocket frames at:

- `ws://<hostname>:8000/ws`

And uses backend camera streaming at:

- `http://<hostname>:8000/camera.mjpeg`

In the in-app Render panel you can toggle:

- `Camera Feed` to show/hide the live camera panel.
- `ML Pose Overlay` to show/hide the model-estimated 2D pose over the camera feed.

## Avatar Selection

The frontend now defaults to a procedural humanoid rig (no FBX dependency).

- Procedural avatar (default): open the app normally.
- Mixamo Y-Bot (optional): add `?avatar=mixamo` to the frontend URL.

Example:

`http://localhost:5173/?avatar=mixamo`

The procedural avatar intentionally uses a minimal set of body primitives
(torso/head + upper/lower arms and legs + feet) to keep visualization clean
while still preserving elbow/knee bends.

## Build

```bash
cd frontend
npm run build
npm run preview
```