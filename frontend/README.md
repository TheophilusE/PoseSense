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

## Build

```bash
cd frontend
npm run build
npm run preview
```