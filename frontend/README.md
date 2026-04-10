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

## Build

```bash
cd frontend
npm run build
npm run preview
```