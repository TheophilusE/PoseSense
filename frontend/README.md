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

The procedural avatar also uses a dedicated high-response retarget profile,
with stronger motion amplification and world-relative root motion so movement
is easier to read.

For procedural mode, the avatar root is grounded to the world floor (`y=0`)
and feet are corrected if they penetrate below the ground plane.

Procedural mode also runs an analytic two-bone IK pass for each leg using
backend intermediate targets (`left_knee`/`left_ankle`, `right_knee`/`right_ankle`)
to keep knees bending naturally and reduce floating-leg artifacts.

## Build

```bash
cd frontend
npm run build
npm run preview
```