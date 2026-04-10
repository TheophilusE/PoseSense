import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { SimpleSkeletonHelper, setWireframeForObject, ServerSkeletonHelper } from './utils/renderUtils.js';

import { PoseStream } from "./network.js";
import { Retargeter } from "./retargeter.js";
import type { PoseFrame } from "./types.js";
import { loadYBotFbx } from './loadfbx.js';
import { quatFromArray, vec3FromArray, slerpQuat, lerpVec3 } from './utils.js';

// Simple TypeScript favicon injector: creates an inline SVG and sets it as the
// page favicon using a data URI. This avoids adding files to the repo and
// works in modern browsers.
function setFaviconSvg(): void {
  const svg = `<?xml version="1.0" encoding="UTF-8"?>` +
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">` +
    `<rect width="64" height="64" rx="10" fill="#0f1724"/>` +
    `<circle cx="32" cy="32" r="26" fill="#1a73e8"/>` +
    `<circle cx="32" cy="20" r="5" fill="#ffffff"/>` +
    `<path d="M32 25 L32 38" stroke="#ffffff" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>` +
    `<path d="M32 30 L22 36" stroke="#ffffff" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>` +
    `<path d="M32 30 L42 36" stroke="#ffffff" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>` +
    `<path d="M32 38 L24 50" stroke="#ffffff" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>` +
    `<path d="M32 38 L40 50" stroke="#ffffff" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>` +
    `</svg>`;

  const url = 'data:image/svg+xml;utf8,' + encodeURIComponent(svg);
  let link = document.querySelector("link[rel~='icon']") as HTMLLinkElement | null;
  if (!link) {
    link = document.createElement('link');
    link.rel = 'icon';
    document.head.appendChild(link);
  }
  link.href = url;
}

setFaviconSvg();

const statusEl = document.getElementById('status')!;
const fpsEl = document.getElementById('fps')!;
const netEl = document.getElementById('net')!;

type PoseLandmark2D = [number, number, number];
const POSE_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
  [11, 12],
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24],
  [23, 25], [24, 26], [25, 27], [26, 28],
  [27, 29], [28, 30], [29, 31], [30, 32],
  [27, 31], [28, 32]
];

// Performance / debug stats panel
const statsEl = document.createElement('div');
statsEl.id = 'stats';
statsEl.style.position = 'fixed';
statsEl.style.right = '12px';
statsEl.style.top = '12px';
statsEl.style.padding = '12px 14px';
// Glassy 'frosted glass' panel — modern sleek typography + layout
statsEl.style.background = 'linear-gradient(180deg, rgba(255,255,255,0.72), rgba(255,255,255,0.48))';
statsEl.style.border = '1px solid rgba(255,255,255,0.6)';
statsEl.style.color = 'rgba(11,18,32,0.92)';
statsEl.style.fontFamily = "Inter, ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial";
statsEl.style.width = '220px';
statsEl.style.fontSize = '13px';
statsEl.style.lineHeight = '1.3';
statsEl.style.borderRadius = '12px';
statsEl.style.boxShadow = '0 12px 40px rgba(16,24,32,0.12)';
// blur backdrop for glass effect (with webkit fallback)
(statsEl.style as any).backdropFilter = 'blur(10px) saturate(120%)';
(statsEl.style as any).webkitBackdropFilter = 'blur(10px) saturate(120%)';
statsEl.style.zIndex = '9999';
statsEl.innerHTML = `
<div id="stat-header" style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;cursor:grab">
  <div style="display:flex;flex-direction:column">
    <div style="font-weight:700;font-size:14px">Render</div>
    <div style="font-size:11px;color:rgba(11,18,32,0.45);letter-spacing:0.6px">stats</div>
  </div>
  <div style="display:flex;align-items:center;gap:8px">
    <button id="stat-reset" title="Reset min/max" style="background:transparent;border:none;cursor:pointer;padding:6px;border-radius:6px;font-weight:700;color:rgba(11,18,32,0.7)">Reset</button>
  </div>
</div>
<div style="display:grid;grid-template-columns:1fr auto;gap:6px 12px;align-items:center">
  <div style="color:rgba(11,18,32,0.6)">FPS</div><div id="stat-fps" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;font-weight:700">--</div>
  <div style="color:rgba(11,18,32,0.6)">Frame</div><div id="stat-frame" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;font-weight:700">-- ms</div>
  <div style="color:rgba(11,18,32,0.6)">Min / Max / Avg</div><div id="stat-minmax" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace">-- / -- / --</div>
  <div style="color:rgba(11,18,32,0.6)">Triangles</div><div id="stat-tris" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace">--</div>
  <div style="color:rgba(11,18,32,0.6)">Draws</div><div id="stat-draw" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace">--</div>
  <div style="color:rgba(11,18,32,0.6)">Camera</div><div id="stat-cam" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace">pos(--)</div>
  <div style="color:rgba(11,18,32,0.6)">Memory</div><div id="stat-mem" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace">-- GB</div>
  <div style="color:rgba(11,18,32,0.6)">GPU Render</div><div id="stat-gpu" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace">-- ms</div>
  <div style="color:rgba(11,18,32,0.6)">Retargeter</div><div id="stat-ret" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace">-- ms</div>
</div>
<div style="display:flex;gap:8px;margin-top:10px;align-items:center">
  <canvas id="spark-frame" style="flex:1;min-width:0;height:40px;border-radius:6px;background:rgba(255,255,255,0.03)"></canvas>
</div>
`;
document.body.appendChild(statsEl);
const statFps = document.getElementById('stat-fps')!;
const statFrame = document.getElementById('stat-frame')!;
const statMinMax = document.getElementById('stat-minmax')!;
const statTris = document.getElementById('stat-tris')!;
const statDraw = document.getElementById('stat-draw')!;
const statCam = document.getElementById('stat-cam')!;
const sparkFrame = document.getElementById('spark-frame') as HTMLCanvasElement | null;
const statToggle = document.getElementById('stat-toggle') as HTMLButtonElement | null;
const statReset = document.getElementById('stat-reset') as HTMLButtonElement | null;
const statMem = document.getElementById('stat-mem')!;
const statGpu = document.getElementById('stat-gpu')!;
const statRet = document.getElementById('stat-ret')!;



// Sparklines: simple circular buffers
const SPARK_LEN = 64;
const frameHistory: number[] = new Array(SPARK_LEN).fill(0);
let sparkIndex = 0;
// load persisted sparklines if present
try {
  const sFr = localStorage.getItem('ps_spark_frame');
  if (sFr) {
    const arr = JSON.parse(sFr) as number[];
    for (let i = 0; i < Math.min(arr.length, SPARK_LEN); ++i) frameHistory[i] = arr[i] ?? 0;
  }
} catch (e) { /* ignore */ }

function drawSparkline(canvas: HTMLCanvasElement | null, data: number[], color = '#1f73ff') {
  if (!canvas) return;
  // Make canvas pixel-ratio aware and responsive
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const rectW = Math.floor((canvas.clientWidth || canvas.width));
  const rectH = Math.floor((canvas.clientHeight || canvas.height));
  canvas.width = rectW * dpr;
  canvas.height = rectH * dpr;
  canvas.style.width = rectW + 'px';
  canvas.style.height = rectH + 'px';
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  // background subtle
  ctx.fillStyle = 'rgba(255,255,255,0.02)';
  ctx.fillRect(0, 0, w, h);

  const max = Math.max(...data, 1e-3);
  const min = Math.min(...data, max);
  const range = Math.max(max - min, 1e-6);

  ctx.beginPath();
  for (let i = 0; i < data.length; ++i) {
    const v = data[(sparkIndex + i) % data.length] ?? 0;
    const x = (i / (data.length - 1)) * w;
    const norm = (v - min) / range;
    const y = h - norm * (h - 4) - 2;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = Math.max(1, 2 * (window.devicePixelRatio || 1));
  ctx.stroke();

  // fill gradient
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, color + '33');
  grad.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.lineTo(w, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  ctx.fillStyle = grad as any;
  ctx.fill();

  // draw current numeric overlay top-right
  const lastVal = data[(sparkIndex + data.length - 1) % data.length] ?? 0;
  ctx.fillStyle = 'rgba(255,255,255,0.9)';
  ctx.font = `${12 * dpr}px ui-monospace, SFMono-Regular, Menlo, Monaco, monospace`;
  ctx.textAlign = 'right';
  ctx.fillText(lastVal.toFixed(1), w - 6 * dpr, 14 * dpr);
}

// Create external toggle so the panel can be hidden but still revealed
const externalToggle = document.createElement('button');
externalToggle.title = 'Toggle stats';
externalToggle.style.position = 'fixed';
externalToggle.style.right = '12px';
externalToggle.style.top = '12px';
externalToggle.style.width = '40px';
externalToggle.style.height = '40px';
externalToggle.style.borderRadius = '8px';
externalToggle.style.background = 'rgba(255,255,255,0.92)';
externalToggle.style.border = '1px solid rgba(0,0,0,0.06)';
externalToggle.style.boxShadow = '0 6px 18px rgba(16,24,32,0.08)';
externalToggle.style.zIndex = '10000';
externalToggle.textContent = '▦';
externalToggle.addEventListener('click', () => {
  if (statsEl.style.display === 'none') statsEl.style.display = 'block'; else statsEl.style.display = 'none';
});
document.body.appendChild(externalToggle);

// internal toggle (hidden) kept for backward compat but not visible
if (statToggle) statToggle.style.display = 'none';

// Reset button clears min/max/avg and spark history
if (statReset) {
  statReset.addEventListener('click', () => {
    statsMin = Number.POSITIVE_INFINITY;
    statsMax = 0;
    statsAvg = 0;
    for (let i = 0; i < frameHistory.length; ++i) frameHistory[i] = 0;
    try { localStorage.removeItem('ps_spark_frame'); } catch (e) { }
  });
}

// Make the panel draggable via header
const header = document.getElementById('stat-header');
if (header) {
  let dragging = false;
  let startX = 0, startY = 0;
  let startLeft = 0, startTop = 0;
  header.addEventListener('pointerdown', (ev) => {
    dragging = true;
    header.setPointerCapture(ev.pointerId);
    startX = ev.clientX;
    startY = ev.clientY;
    const rect = statsEl.getBoundingClientRect();
    startLeft = rect.left;
    startTop = rect.top;
    statsEl.style.right = 'auto';
    statsEl.style.left = `${startLeft}px`;
    statsEl.style.top = `${startTop}px`;
    statsEl.style.cursor = 'grabbing';
  });
  window.addEventListener('pointermove', (ev) => {
    if (!dragging) return;
    const dx = ev.clientX - startX;
    const dy = ev.clientY - startY;
    statsEl.style.left = `${startLeft + dx}px`;
    statsEl.style.top = `${startTop + dy}px`;
  });
  window.addEventListener('pointerup', (ev) => {
    if (!dragging) return;
    dragging = false;
    statsEl.style.cursor = 'grab';
  });
}

// Stats accumulators
let statsCount = 0;
let statsAvg = 0; // average frame ms
let statsMin = Number.POSITIVE_INFINITY;
let statsMax = 0;
let statsLastSampleTime = performance.now();
let statsInterval = 500; // ms between UI updates
let statsAccumFrames = 0;
let statsAccumTime = 0;

// Renderer/scene/camera/controls (same as before)
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Render options (persisted)
const defaultRenderOptions = {
  shadows: true,
  debugSkeleton: false,
  showSkinnedMesh: true,
  wireframe: false,
  serverSkeleton: false,
  cameraFeed: true,
  mlPoseOverlay: true,
};

const renderOptions = (() => {
  try {
    const raw = localStorage.getItem('ps_render_opts');
    if (raw) return { ...defaultRenderOptions, ...JSON.parse(raw) };
  } catch (e) { /* ignore */ }
  return { ...defaultRenderOptions };
})();

// Enable physically-correct lighting and soft shadows for a studio look
renderer.shadowMap.enabled = !!renderOptions.shadows;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
(renderer as any).physicallyCorrectLights = true;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

// Render toolbar (indicator + settings)
const toolbar = document.createElement('div');
toolbar.id = 'render-toolbar';
toolbar.style.position = 'fixed';
toolbar.style.right = '12px';
toolbar.style.bottom = '12px';
toolbar.style.top = 'auto';
toolbar.style.padding = '8px 10px';
toolbar.style.background = 'rgba(255,255,255,0.95)';
toolbar.style.border = '1px solid rgba(0,0,0,0.06)';
toolbar.style.borderRadius = '10px';
toolbar.style.boxShadow = '0 8px 24px rgba(16,24,32,0.08)';
toolbar.style.zIndex = '10001';
toolbar.style.fontFamily = statsEl.style.fontFamily;
toolbar.style.fontSize = '13px';
toolbar.innerHTML = `
  <div style="display:flex;align-items:center;justify-content:space-between;gap:8px">
    <div style="font-weight:700">Render</div>
    <div id="render-indicator" style="font-weight:600;color:#0b1a20;font-size:12px">--</div>
  </div>
  <div id="render-opts" style="display:grid;grid-template-columns:1fr auto;gap:6px 10px;margin-top:8px">
    <div style="color:rgba(11,18,32,0.6)">Shadows</div><div><input id="opt-shadows" type="checkbox"></div>
    <div style="color:rgba(11,18,32,0.6)">Debug Skeleton</div><div><input id="opt-debug-skel" type="checkbox"></div>
    <div style="color:rgba(11,18,32,0.6)">Show Mesh</div><div><input id="opt-show-mesh" type="checkbox"></div>
      <div style="color:rgba(11,18,32,0.6)">Wireframe</div><div><input id="opt-wireframe" type="checkbox"></div>
      <div style="color:rgba(11,18,32,0.6)">Server Skeleton</div><div><input id="opt-server-skel" type="checkbox"></div>
      <div style="color:rgba(11,18,32,0.6)">Camera Feed</div><div><input id="opt-camera-feed" type="checkbox"></div>
      <div style="color:rgba(11,18,32,0.6)">ML Pose Overlay</div><div><input id="opt-ml-overlay" type="checkbox"></div>
  </div>
`;
document.body.appendChild(toolbar);

const optShadows = document.getElementById('opt-shadows') as HTMLInputElement | null;
const optDebugSkel = document.getElementById('opt-debug-skel') as HTMLInputElement | null;
const optShowMesh = document.getElementById('opt-show-mesh') as HTMLInputElement | null;
const optWireframe = document.getElementById('opt-wireframe') as HTMLInputElement | null;
const optServerSkel = document.getElementById('opt-server-skel') as HTMLInputElement | null;
const optCameraFeed = document.getElementById('opt-camera-feed') as HTMLInputElement | null;
const optMlOverlay = document.getElementById('opt-ml-overlay') as HTMLInputElement | null;
const renderIndicator = document.getElementById('render-indicator') as HTMLDivElement | null;

const cameraPanel = document.createElement('div');
cameraPanel.id = 'camera-panel';
cameraPanel.style.position = 'fixed';
cameraPanel.style.left = '12px';
cameraPanel.style.bottom = '12px';
cameraPanel.style.width = '360px';
cameraPanel.style.maxWidth = 'calc(100vw - 24px)';
cameraPanel.style.background = 'rgba(14,16,24,0.76)';
cameraPanel.style.border = '1px solid rgba(255,255,255,0.2)';
cameraPanel.style.borderRadius = '10px';
cameraPanel.style.padding = '8px';
cameraPanel.style.backdropFilter = 'blur(6px)';
cameraPanel.style.zIndex = '10001';
cameraPanel.innerHTML = `
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;color:#e5e9f0;font:12px/1.3 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;">
    <span>Live Camera</span>
    <span id="camera-state">connecting...</span>
  </div>
  <div style="position:relative;width:100%;aspect-ratio:16/9;overflow:hidden;border-radius:8px;background:#10131b;">
    <img id="camera-feed" alt="Live camera stream" style="display:block;width:100%;height:100%;object-fit:cover;" />
    <canvas id="camera-overlay" style="position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;"></canvas>
  </div>
`;
document.body.appendChild(cameraPanel);

const cameraFeedEl = document.getElementById('camera-feed') as HTMLImageElement | null;
const cameraOverlayEl = document.getElementById('camera-overlay') as HTMLCanvasElement | null;
const cameraStateEl = document.getElementById('camera-state') as HTMLSpanElement | null;
const cameraFeedUrl = `${location.protocol}//${location.hostname}:8000/camera.mjpeg`;
let latestPoseLandmarks2d: PoseLandmark2D[] | null = null;
let cameraStreamActive = false;
let cameraRetryTimer: number | null = null;

function clearCameraRetry() {
  if (cameraRetryTimer !== null) {
    window.clearTimeout(cameraRetryTimer);
    cameraRetryTimer = null;
  }
}

function scheduleCameraRetry(delayMs = 1200) {
  if (!renderOptions.cameraFeed || !cameraStreamActive || !cameraFeedEl) return;
  if (cameraRetryTimer !== null) return;

  cameraRetryTimer = window.setTimeout(() => {
    cameraRetryTimer = null;
    if (!renderOptions.cameraFeed || !cameraStreamActive || !cameraFeedEl) return;
    if (cameraStateEl) cameraStateEl.textContent = 'reconnecting...';
    cameraFeedEl.src = `${cameraFeedUrl}?t=${Date.now()}`;
  }, delayMs);
}

function resizeCameraOverlayCanvas() {
  if (!cameraOverlayEl || !cameraFeedEl) return;
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const w = Math.max(1, Math.floor(cameraFeedEl.clientWidth));
  const h = Math.max(1, Math.floor(cameraFeedEl.clientHeight));
  if (cameraOverlayEl.width !== Math.floor(w * dpr) || cameraOverlayEl.height !== Math.floor(h * dpr)) {
    cameraOverlayEl.width = Math.floor(w * dpr);
    cameraOverlayEl.height = Math.floor(h * dpr);
  }
  cameraOverlayEl.style.width = `${w}px`;
  cameraOverlayEl.style.height = `${h}px`;
}

function setCameraFeedEnabled(enabled: boolean) {
  if (!cameraFeedEl || !cameraStateEl) return;

  if (enabled && !cameraStreamActive) {
    cameraStreamActive = true;
    clearCameraRetry();
    cameraStateEl.textContent = 'connecting...';
    cameraFeedEl.src = `${cameraFeedUrl}?t=${Date.now()}`;
  }

  if (!enabled && cameraStreamActive) {
    cameraStreamActive = false;
    clearCameraRetry();
    cameraFeedEl.src = '';
    cameraStateEl.textContent = 'off';
  }
}

function drawCameraPoseOverlay() {
  if (!cameraOverlayEl) return;

  const ctx = cameraOverlayEl.getContext('2d');
  if (!ctx) return;

  resizeCameraOverlayCanvas();
  const w = cameraOverlayEl.width;
  const h = cameraOverlayEl.height;
  ctx.clearRect(0, 0, w, h);

  if (!renderOptions.cameraFeed || !renderOptions.mlPoseOverlay || !latestPoseLandmarks2d) return;

  const minVisibility = 0.2;
  ctx.strokeStyle = 'rgba(60,220,160,0.85)';
  ctx.lineWidth = Math.max(1, 2 * (window.devicePixelRatio || 1));
  for (const [a, b] of POSE_CONNECTIONS) {
    const p1 = latestPoseLandmarks2d[a];
    const p2 = latestPoseLandmarks2d[b];
    if (!p1 || !p2 || p1[2] < minVisibility || p2[2] < minVisibility) continue;
    ctx.beginPath();
    ctx.moveTo(p1[0] * w, p1[1] * h);
    ctx.lineTo(p2[0] * w, p2[1] * h);
    ctx.stroke();
  }

  ctx.fillStyle = 'rgba(255,120,80,0.9)';
  const r = Math.max(1.5, 3.2 * (window.devicePixelRatio || 1));
  for (const p of latestPoseLandmarks2d) {
    if (p[2] < minVisibility) continue;
    ctx.beginPath();
    ctx.arc(p[0] * w, p[1] * h, r, 0, Math.PI * 2);
    ctx.fill();
  }
}

if (cameraFeedEl) {
  cameraFeedEl.addEventListener('load', () => {
    clearCameraRetry();
    if (cameraStateEl) cameraStateEl.textContent = 'live';
    resizeCameraOverlayCanvas();
  });
  cameraFeedEl.addEventListener('error', () => {
    if (cameraStateEl) cameraStateEl.textContent = 'error';
    scheduleCameraRetry();
  });
}

// initialize checkboxes from persisted options
if (optShadows) optShadows.checked = !!renderOptions.shadows;
if (optDebugSkel) optDebugSkel.checked = !!renderOptions.debugSkeleton;
if (optShowMesh) optShowMesh.checked = !!renderOptions.showSkinnedMesh;
if (optWireframe) optWireframe.checked = !!renderOptions.wireframe;
if (optServerSkel) optServerSkel.checked = !!renderOptions.serverSkeleton;
if (optCameraFeed) optCameraFeed.checked = !!renderOptions.cameraFeed;
if (optMlOverlay) optMlOverlay.checked = !!renderOptions.mlPoseOverlay;

function updateRenderIndicator() {
  if (!renderIndicator) return;
  const mode = renderOptions.debugSkeleton ? 'Debug Skeleton' : 'Skinned Mesh';
  const flags: string[] = [];
  if (renderOptions.shadows) flags.push('Shadows');
  if (renderOptions.wireframe) flags.push('Wireframe');
  if (renderOptions.serverSkeleton) flags.push('ServerSkel');
  if (renderOptions.cameraFeed) flags.push('Cam');
  if (renderOptions.mlPoseOverlay) flags.push('MLPose');
  renderIndicator.textContent = `${mode}${flags.length ? ' · ' + flags.join(', ') : ''}`;
}

function hookOptionInputs() {
  if (optShadows) optShadows.addEventListener('change', () => { renderOptions.shadows = optShadows.checked; persistRenderOptions(); applyRenderOptions(); updateRenderIndicator(); });
  if (optDebugSkel) optDebugSkel.addEventListener('change', () => { renderOptions.debugSkeleton = optDebugSkel.checked; persistRenderOptions(); applyRenderOptions(); updateRenderIndicator(); });
  if (optShowMesh) optShowMesh.addEventListener('change', () => { renderOptions.showSkinnedMesh = optShowMesh.checked; persistRenderOptions(); applyRenderOptions(); updateRenderIndicator(); });
  if (optWireframe) optWireframe.addEventListener('change', () => { renderOptions.wireframe = optWireframe.checked; persistRenderOptions(); applyRenderOptions(); updateRenderIndicator(); });
  if (optServerSkel) optServerSkel.addEventListener('change', () => { renderOptions.serverSkeleton = optServerSkel.checked; persistRenderOptions(); applyRenderOptions(); updateRenderIndicator(); });
  if (optCameraFeed) optCameraFeed.addEventListener('change', () => { renderOptions.cameraFeed = optCameraFeed.checked; persistRenderOptions(); applyRenderOptions(); updateRenderIndicator(); });
  if (optMlOverlay) optMlOverlay.addEventListener('change', () => { renderOptions.mlPoseOverlay = optMlOverlay.checked; persistRenderOptions(); applyRenderOptions(); updateRenderIndicator(); });
}
hookOptionInputs();
updateRenderIndicator();

const scene = new THREE.Scene();
// Bright, neutral studio background
scene.background = new THREE.Color(0xf4f7fb);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(2.5, 1.6, 2.8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.0, 0);
controls.enableDamping = true;

// Lights/ground (same as before)
// Studio lighting: subtle sky/fill, strong key, and a rim light
const hemi = new THREE.HemisphereLight(0x9ecfff, 0x444444, 0.6);
scene.add(hemi);

// Key directional light (soft shadow)
const key = new THREE.DirectionalLight(0xffffff, 1.0);
key.position.set(2.5, 6, 4);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.camera.left = -5;
key.shadow.camera.right = 5;
key.shadow.camera.top = 5;
key.shadow.camera.bottom = -5;
key.shadow.camera.near = 0.5;
key.shadow.camera.far = 20;
key.shadow.bias = -0.0005;
scene.add(key);

// Fill light to wash shadows a bit
const fill = new THREE.AmbientLight(0xffffff, 0.35);
scene.add(fill);

// Rim/backlight to separate the model from the background
const rim = new THREE.SpotLight(0xffffff, 0.6, 0, Math.PI / 6, 0.6);
rim.position.set(-3, 4, -2);
rim.castShadow = false;
scene.add(rim);

// Ground: light, slightly glossy for subtle reflections
const groundMat = new THREE.MeshStandardMaterial({ color: 0xf6f7f9, roughness: 0.5, metalness: 0 });
const ground = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), groundMat);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
ground.position.y = 0;
scene.add(ground);

// Backdrop: large curved paper-like plane behind the model
const backdropMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.8, metalness: 0, side: THREE.DoubleSide });
const backdrop = new THREE.Mesh(new THREE.PlaneGeometry(40, 18), backdropMat);
backdrop.position.set(0, 4.5, -6);
scene.add(backdrop);

// Subtle grid helper on the ground for studio reference
const grid = new THREE.GridHelper(20, 40, 0xe6eef6, 0xe6eef6);
grid.material.opacity = 0.6;
(grid.material as THREE.Material).transparent = true;
scene.add(grid);

// Load Y-Bot FBX
let retargeter: Retargeter | null = null;
let skinned: THREE.SkinnedMesh | null = null;
let skeletonHelper: any = null;
let modelRoot: THREE.Object3D | null = null;
let serverSkeletonHelper: ServerSkeletonHelper | null = null;
let modelBonesByName: Map<string, THREE.Object3D> | null = null;

function persistRenderOptions() {
  try { localStorage.setItem('ps_render_opts', JSON.stringify(renderOptions)); } catch (e) { }
}


function applyRenderOptions() {
  setCameraFeedEnabled(!!renderOptions.cameraFeed);
  if (cameraPanel) {
    cameraPanel.style.display = renderOptions.cameraFeed ? 'block' : 'none';
  }
  if (cameraOverlayEl) {
    cameraOverlayEl.style.display = (renderOptions.cameraFeed && renderOptions.mlPoseOverlay) ? 'block' : 'none';
    if (!renderOptions.mlPoseOverlay) {
      const ctx = cameraOverlayEl.getContext('2d');
      if (ctx) ctx.clearRect(0, 0, cameraOverlayEl.width, cameraOverlayEl.height);
    }
  }

  renderer.shadowMap.enabled = !!renderOptions.shadows;
  try { key.castShadow = !!renderOptions.shadows; } catch (e) { }
  if (modelRoot) {
    // enable/disable shadows on all meshes in the model
    modelRoot.traverse((o) => {
      if ((o as THREE.Mesh).isMesh) {
        try {
          (o as any).castShadow = !!renderOptions.shadows;
          (o as any).receiveShadow = !!renderOptions.shadows;
        } catch (e) { }
      }
    });

    // Skeleton helper visibility
    if (skeletonHelper) skeletonHelper.visible = !!renderOptions.debugSkeleton;

    // Mesh visibility: showSkinnedMesh controls the filled mesh. If wireframe is enabled
    // and showSkinnedMesh is false, still show the model so wireframe lines are visible.
    modelRoot.visible = !!renderOptions.showSkinnedMesh || !!renderOptions.wireframe;

    // Wireframe: apply to entire model (but don't change visibility)
    setWireframeForObject(modelRoot, !!renderOptions.wireframe);
    // Server skeleton helper visibility
    if (serverSkeletonHelper) serverSkeletonHelper.visible = !!renderOptions.serverSkeleton;
  }
}

(async () => {
  try {
    const model = await loadYBotFbx('/models/y-bot/y-bot.fbx');

    // Find first SkinnedMesh
    skinned = null;
    model.traverse((obj) => {
      if ((obj as THREE.SkinnedMesh).isSkinnedMesh) skinned = obj as THREE.SkinnedMesh;
    });

    if (!skinned) {
      console.error('No SkinnedMesh found in Y_Bot.fbx');
      statusEl.textContent = 'Model error';
      return;
    }

    // Reset to bind pose
    (skinned as THREE.SkinnedMesh).skeleton.pose();

    scene.add(model);
    modelRoot = model;

    // Apply initial shadow/cast settings from options
    if (skinned) {
      // initial apply on entire model
      // (applyRenderOptions will set per-mesh flags too)
      // create skeleton helper and respect visibility flag
      try {
        skeletonHelper = new SimpleSkeletonHelper(skinned as any);
        skeletonHelper.visible = !!renderOptions.debugSkeleton;
        scene.add(skeletonHelper.mesh);
      } catch (e) {
        console.warn('Failed to create SimpleSkeletonHelper:', e);
        skeletonHelper = null;
      }
      // set wireframe if requested
      setWireframeForObject(modelRoot!, !!renderOptions.wireframe);
      // Build a quick lookup of model bones by several key formats so the
      // ServerSkeletonHelper can place joints at the model's bone positions.
      try {
        const map = new Map<string, THREE.Object3D>();
        if ((skinned as any).skeleton && Array.isArray((skinned as any).skeleton.bones)) {
          for (const b of (skinned as any).skeleton.bones as THREE.Object3D[]) {
            map.set(b.name, b);
            map.set(`mixamorig:${b.name}`, b);
            const norm = b.name.toLowerCase().replace(/[^a-z0-9]/g, '');
            map.set(norm, b);
          }
        }
        modelBonesByName = map;
      } catch (e) { modelBonesByName = null; }
      // Apply full render options to modelRoot
      applyRenderOptions();
    }

    retargeter = new Retargeter(skinned!, {
      rootScale: 1.0 // set to 1.0 because we scaled model by 0.01 already
      // corrections: { LeftForeArm: new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1,0,0), Math.PI/2) }
    });

    statusEl.textContent = 'Model ready';
  } catch (e) {
    console.error(e);
    statusEl.textContent = 'Failed to load Y_Bot.fbx';
  }
})();

// Network
const wsUrl = `ws://${location.hostname}:8000/ws`;
const stream = new PoseStream(wsUrl, (s) => { statusEl.textContent = s; });

// Render loop
const clock = new THREE.Clock();
let frames = 0;
let lastFpsTime = performance.now();

function animate(): void {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  controls.update();

  // update skeleton helper lines if present
  try {
    if (skeletonHelper && (skeletonHelper as any).update) (skeletonHelper as any).update();
  } catch (e) { /* ignore update errors */ }

  // If available, start a GPU time query for the whole frame render
  let gpuQuery: any = null;
  let gl: any;
  try {
    gl = renderer.getContext();
  } catch (e) { gl = null; }

  if (gl && (gl as any).getExtension) {
    try {
      const ext = (gl as any).getExtension('EXT_disjoint_timer_query_webgl2') || (gl as any).getExtension('EXT_disjoint_timer_query');
      if (ext && (gl as any).createQuery) {
        try {
          gpuQuery = (gl as any).createQuery();
          // TIME_ELAPSED_EXT constant is on the extension
          const TIME_ELAPSED = ext.TIME_ELAPSED_EXT || ext.TIME_ELAPSED;
          if (TIME_ELAPSED) (gl as any).beginQuery(TIME_ELAPSED, gpuQuery);
        } catch (e) {
          gpuQuery = null;
        }
      }
    } catch (e) { /* ignore */ }
  }

  const sample = stream.pollInterpolated();
  if (retargeter && sample) {
    if (sample.kind === 'interp') {
      retargeter.applyInterpolated(sample.a, sample.b, sample.alpha);
    } else {
      retargeter.applyHold(sample.data);
    }
  }

  // Update server skeleton overlay (lazy create)
  try {
    if (sample) {
      // Build an interpolated frame when possible so the server overlay matches
      // the retargeter's interpolation timing.
      let frameForViz: any = null;
      if (sample.kind === 'interp') {
        const a = sample.a;
        const b = sample.b;
        const alpha = sample.alpha;
        // Interpolate root position
        const ra = (a.root && Array.isArray(a.root.position)) ? vec3FromArray(a.root.position as [number, number, number]) : new THREE.Vector3();
        const rb = (b.root && Array.isArray(b.root.position)) ? vec3FromArray(b.root.position as [number, number, number]) : new THREE.Vector3();
        const rp = lerpVec3(new THREE.Vector3(), ra, rb, alpha);

        // Interpolate root rotation
        const rqa = (a.root && Array.isArray(a.root.rotation)) ? quatFromArray(a.root.rotation as [number, number, number, number]) : new THREE.Quaternion();
        const rqb = (b.root && Array.isArray(b.root.rotation)) ? quatFromArray(b.root.rotation as [number, number, number, number]) : new THREE.Quaternion();
        const rqi = slerpQuat(new THREE.Quaternion(), rqa, rqb, alpha);

        // Interpolate joints by name
        const ma = new Map<string, any>();
        const mb = new Map<string, any>();
        if (Array.isArray(a.joints)) for (const j of a.joints) ma.set(j.name, j.rotation);
        if (Array.isArray(b.joints)) for (const j of b.joints) mb.set(j.name, j.rotation);
        const jointNames = Array.from(new Set([...(Array.isArray(a.joints) ? a.joints.map((j: any) => j.name) : []), ...(Array.isArray(b.joints) ? b.joints.map((j: any) => j.name) : [])]));
        const joints: any[] = [];
        for (const name of jointNames) {
          const raArr = ma.get(name) as [number, number, number, number] | undefined;
          const rbArr = mb.get(name) as [number, number, number, number] | undefined;
          const qa = raArr ? quatFromArray(raArr) : new THREE.Quaternion();
          const qb = rbArr ? quatFromArray(rbArr) : new THREE.Quaternion();
          const qi = slerpQuat(new THREE.Quaternion(), qa, qb, alpha);
          joints.push({ name, rotation: [qi.w, qi.x, qi.y, qi.z] });
        }

        frameForViz = {
          ...b,
          root: { position: [rp.x, rp.y, rp.z], rotation: [rqi.w, rqi.x, rqi.y, rqi.z] },
          joints
        };
      } else {
        frameForViz = sample.data;
      }

      if (!serverSkeletonHelper && frameForViz) {
        const names = Array.isArray(frameForViz.joints) ? frameForViz.joints.map((j: any) => j.name) : [];
        if (names.length) {
          serverSkeletonHelper = new ServerSkeletonHelper(names);
          serverSkeletonHelper.visible = !!renderOptions.serverSkeleton;
          scene.add(serverSkeletonHelper.mesh);
        }
      }
      if (serverSkeletonHelper && frameForViz) {
        serverSkeletonHelper.updateFromPose(frameForViz, modelBonesByName ?? undefined);
      }
    }
  } catch (e) { /* ignore visualization errors */ }

  // Update network overlay with latest server skeleton/frame metadata
  try {
    const latest = (stream as any).getLatestFrame ? (stream as any).getLatestFrame() : null as any;
    if (latest && netEl) {
      const conf = latest.meta && (latest.meta as any).confidence;
      netEl.textContent = `Net: ${latest.skeleton ?? '—'} · frame ${latest.frame ?? '—'} · fps ${latest.fps ?? '—'} · conf ${typeof conf === 'number' ? conf.toFixed(2) : '—'}`;

      const landmarks2d = latest.meta && (latest.meta as any).pose_landmarks_2d;
      if (Array.isArray(landmarks2d)) {
        latestPoseLandmarks2d = landmarks2d
          .filter((p: any) => Array.isArray(p) && p.length >= 2)
          .map((p: any) => [Number(p[0]), Number(p[1]), Number(p[2] ?? 1)] as PoseLandmark2D);
      }
    }
  } catch (e) { /* ignore UI update errors */ }

  drawCameraPoseOverlay();

  renderer.render(scene, camera);

  // End gpu query and try to resolve previous queries (non-blocking)
  try {
    if (gpuQuery && gl) {
      const ext = (gl as any).getExtension('EXT_disjoint_timer_query_webgl2') || (gl as any).getExtension('EXT_disjoint_timer_query');
      const TIME_ELAPSED = ext && (ext.TIME_ELAPSED_EXT || ext.TIME_ELAPSED);
      if (TIME_ELAPSED) (gl as any).endQuery(TIME_ELAPSED);
      // Try to read an available result from last frame's query (non-blocking)
      if ((gl as any).getQueryParameter) {
        // Read last query result if available
        try {
          const available = (gl as any).getQueryParameter(gpuQuery, (gl as any).QUERY_RESULT_AVAILABLE);
          const disjoint = ext && (gl as any).getParameter(ext.GPU_DISJOINT_EXT);
          if (available && !disjoint) {
            const timeElapsed = (gl as any).getQueryParameter(gpuQuery, (gl as any).QUERY_RESULT);
            // For EXT_disjoint_timer_query, result is in nanoseconds
            if (typeof timeElapsed === 'number') {
              statGpu.textContent = `${(timeElapsed / 1e6).toFixed(2)} ms`;
            }
          }
        } catch (e) { /* ignore retrieval errors */ }
      }
    }
  } catch (e) { /* ignore */ }

  // Update stats accumulators
  const statsNow = performance.now();
  const frameMs = dt * 1000.0;
  statsAccumFrames += 1;
  statsAccumTime += frameMs;
  statsCount += 1;
  // incremental avg
  statsAvg += (frameMs - statsAvg) / statsCount;
  statsMin = Math.min(statsMin, frameMs);
  statsMax = Math.max(statsMax, frameMs);

  // Periodically update the UI
  if (statsNow - statsLastSampleTime >= statsInterval) {
    const avgMsWindow = statsAccumFrames ? (statsAccumTime / statsAccumFrames) : 0;
    const fpsWindow = avgMsWindow > 0 ? (1000.0 / avgMsWindow) : 0;
    statFps.textContent = `${fpsWindow.toFixed(1)}`;
    statFrame.textContent = `${avgMsWindow.toFixed(2)} ms`;
    statMinMax.textContent = `${statsMin.toFixed(2)} / ${statsMax.toFixed(2)} / ${statsAvg.toFixed(2)} ms`;
    // renderer.info contains triangles and draw calls
    const info = renderer.info;
    statTris.textContent = `${info.render.triangles ?? 0}`;
    statDraw.textContent = `${info.render.calls ?? 0}`;
    // camera info
    const pos = camera.position;
    const rot = camera.rotation;
    const rotDeg = `${(rot.x * 180 / Math.PI).toFixed(1)}, ${(rot.y * 180 / Math.PI).toFixed(1)}, ${(rot.z * 180 / Math.PI).toFixed(1)}`;
    statCam.textContent = `pos(${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)}) rot(${rotDeg}) fov ${camera.fov.toFixed(1)}`;

    // device memory (coarse)
    try {
      // navigator.deviceMemory is in GB (approx) when available
      const dm = (navigator as any).deviceMemory;
      statMem.textContent = dm ? `${dm} GB` : `unknown`;
    } catch (e) {
      statMem.textContent = `unknown`;
    }

    // retargeter micro-profiling
    if (retargeter && (retargeter as any).getRetargetStats) {
      try {
        const r = (retargeter as any).getRetargetStats();
        statRet.textContent = `${r.last.toFixed(2)} / ${r.avg.toFixed(2)} ms (bones ${r.boneCount})`;
      } catch (e) {
        statRet.textContent = `--`;
      }
    }

    // reset window accumulators
    statsLastSampleTime = statsNow;
    statsAccumFrames = 0;
    statsAccumTime = 0;
  }

  // Push into sparkline buffers per frame (ms only)
  frameHistory[sparkIndex] = frameMs;
  sparkIndex = (sparkIndex + 1) % SPARK_LEN;

  // persist a small history occasionally
  if (frames % 60 === 0) {
    try { localStorage.setItem('ps_spark_frame', JSON.stringify(frameHistory)); } catch (e) { }
  }

  // Draw sparklines (cheap)
  drawSparkline(sparkFrame, frameHistory, '#ff6b6b');

  frames++;
  const now = performance.now();
  if (now - lastFpsTime > 500) {
    const fps = (frames * 1000) / (now - lastFpsTime);
    fpsEl.textContent = `Render: ${fps.toFixed(1)} fps`;
    frames = 0;
    lastFpsTime = now;
  }
}
animate();

// Resize
window.addEventListener('resize', () => {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  resizeCameraOverlayCanvas();
});
