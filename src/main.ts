import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

// Lightweight in-file skeleton visualizer to avoid importing three/examples
class SimpleSkeletonHelper {
  mesh: THREE.LineSegments;
  bones: THREE.Object3D[];
  private _pos: Float32Array;

  constructor(skinned: THREE.SkinnedMesh) {
    this.bones = skinned.skeleton ? skinned.skeleton.bones.slice() : [];
    this._pos = new Float32Array(this.bones.length * 2 * 3);
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(this._pos, 3));
    const mat = new THREE.LineBasicMaterial({ color: 0x1f73ff });
    this.mesh = new THREE.LineSegments(geom, mat);
    this.mesh.frustumCulled = false;
    this.update();
  }

  update() {
    const attr = this.mesh.geometry.getAttribute('position') as THREE.BufferAttribute;
    const arr = attr.array as Float32Array;
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    let i = 0;
    for (const bone of this.bones) {
      bone.getWorldPosition(a);
      const parent = bone.parent as THREE.Object3D | null;
      if (parent) parent.getWorldPosition(b); else b.copy(a);
      arr[i++] = a.x; arr[i++] = a.y; arr[i++] = a.z;
      arr[i++] = b.x; arr[i++] = b.y; arr[i++] = b.z;
    }
    attr.needsUpdate = true;
    if ((this.mesh.geometry as any).computeBoundingSphere) (this.mesh.geometry as any).computeBoundingSphere();
  }

  set visible(v: boolean) { this.mesh.visible = v; }
  get visible() { return this.mesh.visible; }

  dispose() {
    this.mesh.geometry.dispose();
    (this.mesh.material as any).dispose();
  }
}

import { PoseStream } from "./network.js";
import { Retargeter } from "./retargeter.js";
import type { PoseFrame } from "./types.js";
import { loadYBotFbx } from './loadfbx.js';

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
const renderOptions = (() => {
  try {
    const raw = localStorage.getItem('ps_render_opts');
    if (raw) return JSON.parse(raw);
  } catch (e) { /* ignore */ }
  return { shadows: true, debugSkeleton: false, showSkinnedMesh: true, wireframe: false };
})();

// Enable physically-correct lighting and soft shadows for a studio look
renderer.shadowMap.enabled = !!renderOptions.shadows;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
(renderer as any).physicallyCorrectLights = true;
renderer.outputColorSpace = (THREE as any).SRGBColorSpace ?? (THREE as any).sRGBEncoding; // compatibility
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

// --- Render options UI (shadows, debug skeleton, mesh visibility, wireframe)
const optsContainer = document.createElement('div');
optsContainer.style.display = 'grid';
optsContainer.style.gridTemplateColumns = '1fr auto';
optsContainer.style.gap = '6px 8px';
optsContainer.style.marginTop = '10px';
optsContainer.innerHTML = `
  <div style="color:rgba(11,18,32,0.6)">Shadows</div><div><input id="opt-shadows" type="checkbox"></div>
  <div style="color:rgba(11,18,32,0.6)">Debug Skeleton</div><div><input id="opt-debug-skel" type="checkbox"></div>
  <div style="color:rgba(11,18,32,0.6)">Show Mesh</div><div><input id="opt-show-mesh" type="checkbox"></div>
  <div style="color:rgba(11,18,32,0.6)">Wireframe</div><div><input id="opt-wireframe" type="checkbox"></div>
`;
statsEl.appendChild(optsContainer);

const optShadows = document.getElementById('opt-shadows') as HTMLInputElement | null;
const optDebugSkel = document.getElementById('opt-debug-skel') as HTMLInputElement | null;
const optShowMesh = document.getElementById('opt-show-mesh') as HTMLInputElement | null;
const optWireframe = document.getElementById('opt-wireframe') as HTMLInputElement | null;

// initialize checkboxes from persisted options
if (optShadows) optShadows.checked = !!renderOptions.shadows;
if (optDebugSkel) optDebugSkel.checked = !!renderOptions.debugSkeleton;
if (optShowMesh) optShowMesh.checked = !!renderOptions.showSkinnedMesh;
if (optWireframe) optWireframe.checked = !!renderOptions.wireframe;

function hookOptionInputs() {
  if (optShadows) optShadows.addEventListener('change', () => { renderOptions.shadows = optShadows.checked; persistRenderOptions(); applyRenderOptions(); });
  if (optDebugSkel) optDebugSkel.addEventListener('change', () => { renderOptions.debugSkeleton = optDebugSkel.checked; persistRenderOptions(); applyRenderOptions(); });
  if (optShowMesh) optShowMesh.addEventListener('change', () => { renderOptions.showSkinnedMesh = optShowMesh.checked; persistRenderOptions(); applyRenderOptions(); });
  if (optWireframe) optWireframe.addEventListener('change', () => { renderOptions.wireframe = optWireframe.checked; persistRenderOptions(); applyRenderOptions(); });
}
hookOptionInputs();

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

function persistRenderOptions() {
  try { localStorage.setItem('ps_render_opts', JSON.stringify(renderOptions)); } catch (e) { }
}

function setWireframeForObject(obj: THREE.Object3D, enabled: boolean) {
  obj.traverse((o) => {
    const m = (o as any).material as THREE.Material | THREE.Material[] | undefined;
    if (!m) return;
    if (Array.isArray(m)) {
      for (const mm of m) {
        if ((mm as any).wireframe !== undefined) (mm as any).wireframe = enabled;
      }
    } else {
      if ((m as any).wireframe !== undefined) (m as any).wireframe = enabled;
    }
  });
}

function applyRenderOptions() {
  renderer.shadowMap.enabled = !!renderOptions.shadows;
  try { key.castShadow = !!renderOptions.shadows; } catch (e) { }
  if (skinned) {
    (skinned as any).castShadow = !!renderOptions.shadows;
    (skinned as any).receiveShadow = !!renderOptions.shadows;
    if (skeletonHelper) skeletonHelper.visible = !!renderOptions.debugSkeleton;
    (skinned as any).visible = !!renderOptions.showSkinnedMesh && !renderOptions.debugSkeleton;
    setWireframeForObject(skinned, !!renderOptions.wireframe);
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

    // Apply initial shadow/cast settings from options
    if (skinned) {
      (skinned as any).castShadow = !!renderOptions.shadows;
      (skinned as any).receiveShadow = !!renderOptions.shadows;
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
      setWireframeForObject(skinned, !!renderOptions.wireframe);
      // if debugSkeleton is enabled, hide rendered mesh unless explicitly requested
      (skinned as any).visible = !!renderOptions.showSkinnedMesh && !renderOptions.debugSkeleton;
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

  if (retargeter) {
    const sample = stream.pollInterpolated();
    if (sample) {
      if (sample.kind === 'interp') {
        retargeter.applyInterpolated(sample.a, sample.b, sample.alpha);
      } else {
        retargeter.applyHold(sample.data);
      }
    }
  }

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
});
