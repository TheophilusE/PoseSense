import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

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
    <button id="stat-toggle" title="Toggle stats" style="background:transparent;border:none;cursor:pointer;padding:6px;border-radius:6px">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 5C7 5 2.73 8.11 1 12c1.73 3.89 6 7 11 7s9.27-3.11 11-7c-1.73-3.89-6-7-11-7z" stroke="rgba(11,18,32,0.7)" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="12" cy="12" r="3" stroke="rgba(11,18,32,0.7)" stroke-width="1.2"/>
      </svg>
    </button>
  </div>
</div>
<div style="display:grid;grid-template-columns:1fr auto;gap:6px 12px;align-items:center">
  <div style="color:rgba(11,18,32,0.6)">FPS</div><div id="stat-fps" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;font-weight:700">--</div>
  <div style="color:rgba(11,18,32,0.6)">Frame</div><div id="stat-frame" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;font-weight:700">-- ms</div>
  <div style="color:rgba(11,18,32,0.6)">Min / Max / Avg</div><div id="stat-minmax" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace">-- / -- / --</div>
  <div style="color:rgba(11,18,32,0.6)">Triangles</div><div id="stat-tris" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace">--</div>
  <div style="color:rgba(11,18,32,0.6)">Draws</div><div id="stat-draw" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace">--</div>
  <div style="color:rgba(11,18,32,0.6)">Camera</div><div id="stat-cam" style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace">pos(--)</div>
</div>
<div style="display:flex;gap:8px;margin-top:10px;align-items:center">
  <canvas id="spark-fps" width="200" height="40" style="flex:1;border-radius:6px;background:rgba(255,255,255,0.03)"></canvas>
  <canvas id="spark-frame" width="200" height="40" style="flex:1;border-radius:6px;background:rgba(255,255,255,0.03)"></canvas>
</div>
`;
document.body.appendChild(statsEl);
const statFps = document.getElementById('stat-fps')!;
const statFrame = document.getElementById('stat-frame')!;
const statMinMax = document.getElementById('stat-minmax')!;
const statTris = document.getElementById('stat-tris')!;
const statDraw = document.getElementById('stat-draw')!;
const statCam = document.getElementById('stat-cam')!;
const sparkFps = document.getElementById('spark-fps') as HTMLCanvasElement | null;
const sparkFrame = document.getElementById('spark-frame') as HTMLCanvasElement | null;
const statToggle = document.getElementById('stat-toggle') as HTMLButtonElement | null;

// Sparklines: simple circular buffers
const SPARK_LEN = 64;
const fpsHistory: number[] = new Array(SPARK_LEN).fill(0);
const frameHistory: number[] = new Array(SPARK_LEN).fill(0);
let sparkIndex = 0;

function drawSparkline(canvas: HTMLCanvasElement | null, data: number[], color = '#1f73ff') {
  if (!canvas) return;
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
  ctx.lineWidth = 2;
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
}

// Toggle button behavior
if (statToggle) {
  statToggle.addEventListener('click', () => {
    if (statsEl.style.display === 'none') {
      statsEl.style.display = 'block';
    } else {
      statsEl.style.display = 'none';
    }
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

// Enable physically-correct lighting and soft shadows for a studio look
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
(renderer as any).physicallyCorrectLights = true;
renderer.outputColorSpace = (THREE as any).SRGBColorSpace ?? (THREE as any).sRGBEncoding; // compatibility
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;

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

(async () => {
  try {
    const model = await loadYBotFbx('/models/y-bot/y-bot.fbx');

    // Find first SkinnedMesh
    let skinned: THREE.SkinnedMesh | null = null;
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

    retargeter = new Retargeter(skinned, {
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
    statFps.textContent = `FPS: ${fpsWindow.toFixed(1)}`;
    statFrame.textContent = `Frame ms: ${avgMsWindow.toFixed(2)} ms`;
    statMinMax.textContent = `min: ${statsMin.toFixed(2)} ms / max: ${statsMax.toFixed(2)} ms / avg: ${statsAvg.toFixed(2)} ms`;
    // renderer.info contains triangles and draw calls
    const info = renderer.info;
    statTris.textContent = `Triangles: ${info.render.triangles ?? 0}`;
    statDraw.textContent = `Draw calls: ${info.render.calls ?? 0}`;
    // camera info
    const pos = camera.position;
    const rot = camera.rotation;
    const rotDeg = `${(rot.x * 180/Math.PI).toFixed(1)}, ${(rot.y * 180/Math.PI).toFixed(1)}, ${(rot.z * 180/Math.PI).toFixed(1)}`;
    statCam.textContent = `Cam: pos(${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)}) rot(${rotDeg}) fov: ${camera.fov.toFixed(1)}`;

    // reset window accumulators
  statsLastSampleTime = statsNow;
    statsAccumFrames = 0;
    statsAccumTime = 0;
  }

  // Push into sparkline buffers per frame
  const instantFps = frameMs > 0 ? (1000.0 / frameMs) : 0;
  fpsHistory[sparkIndex] = instantFps;
  frameHistory[sparkIndex] = frameMs;
  sparkIndex = (sparkIndex + 1) % SPARK_LEN;

  // Draw sparklines (cheap)
  drawSparkline(sparkFps, fpsHistory, '#1f73ff');
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
