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

// Renderer/scene/camera/controls (same as before)
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e0f12);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(2.5, 1.6, 2.8);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.0, 0);
controls.enableDamping = true;

// Lights/ground (same as before)
const hemi = new THREE.HemisphereLight(0xffffff, 0x222233, 0.8);
scene.add(hemi);
const dir = new THREE.DirectionalLight(0xffffff, 0.8);
dir.position.set(3, 5, 2);
scene.add(dir);

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(20, 20),
  new THREE.MeshStandardMaterial({ color: 0x1a1f2b, roughness: 1 })
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

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
    skinned.skeleton.pose();

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
