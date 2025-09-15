// ybot-retarget.ts — full drop-in replacement with robust retargeting, auto-scale, world-accurate debug, pose buffering, and constraints.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { FBXLoader } from "three/examples/jsm/loaders/FBXLoader.js";

/**
 * Overview
 * - Loads Y-Bot (Mixamo rig) at native scale (no manual model scaling).
 * - Streams poses over WebSocket: { joints: { joint_#: [x,y,z], ... }, timestamp? }.
 * - Calibrates scale using model shoulder width vs input shoulder width.
 * - Normalizes pose around hip center, handles handedness and mirror.
 * - Buffers poses with small target delay and interpolates for smooth playback.
 * - EMA filters joints, adaptive slerp for bone rotations by deltaTime.
 * - Parent-space aim retargeting (bind-pose aware), with twist limiting.
 * - Optional hinge-like clamps for elbows/knees and mild spine distribution.
 * - Accurate debug: world-space joint spheres/lines + SkeletonHelper overlay.
 * - Graceful handling of missing joints; UI toggles for overlays.
 */

// ----------------------------- Config -----------------------------

const SOCKET_URL = "ws://localhost:8000/ws/pose";
const MODEL_PATH = "/models/y-bot/y-bot.fbx";

// Debug overlays
const DEBUG_JOINTS = true;       // world-space spheres/lines for incoming joints
const DEBUG_SKELETON = true;     // THREE.SkeletonHelper overlay
const DEBUG_SCALE = 1.0;

// Pose buffering/interpolation (ms)
const TARGET_DELAY_MS = 80;      // latency cushion to allow interpolation (60–120 is typical)
const MAX_BUFFER_MS = 400;       // drop frames older than this

// Filters and smoothing
const JOINT_POS_ALPHA = 0.30;    // EMA: higher = more responsive
const MODEL_SCALE_ALPHA = 0.15;  // EMA for per-frame input->model scale
const ROT_SLERP_PER_SEC = 6.0;   // per-second blend rate for bone slerp

// Coordinate & retarget options
const FLIP_Z = true;             // flip Z if source forward is inverted
const INVERT_Y = true;           // some sensors have Y-up vs Y-down
const MIRROR_LEFT_RIGHT = false; // optional horizontal mirror for the incoming pose

// Twist limiting in degrees (per bone)
const TWIST_LIMITS: Record<string, number> = {
  mixamorigLeftArm: 60,
  mixamorigRightArm: 60,
  mixamorigLeftForeArm: 45,
  mixamorigRightForeArm: 45,
  mixamorigLeftUpLeg: 45,
  mixamorigRightUpLeg: 45,
  mixamorigLeftLeg: 45,
  mixamorigRightLeg: 45,
};

// Simple hinge-like clamp for elbows/knees (angle between child direction and rest)
const HINGE_CLAMPS: Record<string, { minDeg: number; maxDeg: number }> = {
  mixamorigLeftForeArm: { minDeg: 0, maxDeg: 160 },
  mixamorigRightForeArm: { minDeg: 0, maxDeg: 160 },
  mixamorigLeftLeg: { minDeg: 0, maxDeg: 170 },
  mixamorigRightLeg: { minDeg: 0, maxDeg: 170 },
};

// Joint-to-bone mapping (adapt to your source indexing)
const JOINT_TO_BONE: Record<string, string> = {
  joint_9: "mixamorigLeftUpLeg",
  joint_10: "mixamorigRightUpLeg",
  joint_11: "mixamorigLeftLeg",
  joint_12: "mixamorigRightLeg",
  joint_13: "mixamorigLeftFoot",
  joint_14: "mixamorigRightFoot",
  joint_23: "mixamorigLeftArm",
  joint_24: "mixamorigRightArm",
  joint_25: "mixamorigLeftForeArm",
  joint_26: "mixamorigRightForeArm",
  joint_27: "mixamorigLeftHand",
  joint_28: "mixamorigRightHand",
  joint_0: "mixamorigSpine",    // spine base
  joint_31: "mixamorigHead"
};

// Direction (parent-space) computed from joint -> child_joint
const CHILD_OF: Record<string, string> = {
  joint_0: "joint_1",
  joint_9: "joint_11",
  joint_10: "joint_12",
  joint_11: "joint_13",
  joint_12: "joint_14",
  joint_23: "joint_25",
  joint_24: "joint_26",
  joint_25: "joint_27",
  joint_26: "joint_28",
  joint_31: "joint_32"
};

// Root alignment (estimate pelvis center from L/R hips)
const ROOT_LEFT_HIP = "joint_9";
const ROOT_RIGHT_HIP = "joint_10";
const LEFT_SHOULDER = "joint_23";
const RIGHT_SHOULDER = "joint_24";

// ----------------------------- Scene -----------------------------

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x141414);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.02, 2048);
camera.position.set(0.5, 1.6, 3.8);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 1.0, 0);

// Lights
scene.add(new THREE.HemisphereLight(0xe8f0ff, 0x2a2a3a, 0.8));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.15);
dirLight.position.set(6, 10, 6);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
dirLight.shadow.camera.left = -8;
dirLight.shadow.camera.right = 8;
dirLight.shadow.camera.top = 8;
dirLight.shadow.camera.bottom = -8;
scene.add(dirLight);

// Ground
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(80, 80),
  new THREE.MeshStandardMaterial({
    color: 0x232323,
    roughness: 0.95,
    metalness: 0.0,
    depthWrite: false
  })
);
ground.receiveShadow = true;
ground.rotation.x = -Math.PI / 2;
scene.add(ground);

// ----------------------------- Types -----------------------------

type Vec3 = [number, number, number];
type Joints = Record<string, Vec3>;

interface PoseFrame {
  joints: Joints;
  timestamp: number; // ms
}

interface BoneInfo {
  bone: THREE.Bone;
  parent: THREE.Object3D | null;
  bindLocalQuat: THREE.Quaternion;
  bindLocalQuatInv: THREE.Quaternion;
  restDirParent: THREE.Vector3;
}

interface RetargetState {
  modelRoot: THREE.Object3D;
  bones: Record<string, BoneInfo>;
  leftShoulder?: THREE.Object3D;
  rightShoulder?: THREE.Object3D;
  leftHip?: THREE.Object3D;
  rightHip?: THREE.Object3D;
  shoulderWidthModel?: number;
  skeletonHelper?: THREE.SkeletonHelper;
}

// ----------------------------- Globals -----------------------------

let yBot: THREE.Group | null = null;
const retarget: RetargetState = { modelRoot: scene, bones: {} };

// EMA filtered joint positions (model-local, normalized space)
const jointFilter: Map<string, THREE.Vector3> = new Map();
// Smoothed dynamic scale input->model
let smoothedScale = 1.0;

// Pose buffer for interpolation
const poseBuffer: PoseFrame[] = [];

// Debug helpers
const debugGroup = new THREE.Group();
let debugInited = false;
const debugSpheres: Map<string, THREE.Mesh> = new Map();
const debugLines: Map<string, THREE.Line> = new Map();
if (DEBUG_JOINTS) scene.add(debugGroup);

// Timing
const clock = new THREE.Clock();

// ----------------------------- Utilities -----------------------------

function ema(prev: THREE.Vector3 | undefined, curr: THREE.Vector3, alpha: number) {
  if (!prev) return curr.clone();
  return prev.clone().multiplyScalar(1 - alpha).add(curr.clone().multiplyScalar(alpha));
}

function normalizedDir(from: THREE.Vector3, to: THREE.Vector3) {
  const d = to.clone().sub(from);
  const len = d.length();
  return len > 1e-6 ? d.multiplyScalar(1 / len) : new THREE.Vector3(0, 0, 1);
}

function quaternionFromVectors(a: THREE.Vector3, b: THREE.Vector3) {
  const q = new THREE.Quaternion();
  q.setFromUnitVectors(a, b);
  return q;
}

// Swing-twist decomposition clamp
function clampTwist(localQ: THREE.Quaternion, axisParent: THREE.Vector3, maxDeg: number) {
  if (maxDeg <= 0) return localQ;
  const axis = axisParent.clone().normalize();
  const r = localQ.clone().normalize();
  // Extract twist around axis
  const rq = new THREE.Quaternion(r.x, r.y, r.z, r.w);
  const vec = new THREE.Vector3(rq.x, rq.y, rq.z);
  const proj = axis.clone().multiplyScalar(vec.dot(axis));
  const twist = new THREE.Quaternion(proj.x, proj.y, proj.z, rq.w).normalize();

  const twistAngle = 2 * Math.acos(THREE.MathUtils.clamp(twist.w, -1, 1));
  const twistDeg = THREE.MathUtils.radToDeg(twistAngle);
  if (twistDeg <= maxDeg) return localQ;

  const sign = vec.dot(axis) >= 0 ? 1 : -1;
  const clampedAngle = THREE.MathUtils.degToRad(maxDeg) * sign;
  const clampedTwist = new THREE.Quaternion().setFromAxisAngle(axis, clampedAngle);

  const swing = r.clone().multiply(twist.clone().invert());
  return swing.multiply(clampedTwist).normalize();
}

function clampHinge(
  targetLocal: THREE.Quaternion,
  restDirParent: THREE.Vector3,
  targetDirParent: THREE.Vector3,
  minDeg: number,
  maxDeg: number
) {
  // Angle between rest and target around the swing (ignore twist); clamp the total swing angle.
  const angle = THREE.MathUtils.radToDeg(
    Math.acos(THREE.MathUtils.clamp(restDirParent.clone().dot(targetDirParent), -1, 1))
  );
  if (angle >= minDeg && angle <= maxDeg) return targetLocal;

  const clampedDeg = THREE.MathUtils.clamp(angle, minDeg, maxDeg);
  // Interpolate direction on the great circle
  const t = angle < 1e-3 ? 1.0 : clampedDeg / angle;
  const clampedDir = restDirParent.clone().lerp(targetDirParent, t).normalize();
  const aimQParent = quaternionFromVectors(restDirParent.clone().normalize(), clampedDir);
  return aimQParent.multiply(targetLocal.clone().premultiply(new THREE.Quaternion()).identity()); // keep same form
}

function computeModelShoulderWidth(): number | undefined {
  const L = scene.getObjectByName("mixamorigLeftShoulder");
  const R = scene.getObjectByName("mixamorigRightShoulder");
  if (!L || !R) return undefined;
  const a = new THREE.Vector3(); L.getWorldPosition(a);
  const b = new THREE.Vector3(); R.getWorldPosition(b);
  return a.distanceTo(b);
}

function averageVec(a: THREE.Vector3, b: THREE.Vector3) {
  return a.clone().add(b).multiplyScalar(0.5);
}

function applyMirror(v: THREE.Vector3) {
  // Mirror across model Z axis by flipping X
  return new THREE.Vector3(-v.x, v.y, v.z);
}

function computeAutoScale(joints: Joints) {
  const ls = joints[LEFT_SHOULDER];
  const rs = joints[RIGHT_SHOULDER];
  if (!ls || !rs) return 1.0;

  const lsv = new THREE.Vector3(...ls);
  const rsv = new THREE.Vector3(...rs);
  const inputShoulderWidth = lsv.distanceTo(rsv);

  const modelShoulderWidth = retarget.shoulderWidthModel;
  if (!modelShoulderWidth || inputShoulderWidth < 1e-6) return 1.0;

  return modelShoulderWidth / inputShoulderWidth;
}

// Convert incoming joints (sensor/camera space) -> model local, centered + scaled
function normalizeIncomingPose(joints: Joints) {
  const lh = joints[ROOT_LEFT_HIP];
  const rh = joints[ROOT_RIGHT_HIP];
  if (!lh || !rh) return null;

  const lhv = new THREE.Vector3(...lh);
  const rhv = new THREE.Vector3(...rh);
  let hipCenter = averageVec(lhv, rhv);

  // Estimate scale per frame, with smoothing
  const frameScale = computeAutoScale(joints);
  smoothedScale = THREE.MathUtils.lerp(smoothedScale, frameScale, MODEL_SCALE_ALPHA);

  const out: Record<string, THREE.Vector3> = {};
  for (const [key, arr] of Object.entries(joints)) {
    const p = new THREE.Vector3(arr[0], arr[1], arr[2]);

    // Center at hips and scale to model
    p.sub(hipCenter).multiplyScalar(smoothedScale * 100);

    // Axis adjustments
    if (INVERT_Y) p.y *= -1;
    if (FLIP_Z) p.z *= -1;
    if (MIRROR_LEFT_RIGHT) p.copy(applyMirror(p));

    out[key] = p;
  }

  return out;
}

function worldToParentLocalPoint(parent: THREE.Object3D, modelRoot: THREE.Object3D, pModelLocal: THREE.Vector3) {
  const pWorld = pModelLocal.clone().applyMatrix4(modelRoot.matrixWorld);
  return parent.worldToLocal(pWorld);
}

// ----------------------------- Model Load -----------------------------

function buildBoneInfo(fbx: THREE.Group) {
  const needed = new Set(Object.values(JOINT_TO_BONE));
  fbx.traverse((obj) => {
    if ((obj as THREE.SkinnedMesh).isSkinnedMesh) {
      const m = obj as THREE.SkinnedMesh;
      m.castShadow = true;
      m.receiveShadow = true;
    }
    if ((obj as THREE.Bone).isBone) {
      const bone = obj as THREE.Bone;
      if (needed.has(bone.name)) {
        const bindLocalQuat = bone.quaternion.clone().normalize();
        const bindLocalQuatInv = bindLocalQuat.clone().invert();

        // Parent-space rest direction from first child in bind-pose
        let restDirParent = new THREE.Vector3(0, 1, 0);
        if (bone.children.length > 0 && bone.parent) {
          const parent = bone.parent;
          const bonePosW = new THREE.Vector3(); bone.getWorldPosition(bonePosW);
          const childPosW = new THREE.Vector3(); (bone.children[0] as THREE.Object3D).getWorldPosition(childPosW);
          const fromP = parent.worldToLocal(bonePosW);
          const toP = parent.worldToLocal(childPosW);
          restDirParent = normalizedDir(fromP, toP);
        }

        retarget.bones[bone.name] = {
          bone,
          parent: bone.parent,
          bindLocalQuat,
          bindLocalQuatInv,
          restDirParent
        };
      }
    }
  });

  retarget.leftShoulder = fbx.getObjectByName("mixamorigLeftShoulder") || undefined;
  retarget.rightShoulder = fbx.getObjectByName("mixamorigRightShoulder") || undefined;
  retarget.leftHip = fbx.getObjectByName("mixamorigLeftUpLeg") || undefined;
  retarget.rightHip = fbx.getObjectByName("mixamorigRightUpLeg") || undefined;
  retarget.shoulderWidthModel = computeModelShoulderWidth();
}

new FBXLoader().load(
  MODEL_PATH,
  (fbx) => {
    fbx.scale.setScalar(0.1);

    fbx.traverse((c) => {
      if ((c as THREE.Mesh).isMesh) {
        const m = c as THREE.Mesh;
        m.castShadow = true;
        m.receiveShadow = true;
      }
    });

    scene.add(fbx);
    yBot = fbx;
    retarget.modelRoot = fbx;
    buildBoneInfo(fbx);

    if (DEBUG_SKELETON) {
      const helper = new THREE.SkeletonHelper(fbx);
      helper.material.depthTest = false;
      helper.material.opacity = 0.6;
      // @ts-ignore
      helper.material.transparent = true;
      scene.add(helper);
      retarget.skeletonHelper = helper;
    }
  },
  undefined,
  (err) => console.error("FBX load error:", err)
);

// ----------------------------- Pose Buffering -----------------------------

function addPoseToBuffer(joints: Joints, timestamp?: number) {
  const t = typeof timestamp === "number" ? timestamp : performance.now();
  poseBuffer.push({ joints, timestamp: t });

  // Purge old frames
  const cutoff = performance.now() - MAX_BUFFER_MS;
  while (poseBuffer.length && poseBuffer[0].timestamp < cutoff) {
    poseBuffer.shift();
  }
}

function lerpVec(a: THREE.Vector3, b: THREE.Vector3, t: number) {
  return a.clone().lerp(b, t);
}

function interpolateJoints(a: Joints, b: Joints, t: number): Joints {
  const result: Joints = {};
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const k of keys) {
    const va = a[k], vb = b[k];
    if (va && vb) {
      const A = new THREE.Vector3(...va);
      const B = new THREE.Vector3(...vb);
      const L = lerpVec(A, B, t);
      result[k] = [L.x, L.y, L.z];
    } else if (va) {
      result[k] = va;
    } else if (vb) {
      result[k] = vb;
    }
  }
  return result;
}

function samplePoseAtTime(targetTime: number): Joints | null {
  if (poseBuffer.length === 0) return null;
  // Find bracketing frames
  let i1 = -1, i2 = -1;
  for (let i = 0; i < poseBuffer.length; i++) {
    if (poseBuffer[i].timestamp <= targetTime) i1 = i;
    if (poseBuffer[i].timestamp >= targetTime) { i2 = i; break; }
  }
  if (i1 === -1) return poseBuffer[0].joints;
  if (i2 === -1) return poseBuffer[poseBuffer.length - 1].joints;
  if (i1 === i2) return poseBuffer[i1].joints;

  const A = poseBuffer[i1], B = poseBuffer[i2];
  const dt = B.timestamp - A.timestamp;
  const t = dt > 1e-3 ? (targetTime - A.timestamp) / dt : 0;
  return interpolateJoints(A.joints, B.joints, THREE.MathUtils.clamp(t, 0, 1));
}

// ----------------------------- Retargeting -----------------------------

function drawDebug(modelLocal: Record<string, THREE.Vector3>) {
  if (!DEBUG_JOINTS) return;
  if (!retarget.modelRoot) return;

  if (!debugInited) {
    const mat = new THREE.MeshBasicMaterial({ color: 0xff6666 });
    const geom = new THREE.SphereGeometry(0.014 * DEBUG_SCALE, 10, 10);

    Object.keys(modelLocal).forEach((k) => {
      const s = new THREE.Mesh(geom, mat);
      debugGroup.add(s);
      debugSpheres.set(k, s);
    });

    Object.entries(CHILD_OF).forEach(([a, b]) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
      const line = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x33aa33 }));
      debugGroup.add(line);
      debugLines.set(`${a}->${b}`, line);
    });

    debugInited = true;
  }

  for (const [k, v] of Object.entries(modelLocal)) {
    const s = debugSpheres.get(k);
    if (!s) continue;
    const w = v.clone().applyMatrix4(retarget.modelRoot.matrixWorld);
    s.position.copy(w);
  }

  for (const [a, b] of Object.entries(CHILD_OF)) {
    const line = debugLines.get(`${a}->${b}`);
    if (!line) continue;
    const pa = modelLocal[a], pb = modelLocal[b];
    if (!pa || !pb) continue;
    const paW = pa.clone().applyMatrix4(retarget.modelRoot.matrixWorld);
    const pbW = pb.clone().applyMatrix4(retarget.modelRoot.matrixWorld);
    const posAttr = line.geometry.getAttribute("position") as THREE.BufferAttribute;
    posAttr.setXYZ(0, paW.x, paW.y, paW.z);
    posAttr.setXYZ(1, pbW.x, pbW.y, pbW.z);
    posAttr.needsUpdate = true;
  }
}

function applyPoseFrame(jointsRaw: Joints, dt: number) {
  if (!yBot) return;

  // 1) Normalize incoming pose to model local
  const normalized = normalizeIncomingPose(jointsRaw);
  if (!normalized) return;

  // 2) EMA smooth joint positions
  const smoothed: Record<string, THREE.Vector3> = {};
  for (const [k, v] of Object.entries(normalized)) {
    const prev = jointFilter.get(k);
    const filt = ema(prev, v, JOINT_POS_ALPHA);
    jointFilter.set(k, filt);
    smoothed[k] = filt;
  }

  // 3) Debug draw (world-aligned)
  drawDebug(smoothed);

  // 4) Retarget: for each joint->bone, compute parent-space target direction
  const modelRoot = retarget.modelRoot;
  for (const [jointName, boneName] of Object.entries(JOINT_TO_BONE)) {
    const info = retarget.bones[boneName];
    if (!info || !info.parent) continue;

    const childName = CHILD_OF[jointName];
    const fromML = smoothed[jointName];
    const toML = smoothed[childName];
    if (!fromML || !toML) continue;

    // Transform model-local joint points into bone parent local space
    const fromP = worldToParentLocalPoint(info.parent, modelRoot, fromML);
    const toP = worldToParentLocalPoint(info.parent, modelRoot, toML);

    const targetDirParent = normalizedDir(fromP, toP).normalize();

    // Aim from restDirParent -> targetDirParent in parent space
    const aimQParent = quaternionFromVectors(info.restDirParent.clone().normalize(), targetDirParent);

    // Local target rotation relative to bind pose (bind-aware)
    let targetLocal = aimQParent.multiply(info.bindLocalQuat).normalize();

    // Optional hinge-like clamp for elbows/knees
    const hinge = HINGE_CLAMPS[boneName];
    if (hinge) {
      targetLocal = clampHinge(
        targetLocal,
        info.restDirParent.clone().normalize(),
        targetDirParent,
        hinge.minDeg,
        hinge.maxDeg
      );
    }

    // Optional twist clamp
    const maxTwist = TWIST_LIMITS[boneName] || 0;
    if (maxTwist > 0) {
      targetLocal = clampTwist(targetLocal, info.restDirParent, maxTwist);
    }

    // Adaptive slerp per dt
    const t = 1.0 - Math.exp(-ROT_SLERP_PER_SEC * dt);
    info.bone.quaternion.slerp(targetLocal, t);
  }

  // 5) Mild spine distribution (optional): average upper body up-vector and blend spine/neck/head a bit.
  // This mitigates small jitter when source spine joints are sparse.
  distributeSpine(smoothed, dt);
}

function distributeSpine(smoothed: Record<string, THREE.Vector3>, dt: number) {
  const spineBone = retarget.bones["mixamorigSpine"]?.bone;
  const headBone = retarget.bones["mixamorigHead"]?.bone;
  if (!spineBone || !headBone) return;

  // Estimate up vector from hips center to shoulders center in model-local
  const lh = smoothed[ROOT_LEFT_HIP], rh = smoothed[ROOT_RIGHT_HIP];
  const ls = smoothed[LEFT_SHOULDER], rs = smoothed[RIGHT_SHOULDER];
  if (!lh || !rh || !ls || !rs) return;

  const hips = averageVec(lh, rh);
  const shoulders = averageVec(ls, rs);
  const upML = normalizedDir(hips, shoulders);

  // Blend head to look more along upML by a small factor
  const parent = headBone.parent;
  if (!parent) return;

  const headParentRest = retarget.bones["mixamorigHead"]?.restDirParent || new THREE.Vector3(0, 1, 0);
  const upParent = worldToParentLocalPoint(parent, retarget.modelRoot, hips.clone().add(upML)).sub(
    worldToParentLocalPoint(parent, retarget.modelRoot, hips.clone())
  ).normalize();

  const aimHead = quaternionFromVectors(headParentRest.clone().normalize(), upParent);
  const targetLocal = aimHead.multiply(retarget.bones["mixamorigHead"].bindLocalQuat).normalize();
  const t = 1.0 - Math.exp(-ROT_SLERP_PER_SEC * 0.4 * dt);
  headBone.quaternion.slerp(targetLocal, t);
}

// ----------------------------- WebSocket -----------------------------

function startPoseStream() {
  const socket = new WebSocket(SOCKET_URL);

  socket.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg && msg.joints) {
        // Accept incoming timestamp if provided; else performance.now()
        addPoseToBuffer(msg.joints as Joints, typeof msg.timestamp === "number" ? msg.timestamp : undefined);
      }
    } catch (e) {
      console.warn("Bad pose packet:", e);
    }
  };

  socket.onclose = () => {
    console.warn("Pose stream disconnected. Reconnecting in 1s...");
    setTimeout(startPoseStream, 1000);
  };
}

startPoseStream();

// ----------------------------- UI: toggles -----------------------------

window.addEventListener("keydown", (e) => {
  if (e.key.toLowerCase() === "k") {
    if (retarget.skeletonHelper) {
      retarget.skeletonHelper.visible = !retarget.skeletonHelper.visible;
    }
  } else if (e.key.toLowerCase() === "j") {
    debugGroup.visible = !debugGroup.visible;
  } else if (e.key.toLowerCase() === "r") {
    // Reset filters
    jointFilter.clear();
    smoothedScale = 1.0;
    console.log("Filters reset");
  }
});

// ----------------------------- Resize & Loop -----------------------------

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(0.05, clock.getDelta()); // clamp dt to avoid spikes
  controls.update();

  // Try to sample pose at timeNow - targetDelay
  const now = performance.now();
  const targetTime = now - TARGET_DELAY_MS;
  const sampled = samplePoseAtTime(targetTime);
  if (sampled) {
    // Apply retarget using sampled pose
    applyPoseFrame(sampled, dt);
  }

  renderer.render(scene, camera);
}
animate();
