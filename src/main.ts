// ybot-retarget.ts
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { FBXLoader } from "three/examples/jsm/loaders/FBXLoader.js";

/**
 * High-level overview:
 * - Loads Y-Bot (Mixamo rig).
 * - Receives joints via WebSocket: { joints: { joint_#: [x,y,z], ... }, timestamp }
 * - Normalizes incoming pose (center, scale, orientation).
 * - Retargets to bones using parent-space aim (bind-pose aware).
 * - Smooths joints and rotations to avoid jitter and flipping.
 */

// ----------------------------- Config -----------------------------

const SOCKET_URL = "ws://localhost:8000/ws/pose";
const MODEL_PATH = "/models/y-bot/y-bot.fbx";
const MODEL_SCALE = 0.01;

// EMA smoothing factors
const JOINT_POS_ALPHA = 0.35; // 0..1, higher = more responsive, more jitter
const BONE_ROT_SLERP = 0.35;  // 0..1 per frame

// Optional twist limiting in degrees (per bone name match)
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

// Joint-to-bone mapping (adapt this to your source)
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

// Debug overlay
const DEBUG_DRAW = true;      // draw input joints/segments
const DEBUG_SCALE = 1.0;      // scale for debug gizmos

// ----------------------------- Scene -----------------------------

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1b1b1b);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.05, 200);
camera.position.set(0, 1.5, 3.5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.shadowMap.enabled = true;
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 1.0, 0);

// Lights
scene.add(new THREE.HemisphereLight(0xffffff, 0x333344, 0.9));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.1);
dirLight.position.set(5, 10, 6);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
scene.add(dirLight);

// Ground
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(50, 50),
  new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 1.0, metalness: 0.0, depthWrite: false })
);
ground.receiveShadow = true;
ground.rotation.x = -Math.PI / 2;
scene.add(ground);

// ----------------------------- Types -----------------------------

type Joints = Record<string, [number, number, number]>;

interface BoneInfo {
  bone: THREE.Bone;
  parent: THREE.Object3D | null;
  bindLocalQuat: THREE.Quaternion;       // bone.quaternion at bind
  bindLocalQuatInv: THREE.Quaternion;
  restDirParent: THREE.Vector3;          // direction in parent space (bind pose)
}

interface RetargetState {
  modelRoot: THREE.Object3D;
  bones: Record<string, BoneInfo>;
  leftShoulder?: THREE.Object3D;
  rightShoulder?: THREE.Object3D;
  leftHip?: THREE.Object3D;
  rightHip?: THREE.Object3D;
  shoulderWidthModel?: number;
}

// ----------------------------- Globals -----------------------------

let yBot: THREE.Group | null = null;
const retarget: RetargetState = { modelRoot: scene, bones: {} };
const jointFilter: Map<string, THREE.Vector3> = new Map(); // EMA filtered joint positions

// Debug helpers
const debugGroup = new THREE.Group();
if (DEBUG_DRAW) scene.add(debugGroup);

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
  // Both assumed normalized
  const q = new THREE.Quaternion();
  q.setFromUnitVectors(a, b);
  return q;
}

// Swing-twist decomposition to clamp twist around axis
function clampTwist(localQ: THREE.Quaternion, axisParent: THREE.Vector3, maxDeg: number) {
  if (maxDeg <= 0) return localQ;
  const axis = axisParent.clone().normalize();
  const twistAxis = new THREE.Vector3(axis.x, axis.y, axis.z);

  // Project rotation axis on twist axis to get twist-only quat
  const r = localQ.clone();
  const ra = new THREE.Vector3(r.x, r.y, r.z);
  const proj = twistAxis.clone().multiplyScalar(ra.dot(twistAxis));
  const twist = new THREE.Quaternion(proj.x, proj.y, proj.z, r.w).normalize();

  const twistAngle = 2 * Math.acos(THREE.MathUtils.clamp(twist.w, -1, 1));
  const twistDeg = THREE.MathUtils.radToDeg(twistAngle);
  if (twistDeg <= maxDeg) return localQ;

  const sign = ra.dot(twistAxis) >= 0 ? 1 : -1;
  const clampedAngle = THREE.MathUtils.degToRad(maxDeg) * sign;
  const clampedTwist = new THREE.Quaternion().setFromAxisAngle(twistAxis, clampedAngle);

  // swing = r * twist^{-1}
  const swing = r.clone().multiply(twist.clone().invert());
  const clamped = swing.multiply(clampedTwist);
  return clamped.normalize();
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

// Convert incoming joints (camera space) to model local space with centering/scaling
function normalizeIncomingPose(joints: Joints, modelRoot: THREE.Object3D) {
  const lh = joints[ROOT_LEFT_HIP];
  const rh = joints[ROOT_RIGHT_HIP];
  if (!lh || !rh) return null;

  const lhv = new THREE.Vector3(...lh);
  const rhv = new THREE.Vector3(...rh);
  const hipCenter = averageVec(lhv, rhv);

  // Shoulder width for scale (if available)
  let scale = 1.0;
  const ls = joints[LEFT_SHOULDER];
  const rs = joints[RIGHT_SHOULDER];
  if (ls && rs) {
    const lsv = new THREE.Vector3(...ls);
    const rsv = new THREE.Vector3(...rs);
    const shoulderWidthInput = lsv.distanceTo(rsv);
    const shoulderWidthModel = retarget.shoulderWidthModel;
    if (shoulderWidthInput > 1e-6 && shoulderWidthModel && shoulderWidthModel > 1e-6) {
      scale = shoulderWidthModel / shoulderWidthInput;
    }
  }

  // Build transform: center at model root origin, scale
  const out: Record<string, THREE.Vector3> = {};
  Object.entries(joints).forEach(([key, v]) => {
    const p = new THREE.Vector3(v[0], v[1], v[2]);

    // Center at hips and scale
    p.sub(hipCenter).multiplyScalar(scale);

    // Optional: flip Z if your input forward/back is inverted
    // p.z *= -1; // uncomment if needed

    // Move into model local space (model is at origin already)
    out[key] = p;
  });

  // Slight forward offset to stand in front of origin (optional)
  // for (const v of Object.values(out)) v.z -= 0.0;

  return out;
}

// ----------------------------- Model Load -----------------------------

function buildBoneInfo(fbx: THREE.Group) {
  const needed = new Set(Object.values(JOINT_TO_BONE));
  fbx.traverse((obj) => {
    if ((obj as THREE.SkinnedMesh).isSkinnedMesh) {
      (obj as THREE.SkinnedMesh).castShadow = true;
      (obj as THREE.SkinnedMesh).receiveShadow = true;
    }
    if ((obj as THREE.Bone).isBone) {
      const bone = obj as THREE.Bone;
      if (needed.has(bone.name)) {
        // Capture bind local quaternion
        const bindLocalQuat = bone.quaternion.clone().normalize();
        const bindLocalQuatInv = bindLocalQuat.clone().invert();

        // Parent-space rest direction: from bone to first child (bind pose)
        let restDirParent = new THREE.Vector3(0, 1, 0);
        if (bone.children.length > 0) {
          const parent = bone.parent!;
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
    fbx.scale.setScalar(MODEL_SCALE);
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
  },
  undefined,
  (err) => console.error("FBX load error:", err)
);

// ----------------------------- Retargeting -----------------------------

function applyPose(jointsRaw: Joints) {
  if (!yBot) return;

  // 1) Normalize incoming pose into model local space
  const pose = normalizeIncomingPose(jointsRaw, retarget.modelRoot);
  if (!pose) return;

  // 2) Smooth joint positions (EMA)
  const smoothed: Record<string, THREE.Vector3> = {};
  for (const [k, v] of Object.entries(pose)) {
    const curr = v;
    const prev = jointFilter.get(k);
    const filt = ema(prev, curr, JOINT_POS_ALPHA);
    jointFilter.set(k, filt);
    smoothed[k] = filt;
  }

  // 3) Optional debug draw
  if (DEBUG_DRAW) drawDebug(smoothed);

  // 4) For each mapped joint -> bone, compute parent-space aim and apply to local rotation
  for (const [jointName, boneName] of Object.entries(JOINT_TO_BONE)) {
    const info = retarget.bones[boneName];
    if (!info || !info.parent) continue;

    const childName = CHILD_OF[jointName];
    const from = smoothed[jointName];
    const to = smoothed[childName];
    if (!from || !to) continue;

    // Parent-space target direction
    const targetDir = normalizedDir(from, to).normalize();

    // Ensure we work in parent space
    // Since 'from' and 'to' are already in model local, and bones are local to parent,
    // for direction we can use those vectors directly relative to parent origin.
    // Align restDirParent -> targetDir
    const aimQParent = quaternionFromVectors(info.restDirParent.clone().normalize(), targetDir);

    // Local rotation = aim * bind
    let targetLocal = aimQParent.multiply(info.bindLocalQuat).normalize();

    // Optional twist limit around bone's local rest axis (approx)
    const maxTwist = TWIST_LIMITS[boneName] || 0;
    if (maxTwist > 0) {
      targetLocal = clampTwist(targetLocal, info.restDirParent, maxTwist);
    }

    // Smooth rotation towards target
    info.bone.quaternion.slerp(targetLocal, BONE_ROT_SLERP);
  }
}

// ----------------------------- Debug -----------------------------

let debugInited = false;
const debugSpheres: Map<string, THREE.Mesh> = new Map();
const debugLines: Map<string, THREE.Line> = new Map();

function drawDebug(joints: Record<string, THREE.Vector3>) {
  if (!debugInited) {
    // Make small spheres for each joint, and lines for key bone segments
    const mat = new THREE.MeshBasicMaterial({ color: 0x66ccff });
    const geom = new THREE.SphereGeometry(0.01 * DEBUG_SCALE, 8, 8);

    Object.keys(joints).forEach((k) => {
      const s = new THREE.Mesh(geom, mat);
      debugGroup.add(s);
      debugSpheres.set(k, s);
    });

    Object.entries(CHILD_OF).forEach(([a, b]) => {
      const g = new THREE.BufferGeometry();
      const pts = new Float32Array(6);
      g.setAttribute("position", new THREE.BufferAttribute(pts, 3));
      const line = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x33aa33 }));
      debugGroup.add(line);
      debugLines.set(`${a}->${b}`, line);
    });

    debugInited = true;
  }

  for (const [k, v] of Object.entries(joints)) {
    const s = debugSpheres.get(k);
    if (s) s.position.copy(v);
  }

  for (const [a, b] of Object.entries(CHILD_OF)) {
    const line = debugLines.get(`${a}->${b}`);
    if (!line) continue;
    const pa = joints[a], pb = joints[b];
    if (!pa || !pb) continue;
    const posAttr = line.geometry.getAttribute("position") as THREE.BufferAttribute;
    posAttr.setXYZ(0, pa.x, pa.y, pa.z);
    posAttr.setXYZ(1, pb.x, pb.y, pb.z);
    posAttr.needsUpdate = true;
  }
}

// ----------------------------- WebSocket -----------------------------

function startPoseStream() {
  const socket = new WebSocket(SOCKET_URL);

  socket.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg && msg.joints) applyPose(msg.joints as Joints);
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

// ----------------------------- Resize & Loop -----------------------------

window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
