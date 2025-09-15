import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { FBXLoader } from "three/examples/jsm/loaders/FBXLoader.js";

// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x202020);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 1.5, 3);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1, 0);
controls.update();

// Lights
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.0));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
dirLight.position.set(5, 10, 7.5);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
scene.add(dirLight);

// Ground
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(50, 50),
  new THREE.MeshStandardMaterial({ color: 0x222222, depthWrite: false })
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// --- Skeleton mapping ---
const jointToBone: Record<string, string> = {
  l_hip: "mixamorigLeftUpLeg",
  l_knee: "mixamorigLeftLeg",
  l_ankle: "mixamorigLeftFoot",
  r_hip: "mixamorigRightUpLeg",
  r_knee: "mixamorigRightLeg",
  r_ankle: "mixamorigRightFoot",
  l_shoulder: "mixamorigLeftArm",
  l_elbow: "mixamorigLeftForeArm",
  l_wrist: "mixamorigLeftHand",
  r_shoulder: "mixamorigRightArm",
  r_elbow: "mixamorigRightForeArm",
  r_wrist: "mixamorigRightHand",
  neck: "mixamorigNeck",
  head: "mixamorigHead"
};

// Store bone references and rest directions
interface BoneInfo {
  bone: THREE.Bone;
  restForward: THREE.Vector3;
}
const bones: Record<string, BoneInfo> = {};

let yBot: THREE.Group | null = null;

// Load FBX
const loader = new FBXLoader();
loader.load("/models/y-bot/y-bot.fbx", (fbx) => {
  fbx.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) {
      const mesh = child as THREE.Mesh;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    }
    if ((child as THREE.Bone).isBone) {
      const bone = child as THREE.Bone;
      if (Object.values(jointToBone).includes(bone.name)) {
        const forward = new THREE.Vector3();
        if (bone.children.length > 0) {
          forward.subVectors(
            bone.children[0].getWorldPosition(new THREE.Vector3()),
            bone.getWorldPosition(new THREE.Vector3())
          ).normalize();
        } else {
          forward.set(0, 1, 0);
        }
        bones[bone.name] = { bone, restForward: forward };
      }
    }
  });
  fbx.scale.setScalar(0.01);
  scene.add(fbx);
  yBot = fbx;
});

// --- Pose application helpers ---
function computeAimRotation(
  restForward: THREE.Vector3,
  fromPos: THREE.Vector3,
  toPos: THREE.Vector3
): THREE.Quaternion {
  const targetDir = new THREE.Vector3().subVectors(toPos, fromPos).normalize();
  const q = new THREE.Quaternion();
  q.setFromUnitVectors(restForward.clone().normalize(), targetDir);
  return q;
}

function findChildJoint(jointName: string, joints: Record<string, number[]>): string | null {
  const hierarchy: Record<string, string> = {
    l_hip: "l_knee",
    l_knee: "l_ankle",
    r_hip: "r_knee",
    r_knee: "r_ankle",
    l_shoulder: "l_elbow",
    l_elbow: "l_wrist",
    r_shoulder: "r_elbow",
    r_elbow: "r_wrist",
    neck: "head"
  };
  return hierarchy[jointName] || null;
}

// --- WebSocket connection ---
function connectPoseStream(onFrame: (frame: any) => void) {
  const socket = new WebSocket("ws://localhost:8000/ws/pose");

  socket.onmessage = (event) => {
    const frame = JSON.parse(event.data);
    onFrame(frame);
  };

  socket.onclose = () => {
    console.log("Pose stream closed, reconnecting...");
    setTimeout(() => connectPoseStream(onFrame), 1000);
  };
}

// --- Apply pose to skeleton ---
connectPoseStream((frame) => {
  if (!yBot) return;
  const joints = frame.joints;

  for (const [jointName, boneName] of Object.entries(jointToBone)) {
    const boneInfo = bones[boneName];
    if (!boneInfo) continue;

    const childJointName = findChildJoint(jointName, joints);
    if (!childJointName) continue;

    const fromPos = joints[jointName];
    const toPos = joints[childJointName];
    if (!fromPos || !toPos) continue;

    const fromVec = new THREE.Vector3(...fromPos);
    const toVec = new THREE.Vector3(...toPos);

    const q = computeAimRotation(boneInfo.restForward, fromVec, toVec);
    boneInfo.bone.quaternion.slerp(q, 0.5); // smooth interpolation
  }
});

// --- Resize handling ---
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- Animation loop ---
function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();
