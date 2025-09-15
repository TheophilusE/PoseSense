import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { FBXLoader } from "three/examples/jsm/loaders/FBXLoader.js";

// === Constants ===
const SOCKET_URL = "ws://localhost:8000/ws/pose";
const MODEL_PATH = "/models/y-bot/y-bot.fbx";
const SCALE_FACTOR = 0.01;

// === Joint-to-Bone Mapping ===
const jointToBone: Record<string, string> = {
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
  joint_0: "mixamorigSpine",
  joint_31: "mixamorigHead"
};

const jointHierarchy: Record<string, string> = {
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

// === Scene Setup ===
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

scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.0));

const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
dirLight.position.set(5, 10, 7.5);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
scene.add(dirLight);

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(50, 50),
  new THREE.MeshStandardMaterial({ color: 0x222222, depthWrite: false })
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// === Bone Registry ===
interface BoneInfo {
  bone: THREE.Bone;
  restForward: THREE.Vector3;
}
const bones: Record<string, BoneInfo> = {};
let yBot: THREE.Group | null = null;

// === Load Y-Bot Model ===
new FBXLoader().load(MODEL_PATH, (fbx) => {
  fbx.scale.setScalar(SCALE_FACTOR);
  fbx.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) {
      const mesh = child as THREE.Mesh;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    }
    if ((child as THREE.Bone).isBone) {
      const bone = child as THREE.Bone;
      if (Object.values(jointToBone).includes(bone.name)) {
        const forward = bone.children.length
          ? new THREE.Vector3().subVectors(
              bone.children[0].getWorldPosition(new THREE.Vector3()),
              bone.getWorldPosition(new THREE.Vector3())
            ).normalize()
          : new THREE.Vector3(0, 1, 0);
        bones[bone.name] = { bone, restForward: forward };
      }
    }
  });
  scene.add(fbx);
  yBot = fbx;
});

// === Pose Application ===
function applyPose(joints: Record<string, number[]>) {
  if (!yBot) return;

  for (const [jointName, boneName] of Object.entries(jointToBone)) {
    const boneInfo = bones[boneName];
    if (!boneInfo) continue;

    const childName = jointHierarchy[jointName];
    const from = joints[jointName];
    const to = joints[childName];
    if (!from || !to) continue;

    const fromVec = new THREE.Vector3(...from);
    const toVec = new THREE.Vector3(...to);
    const targetDir = new THREE.Vector3().subVectors(toVec, fromVec).normalize();

    const q = new THREE.Quaternion().setFromUnitVectors(
      boneInfo.restForward.clone().normalize(),
      targetDir
    );
    boneInfo.bone.quaternion.slerp(q, 0.5);
  }
}

// === WebSocket Connection ===
function startPoseStream() {
  const socket = new WebSocket(SOCKET_URL);

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.joints) applyPose(data.joints);
  };

  socket.onclose = () => {
    console.warn("Pose stream disconnected. Reconnecting...");
    setTimeout(startPoseStream, 1000);
  };
}
startPoseStream();

// === Responsive Resize ===
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// === Animation Loop ===
function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();
