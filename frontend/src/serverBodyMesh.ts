import * as THREE from 'three';
import type { PoseFrame } from './types.js';

type JointKey =
  | 'pelvis'
  | 'chest'
  | 'neck'
  | 'head'
  | 'left_shoulder'
  | 'left_elbow'
  | 'left_wrist'
  | 'left_hand'
  | 'right_shoulder'
  | 'right_elbow'
  | 'right_wrist'
  | 'right_hand'
  | 'left_hip'
  | 'left_knee'
  | 'left_ankle'
  | 'left_foot'
  | 'left_foot_index'
  | 'right_hip'
  | 'right_knee'
  | 'right_ankle'
  | 'right_foot'
  | 'right_foot_index';

type SegmentSpec = {
  a: JointKey;
  b: JointKey;
  radius: number;
  color: number;
};

const JOINT_KEYS: JointKey[] = [
  'pelvis',
  'chest',
  'neck',
  'head',
  'left_shoulder',
  'left_elbow',
  'left_wrist',
  'left_hand',
  'right_shoulder',
  'right_elbow',
  'right_wrist',
  'right_hand',
  'left_hip',
  'left_knee',
  'left_ankle',
  'left_foot',
  'left_foot_index',
  'right_hip',
  'right_knee',
  'right_ankle',
  'right_foot',
  'right_foot_index',
];

const SEGMENTS: SegmentSpec[] = [
  { a: 'pelvis', b: 'chest', radius: 0.07, color: 0x5577aa },
  { a: 'chest', b: 'neck', radius: 0.06, color: 0x5577aa },
  { a: 'neck', b: 'head', radius: 0.055, color: 0x5577aa },

  { a: 'chest', b: 'left_shoulder', radius: 0.05, color: 0x59a773 },
  { a: 'left_shoulder', b: 'left_elbow', radius: 0.045, color: 0x59a773 },
  { a: 'left_elbow', b: 'left_wrist', radius: 0.038, color: 0x59a773 },
  { a: 'left_wrist', b: 'left_hand', radius: 0.034, color: 0x59a773 },

  { a: 'chest', b: 'right_shoulder', radius: 0.05, color: 0x59a773 },
  { a: 'right_shoulder', b: 'right_elbow', radius: 0.045, color: 0x59a773 },
  { a: 'right_elbow', b: 'right_wrist', radius: 0.038, color: 0x59a773 },
  { a: 'right_wrist', b: 'right_hand', radius: 0.034, color: 0x59a773 },

  { a: 'pelvis', b: 'left_hip', radius: 0.055, color: 0x4f5f79 },
  { a: 'left_hip', b: 'left_knee', radius: 0.052, color: 0x4f5f79 },
  { a: 'left_knee', b: 'left_ankle', radius: 0.048, color: 0x4f5f79 },
  { a: 'left_ankle', b: 'left_foot', radius: 0.042, color: 0x4f5f79 },
  { a: 'left_foot', b: 'left_foot_index', radius: 0.032, color: 0x4f5f79 },

  { a: 'pelvis', b: 'right_hip', radius: 0.055, color: 0x4f5f79 },
  { a: 'right_hip', b: 'right_knee', radius: 0.052, color: 0x4f5f79 },
  { a: 'right_knee', b: 'right_ankle', radius: 0.048, color: 0x4f5f79 },
  { a: 'right_ankle', b: 'right_foot', radius: 0.042, color: 0x4f5f79 },
  { a: 'right_foot', b: 'right_foot_index', radius: 0.032, color: 0x4f5f79 },
];

function readTargetVec3(targets: Record<string, unknown>, key: string, out: THREE.Vector3): boolean {
  const raw = targets[key];
  if (!Array.isArray(raw) || raw.length < 3) return false;

  const x = Number(raw[0]);
  const y = Number(raw[1]);
  const z = Number(raw[2]);
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return false;

  out.set(x, y, z);
  return true;
}

export class ServerBodyMesh {
  public readonly root = new THREE.Group();

  private readonly _segmentMeshByKey = new Map<string, THREE.Mesh>();
  private readonly _jointMeshByKey = new Map<JointKey, THREE.Mesh>();
  private readonly _smoothedPointByKey = new Map<JointKey, THREE.Vector3>();
  private readonly _pointSmoothing = 0.36;
  private readonly _rootMotionGainXZ = 1.0;
  private readonly _rootMotionGainY = 0.68;
  private readonly _rootPosSmoothing = 0.22;
  private readonly _frameYawFix = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI);
  private readonly _identity = new THREE.Quaternion();

  private _rootInitialized = false;
  private readonly _rootSourceOrigin = new THREE.Vector3();
  private readonly _rootPosFiltered = new THREE.Vector3();
  private readonly _rootTargetPos = new THREE.Vector3();

  private readonly _tmpA = new THREE.Vector3();
  private readonly _tmpB = new THREE.Vector3();
  private readonly _tmpDir = new THREE.Vector3();
  private readonly _tmpMid = new THREE.Vector3();
  private readonly _upAxis = new THREE.Vector3(0, 1, 0);
  private readonly _tmpRootRaw = new THREE.Vector3();
  private readonly _tmpRootYawed = new THREE.Vector3();

  constructor() {
    this.root.name = 'ServerBodyMeshRoot';

    const segmentGeometry = new THREE.CylinderGeometry(1, 1, 1, 12, 1, false);
    for (const seg of SEGMENTS) {
      const material = new THREE.MeshStandardMaterial({
        color: seg.color,
        roughness: 0.55,
        metalness: 0.05,
      });
      const mesh = new THREE.Mesh(segmentGeometry, material);
      mesh.name = `ServerSeg_${seg.a}_${seg.b}`;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.scale.set(seg.radius, 1, seg.radius);
      this.root.add(mesh);
      this._segmentMeshByKey.set(`${seg.a}:${seg.b}`, mesh);
    }

    const jointGeometry = new THREE.SphereGeometry(1, 12, 10);
    for (const key of JOINT_KEYS) {
      const isCore = key === 'pelvis' || key === 'chest' || key === 'neck' || key === 'head';
      const material = new THREE.MeshStandardMaterial({
        color: isCore ? 0x344564 : 0x2f3d54,
        roughness: 0.52,
        metalness: 0.04,
      });
      const mesh = new THREE.Mesh(jointGeometry, material);
      mesh.name = `ServerJoint_${key}`;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      const r = isCore ? 0.045 : 0.03;
      mesh.scale.set(r, r, r);
      this.root.add(mesh);
      this._jointMeshByKey.set(key, mesh);
    }
  }

  private updateSegment(mesh: THREE.Mesh, a: THREE.Vector3, b: THREE.Vector3, radius: number): void {
    this._tmpDir.copy(b).sub(a);
    const len = this._tmpDir.length();
    if (len < 1e-5) {
      mesh.visible = false;
      return;
    }

    mesh.visible = true;
    mesh.position.copy(this._tmpMid.copy(a).add(b).multiplyScalar(0.5));
    mesh.quaternion.setFromUnitVectors(this._upAxis, this._tmpDir.normalize());
    mesh.scale.set(radius, len, radius);
  }

  private resolveRootSourceRaw(frame: PoseFrame, targets: Record<string, unknown>, out: THREE.Vector3): boolean {
    const rawRoot = frame.root?.position;
    if (Array.isArray(rawRoot) && rawRoot.length >= 3) {
      const x = Number(rawRoot[0]);
      const y = Number(rawRoot[1]);
      const z = Number(rawRoot[2]);
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        out.set(x, y, z);
        return true;
      }
    }

    return readTargetVec3(targets, 'pelvis', out);
  }

  private updateRootMotion(rootRaw: THREE.Vector3): void {
    this._tmpRootYawed.copy(rootRaw).applyQuaternion(this._frameYawFix);

    if (!this._rootInitialized) {
      this._rootInitialized = true;
      this._rootSourceOrigin.copy(this._tmpRootYawed);
      this._rootPosFiltered.set(0, 0, 0);
      this.root.position.set(0, 0, 0);
    }

    const delta = this._tmpA.copy(this._tmpRootYawed).sub(this._rootSourceOrigin);
    this._rootTargetPos.set(
      delta.x * this._rootMotionGainXZ,
      delta.y * this._rootMotionGainY,
      delta.z * this._rootMotionGainXZ,
    );
    this._rootPosFiltered.lerp(this._rootTargetPos, this._rootPosSmoothing);
    this.root.position.copy(this._rootPosFiltered);
    this.root.quaternion.copy(this._identity);
  }

  updateFromPose(frame: PoseFrame): boolean {
    const meta = frame.meta as any;
    const targets = (meta?.intermediate_targets ?? meta?.intermediate_targets_raw) as Record<string, unknown> | undefined;
    if (!targets) {
      this.root.visible = false;
      return false;
    }

    const hasRootSource = this.resolveRootSourceRaw(frame, targets, this._tmpRootRaw);
    if (hasRootSource) {
      this.updateRootMotion(this._tmpRootRaw);
    }

    const points = new Map<JointKey, THREE.Vector3>();
    for (const key of JOINT_KEYS) {
      if (!readTargetVec3(targets, key, this._tmpA)) continue;

      if (hasRootSource) {
        this._tmpA.sub(this._tmpRootRaw);
      }
      this._tmpA.applyQuaternion(this._frameYawFix);
      const prev = this._smoothedPointByKey.get(key);
      if (!prev) {
        const seeded = this._tmpA.clone();
        this._smoothedPointByKey.set(key, seeded);
        points.set(key, seeded);
      } else {
        prev.lerp(this._tmpA, this._pointSmoothing);
        points.set(key, prev);
      }
    }

    if (points.size < 6) {
      this.root.visible = false;
      return false;
    }

    this.root.visible = true;

    for (const key of JOINT_KEYS) {
      const mesh = this._jointMeshByKey.get(key);
      if (!mesh) continue;
      const p = points.get(key);
      if (!p) {
        mesh.visible = false;
        continue;
      }
      mesh.visible = true;
      mesh.position.copy(p);
    }

    for (const seg of SEGMENTS) {
      const mesh = this._segmentMeshByKey.get(`${seg.a}:${seg.b}`);
      if (!mesh) continue;

      const a = points.get(seg.a);
      const b = points.get(seg.b);
      if (!a || !b) {
        mesh.visible = false;
        continue;
      }

      this.updateSegment(mesh, a, b, seg.radius);
    }

    return true;
  }

  getFollowAnchorWorld(out: THREE.Vector3): boolean {
    const pelvis = this._jointMeshByKey.get('pelvis');
    if (!pelvis || !pelvis.visible) return false;

    pelvis.getWorldPosition(out);
    out.y += 0.92;
    return true;
  }

  getMinFootWorldY(): number | null {
    const keys: JointKey[] = ['left_foot_index', 'right_foot_index', 'left_foot', 'right_foot', 'left_ankle', 'right_ankle'];
    let minY = Number.POSITIVE_INFINITY;
    let found = false;
    for (const key of keys) {
      const joint = this._jointMeshByKey.get(key);
      if (!joint || !joint.visible) continue;
      joint.getWorldPosition(this._tmpB);
      minY = Math.min(minY, this._tmpB.y);
      found = true;
    }
    return found ? minY : null;
  }

  dispose(): void {
    this.root.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.geometry.dispose();
      if (Array.isArray(mesh.material)) {
        for (const m of mesh.material) m.dispose();
      } else {
        mesh.material.dispose();
      }
    });
  }
}
