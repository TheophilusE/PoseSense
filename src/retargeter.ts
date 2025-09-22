// src/retargeter.ts
import { Quaternion, SkinnedMesh, Vector3 } from 'three';
import type { PoseFrame } from './types.js';
import { quatFromArray, vec3FromArray, slerpQuat, lerpVec3, clamp } from './utils.js';

type BoneMap = Record<string, string>;

export class Retargeter {
  private mesh: SkinnedMesh;
  private skeleton: SkinnedMesh['skeleton'];
  private bonesByName = new Map<string, THREE.Bone>();
  private corrections = new Map<string, Quaternion>();
  private jointWeights = new Map<string, number>();
  private rootBone: THREE.Bone;
  private rootScale: number;

  // Temp objects
  private _qa = new Quaternion();
  private _qb = new Quaternion();
  private _qr = new Quaternion();
  private _va = new Vector3();
  private _vb = new Vector3();
  private _vr = new Vector3();

  // Server->Model joint name mapping (extend as needed)
  private mapName: (n: string) => string | null = (n: string) => {
    switch (n) {
      case 'Spine': return 'Spine';
      case 'LeftUpLeg': return 'LeftUpLeg';
      case 'LeftLeg': return 'LeftLeg';
      case 'RightUpLeg': return 'RightUpLeg';
      case 'RightLeg': return 'RightLeg';
      case 'LeftShoulder': return 'LeftShoulder';
      case 'LeftArm': return 'LeftArm';
      case 'LeftForeArm': return 'LeftForeArm';
      case 'RightShoulder': return 'RightShoulder';
      case 'RightArm': return 'RightArm';
      case 'RightForeArm': return 'RightForeArm';
      default: return null;
    }
  };

  constructor(skinnedMesh: SkinnedMesh, options?: { rootScale?: number; corrections?: Record<string, Quaternion> }) {
    this.mesh = skinnedMesh;
    this.skeleton = skinnedMesh.skeleton;
    this.rootScale = options?.rootScale ?? 1.0;

    for (const b of this.skeleton.bones) {
      this.bonesByName.set(b.name, b);
    }

    this.rootBone =
      this.bonesByName.get('Hips') ??
      this.bonesByName.get('mixamorig:Hips') ??
      this.skeleton.bones[0];

    // Optional per-bone corrective quats for axis alignment
    if (options?.corrections) {
      for (const [k, v] of Object.entries(options.corrections)) {
        this.corrections.set(k, v.clone().normalize());
      }
    }

    // Default weights = 1
    const defaultNames = [
      'Spine', 'LeftUpLeg', 'LeftLeg', 'RightUpLeg', 'RightLeg',
      'LeftShoulder', 'LeftArm', 'LeftForeArm', 'RightShoulder', 'RightArm', 'RightForeArm'
    ];
    defaultNames.forEach(n => this.jointWeights.set(n, 1.0));
  }

  applyInterpolated(a: PoseFrame, b: PoseFrame, alphaIn: number): void {
    const alpha = clamp(alphaIn, 0, 1);

    // Root pose
    const pa = vec3FromArray(a.root.position).multiplyScalar(this.rootScale);
    const pb = vec3FromArray(b.root.position).multiplyScalar(this.rootScale);
    const qa = quatFromArray(a.root.rotation);
    const qb = quatFromArray(b.root.rotation);

    const p = lerpVec3(this._vr, pa, pb, alpha);
    const q = slerpQuat(this._qr, qa, qb, alpha);

    this.rootBone.position.copy(p);
    this.rootBone.quaternion.copy(q);

    // Build maps for joints
    const ja = new Map<string, [number, number, number, number]>(a.joints.map(j => [j.name, j.rotation]));
    const jb = new Map<string, [number, number, number, number]>(b.joints.map(j => [j.name, j.rotation]));

    // Iterate through available joints
    for (const [srvName, rotA] of ja.entries()) {
      const tname = this.mapName(srvName);
      if (!tname) continue;

      const bone =
        this.bonesByName.get(tname) ??
        this.bonesByName.get(`mixamorig:${tname}`);
      if (!bone) continue;

      const rotB = jb.get(srvName) ?? rotA;

      const qA = quatFromArray(rotA);
      const qB = quatFromArray(rotB);
      const qL = slerpQuat(this._qa, qA, qB, alpha);

      const corr = this.corrections.get(tname);
      const finalLocal = corr ? this._qb.copy(corr).multiply(qL) : qL;

      const w = this.jointWeights.get(tname) ?? 1.0;
      if (w >= 1.0) {
        bone.quaternion.copy(finalLocal);
      } else if (w <= 0) {
        // leave as-is
      } else {
        bone.quaternion.slerp(finalLocal, w);
      }
    }

    // Upload to GPU
    this.skeleton.update();
  }

  applyHold(f: PoseFrame): void {
    this.applyInterpolated(f, f, 0);
  }
}
