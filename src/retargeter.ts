// src/retargeter.ts
import { Quaternion, SkinnedMesh, Vector3, Bone, Object3D } from 'three';
import type { PoseFrame } from './types.js';
import { quatFromArray, vec3FromArray, slerpQuat, lerpVec3, clamp } from './utils.js';

type BoneMap = Record<string, string>;

export class Retargeter {
  private mesh: SkinnedMesh;
  private skeleton: SkinnedMesh['skeleton'];
  private bonesByName = new Map<string, Bone>();
  private corrections = new Map<string, Quaternion>();
  private jointWeights = new Map<string, number>();
  private rootBone: Bone;
  private rootScale: number;
  private _loggedMappings = false;

  // Temp objects
  private _qa = new Quaternion();
  private _qb = new Quaternion();
  private _qr = new Quaternion();
  private _loggedSpine = false;
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

    // Debug: list available bone names (helps detect naming mismatches)
    try {
      // eslint-disable-next-line no-console
      console.log('Retargeter: skeleton bones:', Array.from(this.bonesByName.keys()).join(', '));
    } catch (e) {
      // ignore in non-browser environments
    }

    this.rootBone = (
      this.findBoneByName('Hips') ??
      this.findBoneByName('mixamorigHips') ??
      this.skeleton.bones[0]
    )!;

    // Optional per-bone corrective quats for axis alignment
    if (options?.corrections) {
      for (const [k, v] of Object.entries(options.corrections)) {
        this.corrections.set(k, v.clone().normalize());
      }
    }

    // Auto-compute per-bone correction quaternions by comparing the model's
    // rest directions with the server's rest directions (same as backend
    // processor.rest_dirs). This helps align axes between MediaPipe-derived
    // rotations and the Mixamo bone axes.
    const serverRestDirs: Record<string, [number, number, number]> = {
      Spine: [0, 1, 0],
      LeftUpLeg: [0, -1, 0],
      LeftLeg: [0, -1, 0],
      RightUpLeg: [0, -1, 0],
      RightLeg: [0, -1, 0],
      LeftShoulder: [-1, 0, 0],
      RightShoulder: [1, 0, 0],
      LeftArm: [-1, 0, 0],
      LeftForeArm: [-1, 0, 0],
      RightArm: [1, 0, 0],
      RightForeArm: [1, 0, 0],
    };

    // Ensure skeleton is in bind pose so world positions represent rest pose
    try {
      this.skeleton.pose();
    } catch (e) {
      // ignore if not available
    }

    for (const [name, dirArr] of Object.entries(serverRestDirs)) {
      const bone = this.findBoneByName(name);
      if (!bone) continue;

      // find a child bone to infer the model's rest direction for this bone
      const childBone = bone.children.find((c) => (c as Bone).isBone) as Bone | undefined
        ?? this.skeleton.bones.find(b => b.parent === bone);
      if (!childBone) continue;

      const a = new Vector3();
      const b = new Vector3();
      bone.getWorldPosition(a);
      childBone.getWorldPosition(b);
      const modelDir = b.sub(a).normalize();

      const serverDir = new Vector3(dirArr[0], dirArr[1], dirArr[2]).normalize();

      // corr maps serverRest -> modelRest. We'll store its inverse so later
      // we can do final = qL * corrInv (see reasoning in code comments).
      const corr = new Quaternion().setFromUnitVectors(serverDir, modelDir);
      const corrInv = corr.clone().invert();
      this.corrections.set(name, corrInv.normalize());
    }

    // Default weights = 1
    const defaultNames = [
      'Spine', 'LeftUpLeg', 'LeftLeg', 'RightUpLeg', 'RightLeg',
      'LeftShoulder', 'LeftArm', 'LeftForeArm', 'RightShoulder', 'RightArm', 'RightForeArm'
    ];
    defaultNames.forEach(n => this.jointWeights.set(n, 1.0));
  }

  /**
   * Find a bone by server target name using several fallback strategies:
   *  - exact match
   *  - prefixed (mixamorig:name)
   *  - normalized compare (strip non-alphanum, lower-case)
   */
  private findBoneByName(tname: string): Bone | undefined {
    const exact = this.bonesByName.get(tname);
    if (exact) return exact;

    const mixamo = this.bonesByName.get(`mixamorig:${tname}`);
    if (mixamo) return mixamo;

    const normalize = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, '');
    const target = normalize(tname);
    for (const [name, bone] of this.bonesByName.entries()) {
      if (normalize(name) === target) return bone;
    }
    for (const [name, bone] of this.bonesByName.entries()) {
      if (normalize(name).endsWith(target)) return bone;
    }
    return undefined;
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
      // On first frame, log mapping results to help debug missing bones
      if (!this._loggedMappings) {
        const mb = this.findBoneByName(this.mapName(srvName) ?? srvName);
        // eslint-disable-next-line no-console
        console.log(`Retargeter mapping: ${srvName} -> ${mb ? mb.name : '<missing>'}`);
      }
      const tname = this.mapName(srvName);
      if (!tname) continue;

      const bone = this.findBoneByName(tname);
      if (!bone) continue;

      const rotB = jb.get(srvName) ?? rotA;

      const qA = quatFromArray(rotA);
      const qB = quatFromArray(rotB);
      const qL = slerpQuat(this._qa, qA, qB, alpha);

      const corr = this.corrections.get(tname);

      // Server quaternions are provided in a common/world-like frame.
      // We stored per-bone correction quaternions (as the inverse of
      // server->model) in `this.corrections`. To convert the incoming
      // server quaternion qL into the model bone frame we multiply on the
      // right by the stored correction: final = qL * corrInv
      const desiredWorld = corr ? this._qb.copy(qL).multiply(corr) : qL.clone();

      // Get parent world quaternion (Object3D.getWorldQuaternion writes into target)
      const parent = bone.parent as Object3D | null;
      if (parent) {
        parent.getWorldQuaternion(this._qr);
        // Debug: log spine quaternions once to inspect why center bone is stationary
        if (!this._loggedSpine && (tname === 'Spine' || bone.name.toLowerCase().includes('spine'))) {
          // desiredWorld in _qb (we set below), parent world in _qr
          // compute local for logging
          const parentInv = this._qa.copy(this._qr).invert();
          const localForLog = parentInv.clone().multiply(desiredWorld);
          // eslint-disable-next-line no-console
          console.log('Retargeter debug (Spine):', JSON.stringify({
            srvName, tname, boneName: bone.name,
            qL: qL.toArray().map(n => Number(n.toFixed(6))),
            desiredWorld: desiredWorld.toArray().map(n => Number(n.toFixed(6))),
            parentWorld: this._qr.toArray().map(n => Number(n.toFixed(6))),
            computedLocal: localForLog.toArray().map(n => Number(n.toFixed(6)))
          }));
          this._loggedSpine = true;
        }
        // parent world inverse
        this._qr.invert();
        // local = parentInv * desiredWorld
        const local = this._qa.copy(this._qr).multiply(desiredWorld);

        // Special-case: distribute spine rotation across multiple spine bones
        if (tname === 'Spine') {
          const spineNames = ['Spine', 'Spine1', 'Spine2'];
          const spineWeights = [1.0, 0.5, 0.25];
          for (let i = 0; i < spineNames.length; ++i) {
            const sName = spineNames[i]!;
            const b = this.findBoneByName(sName);
            if (!b) continue;
            const sw = spineWeights[i] ?? 1.0;
            if (sw >= 1.0) {
              b.quaternion.copy(local);
            } else {
              b.quaternion.slerp(local, sw);
            }
          }
        } else {
          const w = this.jointWeights.get(tname) ?? 1.0;
          if (w >= 1.0) {
            bone.quaternion.copy(local);
          } else if (w <= 0) {
            // leave as-is
          } else {
            bone.quaternion.slerp(local, w);
          }
        }
      } else {
        // No parent (shouldn't usually happen) — apply directly
        const w = this.jointWeights.get(tname) ?? 1.0;
        if (w >= 1.0) {
          bone.quaternion.copy(desiredWorld);
        } else if (w <= 0) {
          // leave as-is
        } else {
          bone.quaternion.slerp(desiredWorld, w);
        }
      }
    }

    if (!this._loggedMappings) this._loggedMappings = true;

    // Upload to GPU
    this.skeleton.update();
  }

  applyHold(f: PoseFrame): void {
    this.applyInterpolated(f, f, 0);
  }
}
