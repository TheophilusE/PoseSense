// src/retargeter.ts
import { Quaternion, SkinnedMesh, Vector3, Bone, Object3D, MathUtils } from 'three';
import type { PoseFrame } from './types.js';
import { quatFromArray, vec3FromArray, slerpQuat, lerpVec3, clamp } from './utils.js';

type JointConstraint = {
  maxSwingDeg: number;
  twistMinDeg: number;
  twistMaxDeg: number;
  smoothing: number;
  maxStepDeg: number;
};

export class Retargeter {
  private mesh: SkinnedMesh;
  private skeleton: SkinnedMesh['skeleton'];
  private bonesByName = new Map<string, Bone>();
  private corrections = new Map<string, Quaternion>();
  private constraints = new Map<string, JointConstraint>();
  private bindLocalByBoneName = new Map<string, Quaternion>();
  private axisLocalByJoint = new Map<string, Vector3>();
  private prevLocalByBoneName = new Map<string, Quaternion>();
  private rootBone: Bone;
  private rootScale: number;
  private _loggedMappings = false;

  private rootBindPos = new Vector3();
  private rootBindRot = new Quaternion();
  private rootSourceOrigin = new Vector3();
  private rootOriginInitialized = false;
  private rootMotionGainXZ = 1.8;
  private rootMotionGainY = 1.15;

  // Temp objects reused each frame.
  private _qa = new Quaternion();
  private _qb = new Quaternion();
  private _qr = new Quaternion();
  private _vr = new Vector3();
  private _identity = new Quaternion();
  private _tmpForward = new Vector3();
  private _upAxis = new Vector3(0, 1, 0);

  private _rootPosFiltered = new Vector3();
  private _rootRotFiltered = new Quaternion();
  private _rootInitialized = false;

  // Micro-profiling
  private _retargetTimes: number[] = new Array(128).fill(0);
  private _retargetIndex = 0;
  private _lastRetargetMs = 0;
  private _lastBoneCount = 0;

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

  private defaultConstraint: JointConstraint = {
    maxSwingDeg: 80,
    twistMinDeg: -45,
    twistMaxDeg: 45,
    smoothing: 0.35,
    maxStepDeg: 25,
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

    // Ensure a stable bind/reference pose before extracting rest data.
    try {
      this.skeleton.pose();
    } catch (e) {
      // ignore if not available
    }

    this.rootBone = (
      this.findBoneByName('Hips') ??
      this.findBoneByName('mixamorigHips') ??
      this.skeleton.bones[0]
    )!;

    this.rootBindPos.copy(this.rootBone.position);
    this.rootBindRot.copy(this.rootBone.quaternion);

    for (const b of this.skeleton.bones) {
      this.bindLocalByBoneName.set(b.name, b.quaternion.clone());
      this.prevLocalByBoneName.set(b.name, b.quaternion.clone());
    }

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

      // Convert rest axis to local-bone space for swing/twist constraints.
      const wq = new Quaternion();
      bone.getWorldQuaternion(wq);
      const axisLocal = modelDir.clone().applyQuaternion(wq.invert()).normalize();
      if (axisLocal.lengthSq() > 1e-6) {
        this.axisLocalByJoint.set(name, axisLocal);
      }
    }

    // Biomechanical constraints tuned for noisy webcam motion.
    this.constraints.set('Spine', { maxSwingDeg: 35, twistMinDeg: -25, twistMaxDeg: 25, smoothing: 0.30, maxStepDeg: 16 });
    this.constraints.set('LeftUpLeg', { maxSwingDeg: 75, twistMinDeg: -35, twistMaxDeg: 35, smoothing: 0.35, maxStepDeg: 24 });
    this.constraints.set('RightUpLeg', { maxSwingDeg: 75, twistMinDeg: -35, twistMaxDeg: 35, smoothing: 0.35, maxStepDeg: 24 });
    this.constraints.set('LeftLeg', { maxSwingDeg: 110, twistMinDeg: -20, twistMaxDeg: 20, smoothing: 0.45, maxStepDeg: 26 });
    this.constraints.set('RightLeg', { maxSwingDeg: 110, twistMinDeg: -20, twistMaxDeg: 20, smoothing: 0.45, maxStepDeg: 26 });
    this.constraints.set('LeftShoulder', { maxSwingDeg: 70, twistMinDeg: -18, twistMaxDeg: 18, smoothing: 0.38, maxStepDeg: 22 });
    this.constraints.set('RightShoulder', { maxSwingDeg: 70, twistMinDeg: -18, twistMaxDeg: 18, smoothing: 0.38, maxStepDeg: 22 });
    this.constraints.set('LeftArm', { maxSwingDeg: 120, twistMinDeg: -22, twistMaxDeg: 22, smoothing: 0.40, maxStepDeg: 24 });
    this.constraints.set('RightArm', { maxSwingDeg: 120, twistMinDeg: -22, twistMaxDeg: 22, smoothing: 0.40, maxStepDeg: 24 });
    this.constraints.set('LeftForeArm', { maxSwingDeg: 145, twistMinDeg: -35, twistMaxDeg: 35, smoothing: 0.46, maxStepDeg: 26 });
    this.constraints.set('RightForeArm', { maxSwingDeg: 145, twistMinDeg: -35, twistMaxDeg: 35, smoothing: 0.46, maxStepDeg: 26 });

    // Mixamo arms often look over-twisted with camera noise; use conservative roll corrections.
    for (const j of ['LeftShoulder', 'RightShoulder', 'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm']) {
      this.corrections.set(j, new Quaternion());
    }
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

  private decomposeSwingTwist(q: Quaternion, twistAxisLocal: Vector3): { swing: Quaternion; twist: Quaternion } {
    const axis = twistAxisLocal.clone().normalize();
    const v = new Vector3(q.x, q.y, q.z);
    const proj = axis.clone().multiplyScalar(v.dot(axis));
    const twist = new Quaternion(proj.x, proj.y, proj.z, q.w);
    if (twist.lengthSq() < 1e-10) {
      twist.identity();
    } else {
      twist.normalize();
    }
    const swing = q.clone().multiply(twist.clone().invert()).normalize();
    return { swing, twist };
  }

  private quatAngleRad(q: Quaternion): number {
    const w = MathUtils.clamp(q.w, -1, 1);
    let angle = 2 * Math.acos(w);
    if (angle > Math.PI) angle = (Math.PI * 2) - angle;
    return angle;
  }

  private signedTwistAngleRad(q: Quaternion, twistAxisLocal: Vector3): number {
    const axis = twistAxisLocal.clone().normalize();
    const w = MathUtils.clamp(q.w, -1, 1);
    let angle = 2 * Math.acos(w);
    const s = Math.sqrt(Math.max(1 - (w * w), 0));

    if (s < 1e-6) return 0;

    const qAxis = new Vector3(q.x / s, q.y / s, q.z / s).normalize();
    if (qAxis.dot(axis) < 0) angle = -angle;
    if (angle > Math.PI) angle -= Math.PI * 2;
    if (angle < -Math.PI) angle += Math.PI * 2;
    return angle;
  }

  private constrainLocalQuaternion(localTarget: Quaternion, bindLocal: Quaternion, axisLocal: Vector3, cfg: JointConstraint): Quaternion {
    // Delta from bind pose in bone-local space.
    const delta = bindLocal.clone().invert().multiply(localTarget).normalize();
    const { swing, twist } = this.decomposeSwingTwist(delta, axisLocal);

    const maxSwingRad = MathUtils.degToRad(cfg.maxSwingDeg);
    const swingAngle = this.quatAngleRad(swing);
    let clampedSwing = swing.clone();
    if (swingAngle > maxSwingRad && swingAngle > 1e-6) {
      const w = MathUtils.clamp(swing.w, -1, 1);
      const s = Math.sqrt(Math.max(1 - (w * w), 0));
      const axis = s < 1e-6
        ? axisLocal.clone().normalize()
        : new Vector3(swing.x / s, swing.y / s, swing.z / s).normalize();
      clampedSwing = new Quaternion().setFromAxisAngle(axis, maxSwingRad).normalize();
    }

    const twistAngle = this.signedTwistAngleRad(twist, axisLocal);
    const minTwist = MathUtils.degToRad(cfg.twistMinDeg);
    const maxTwist = MathUtils.degToRad(cfg.twistMaxDeg);
    const clampedTwistAngle = clamp(twistAngle, minTwist, maxTwist);
    const clampedTwist = new Quaternion().setFromAxisAngle(axisLocal.clone().normalize(), clampedTwistAngle);

    const constrainedDelta = clampedSwing.multiply(clampedTwist).normalize();
    return bindLocal.clone().multiply(constrainedDelta).normalize();
  }

  private applyConstrainedLocalRotation(bone: Bone, localTarget: Quaternion, jointName: string): void {
    const cfg = this.constraints.get(jointName) ?? this.defaultConstraint;
    const bindLocal = this.bindLocalByBoneName.get(bone.name) ?? bone.quaternion.clone();
    const axisLocal = this.axisLocalByJoint.get(jointName) ?? new Vector3(1, 0, 0);

    const constrained = this.constrainLocalQuaternion(localTarget, bindLocal, axisLocal, cfg);
    const prev = this.prevLocalByBoneName.get(bone.name) ?? bindLocal.clone();

    const maxStepRad = MathUtils.degToRad(cfg.maxStepDeg);
    const angleToTarget = prev.angleTo(constrained);
    let stepLimited = constrained;
    if (angleToTarget > maxStepRad && maxStepRad > 1e-6) {
      const t = maxStepRad / Math.max(angleToTarget, 1e-6);
      stepLimited = prev.clone().slerp(constrained, t);
    }

    const smoothed = prev.clone().slerp(stepLimited, clamp(cfg.smoothing, 0.01, 1.0));
    bone.quaternion.copy(smoothed);
    this.prevLocalByBoneName.set(bone.name, smoothed.clone());
  }

  private extractYawQuaternion(worldQ: Quaternion): Quaternion {
    const flatForward = this._tmpForward.set(0, 0, 1).applyQuaternion(worldQ);
    flatForward.y = 0;
    if (flatForward.lengthSq() < 1e-8) {
      return this._identity.clone();
    }
    flatForward.normalize();
    const yaw = Math.atan2(flatForward.x, flatForward.z);
    return new Quaternion().setFromAxisAngle(this._upAxis, yaw);
  }

  applyInterpolated(a: PoseFrame, b: PoseFrame, alphaIn: number): void {
    const tStart = (typeof performance !== 'undefined') ? performance.now() : Date.now();
    const alpha = clamp(alphaIn, 0, 1);

    // Root pose
    const pa = vec3FromArray(a.root.position).multiplyScalar(this.rootScale);
    const pb = vec3FromArray(b.root.position).multiplyScalar(this.rootScale);
    const qa = quatFromArray(a.root.rotation);
    const qb = quatFromArray(b.root.rotation);

    const p = lerpVec3(this._vr, pa, pb, alpha);
    const q = slerpQuat(this._qr, qa, qb, alpha);

    if (!this.rootOriginInitialized) {
      this.rootSourceOrigin.copy(p);
      this.rootOriginInitialized = true;
    }

    const delta = this._vr.copy(p).sub(this.rootSourceOrigin);
    const rootTargetPos = new Vector3(
      this.rootBindPos.x + delta.x * this.rootMotionGainXZ,
      this.rootBindPos.y + delta.y * this.rootMotionGainY,
      this.rootBindPos.z + delta.z * this.rootMotionGainXZ,
    );
    const rootTargetRot = this.rootBindRot.clone().multiply(this.extractYawQuaternion(q));

    // Smooth root motion while preserving visible translation from the pose midpoint.
    if (!this._rootInitialized) {
      this._rootPosFiltered.copy(rootTargetPos);
      this._rootRotFiltered.copy(rootTargetRot);
      this._rootInitialized = true;
    } else {
      this._rootPosFiltered.lerp(rootTargetPos, 0.28);
      this._rootRotFiltered.slerp(rootTargetRot, 0.24);
    }

    this.rootBone.position.copy(this._rootPosFiltered);
    this.rootBone.quaternion.copy(this._rootRotFiltered);

    // Build maps for joints
    const ja = new Map<string, [number, number, number, number]>(a.joints.map(j => [j.name, j.rotation]));
    const jb = new Map<string, [number, number, number, number]>(b.joints.map(j => [j.name, j.rotation]));

    // Iterate through available joints
    let boneCount = 0;
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
  boneCount++;

      const rotB = jb.get(srvName) ?? rotA;

      const qA = quatFromArray(rotA);
      const qB = quatFromArray(rotB);
      const qL = slerpQuat(this._qa, qA, qB, alpha);

      const corr = this.corrections.get(tname);

      // If the incoming frames indicate they are already provided in
      // model-relative space (server meta.model_space === 'mixamo'),
      // don't apply corrections — the backend already converted them.
      const serverIsModelSpace = (a.meta && (a.meta as any).model_space === 'mixamo') ||
        (b.meta && (b.meta as any).model_space === 'mixamo');

      // Server quaternions are provided either in a common/world-like
      // frame (older backend) or already in model space (new backend).
      // If we have a per-bone correction and the server did not already
      // convert into model space, apply it by right-multiplying.
      const desiredWorld = (!serverIsModelSpace && corr)
        ? this._qb.copy(qL).multiply(corr)
        : qL.clone();

      // Get parent world quaternion (Object3D.getWorldQuaternion writes into target)
      const parent = bone.parent as Object3D | null;
      if (parent) {
        parent.getWorldQuaternion(this._qr);
        // parent world inverse
        this._qr.invert();
        // local = parentInv * desiredWorld
        const local = this._qa.copy(this._qr).multiply(desiredWorld);

        // Distribute torso bending across the full spine chain.
        if (tname === 'Spine') {
          const spineNames = ['Spine', 'Spine1', 'Spine2', 'Neck', 'Head'];
          const spineWeights = [1.0, 0.55, 0.30, 0.18, 0.10];

          const baseBind = this.bindLocalByBoneName.get(bone.name) ?? bone.quaternion.clone();
          const spineDelta = baseBind.clone().invert().multiply(local).normalize();

          for (let i = 0; i < spineNames.length; ++i) {
            const sName = spineNames[i]!;
            const sb = this.findBoneByName(sName);
            if (!sb) continue;
            const sw = spineWeights[i] ?? 1.0;

            const bind = this.bindLocalByBoneName.get(sb.name) ?? sb.quaternion.clone();
            const weightedDelta = this._identity.clone().slerp(spineDelta, clamp(sw, 0, 1));
            const spineLocalTarget = bind.clone().multiply(weightedDelta).normalize();
            this.applyConstrainedLocalRotation(sb, spineLocalTarget, 'Spine');
          }
        } else {
          this.applyConstrainedLocalRotation(bone, local, tname);
        }
      } else {
        // No parent (shouldn't usually happen) — apply directly
        this.applyConstrainedLocalRotation(bone, desiredWorld, tname);
      }
    }

    if (!this._loggedMappings) this._loggedMappings = true;

    // Upload to GPU
    this.skeleton.update();

    // record profiling info
    const tEnd = (typeof performance !== 'undefined') ? performance.now() : Date.now();
    const elapsed = tEnd - tStart;
    this._lastRetargetMs = elapsed;
    this._lastBoneCount = boneCount;
    this._retargetTimes[this._retargetIndex] = elapsed;
    this._retargetIndex = (this._retargetIndex + 1) % this._retargetTimes.length;
  }

  applyHold(f: PoseFrame): void {
    this.applyInterpolated(f, f, 0);
  }

  // Returns simple profiling stats for the retargeter (ms)
  getRetargetStats() {
    const times = this._retargetTimes.filter(v => v > 0);
    if (times.length === 0) return { last: this._lastRetargetMs, avg: 0, min: 0, max: 0, boneCount: this._lastBoneCount };
    let sum = 0; let min = Number.POSITIVE_INFINITY; let max = 0;
    for (const t of times) { sum += t; min = Math.min(min, t); max = Math.max(max, t); }
    return { last: this._lastRetargetMs, avg: sum / times.length, min, max, boneCount: this._lastBoneCount };
  }
}
