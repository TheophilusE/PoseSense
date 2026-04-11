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
  private axisLocalByBoneName = new Map<string, Vector3>();
  private prevLocalByBoneName = new Map<string, Quaternion>();
  private rootBone: Bone;
  private rootDriver: Object3D | null = null;
  private rootScale: number;
  private _loggedMappings = false;

  private rootBindPos = new Vector3();
  private rootBindRot = new Quaternion();
  private rootDriverBindPos = new Vector3();
  private rootDriverBindRot = new Quaternion();
  private rootSourceOrigin = new Vector3();
  private rootOriginInitialized = false;
  private rootMotionGainXZ = 1.0;
  private rootMotionGainY = 0.55;
  private root2dOrigin = new Vector3();
  private root2dInitialized = false;
  private root2dRefShoulderWidth = 0;
  private root2dGainX = 2.2;
  private root2dGainY = 1.8;
  private root2dGainZ = 0.9;

  // Temp objects reused each frame.
  private _qa = new Quaternion();
  private _qb = new Quaternion();
  private _qr = new Quaternion();
  private _vr = new Vector3();
  private _vs = new Vector3();
  private _vt = new Vector3();
  private _vu = new Vector3();
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

    this.rootDriver = this.resolveRootDriver(this.rootBone) ?? this.mesh.parent;

    this.rootBindPos.copy(this.rootBone.position);
    this.rootBindRot.copy(this.rootBone.quaternion);
    if (this.rootDriver) {
      this.rootDriverBindPos.copy(this.rootDriver.position);
      this.rootDriverBindRot.copy(this.rootDriver.quaternion);
    }

    for (const b of this.skeleton.bones) {
      this.bindLocalByBoneName.set(b.name, b.quaternion.clone());
      this.prevLocalByBoneName.set(b.name, b.quaternion.clone());
    }

    // Cache each bone's rest axis in local space by looking at its child direction.
    for (const bone of this.skeleton.bones) {
      const childBone = bone.children.find((c) => (c as Bone).isBone) as Bone | undefined
        ?? this.skeleton.bones.find(b => b.parent === bone);
      if (!childBone) continue;

      const a = new Vector3();
      const b = new Vector3();
      bone.getWorldPosition(a);
      childBone.getWorldPosition(b);
      const modelDir = b.sub(a).normalize();
      if (modelDir.lengthSq() < 1e-10) continue;

      const wq = new Quaternion();
      bone.getWorldQuaternion(wq);
      const axisLocal = modelDir.clone().applyQuaternion(wq.invert()).normalize();
      if (axisLocal.lengthSq() > 1e-6) {
        this.axisLocalByBoneName.set(bone.name, axisLocal);
      }
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
        this.axisLocalByBoneName.set(bone.name, axisLocal);
      }
    }

    // Biomechanical constraints tuned for noisy webcam motion.
    this.constraints.set('Spine', { maxSwingDeg: 32, twistMinDeg: -22, twistMaxDeg: 22, smoothing: 0.42, maxStepDeg: 18 });
    this.constraints.set('Spine1', { maxSwingDeg: 28, twistMinDeg: -18, twistMaxDeg: 18, smoothing: 0.46, maxStepDeg: 16 });
    this.constraints.set('Spine2', { maxSwingDeg: 24, twistMinDeg: -15, twistMaxDeg: 15, smoothing: 0.50, maxStepDeg: 14 });
    this.constraints.set('Neck', { maxSwingDeg: 22, twistMinDeg: -18, twistMaxDeg: 18, smoothing: 0.54, maxStepDeg: 13 });
    this.constraints.set('Head', { maxSwingDeg: 18, twistMinDeg: -16, twistMaxDeg: 16, smoothing: 0.58, maxStepDeg: 12 });
    this.constraints.set('LeftUpLeg', { maxSwingDeg: 75, twistMinDeg: -35, twistMaxDeg: 35, smoothing: 0.35, maxStepDeg: 24 });
    this.constraints.set('RightUpLeg', { maxSwingDeg: 75, twistMinDeg: -35, twistMaxDeg: 35, smoothing: 0.35, maxStepDeg: 24 });
    this.constraints.set('LeftLeg', { maxSwingDeg: 110, twistMinDeg: -20, twistMaxDeg: 20, smoothing: 0.45, maxStepDeg: 26 });
    this.constraints.set('RightLeg', { maxSwingDeg: 110, twistMinDeg: -20, twistMaxDeg: 20, smoothing: 0.45, maxStepDeg: 26 });
    this.constraints.set('LeftShoulder', { maxSwingDeg: 95, twistMinDeg: -18, twistMaxDeg: 18, smoothing: 0.52, maxStepDeg: 30 });
    this.constraints.set('RightShoulder', { maxSwingDeg: 95, twistMinDeg: -18, twistMaxDeg: 18, smoothing: 0.52, maxStepDeg: 30 });
    this.constraints.set('LeftArm', { maxSwingDeg: 145, twistMinDeg: -28, twistMaxDeg: 28, smoothing: 0.54, maxStepDeg: 34 });
    this.constraints.set('RightArm', { maxSwingDeg: 145, twistMinDeg: -28, twistMaxDeg: 28, smoothing: 0.54, maxStepDeg: 34 });
    this.constraints.set('LeftForeArm', { maxSwingDeg: 155, twistMinDeg: -36, twistMaxDeg: 36, smoothing: 0.58, maxStepDeg: 36 });
    this.constraints.set('RightForeArm', { maxSwingDeg: 155, twistMinDeg: -36, twistMaxDeg: 36, smoothing: 0.58, maxStepDeg: 36 });
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

  private resolveRootDriver(fromBone: Bone): Object3D | null {
    let p: Object3D | null = fromBone.parent;
    while (p && (p as any).isBone) {
      p = p.parent;
    }
    return p;
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
    const axisLocal = this.axisLocalByBoneName.get(bone.name)
      ?? this.axisLocalByJoint.get(jointName)
      ?? new Vector3(1, 0, 0);

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

  private extractRootPosition(frame: PoseFrame, out: Vector3): Vector3 {
    out.copy(vec3FromArray(frame.root.position)).multiplyScalar(this.rootScale);

    const pose2d = frame.meta && (frame.meta as any).pose_landmarks_2d;
    if (!Array.isArray(pose2d) || pose2d.length < 25) {
      return out;
    }

    const ls = pose2d[11];
    const rs = pose2d[12];
    const lh = pose2d[23];
    const rh = pose2d[24];
    if (![ls, rs, lh, rh].every((p) => Array.isArray(p) && p.length >= 2)) {
      return out;
    }

    const hipX = (Number(lh[0]) + Number(rh[0])) * 0.5;
    const hipY = (Number(lh[1]) + Number(rh[1])) * 0.5;
    const shoulderW = Math.hypot(Number(ls[0]) - Number(rs[0]), Number(ls[1]) - Number(rs[1]));

    if (!Number.isFinite(hipX) || !Number.isFinite(hipY) || !Number.isFinite(shoulderW) || shoulderW < 1e-4) {
      return out;
    }

    if (!this.root2dInitialized) {
      this.root2dOrigin.set(hipX, hipY, 0);
      this.root2dRefShoulderWidth = shoulderW;
      this.root2dInitialized = true;
    }

    const dx = (hipX - this.root2dOrigin.x) * this.root2dGainX;
    const dy = (this.root2dOrigin.y - hipY) * this.root2dGainY;
    const depthRatio = this.root2dRefShoulderWidth / Math.max(shoulderW, 1e-4);
    const dz = (depthRatio - 1.0) * this.root2dGainZ;

    out.add(this._vu.set(dx, dy, dz));
    return out;
  }

  applyInterpolated(a: PoseFrame, b: PoseFrame, alphaIn: number): void {
    const tStart = (typeof performance !== 'undefined') ? performance.now() : Date.now();
    const alpha = clamp(alphaIn, 0, 1);

    // Root pose
    const pa = this.extractRootPosition(a, this._vs);
    const pb = this.extractRootPosition(b, this._vt);
    const qa = quatFromArray(a.root.rotation);
    const qb = quatFromArray(b.root.rotation);

    const p = lerpVec3(this._vr, pa, pb, alpha);
    const q = slerpQuat(this._qr, qa, qb, alpha);

    if (!this.rootOriginInitialized) {
      this.rootSourceOrigin.copy(p);
      this.rootOriginInitialized = true;
    }

    const delta = this._vr.copy(p).sub(this.rootSourceOrigin);
    const rootPosBase = this.rootDriver ? this.rootDriverBindPos : this.rootBindPos;
    const rootRotBase = this.rootDriver ? this.rootDriverBindRot : this.rootBindRot;
    const rootTargetPos = new Vector3(
      rootPosBase.x + delta.x * this.rootMotionGainXZ,
      rootPosBase.y + delta.y * this.rootMotionGainY,
      rootPosBase.z + delta.z * this.rootMotionGainXZ,
    );
    const rootTargetRot = rootRotBase.clone().multiply(this.extractYawQuaternion(q));

    // Smooth root motion while preserving visible translation from the pose midpoint.
    if (!this._rootInitialized) {
      this._rootPosFiltered.copy(rootTargetPos);
      this._rootRotFiltered.copy(rootTargetRot);
      this._rootInitialized = true;
    } else {
      this._rootPosFiltered.lerp(rootTargetPos, 0.28);
      this._rootRotFiltered.slerp(rootTargetRot, 0.24);
    }

    if (this.rootDriver) {
      this.rootDriver.position.copy(this._rootPosFiltered);
      this.rootDriver.quaternion.copy(this._rootRotFiltered);

      // Keep hips in bind local pose and drive global motion on a non-bone
      // parent object. This prevents a long root-to-hips connector artifact.
      this.rootBone.position.copy(this.rootBindPos);
      this.rootBone.quaternion.copy(this.rootBindRot);
    } else {
      this.rootBone.position.copy(this._rootPosFiltered);
      this.rootBone.quaternion.copy(this._rootRotFiltered);
    }

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
          const spineWeights = [0.72, 0.46, 0.30, 0.18, 0.10];

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
            this.applyConstrainedLocalRotation(sb, spineLocalTarget, sName);
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
