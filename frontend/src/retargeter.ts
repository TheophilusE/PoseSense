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

type MotionProfile = 'default' | 'procedural';

type RetargeterOptions = {
  rootScale?: number;
  corrections?: Record<string, Quaternion>;
  frameYawDeg?: number;
  motionProfile?: MotionProfile;
  enableLegIK?: boolean;
  rootDriverObject?: Object3D | null;
  jointMotionGain?: number;
  rootPosSmoothing?: number;
  rootRotSmoothing?: number;
  rootMotionGainXZ?: number;
  rootMotionGainY?: number;
  root2dGainX?: number;
  root2dGainY?: number;
  root2dGainZ?: number;
  groundCharacter?: boolean;
  groundY?: number;
};

type FootLockState = {
  initialized: boolean;
  locked: boolean;
  lockPos: Vector3;
  prevTarget: Vector3;
};

type HeldPositionState = {
  initialized: boolean;
  value: Vector3;
};

export class Retargeter {
  private mesh: SkinnedMesh;
  private skeleton: SkinnedMesh['skeleton'];
  private bonesByName = new Map<string, Bone>();
  private corrections = new Map<string, Quaternion>();
  private sourceRestDirByJoint = new Map<string, Vector3>();
  private modelRestDirByJoint = new Map<string, Vector3>();
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
  private root2dOffsetFiltered = new Vector3();
  private root2dOffsetSmoothing = 0.22;
  private root2dOffsetDecay = 0.08;
  private root2dDepthRatioMin = 0.74;
  private root2dDepthRatioMax = 1.30;
  private root2dMinShoulderScale = 0.42;
  private root2dMinVisibility = 0.20;
  private root2dMaxLateralOffset = 1.25;
  private root2dMaxVerticalOffset = 1.05;
  private root2dMaxDepthOffset = 0.46;
  private rootPosSmoothing = 0.28;
  private rootRotSmoothing = 0.24;
  private jointMotionGain = 1.0;
  private jointGainByName = new Map<string, number>();
  private keepGrounded = false;
  private groundY = 0.0;
  private groundPenetrationEpsilon = 0.008;
  private enableLegIK = false;
  private legTargetSmoothing = 0.58;
  private footPlantHeight = 0.055;
  private footPlantVelEnter = 0.014;
  private footPlantVelExit = 0.045;
  private footLiftUnlockHeight = 0.10;
  private footLiftUnlockRelative = 0.052;
  private jumpUnlockHeight = 0.065;
  private airborneTargetSmoothing = 0.34;
  private enforceFlatFeet = true;
  private flatFeetForwardBlend = 0.18;
  private flatFeetSmoothing = 0.30;
  private pelvisShiftStrengthSingle = 0.22;
  private pelvisShiftStrengthDouble = 0.10;
  private pelvisShiftMaxStep = 0.05;
  private kneeHintSmoothing = 0.62;

  private leftFootLock: FootLockState = {
    initialized: false,
    locked: false,
    lockPos: new Vector3(),
    prevTarget: new Vector3(),
  };
  private rightFootLock: FootLockState = {
    initialized: false,
    locked: false,
    lockPos: new Vector3(),
    prevTarget: new Vector3(),
  };
  private leftKneeHintState: HeldPositionState = {
    initialized: false,
    value: new Vector3(),
  };
  private rightKneeHintState: HeldPositionState = {
    initialized: false,
    value: new Vector3(),
  };

  private leftUpLegBone: Bone | null = null;
  private rightUpLegBone: Bone | null = null;
  private leftLegBone: Bone | null = null;
  private rightLegBone: Bone | null = null;
  private leftFootBone: Bone | null = null;
  private rightFootBone: Bone | null = null;
  private leftToeBone: Bone | null = null;
  private rightToeBone: Bone | null = null;
  private leftUpperLen = 0.42;
  private rightUpperLen = 0.42;
  private leftLowerLen = 0.42;
  private rightLowerLen = 0.42;
  private leftLegTargetFiltered = new Vector3();
  private rightLegTargetFiltered = new Vector3();
  private leftLegTargetInitialized = false;
  private rightLegTargetInitialized = false;

  // Temp objects reused each frame.
  private _qa = new Quaternion();
  private _qb = new Quaternion();
  private _qr = new Quaternion();
  private _vr = new Vector3();
  private _vs = new Vector3();
  private _vt = new Vector3();
  private _vu = new Vector3();
  private _vw = new Vector3();
  private _vx = new Vector3();
  private _vy = new Vector3();
  private _vz = new Vector3();
  private _ikV1 = new Vector3();
  private _ikV2 = new Vector3();
  private _ikV3 = new Vector3();
  private _ikV4 = new Vector3();
  private _ikV5 = new Vector3();
  private _ikV6 = new Vector3();
  private _ikV7 = new Vector3();
  private _ikV8 = new Vector3();
  private _ikV9 = new Vector3();
  private _ikV10 = new Vector3();
  private _supportTarget = new Vector3();
  private _kneeHintL = new Vector3();
  private _kneeHintR = new Vector3();
  private _kneePoleL = new Vector3();
  private _kneePoleR = new Vector3();
  private _identity = new Quaternion();
  private _ikQ1 = new Quaternion();
  private _ikQ2 = new Quaternion();
  private _ikQ3 = new Quaternion();
  private _ikQ4 = new Quaternion();
  private _ikQ5 = new Quaternion();
  private _tmpForward = new Vector3();
  private _upAxis = new Vector3(0, 1, 0);
  private _frameYawFix = new Quaternion();

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
      case 'Chest': return 'Spine2';
      case 'Neck': return 'Neck';
      case 'Head': return 'Head';
      case 'LeftUpLeg': return 'LeftUpLeg';
      case 'LeftLeg': return 'LeftLeg';
      case 'LeftFoot': return 'LeftFoot';
      case 'RightUpLeg': return 'RightUpLeg';
      case 'RightLeg': return 'RightLeg';
      case 'RightFoot': return 'RightFoot';
      case 'LeftShoulder': return 'LeftShoulder';
      case 'LeftArm': return 'LeftArm';
      case 'LeftForeArm': return 'LeftForeArm';
      case 'LeftHand': return 'LeftHand';
      case 'RightShoulder': return 'RightShoulder';
      case 'RightArm': return 'RightArm';
      case 'RightForeArm': return 'RightForeArm';
      case 'RightHand': return 'RightHand';
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

  constructor(skinnedMesh: SkinnedMesh, options?: RetargeterOptions) {
    this.mesh = skinnedMesh;
    this.skeleton = skinnedMesh.skeleton;
    this.rootScale = options?.rootScale ?? 1.0;
    const frameYawDeg = options?.frameYawDeg ?? 0;
    this._frameYawFix.setFromAxisAngle(this._upAxis, MathUtils.degToRad(frameYawDeg));

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

    const preferredParent = (this.mesh.parent && !(this.mesh.parent as any).isBone)
      ? this.mesh.parent
      : null;
    this.rootDriver = options?.rootDriverObject ?? preferredParent ?? this.resolveRootDriver(this.rootBone);
    this.leftUpLegBone = this.findBoneByName('LeftUpLeg') ?? null;
    this.rightUpLegBone = this.findBoneByName('RightUpLeg') ?? null;
    this.leftLegBone = this.findBoneByName('LeftLeg') ?? null;
    this.rightLegBone = this.findBoneByName('RightLeg') ?? null;
    this.leftFootBone = this.findBoneByName('LeftFoot') ?? null;
    this.rightFootBone = this.findBoneByName('RightFoot') ?? null;
    this.leftToeBone = this.findBoneByName('LeftToeBase') ?? null;
    this.rightToeBone = this.findBoneByName('RightToeBase') ?? null;
    this.leftUpperLen = this.leftLegBone ? Math.max(this.leftLegBone.position.length(), 0.12) : 0.42;
    this.rightUpperLen = this.rightLegBone ? Math.max(this.rightLegBone.position.length(), 0.12) : 0.42;
    this.leftLowerLen = this.leftFootBone ? Math.max(this.leftFootBone.position.length(), 0.12) : 0.42;
    this.rightLowerLen = this.rightFootBone ? Math.max(this.rightFootBone.position.length(), 0.12) : 0.42;

    this.rootBindPos.copy(this.rootBone.position);
    this.rootBindRot.copy(this.rootBone.quaternion);
    if (this.rootDriver) {
      this.rootDriverBindPos.copy(this.rootDriver.position);
      this.rootDriverBindRot.copy(this.rootDriver.quaternion);
    }
    this.keepGrounded = !!options?.groundCharacter;
    this.groundY = typeof options?.groundY === 'number'
      ? options.groundY
      : (this.rootDriver ? this.rootDriverBindPos.y : this.rootBindPos.y);

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
      Spine2: [0, 1, 0],
      Neck: [0, 1, 0],
      Head: [0, 1, 0],
      LeftUpLeg: [0, -1, 0],
      LeftLeg: [0, -1, 0],
      LeftFoot: [0, 0, 1],
      RightUpLeg: [0, -1, 0],
      RightLeg: [0, -1, 0],
      RightFoot: [0, 0, 1],
      LeftShoulder: [-1, 0, 0],
      RightShoulder: [1, 0, 0],
      LeftArm: [-1, 0, 0],
      LeftForeArm: [-1, 0, 0],
      LeftHand: [-1, 0, 0],
      RightArm: [1, 0, 0],
      RightForeArm: [1, 0, 0],
      RightHand: [1, 0, 0],
    };

    for (const [name, dirArr] of Object.entries(serverRestDirs)) {
      this.sourceRestDirByJoint.set(name, new Vector3(dirArr[0], dirArr[1], dirArr[2]).normalize());

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
      this.modelRestDirByJoint.set(name, modelDir.clone());

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
    this.constraints.set('LeftFoot', { maxSwingDeg: 65, twistMinDeg: -20, twistMaxDeg: 20, smoothing: 0.50, maxStepDeg: 22 });
    this.constraints.set('RightFoot', { maxSwingDeg: 65, twistMinDeg: -20, twistMaxDeg: 20, smoothing: 0.50, maxStepDeg: 22 });
    this.constraints.set('LeftShoulder', { maxSwingDeg: 95, twistMinDeg: -18, twistMaxDeg: 18, smoothing: 0.52, maxStepDeg: 30 });
    this.constraints.set('RightShoulder', { maxSwingDeg: 95, twistMinDeg: -18, twistMaxDeg: 18, smoothing: 0.52, maxStepDeg: 30 });
    this.constraints.set('LeftArm', { maxSwingDeg: 145, twistMinDeg: -28, twistMaxDeg: 28, smoothing: 0.54, maxStepDeg: 34 });
    this.constraints.set('RightArm', { maxSwingDeg: 145, twistMinDeg: -28, twistMaxDeg: 28, smoothing: 0.54, maxStepDeg: 34 });
    this.constraints.set('LeftForeArm', { maxSwingDeg: 155, twistMinDeg: -36, twistMaxDeg: 36, smoothing: 0.58, maxStepDeg: 36 });
    this.constraints.set('RightForeArm', { maxSwingDeg: 155, twistMinDeg: -36, twistMaxDeg: 36, smoothing: 0.58, maxStepDeg: 36 });
    this.constraints.set('LeftHand', { maxSwingDeg: 70, twistMinDeg: -30, twistMaxDeg: 30, smoothing: 0.60, maxStepDeg: 28 });
    this.constraints.set('RightHand', { maxSwingDeg: 70, twistMinDeg: -30, twistMaxDeg: 30, smoothing: 0.60, maxStepDeg: 28 });

    const motionProfile = options?.motionProfile ?? 'default';
    if (motionProfile === 'procedural') {
      this.applyProceduralMotionProfile();
    }
    this.enableLegIK = options?.enableLegIK ?? (motionProfile === 'procedural');

    if (typeof options?.jointMotionGain === 'number') {
      this.jointMotionGain = Math.max(0.5, options.jointMotionGain);
    }
    if (typeof options?.rootPosSmoothing === 'number') {
      this.rootPosSmoothing = clamp(options.rootPosSmoothing, 0.01, 1.0);
    }
    if (typeof options?.rootRotSmoothing === 'number') {
      this.rootRotSmoothing = clamp(options.rootRotSmoothing, 0.01, 1.0);
    }
    if (typeof options?.rootMotionGainXZ === 'number') {
      this.rootMotionGainXZ = Math.max(0.0, options.rootMotionGainXZ);
    }
    if (typeof options?.rootMotionGainY === 'number') {
      this.rootMotionGainY = Math.max(0.0, options.rootMotionGainY);
    }
    if (typeof options?.root2dGainX === 'number') {
      this.root2dGainX = Math.max(0.0, options.root2dGainX);
    }
    if (typeof options?.root2dGainY === 'number') {
      this.root2dGainY = Math.max(0.0, options.root2dGainY);
    }
    if (typeof options?.root2dGainZ === 'number') {
      this.root2dGainZ = Math.max(0.0, options.root2dGainZ);
    }
  }

  private applyProceduralMotionProfile(): void {
    // Procedural avatar can tolerate a punchier response than Mixamo.
    this.rootMotionGainXZ = 1.9;
    this.rootMotionGainY = 0.0;
    this.root2dGainX = 3.4;
    this.root2dGainY = 2.9;
    this.root2dGainZ = 1.4;
    this.root2dOffsetSmoothing = 0.20;
    this.root2dOffsetDecay = 0.06;
    this.root2dDepthRatioMax = 1.22;
    this.root2dMaxDepthOffset = 0.36;
    this.rootPosSmoothing = 0.58;
    this.rootRotSmoothing = 0.50;
    this.jointMotionGain = 1.18;
    this.footPlantHeight = 0.058;
    this.footPlantVelEnter = 0.013;
    this.footPlantVelExit = 0.040;
    this.footLiftUnlockHeight = 0.096;
    this.footLiftUnlockRelative = 0.048;
    this.jumpUnlockHeight = 0.052;
    this.airborneTargetSmoothing = 0.28;
    this.flatFeetForwardBlend = 0.12;
    this.flatFeetSmoothing = 0.36;
    this.pelvisShiftStrengthSingle = 0.24;
    this.pelvisShiftStrengthDouble = 0.12;
    this.pelvisShiftMaxStep = 0.055;

    this.defaultConstraint = {
      maxSwingDeg: 108,
      twistMinDeg: -62,
      twistMaxDeg: 62,
      smoothing: 0.64,
      maxStepDeg: 56,
    };

    for (const [name, cfg] of this.constraints.entries()) {
      this.constraints.set(name, {
        maxSwingDeg: cfg.maxSwingDeg * 1.2,
        twistMinDeg: cfg.twistMinDeg * 1.28,
        twistMaxDeg: cfg.twistMaxDeg * 1.28,
        smoothing: clamp(cfg.smoothing + 0.20, 0.01, 1.0),
        maxStepDeg: cfg.maxStepDeg * 1.9,
      });
    }

    // Lower body should remain stable and grounded while upper body can be expressive.
    this.constraints.set('LeftUpLeg', { maxSwingDeg: 74, twistMinDeg: -22, twistMaxDeg: 22, smoothing: 0.52, maxStepDeg: 18 });
    this.constraints.set('RightUpLeg', { maxSwingDeg: 74, twistMinDeg: -22, twistMaxDeg: 22, smoothing: 0.52, maxStepDeg: 18 });
    this.constraints.set('LeftLeg', { maxSwingDeg: 92, twistMinDeg: -16, twistMaxDeg: 16, smoothing: 0.56, maxStepDeg: 16 });
    this.constraints.set('RightLeg', { maxSwingDeg: 92, twistMinDeg: -16, twistMaxDeg: 16, smoothing: 0.56, maxStepDeg: 16 });
    this.constraints.set('LeftFoot', { maxSwingDeg: 42, twistMinDeg: -14, twistMaxDeg: 14, smoothing: 0.62, maxStepDeg: 14 });
    this.constraints.set('RightFoot', { maxSwingDeg: 42, twistMinDeg: -14, twistMaxDeg: 14, smoothing: 0.62, maxStepDeg: 14 });

    this.jointGainByName.clear();
    const expressive = ['Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'RightShoulder', 'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand'];
    const lowerBody = ['LeftUpLeg', 'RightUpLeg', 'LeftLeg', 'RightLeg', 'LeftFoot', 'RightFoot'];
    for (const n of expressive) this.jointGainByName.set(n, 1.38);
    for (const n of lowerBody) this.jointGainByName.set(n, 0.88);
  }

  private updateFootLockState(state: FootLockState, targetWorld: Vector3): void {
    if (!state.initialized) {
      state.initialized = true;
      state.prevTarget.copy(targetWorld);
      state.lockPos.copy(targetWorld);
      return;
    }

    const speed = targetWorld.distanceTo(state.prevTarget);
    const verticalSpeed = Math.abs(targetWorld.y - state.prevTarget.y);
    const lift = targetWorld.y - this.groundY;
    const liftFromLock = targetWorld.y - state.lockPos.y;

    if (state.locked) {
      const shouldUnlock =
        (lift > this.footLiftUnlockHeight)
        || (liftFromLock > this.footLiftUnlockRelative)
        || (verticalSpeed > (this.footPlantVelExit * 0.6))
        || (speed > this.footPlantVelExit);
      if (shouldUnlock) {
        state.locked = false;
        state.lockPos.copy(targetWorld);
      } else {
        // Allow tiny drift to avoid jitter when locked.
        state.lockPos.lerp(targetWorld, 0.03);
      }
    } else if (this.keepGrounded) {
      const shouldLock =
        (lift < this.footPlantHeight)
        && (speed < this.footPlantVelEnter)
        && (verticalSpeed < this.footPlantVelEnter);
      if (shouldLock) {
        state.locked = true;
        state.lockPos.copy(targetWorld);
      } else {
        state.lockPos.copy(targetWorld);
      }
    } else {
      state.lockPos.copy(targetWorld);
    }

    state.prevTarget.copy(targetWorld);
  }

  private applySupportPelvisShift(): void {
    if (!this.keepGrounded) return;

    const mover = this.rootDriver ?? this.rootBone;
    const leftLocked = this.leftFootLock.locked;
    const rightLocked = this.rightFootLock.locked;

    if (!leftLocked && !rightLocked) return;

    this._supportTarget.set(0, 0, 0);
    let count = 0;
    if (leftLocked) {
      this._supportTarget.add(this.leftFootLock.lockPos);
      count += 1;
    }
    if (rightLocked) {
      this._supportTarget.add(this.rightFootLock.lockPos);
      count += 1;
    }
    if (count <= 0) return;

    this._supportTarget.multiplyScalar(1 / count);
    const strength = count === 1 ? this.pelvisShiftStrengthSingle : this.pelvisShiftStrengthDouble;

    const dx = (this._supportTarget.x - mover.position.x) * strength;
    const dz = (this._supportTarget.z - mover.position.z) * strength;
    const mag = Math.hypot(dx, dz);
    let sx = dx;
    let sz = dz;
    if (mag > this.pelvisShiftMaxStep && mag > 1e-6) {
      const scale = this.pelvisShiftMaxStep / mag;
      sx *= scale;
      sz *= scale;
    }

    mover.position.x += sx;
    mover.position.z += sz;
    this._rootPosFiltered.x += sx;
    this._rootPosFiltered.z += sz;
  }

  private resolveKneeHintWorld(
    a: PoseFrame,
    b: PoseFrame,
    alpha: number,
    kneeKey: string,
    state: HeldPositionState,
    fallback: Vector3,
    out: Vector3,
  ): boolean {
    const hasKnee = this.getInterpolatedIntermediateRelative(a, b, alpha, kneeKey, this._ikV4);
    if (hasKnee) {
      this._ikV4.add(this._rootPosFiltered);
      if (!state.initialized) {
        state.initialized = true;
        state.value.copy(this._ikV4);
      } else {
        state.value.lerp(this._ikV4, this.kneeHintSmoothing);
      }
      out.copy(state.value);
      return true;
    }

    if (!state.initialized) {
      state.initialized = true;
      state.value.copy(fallback);
    }
    out.copy(state.value);
    return false;
  }

  private amplifyLocalTarget(localTarget: Quaternion, bindLocal: Quaternion, jointName: string): Quaternion {
    const gain = clamp(this.jointGainByName.get(jointName) ?? this.jointMotionGain, 0.5, 2.6);
    if (Math.abs(gain - 1.0) < 1e-3) {
      return localTarget;
    }

    const delta = bindLocal.clone().invert().multiply(localTarget).normalize();
    const w = MathUtils.clamp(delta.w, -1, 1);
    const angle = 2 * Math.acos(w);
    if (angle < 1e-6) {
      return localTarget;
    }

    const s = Math.sqrt(Math.max(1 - (w * w), 0));
    const axis = s < 1e-6
      ? new Vector3(1, 0, 0)
      : new Vector3(delta.x / s, delta.y / s, delta.z / s).normalize();
    const amplifiedDelta = new Quaternion()
      .setFromAxisAngle(axis, angle * gain)
      .normalize();
    return bindLocal.clone().multiply(amplifiedDelta).normalize();
  }

  private getMinTrackedFootY(): number | null {
    const probes: Array<Bone | null> = [
      this.leftFootBone,
      this.rightFootBone,
      this.leftToeBone,
      this.rightToeBone,
    ];
    let found = false;
    let minY = Number.POSITIVE_INFINITY;
    for (const bone of probes) {
      if (!bone) continue;
      bone.getWorldPosition(this._vw);
      minY = Math.min(minY, this._vw.y);
      found = true;
    }
    return found ? minY : null;
  }

  private applyFootGroundingCorrection(): void {
    const minFootY = this.getMinTrackedFootY();
    if (minFootY === null) return;

    const penetration = this.groundY - minFootY;
    if (penetration <= this.groundPenetrationEpsilon) return;

    const lift = penetration * 0.92;
    const mover = this.rootDriver ?? this.rootBone;
    mover.position.y += lift;
    this._rootPosFiltered.y += lift;
  }

  private readIntermediateTargetRelative(frame: PoseFrame, key: string, out: Vector3): boolean {
    const meta = frame.meta as any;
    const map = meta?.intermediate_targets ?? meta?.intermediate_targets_raw;
    const raw = map?.[key];
    if (!Array.isArray(raw) || raw.length < 3) return false;

    const x = Number(raw[0]);
    const y = Number(raw[1]);
    const z = Number(raw[2]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return false;

    const root = frame.root?.position;
    if (!Array.isArray(root) || root.length < 3) return false;
    const rx = Number(root[0]);
    const ry = Number(root[1]);
    const rz = Number(root[2]);
    if (!Number.isFinite(rx) || !Number.isFinite(ry) || !Number.isFinite(rz)) return false;

    out.set(x - rx, y - ry, z - rz)
      .multiplyScalar(this.rootScale)
      .applyQuaternion(this._frameYawFix);
    return true;
  }

  private getInterpolatedIntermediateRelative(a: PoseFrame, b: PoseFrame, alpha: number, key: string, out: Vector3): boolean {
    const hasA = this.readIntermediateTargetRelative(a, key, this._ikV1);
    const hasB = this.readIntermediateTargetRelative(b, key, this._ikV2);
    if (!hasA && !hasB) return false;
    if (hasA && hasB) {
      out.copy(this._ikV1).lerp(this._ikV2, alpha);
      return true;
    }
    out.copy(hasA ? this._ikV1 : this._ikV2);
    return true;
  }

  private getLegTargetWorld(a: PoseFrame, b: PoseFrame, alpha: number, ankleKey: string, out: Vector3): boolean {
    const ok = this.getInterpolatedIntermediateRelative(a, b, alpha, ankleKey, this._ikV3);
    if (!ok) return false;

    out.copy(this._rootPosFiltered).add(this._ikV3);
    if (this.keepGrounded) {
      // Prevent below-floor targets while preserving true lift trajectories.
      out.y = Math.max(out.y, this.groundY + 0.014);
    }
    return true;
  }

  private getLegPoleWorld(a: PoseFrame, b: PoseFrame, alpha: number, kneeKey: string, fallback: Vector3, out: Vector3): Vector3 {
    const ok = this.getInterpolatedIntermediateRelative(a, b, alpha, kneeKey, this._ikV4);
    if (ok) {
      return out.copy(this._rootPosFiltered).add(this._ikV4);
    }
    return out.copy(fallback);
  }

  private solveTwoBoneIk(
    upper: Bone,
    lower: Bone,
    end: Bone,
    targetWorld: Vector3,
    poleWorld: Vector3,
    kneeHintWorld: Vector3 | null,
    lenUpper: number,
    lenLower: number,
    upperJointName: string,
    lowerJointName: string,
  ): void {
    this.mesh.updateMatrixWorld(true);

    upper.getWorldPosition(this._ikV5);
    lower.getWorldPosition(this._ikV6);
    end.getWorldPosition(this._ikV7);

    const toTarget = this._ikV8.copy(targetWorld).sub(this._ikV5);
    let dist = toTarget.length();
    if (dist < 1e-6) return;

    const minReach = Math.max(Math.abs(lenUpper - lenLower) + 1e-4, 1e-4);
    const maxReach = Math.max(lenUpper + lenLower - 1e-4, minReach + 1e-4);
    dist = clamp(dist, minReach, maxReach);
    const dir = toTarget.normalize();

    const bend = this._ikV9.copy(poleWorld).sub(this._ikV5);
    bend.sub(this._ikV10.copy(dir).multiplyScalar(bend.dot(dir)));
    if (bend.lengthSq() < 1e-8) {
      bend.copy(this._ikV6).sub(this._ikV5);
      bend.sub(this._ikV10.copy(dir).multiplyScalar(bend.dot(dir)));
    }
    if (bend.lengthSq() < 1e-8) {
      bend.set(0, 0, 1);
      bend.sub(this._ikV10.copy(dir).multiplyScalar(bend.dot(dir)));
    }
    bend.normalize();

    const x = ((lenUpper * lenUpper) - (lenLower * lenLower) + (dist * dist)) / (2 * dist);
    const y = Math.sqrt(Math.max((lenUpper * lenUpper) - (x * x), 0));
    const midTarget = this._vx.copy(this._ikV5).addScaledVector(dir, x).addScaledVector(bend, y);
    const endTarget = this._vy.copy(this._ikV5).addScaledVector(dir, dist);

    if (kneeHintWorld) {
      // Pull the mid joint toward the observed knee while preserving upper length.
      const desiredMid = this._ikV10.copy(kneeHintWorld).sub(this._ikV5);
      if (desiredMid.lengthSq() > 1e-8) {
        desiredMid.normalize().multiplyScalar(lenUpper).add(this._ikV5);
        midTarget.lerp(desiredMid, 0.9);
      }
    }

    upper.getWorldQuaternion(this._ikQ1);
    const curUpperDir = this._vz.copy(this._ikV6).sub(this._ikV5).normalize();
    const desUpperDir = this._ikV1.copy(midTarget).sub(this._ikV5).normalize();
    if (curUpperDir.lengthSq() > 1e-8 && desUpperDir.lengthSq() > 1e-8) {
      this._ikQ2.setFromUnitVectors(curUpperDir, desUpperDir);
      this._ikQ3.copy(this._ikQ2).multiply(this._ikQ1);

      const parent = upper.parent as Object3D | null;
      if (parent) {
        parent.getWorldQuaternion(this._ikQ4);
        this._ikQ4.invert();
        const upperLocal = this._ikQ5.copy(this._ikQ4).multiply(this._ikQ3);
        this.applyConstrainedLocalRotation(upper, upperLocal, upperJointName);
      }
    }

    this.mesh.updateMatrixWorld(true);
    lower.getWorldPosition(this._ikV6);
    end.getWorldPosition(this._ikV7);
    lower.getWorldQuaternion(this._ikQ1);

    const curLowerDir = this._ikV2.copy(this._ikV7).sub(this._ikV6).normalize();
    const desLowerDir = this._ikV3.copy(endTarget).sub(this._ikV6).normalize();
    if (curLowerDir.lengthSq() > 1e-8 && desLowerDir.lengthSq() > 1e-8) {
      this._ikQ2.setFromUnitVectors(curLowerDir, desLowerDir);
      this._ikQ3.copy(this._ikQ2).multiply(this._ikQ1);

      const parent = lower.parent as Object3D | null;
      if (parent) {
        parent.getWorldQuaternion(this._ikQ4);
        this._ikQ4.invert();
        const lowerLocal = this._ikQ5.copy(this._ikQ4).multiply(this._ikQ3);
        this.applyConstrainedLocalRotation(lower, lowerLocal, lowerJointName);
      }
    }
  }

  private applyFlatFootPose(footBone: Bone, toeBone: Bone | null, jointName: 'LeftFoot' | 'RightFoot'): void {
    const parent = footBone.parent as Object3D | null;
    if (!parent) return;

    const axisLocal = (this.axisLocalByBoneName.get(footBone.name) ?? this._tmpForward.set(0, 0, 1)).clone().normalize();
    if (axisLocal.lengthSq() < 1e-8) return;

    footBone.getWorldPosition(this._ikV1);
    footBone.getWorldQuaternion(this._ikQ1);

    const desiredForward = this._ikV2.set(0, 0, 1).applyQuaternion(this._rootRotFiltered);
    desiredForward.y = 0;
    if (desiredForward.lengthSq() < 1e-8) {
      desiredForward.set(0, 0, 1);
    } else {
      desiredForward.normalize();
    }

    if (toeBone) {
      toeBone.getWorldPosition(this._ikV3);
      const observedForward = this._ikV4.copy(this._ikV3).sub(this._ikV1);
      observedForward.y = 0;
      if (observedForward.lengthSq() > 1e-8) {
        observedForward.normalize();
        desiredForward.lerp(observedForward, this.flatFeetForwardBlend).normalize();
      }
    }

    const currentForward = this._ikV5.copy(axisLocal).applyQuaternion(this._ikQ1).normalize();
    const currentFlat = this._ikV6.copy(currentForward);
    currentFlat.y = 0;
    if (currentFlat.lengthSq() < 1e-8) {
      currentFlat.copy(desiredForward);
    } else {
      currentFlat.normalize();
    }

    this._ikQ2.setFromUnitVectors(currentFlat, desiredForward);
    const desiredWorld = this._ikQ3.copy(this._ikQ2).multiply(this._ikQ1).normalize();

    const forwardAfterYaw = this._ikV7.copy(axisLocal).applyQuaternion(desiredWorld).normalize();
    const forwardFlat = this._ikV8.copy(forwardAfterYaw);
    forwardFlat.y = 0;
    if (forwardFlat.lengthSq() > 1e-8) {
      forwardFlat.normalize();
      this._ikQ4.setFromUnitVectors(forwardAfterYaw, forwardFlat);
      desiredWorld.premultiply(this._ikQ4).normalize();
    }

    parent.getWorldQuaternion(this._ikQ5);
    this._ikQ5.invert();
    const desiredLocal = this._ikQ2.copy(this._ikQ5).multiply(desiredWorld).normalize();

    const prevLocal = this.prevLocalByBoneName.get(footBone.name) ?? footBone.quaternion.clone();
    const blendedLocal = this._ikQ3.copy(prevLocal).slerp(desiredLocal, this.flatFeetSmoothing).normalize();
    this.applyConstrainedLocalRotation(footBone, blendedLocal, jointName);
  }

  private applyFlatFootOrientation(): void {
    if (!this.enforceFlatFeet) return;

    this.mesh.updateMatrixWorld(true);
    if (this.leftFootBone) {
      this.applyFlatFootPose(this.leftFootBone, this.leftToeBone, 'LeftFoot');
    }
    if (this.rightFootBone) {
      this.applyFlatFootPose(this.rightFootBone, this.rightToeBone, 'RightFoot');
    }
  }

  private applyProceduralLegIk(a: PoseFrame, b: PoseFrame, alpha: number): void {
    if (!this.enableLegIK) return;
    if (!this.leftUpLegBone || !this.leftLegBone || !this.leftFootBone) return;
    if (!this.rightUpLegBone || !this.rightLegBone || !this.rightFootBone) return;

    const hasLeftTarget = this.getLegTargetWorld(a, b, alpha, 'left_ankle', this._ikV4);
    const hasRightTarget = this.getLegTargetWorld(a, b, alpha, 'right_ankle', this._ikV5);

    if (!hasLeftTarget && !hasRightTarget) return;

    if (hasLeftTarget) {
      this.updateFootLockState(this.leftFootLock, this._ikV4);
    }
    if (hasRightTarget) {
      this.updateFootLockState(this.rightFootLock, this._ikV5);
    }

    if (hasLeftTarget && hasRightTarget) {
      const leftLift = this._ikV4.y - this.groundY;
      const rightLift = this._ikV5.y - this.groundY;
      if (leftLift > this.jumpUnlockHeight && rightLift > this.jumpUnlockHeight) {
        this.leftFootLock.locked = false;
        this.rightFootLock.locked = false;
        this.leftFootLock.lockPos.copy(this._ikV4);
        this.rightFootLock.lockPos.copy(this._ikV5);
      }
    }

    this.applySupportPelvisShift();

    if (hasLeftTarget) {
      const leftDesired = this.leftFootLock.locked ? this.leftFootLock.lockPos : this._ikV4;
      const leftFallbackPole = this._ikV6.copy(this.leftLegTargetInitialized ? this.leftLegTargetFiltered : leftDesired).add(new Vector3(0, 0, 0.25));
      const leftKneeAvailable = this.resolveKneeHintWorld(
        a,
        b,
        alpha,
        'left_knee',
        this.leftKneeHintState,
        leftFallbackPole,
        this._kneeHintL,
      );

      // When knee data drops out, keep leg target stationary to avoid drifting knees.
      if (!leftKneeAvailable && this.leftLegTargetInitialized) {
        leftDesired.copy(this.leftLegTargetFiltered);
      }

      if (!this.leftLegTargetInitialized) {
        this.leftLegTargetFiltered.copy(leftDesired);
        this.leftLegTargetInitialized = true;
      } else {
        const leftLift = leftDesired.y - this.groundY;
        const leftBlend = leftLift > (this.footPlantHeight + 0.02)
          ? this.airborneTargetSmoothing
          : this.legTargetSmoothing;
        this.leftLegTargetFiltered.lerp(leftDesired, leftBlend);
      }
      const leftPole = this._kneePoleL.copy(this._kneeHintL);
      this.solveTwoBoneIk(
        this.leftUpLegBone,
        this.leftLegBone,
        this.leftFootBone,
        this.leftLegTargetFiltered,
        leftPole,
        this._kneeHintL,
        this.leftUpperLen,
        this.leftLowerLen,
        'LeftUpLeg',
        'LeftLeg',
      );
    }

    if (hasRightTarget) {
      const rightDesired = this.rightFootLock.locked ? this.rightFootLock.lockPos : this._ikV5;
      const rightFallbackPole = this._ikV8.copy(this.rightLegTargetInitialized ? this.rightLegTargetFiltered : rightDesired).add(new Vector3(0, 0, 0.25));
      const rightKneeAvailable = this.resolveKneeHintWorld(
        a,
        b,
        alpha,
        'right_knee',
        this.rightKneeHintState,
        rightFallbackPole,
        this._kneeHintR,
      );

      if (!rightKneeAvailable && this.rightLegTargetInitialized) {
        rightDesired.copy(this.rightLegTargetFiltered);
      }

      if (!this.rightLegTargetInitialized) {
        this.rightLegTargetFiltered.copy(rightDesired);
        this.rightLegTargetInitialized = true;
      } else {
        const rightLift = rightDesired.y - this.groundY;
        const rightBlend = rightLift > (this.footPlantHeight + 0.02)
          ? this.airborneTargetSmoothing
          : this.legTargetSmoothing;
        this.rightLegTargetFiltered.lerp(rightDesired, rightBlend);
      }
      const rightPole = this._kneePoleR.copy(this._kneeHintR);
      this.solveTwoBoneIk(
        this.rightUpLegBone,
        this.rightLegBone,
        this.rightFootBone,
        this.rightLegTargetFiltered,
        rightPole,
        this._kneeHintR,
        this.rightUpperLen,
        this.rightLowerLen,
        'RightUpLeg',
        'RightLeg',
      );
    }

    this.applyFlatFootOrientation();
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

    const targetWithGain = this.amplifyLocalTarget(localTarget, bindLocal, jointName);
    const constrained = this.constrainLocalQuaternion(targetWithGain, bindLocal, axisLocal, cfg);
    const prev = this.prevLocalByBoneName.get(bone.name) ?? bindLocal.clone();

    const maxStepRad = MathUtils.degToRad(cfg.maxStepDeg);
    const angleToTarget = prev.angleTo(constrained);
    let stepLimited = constrained;
    if (angleToTarget > maxStepRad && maxStepRad > 1e-6) {
      const t = maxStepRad / Math.max(angleToTarget, 1e-6);
      stepLimited = prev.clone().slerp(constrained, t);
    }

    // Velocity-aware smoothing: keep tiny motions stable but react quickly to intentional movement.
    const motionIntent = clamp(angleToTarget / MathUtils.degToRad(22), 0, 1);
    const adaptiveSmoothing = clamp(
      cfg.smoothing + ((1 - cfg.smoothing) * motionIntent * 0.85),
      0.01,
      1.0,
    );
    const smoothed = prev.clone().slerp(stepLimited, adaptiveSmoothing);
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

    const decay2dOffset = () => {
      this.root2dOffsetFiltered.lerp(this._ikV9.set(0, 0, 0), this.root2dOffsetDecay);
      out.add(this.root2dOffsetFiltered);
      return out;
    };

    const pose2d = frame.meta && (frame.meta as any).pose_landmarks_2d;
    if (!Array.isArray(pose2d) || pose2d.length < 25) {
      return decay2dOffset();
    }

    const ls = pose2d[11];
    const rs = pose2d[12];
    const lh = pose2d[23];
    const rh = pose2d[24];
    if (![ls, rs, lh, rh].every((p) => Array.isArray(p) && p.length >= 2)) {
      return decay2dOffset();
    }

    const vis = [ls, rs, lh, rh]
      .map((p) => Number((p as any)[2] ?? 1.0))
      .filter((v) => Number.isFinite(v));
    const minVis = vis.length ? Math.min(...vis) : 1.0;
    if (minVis < this.root2dMinVisibility) {
      return decay2dOffset();
    }

    const hipX = (Number(lh[0]) + Number(rh[0])) * 0.5;
    const hipY = (Number(lh[1]) + Number(rh[1])) * 0.5;
    const shoulderW = Math.hypot(Number(ls[0]) - Number(rs[0]), Number(ls[1]) - Number(rs[1]));

    if (!Number.isFinite(hipX) || !Number.isFinite(hipY) || !Number.isFinite(shoulderW) || shoulderW < 1e-4) {
      return decay2dOffset();
    }

    if (!this.root2dInitialized) {
      this.root2dOrigin.set(hipX, hipY, 0);
      this.root2dRefShoulderWidth = shoulderW;
      this.root2dInitialized = true;
      this.root2dOffsetFiltered.set(0, 0, 0);
    }

    const stableShoulderW = Math.max(
      shoulderW,
      this.root2dRefShoulderWidth * this.root2dMinShoulderScale,
      1e-4,
    );
    const depthRatio = clamp(
      this.root2dRefShoulderWidth / stableShoulderW,
      this.root2dDepthRatioMin,
      this.root2dDepthRatioMax,
    );

    const rawDx = (hipX - this.root2dOrigin.x) * this.root2dGainX;
    const rawDy = (this.root2dOrigin.y - hipY) * this.root2dGainY;
    const rawDz = (depthRatio - 1.0) * this.root2dGainZ;
    const targetOffset = this._vu.set(
      clamp(rawDx, -this.root2dMaxLateralOffset, this.root2dMaxLateralOffset),
      clamp(rawDy, -this.root2dMaxVerticalOffset, this.root2dMaxVerticalOffset),
      clamp(rawDz, -this.root2dMaxDepthOffset, this.root2dMaxDepthOffset),
    );

    this.root2dOffsetFiltered.lerp(targetOffset, this.root2dOffsetSmoothing);
    out.add(this.root2dOffsetFiltered);
    return out;
  }

  applyInterpolated(a: PoseFrame, b: PoseFrame, alphaIn: number): void {
    const tStart = (typeof performance !== 'undefined') ? performance.now() : Date.now();
    const alpha = clamp(alphaIn, 0, 1);

    // Root pose
    const pa = this.extractRootPosition(a, this._vs).applyQuaternion(this._frameYawFix);
    const pb = this.extractRootPosition(b, this._vt).applyQuaternion(this._frameYawFix);
    const qa = this._qa.copy(this._frameYawFix).multiply(quatFromArray(a.root.rotation));
    const qb = this._qb.copy(this._frameYawFix).multiply(quatFromArray(b.root.rotation));

    const p = lerpVec3(this._vr, pa, pb, alpha);
    const q = slerpQuat(this._qr, qa, qb, alpha);

    if (!this.rootOriginInitialized) {
      this.rootSourceOrigin.copy(p);
      this.rootOriginInitialized = true;
    }

    const delta = this._vr.copy(p).sub(this.rootSourceOrigin);
    const rootPosBase = this.rootDriver ? this.rootDriverBindPos : this.rootBindPos;
    const rootRotBase = this.rootDriver ? this.rootDriverBindRot : this.rootBindRot;
    const targetY = this.keepGrounded
      ? this.groundY
      : (rootPosBase.y + delta.y * this.rootMotionGainY);
    const rootTargetPos = new Vector3(
      rootPosBase.x + delta.x * this.rootMotionGainXZ,
      targetY,
      rootPosBase.z + delta.z * this.rootMotionGainXZ,
    );
    const rootTargetRot = rootRotBase.clone().multiply(this.extractYawQuaternion(q));

    // Smooth root motion while preserving visible translation from the pose midpoint.
    if (!this._rootInitialized) {
      this._rootPosFiltered.copy(rootTargetPos);
      this._rootRotFiltered.copy(rootTargetRot);
      this._rootInitialized = true;
    } else {
      const posIntent = clamp(this._ikV10.copy(rootTargetPos).sub(this._rootPosFiltered).length() / 0.08, 0, 1);
      const rotIntent = clamp(this._rootRotFiltered.angleTo(rootTargetRot) / MathUtils.degToRad(30), 0, 1);
      const posBlend = clamp(this.rootPosSmoothing + ((1 - this.rootPosSmoothing) * posIntent * 0.7), 0.01, 1.0);
      const rotBlend = clamp(this.rootRotSmoothing + ((1 - this.rootRotSmoothing) * rotIntent * 0.7), 0.01, 1.0);
      this._rootPosFiltered.lerp(rootTargetPos, posBlend);
      this._rootRotFiltered.slerp(rootTargetRot, rotBlend);
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
      const qLf = this._qb.copy(this._frameYawFix).multiply(qL);

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
      let desiredWorld = qLf.clone();
      if (!serverIsModelSpace) {
        const sourceRest = this.sourceRestDirByJoint.get(srvName);
        const targetRest = this.modelRestDirByJoint.get(tname);

        if (sourceRest && targetRest) {
          const modelToSource = this._qb.setFromUnitVectors(targetRest, sourceRest);
          desiredWorld = desiredWorld.multiply(modelToSource);
        } else if (corr) {
          desiredWorld = desiredWorld.multiply(corr);
        }
      }

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

    if (this.enableLegIK) {
      this.applyProceduralLegIk(a, b, alpha);
    }

    // Upload to GPU
    this.skeleton.update();

    if (this.keepGrounded) {
      this.applyFootGroundingCorrection();
    }

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
