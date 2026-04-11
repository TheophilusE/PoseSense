"""Canonical skeleton bridge with intermediate IK solving.

This module converts MediaPipe world landmarks into a stable canonical
humanoid pose for downstream retargeting (for example to Mixamo).

Pipeline:
1) Build intermediate humanoid targets (pelvis/chest/neck/head + limbs).
2) Solve 2-bone IK for arms/legs using fixed chain lengths per frame.
3) Convert solved segment directions into canonical joint quaternions.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np

try:
    from .posemath import (
        normalize,
        quat_identity,
        quat_from_two_vectors,
        quat_mul,
    )
except ImportError:  # Allows running this file directly from backend/.
    from posemath import (
        normalize,
        quat_identity,
        quat_from_two_vectors,
        quat_mul,
    )


CANONICAL_JOINTS: List[str] = [
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
]


CANONICAL_REST_DIRS: Dict[str, np.ndarray] = {
    "Spine": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "Chest": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "Neck": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "Head": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "LeftUpLeg": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "LeftLeg": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "LeftFoot": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "RightUpLeg": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "RightLeg": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "RightFoot": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "LeftShoulder": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "LeftArm": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "LeftForeArm": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "LeftHand": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "RightShoulder": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "RightArm": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "RightForeArm": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "RightHand": np.array([1.0, 0.0, 0.0], dtype=np.float64),
}


LM = {
    "NOSE": 0,
    "L_EYE": 2,
    "R_EYE": 5,
    "L_EAR": 7,
    "R_EAR": 8,
    "L_SHOULDER": 11,
    "R_SHOULDER": 12,
    "L_ELBOW": 13,
    "R_ELBOW": 14,
    "L_WRIST": 15,
    "R_WRIST": 16,
    "L_PINKY": 17,
    "R_PINKY": 18,
    "L_INDEX": 19,
    "R_INDEX": 20,
    "L_THUMB": 21,
    "R_THUMB": 22,
    "L_HIP": 23,
    "R_HIP": 24,
    "L_KNEE": 25,
    "R_KNEE": 26,
    "L_ANKLE": 27,
    "R_ANKLE": 28,
    "L_HEEL": 29,
    "R_HEEL": 30,
    "L_FOOT_INDEX": 31,
    "R_FOOT_INDEX": 32,
}


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) * 0.5


def bone_dir(a: np.ndarray, b: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    v = b - a
    n = np.linalg.norm(v)
    if n < 1e-8:
        if fallback is None:
            return np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return normalize(fallback)
    return v / n


def _vec3(x: float, y: float, z: float) -> np.ndarray:
    return np.array([x, y, z], dtype=np.float64)


def _safe_mean(points: List[np.ndarray], default: np.ndarray) -> np.ndarray:
    if not points:
        return default.copy()
    return np.mean(np.stack(points, axis=0), axis=0)


def _safe_len(a: np.ndarray, b: np.ndarray, fallback: float) -> float:
    n = float(np.linalg.norm(b - a))
    if not np.isfinite(n) or n < 1e-6:
        return float(max(fallback, 1e-3))
    return n


def _project_to_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return v - normal * float(np.dot(v, normal))


def _pole_hint(root: np.ndarray, mid: np.ndarray, end: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    d = bone_dir(root, end, fallback=fallback)
    bend = _project_to_plane(mid - root, d)
    if np.linalg.norm(bend) < 1e-6:
        bend = _project_to_plane(fallback, d)
    if np.linalg.norm(bend) < 1e-6:
        bend = _project_to_plane(_vec3(0.0, 0.0, 1.0), d)
    return normalize(bend)


def _solve_two_bone_chain(
    root: np.ndarray,
    target: np.ndarray,
    pole: np.ndarray,
    len_upper: float,
    len_lower: float,
) -> Tuple[np.ndarray, np.ndarray]:
    to_target = target - root
    dist = float(np.linalg.norm(to_target))
    if dist < 1e-8:
        dist = 1e-8
        to_target = _vec3(0.0, -1.0, 0.0)
    dir_rt = to_target / dist

    min_d = abs(len_upper - len_lower) + 1e-5
    max_d = max(len_upper + len_lower - 1e-5, min_d + 1e-5)
    d = float(np.clip(dist, min_d, max_d))

    pole_plane = _project_to_plane(pole, dir_rt)
    if np.linalg.norm(pole_plane) < 1e-6:
        pole_plane = _project_to_plane(_vec3(0.0, 0.0, 1.0), dir_rt)
    if np.linalg.norm(pole_plane) < 1e-6:
        pole_plane = _project_to_plane(_vec3(1.0, 0.0, 0.0), dir_rt)
    bend_dir = normalize(pole_plane)

    x = (len_upper * len_upper - len_lower * len_lower + d * d) / (2.0 * d)
    y_sq = max(len_upper * len_upper - x * x, 0.0)
    y = np.sqrt(y_sq)

    mid = root + dir_rt * x + bend_dir * y
    end = root + dir_rt * d
    return mid, end


def estimate_scale(lmk: np.ndarray) -> float:
    """Estimate person scale using robust shoulder/hip width."""
    try:
        ls = lmk[LM["L_SHOULDER"]]
        rs = lmk[LM["R_SHOULDER"]]
        lh = lmk[LM["L_HIP"]]
        rh = lmk[LM["R_HIP"]]
        shoulder = np.linalg.norm(ls - rs)
        hip = np.linalg.norm(lh - rh)
        return float(max(shoulder, hip, 0.01))
    except Exception:
        return 1.0


def _build_intermediate_targets(lmk: np.ndarray) -> Dict[str, np.ndarray]:
    ls = lmk[LM["L_SHOULDER"]]
    rs = lmk[LM["R_SHOULDER"]]
    lh = lmk[LM["L_HIP"]]
    rh = lmk[LM["R_HIP"]]
    le = lmk[LM["L_ELBOW"]]
    re = lmk[LM["R_ELBOW"]]
    lw = lmk[LM["L_WRIST"]]
    rw = lmk[LM["R_WRIST"]]
    lk = lmk[LM["L_KNEE"]]
    rk = lmk[LM["R_KNEE"]]
    la = lmk[LM["L_ANKLE"]]
    ra = lmk[LM["R_ANKLE"]]
    ltoe = lmk[LM["L_FOOT_INDEX"]]
    rtoe = lmk[LM["R_FOOT_INDEX"]]
    lheel = lmk[LM["L_HEEL"]]
    rheel = lmk[LM["R_HEEL"]]
    nose = lmk[LM["NOSE"]]
    leye = lmk[LM["L_EYE"]]
    reye = lmk[LM["R_EYE"]]
    lear = lmk[LM["L_EAR"]]
    rear = lmk[LM["R_EAR"]]

    lhand = _safe_mean(
        [lmk[LM["L_INDEX"]], lmk[LM["L_PINKY"]], lmk[LM["L_THUMB"]]],
        default=lw,
    )
    rhand = _safe_mean(
        [lmk[LM["R_INDEX"]], lmk[LM["R_PINKY"]], lmk[LM["R_THUMB"]]],
        default=rw,
    )

    pelvis = midpoint(lh, rh)
    chest = midpoint(ls, rs)
    head = _safe_mean([nose, leye, reye, lear, rear], default=nose)
    neck = midpoint(chest, head)

    return {
        "pelvis": pelvis,
        "chest": chest,
        "neck": neck,
        "head": head,
        "left_shoulder": ls,
        "right_shoulder": rs,
        "left_elbow": le,
        "right_elbow": re,
        "left_wrist": lw,
        "right_wrist": rw,
        "left_hand": lhand,
        "right_hand": rhand,
        "left_hip": lh,
        "right_hip": rh,
        "left_knee": lk,
        "right_knee": rk,
        "left_ankle": la,
        "right_ankle": ra,
        "left_heel": lheel,
        "right_heel": rheel,
        "left_foot_index": ltoe,
        "right_foot_index": rtoe,
        "left_foot": midpoint(ltoe, lheel),
        "right_foot": midpoint(rtoe, rheel),
    }


def _solve_intermediate_targets(raw: Dict[str, np.ndarray], scale: float) -> Dict[str, np.ndarray]:
    solved = {k: v.copy() for k, v in raw.items()}

    pelvis = raw["pelvis"]
    chest_raw = raw["chest"]
    neck_raw = raw["neck"]
    head_raw = raw["head"]

    spine_len = max(_safe_len(pelvis, chest_raw, 1.2 * scale), 0.25 * scale)
    neck_len = max(_safe_len(chest_raw, neck_raw, 0.25 * scale), 0.08 * scale)
    head_len = max(_safe_len(neck_raw, head_raw, 0.32 * scale), 0.12 * scale)

    spine_dir = bone_dir(pelvis, chest_raw, fallback=_vec3(0.0, 1.0, 0.0))
    chest = pelvis + spine_dir * spine_len
    chest_shift = chest - chest_raw

    solved["pelvis"] = pelvis.copy()
    solved["chest"] = chest
    solved["left_shoulder"] = raw["left_shoulder"] + chest_shift
    solved["right_shoulder"] = raw["right_shoulder"] + chest_shift

    neck_dir = bone_dir(chest, neck_raw + chest_shift, fallback=spine_dir)
    neck = chest + neck_dir * neck_len
    head_dir = bone_dir(neck, head_raw + chest_shift, fallback=neck_dir)
    head = neck + head_dir * head_len
    solved["neck"] = neck
    solved["head"] = head

    upper_arm = max(
        0.5 * (
            _safe_len(raw["left_shoulder"], raw["left_elbow"], 0.75 * scale)
            + _safe_len(raw["right_shoulder"], raw["right_elbow"], 0.75 * scale)
        ),
        0.16 * scale,
    )
    lower_arm = max(
        0.5 * (
            _safe_len(raw["left_elbow"], raw["left_wrist"], 0.7 * scale)
            + _safe_len(raw["right_elbow"], raw["right_wrist"], 0.7 * scale)
        ),
        0.16 * scale,
    )
    upper_leg = max(
        0.5 * (
            _safe_len(raw["left_hip"], raw["left_knee"], 0.95 * scale)
            + _safe_len(raw["right_hip"], raw["right_knee"], 0.95 * scale)
        ),
        0.2 * scale,
    )
    lower_leg = max(
        0.5 * (
            _safe_len(raw["left_knee"], raw["left_ankle"], 0.9 * scale)
            + _safe_len(raw["right_knee"], raw["right_ankle"], 0.9 * scale)
        ),
        0.2 * scale,
    )

    l_sh = solved["left_shoulder"]
    r_sh = solved["right_shoulder"]

    left_arm_pole = _pole_hint(
        l_sh,
        raw["left_elbow"],
        raw["left_wrist"],
        fallback=_vec3(-1.0, 0.0, 0.0),
    )
    right_arm_pole = _pole_hint(
        r_sh,
        raw["right_elbow"],
        raw["right_wrist"],
        fallback=_vec3(1.0, 0.0, 0.0),
    )

    l_elbow, l_wrist = _solve_two_bone_chain(
        l_sh,
        raw["left_wrist"] + chest_shift,
        left_arm_pole,
        upper_arm,
        lower_arm,
    )
    r_elbow, r_wrist = _solve_two_bone_chain(
        r_sh,
        raw["right_wrist"] + chest_shift,
        right_arm_pole,
        upper_arm,
        lower_arm,
    )
    solved["left_elbow"] = l_elbow
    solved["left_wrist"] = l_wrist
    solved["right_elbow"] = r_elbow
    solved["right_wrist"] = r_wrist

    left_leg_pole = _pole_hint(
        raw["left_hip"],
        raw["left_knee"],
        raw["left_ankle"],
        fallback=_vec3(0.0, 0.0, 1.0),
    )
    right_leg_pole = _pole_hint(
        raw["right_hip"],
        raw["right_knee"],
        raw["right_ankle"],
        fallback=_vec3(0.0, 0.0, 1.0),
    )

    l_knee, l_ankle = _solve_two_bone_chain(
        raw["left_hip"],
        raw["left_ankle"],
        left_leg_pole,
        upper_leg,
        lower_leg,
    )
    r_knee, r_ankle = _solve_two_bone_chain(
        raw["right_hip"],
        raw["right_ankle"],
        right_leg_pole,
        upper_leg,
        lower_leg,
    )
    solved["left_knee"] = l_knee
    solved["left_ankle"] = l_ankle
    solved["right_knee"] = r_knee
    solved["right_ankle"] = r_ankle

    # Keep distal effectors from tracking landmarks so feet and hands preserve
    # rough orientation cues while the chain lengths remain stable.
    solved["left_hand"] = raw["left_hand"] + chest_shift
    solved["right_hand"] = raw["right_hand"] + chest_shift

    return solved


def compute_root_transform(targets: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    pelvis = targets["pelvis"]
    chest = targets["chest"]
    lh = targets["left_hip"]
    rh = targets["right_hip"]
    ls = targets["left_shoulder"]
    rs = targets["right_shoulder"]

    up = normalize(chest - pelvis)
    right = normalize((rh - lh) + (rs - ls) * 0.5)
    fwd = normalize(np.cross(right, up))
    if np.linalg.norm(fwd) < 1e-6:
        fwd = _vec3(0.0, 0.0, 1.0)

    q_up = quat_from_two_vectors(_vec3(0.0, 1.0, 0.0), up)
    q_fwd = quat_from_two_vectors(_vec3(0.0, 0.0, 1.0), fwd)
    root_q = quat_mul(q_fwd, q_up)
    return pelvis, root_q


def _joints_from_targets(t: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    out["Spine"] = quat_from_two_vectors(CANONICAL_REST_DIRS["Spine"], bone_dir(t["pelvis"], t["chest"]))
    out["Chest"] = quat_from_two_vectors(CANONICAL_REST_DIRS["Chest"], bone_dir(t["chest"], t["neck"]))
    out["Neck"] = quat_from_two_vectors(CANONICAL_REST_DIRS["Neck"], bone_dir(t["neck"], t["head"]))
    out["Head"] = quat_from_two_vectors(CANONICAL_REST_DIRS["Head"], bone_dir(t["neck"], t["head"]))

    out["LeftUpLeg"] = quat_from_two_vectors(CANONICAL_REST_DIRS["LeftUpLeg"], bone_dir(t["left_hip"], t["left_knee"]))
    out["LeftLeg"] = quat_from_two_vectors(CANONICAL_REST_DIRS["LeftLeg"], bone_dir(t["left_knee"], t["left_ankle"]))
    out["LeftFoot"] = quat_from_two_vectors(CANONICAL_REST_DIRS["LeftFoot"], bone_dir(t["left_ankle"], t["left_foot_index"]))

    out["RightUpLeg"] = quat_from_two_vectors(CANONICAL_REST_DIRS["RightUpLeg"], bone_dir(t["right_hip"], t["right_knee"]))
    out["RightLeg"] = quat_from_two_vectors(CANONICAL_REST_DIRS["RightLeg"], bone_dir(t["right_knee"], t["right_ankle"]))
    out["RightFoot"] = quat_from_two_vectors(CANONICAL_REST_DIRS["RightFoot"], bone_dir(t["right_ankle"], t["right_foot_index"]))

    out["LeftShoulder"] = quat_from_two_vectors(CANONICAL_REST_DIRS["LeftShoulder"], bone_dir(t["chest"], t["left_shoulder"]))
    out["LeftArm"] = quat_from_two_vectors(CANONICAL_REST_DIRS["LeftArm"], bone_dir(t["left_shoulder"], t["left_elbow"]))
    out["LeftForeArm"] = quat_from_two_vectors(CANONICAL_REST_DIRS["LeftForeArm"], bone_dir(t["left_elbow"], t["left_wrist"]))
    out["LeftHand"] = quat_from_two_vectors(CANONICAL_REST_DIRS["LeftHand"], bone_dir(t["left_wrist"], t["left_hand"]))

    out["RightShoulder"] = quat_from_two_vectors(CANONICAL_REST_DIRS["RightShoulder"], bone_dir(t["chest"], t["right_shoulder"]))
    out["RightArm"] = quat_from_two_vectors(CANONICAL_REST_DIRS["RightArm"], bone_dir(t["right_shoulder"], t["right_elbow"]))
    out["RightForeArm"] = quat_from_two_vectors(CANONICAL_REST_DIRS["RightForeArm"], bone_dir(t["right_elbow"], t["right_wrist"]))
    out["RightHand"] = quat_from_two_vectors(CANONICAL_REST_DIRS["RightHand"], bone_dir(t["right_wrist"], t["right_hand"]))

    return out


def _v_to_list(v: np.ndarray) -> List[float]:
    return [float(v[0]), float(v[1]), float(v[2])]


def _meta_from_targets(scale: float, raw: Dict[str, np.ndarray], solved: Dict[str, np.ndarray], joints: Dict[str, np.ndarray]) -> Dict:
    pelvis = solved["pelvis"]
    inv_scale = 1.0 / max(float(scale), 1e-6)

    segment_dirs = {
        "spine": _v_to_list(bone_dir(solved["pelvis"], solved["chest"])),
        "chest": _v_to_list(bone_dir(solved["chest"], solved["neck"])),
        "neck": _v_to_list(bone_dir(solved["neck"], solved["head"])),
        "left_upper_arm": _v_to_list(bone_dir(solved["left_shoulder"], solved["left_elbow"])),
        "left_lower_arm": _v_to_list(bone_dir(solved["left_elbow"], solved["left_wrist"])),
        "left_hand": _v_to_list(bone_dir(solved["left_wrist"], solved["left_hand"])),
        "right_upper_arm": _v_to_list(bone_dir(solved["right_shoulder"], solved["right_elbow"])),
        "right_lower_arm": _v_to_list(bone_dir(solved["right_elbow"], solved["right_wrist"])),
        "right_hand": _v_to_list(bone_dir(solved["right_wrist"], solved["right_hand"])),
        "left_upper_leg": _v_to_list(bone_dir(solved["left_hip"], solved["left_knee"])),
        "left_lower_leg": _v_to_list(bone_dir(solved["left_knee"], solved["left_ankle"])),
        "left_foot": _v_to_list(bone_dir(solved["left_ankle"], solved["left_foot_index"])),
        "right_upper_leg": _v_to_list(bone_dir(solved["right_hip"], solved["right_knee"])),
        "right_lower_leg": _v_to_list(bone_dir(solved["right_knee"], solved["right_ankle"])),
        "right_foot": _v_to_list(bone_dir(solved["right_ankle"], solved["right_foot_index"])),
    }

    return {
        "scale": float(scale),
        "joint_names": list(joints.keys()),
        "model_space": "canonical",
        "landmark_schema": "mediapipe_pose_33",
        "intermediate_targets": {k: _v_to_list(v) for k, v in solved.items()},
        "intermediate_targets_raw": {k: _v_to_list(v) for k, v in raw.items()},
        "intermediate_targets_normalized": {
            k: _v_to_list((v - pelvis) * inv_scale)
            for k, v in solved.items()
        },
        "segment_directions": segment_dirs,
    }


def landmarks_to_canonical(lmk3d: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict]:
    """Convert MediaPipe world landmarks into canonical pose data."""
    if lmk3d is None or getattr(lmk3d, "shape", (0,))[0] < 33:
        return {}, np.array([0.0, 0.0, 0.0]), quat_identity(), {"scale": 1.0, "model_space": "canonical"}

    scale = estimate_scale(lmk3d)
    raw_targets = _build_intermediate_targets(lmk3d)
    solved_targets = _solve_intermediate_targets(raw_targets, scale)

    root_pos, root_rot = compute_root_transform(solved_targets)
    joints = _joints_from_targets(solved_targets)

    joints_out: Dict[str, np.ndarray] = {}
    for name, quat in joints.items():
        n = np.linalg.norm(quat)
        if n < 1e-8:
            joints_out[name] = quat_identity()
        else:
            joints_out[name] = quat / n

    meta = _meta_from_targets(scale, raw_targets, solved_targets, joints_out)
    return joints_out, root_pos, root_rot, meta
