"""
Canonical skeleton conversion module

This module implements a clear, well-documented pipeline that converts
MediaPipe 3D landmarks into a canonical server-side skeleton representation.
The canonical representation is a stable intermediate format that the
frontend can retarget into any target rig (Mixamo, custom rigs, etc.).

Design goals:
- Explicit canonical joint list and rest directions.
- Deterministic, testable math: position, orientation quaternions (w,x,y,z).
- Clear metadata in returned payload (joint names, model_space, scale).
- Efficient (vectorized numpy operations) and readable.

Functions provided:
- landmarks_to_canonical(lmk3d) -> (joints_dict, root_pos, root_rot, meta)

Notes on coordinate frames:
- MediaPipe provides `pose_world_landmarks` roughly in meters in a
  camera/world-aligned coordinate system. We treat that as the observed
  world frame `W`.
- The canonical skeleton uses a simple rest orientation (T-pose like)
  expressed as a set of rest bone directions. All joint quaternions are
  returned relative to the canonical model root (root-relative frame).

"""

from typing import Dict, Tuple, List
import numpy as np
try:
    from .posemath import (
        normalize, quat_identity, quat_from_two_vectors, quat_mul,
    )
except ImportError:  # Allows running this file directly from backend/.
    from posemath import (
        normalize, quat_identity, quat_from_two_vectors, quat_mul,
    )

# --- Canonical joint list ----------------------------------------------------
CANONICAL_JOINTS: List[str] = [
    "Spine",
    "LeftUpLeg", "LeftLeg",
    "RightUpLeg", "RightLeg",
    "LeftShoulder", "LeftArm", "LeftForeArm",
    "RightShoulder", "RightArm", "RightForeArm",
]

# Rest directions used by the canonical skeleton. These are expressed in
# canonical model space and should be considered the server's internal
# standard. They are intentionally simple (axis-aligned) so downstream
# retargeters can derive per-model corrections.
CANONICAL_REST_DIRS: Dict[str, np.ndarray] = {
    "Spine":        np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "LeftUpLeg":    np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "LeftLeg":      np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "RightUpLeg":   np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "RightLeg":     np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "LeftShoulder": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "RightShoulder":np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "LeftArm":      np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "LeftForeArm":  np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "RightArm":     np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "RightForeArm": np.array([1.0, 0.0, 0.0], dtype=np.float64),
}

# MediaPipe landmark indices used by the backend. BlazePose-lite subset.
LM = {
    "NOSE": 0,
    "L_SHOULDER": 11, "R_SHOULDER": 12,
    "L_ELBOW": 13, "R_ELBOW": 14,
    "L_WRIST": 15, "R_WRIST": 16,
    "L_HIP": 23, "R_HIP": 24,
    "L_KNEE": 25, "R_KNEE": 26,
    "L_ANKLE": 27, "R_ANKLE": 28,
}


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) * 0.5


def bone_dir(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    v = b - a
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return v / n


def estimate_scale(lmk: np.ndarray) -> float:
    """Estimate human scale from shoulder/hip widths (robust to missing data)."""
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


def compute_root_transform(lmk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute canonical root position and a coarse root rotation quaternion.

    Returns (pos, rot) where rot is a quaternion (w,x,y,z) mapping canonical
    forward/up into observed axes. This is intentionally simple: it aligns
    canonical up (0,1,0) to the observed shoulder->pelvis up and canonical
    forward (0,0,1) to the cross(right, up).
    """
    lh = lmk[LM["L_HIP"]]
    rh = lmk[LM["R_HIP"]]
    ls = lmk[LM["L_SHOULDER"]]
    rs = lmk[LM["R_SHOULDER"]]

    pelvis = midpoint(lh, rh)
    shoulders = midpoint(ls, rs)

    up = normalize(shoulders - pelvis)
    right = normalize(rh - lh)
    fwd = normalize(np.cross(right, up))

    # Construct root rotation from canonical axes to observed axes.
    q_up = quat_from_two_vectors(np.array([0.0, 1.0, 0.0]), up)
    q_fwd = quat_from_two_vectors(np.array([0.0, 0.0, 1.0]), fwd)

    # Compose: apply up alignment then forward alignment. Order matters; this
    # simple composition suffices for a stable root orientation.
    root_q = quat_mul(q_fwd, q_up)
    return pelvis, root_q


def joints_from_landmarks(lmk3d: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute canonical joint quaternions from landmarks.

    This function computes a quaternion for each canonical joint that
    rotates the canonical rest direction into the currently observed
    bone direction (in world/observed coordinates). The returned quaternions
    are expressed in the observed world frame. The frontend will convert
    those into model-local rotations during retargeting.
    """
    out: Dict[str, np.ndarray] = {}

    chest = midpoint(lmk3d[LM["L_SHOULDER"]], lmk3d[LM["R_SHOULDER"]])
    pelvis = midpoint(lmk3d[LM["L_HIP"]], lmk3d[LM["R_HIP"]])

    # Spine: pelvis -> chest
    spine_dir = bone_dir(pelvis, chest)
    out["Spine"] = quat_from_two_vectors(CANONICAL_REST_DIRS["Spine"], spine_dir)

    # Left leg
    L_up = bone_dir(lmk3d[LM["L_HIP"]], lmk3d[LM["L_KNEE"]])
    L_lo = bone_dir(lmk3d[LM["L_KNEE"]], lmk3d[LM["L_ANKLE"]])
    out["LeftUpLeg"] = quat_from_two_vectors(CANONICAL_REST_DIRS["LeftUpLeg"], L_up)
    out["LeftLeg"] = quat_from_two_vectors(CANONICAL_REST_DIRS["LeftLeg"], L_lo)

    # Right leg
    R_up = bone_dir(lmk3d[LM["R_HIP"]], lmk3d[LM["R_KNEE"]])
    R_lo = bone_dir(lmk3d[LM["R_KNEE"]], lmk3d[LM["R_ANKLE"]])
    out["RightUpLeg"] = quat_from_two_vectors(CANONICAL_REST_DIRS["RightUpLeg"], R_up)
    out["RightLeg"] = quat_from_two_vectors(CANONICAL_REST_DIRS["RightLeg"], R_lo)

    # Shoulders/arms
    spine_top = chest
    L_sh = bone_dir(spine_top, lmk3d[LM["L_SHOULDER"]])
    R_sh = bone_dir(spine_top, lmk3d[LM["R_SHOULDER"]])
    out["LeftShoulder"] = quat_from_two_vectors(CANONICAL_REST_DIRS["LeftShoulder"], L_sh)
    out["RightShoulder"] = quat_from_two_vectors(CANONICAL_REST_DIRS["RightShoulder"], R_sh)

    L_arm = bone_dir(lmk3d[LM["L_SHOULDER"]], lmk3d[LM["L_ELBOW"]])
    L_fore = bone_dir(lmk3d[LM["L_ELBOW"]], lmk3d[LM["L_WRIST"]])
    out["LeftArm"] = quat_from_two_vectors(CANONICAL_REST_DIRS["LeftArm"], L_arm)
    out["LeftForeArm"] = quat_from_two_vectors(CANONICAL_REST_DIRS["LeftForeArm"], L_fore)

    R_arm = bone_dir(lmk3d[LM["R_SHOULDER"]], lmk3d[LM["R_ELBOW"]])
    R_fore = bone_dir(lmk3d[LM["R_ELBOW"]], lmk3d[LM["R_WRIST"]])
    out["RightArm"] = quat_from_two_vectors(CANONICAL_REST_DIRS["RightArm"], R_arm)
    out["RightForeArm"] = quat_from_two_vectors(CANONICAL_REST_DIRS["RightForeArm"], R_fore)

    return out


def landmarks_to_canonical(lmk3d: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict]:
    """Top-level conversion: landmarks -> canonical pose payload.

    Returns:
      joints: dict { name: quaternion (w,x,y,z) }
      root_pos: np.ndarray (3)
      root_rot: np.ndarray quaternion (w,x,y,z)
      meta: dict with scale, joint_names, and model_space='canonical'
    """
    if lmk3d is None:
        return {}, np.array([0.0, 0.0, 0.0]), quat_identity(), {"scale": 1.0, "model_space": "canonical"}

    scale = estimate_scale(lmk3d)
    root_pos, root_rot = compute_root_transform(lmk3d)

    joints = joints_from_landmarks(lmk3d)

    # Normalize output quaternions and convert to list format for JSON
    joints_out: Dict[str, np.ndarray] = {}
    for k, q in joints.items():
        qn = q / np.linalg.norm(q)
        joints_out[k] = qn

    meta = {
        "scale": float(scale),
        "joint_names": list(joints_out.keys()),
        "model_space": "canonical",
    }
    return joints_out, root_pos, root_rot, meta
