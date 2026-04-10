import numpy as np
try:
    from .posemath import (normalize, quat_identity, quat_from_two_vectors, quat_mul, quat_conjugate, swing_twist_decomposition)
except ImportError:  # Allows running this file directly from backend/.
    from posemath import (normalize, quat_identity, quat_from_two_vectors, quat_mul, quat_conjugate, swing_twist_decomposition)

# MediaPipe indices for BlazePose (subset)
LM = {
    "NOSE": 0,
    "L_SHOULDER": 11, "R_SHOULDER": 12,
    "L_ELBOW": 13, "R_ELBOW": 14,
    "L_WRIST": 15, "R_WRIST": 16,
    "L_HIP": 23, "R_HIP": 24,
    "L_KNEE": 25, "R_KNEE": 26,
    "L_ANKLE": 27, "R_ANKLE": 28,
    "L_EAR": 7, "R_EAR": 8,
}

def midpoint(a, b):
    return (a + b) * 0.5

def bone_dir(a, b):
    v = b - a
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.array([0,1,0], dtype=np.float64)

def estimate_scale(lmk):
    # Use shoulder width and hip width as proxies
    ls, rs = lmk[LM["L_SHOULDER"]], lmk[LM["R_SHOULDER"]]
    lh, rh = lmk[LM["L_HIP"]], lmk[LM["R_HIP"]]
    shoulder = np.linalg.norm(ls - rs)
    hip = np.linalg.norm(lh - rh)
    return max(shoulder, hip)

def compute_root_transform(lmk, rest_up=np.array([0,1,0.0]), rest_fwd=np.array([0,0,1.0])):
    lh, rh = lmk[LM["L_HIP"]], lmk[LM["R_HIP"]]
    ls, rs = lmk[LM["L_SHOULDER"]], lmk[LM["R_SHOULDER"]]
    pelvis = midpoint(lh, rh)
    shoulders = midpoint(ls, rs)

    up = normalize(shoulders - pelvis)
    right = normalize(rh - lh)
    fwd = normalize(np.cross(right, up))  # ensure orthonormal

    # Map rest axes to observed axes. We align rest_up -> up, rest_fwd -> fwd.
    q_up = quat_from_two_vectors(rest_up, up)
    # Constrain yaw using forward vector to reduce roll drift
    fwd_est = fwd
    fwd_rest_rot = quat_from_two_vectors(normalize([0,0,1.0]), fwd_est)
    # Compose
    q_root = fwd_rest_rot  # This simple strategy is practical; refine with swing-twist if needed.
    pos = pelvis
    return pos, q_root

def joint_from_bone(rest_dir, curr_dir, parent_correction=None):
    q = quat_from_two_vectors(rest_dir, curr_dir)
    if parent_correction is not None:
        q = quat_mul(parent_correction, q)
    return q

def retarget_landmarks_to_mixamo(lmk3d, rest_pose_dirs):
    """
    lmk3d: np.ndarray [N,3] MediaPipe 3D landmarks in consistent world coordinates.
    rest_pose_dirs: dict of bone default directions in model space (Mixamo T-pose).
    Returns: dict { joint_name: quaternion (w,x,y,z) }, root position/rotation, and meta.
    """
    out = {}
    scale = estimate_scale(lmk3d)
    root_pos, root_rot = compute_root_transform(lmk3d)

    # Key joints: compute current bone directions
    hip_L, hip_R = lmk3d[LM["L_HIP"]], lmk3d[LM["R_HIP"]]
    knee_L, knee_R = lmk3d[LM["L_KNEE"]], lmk3d[LM["R_KNEE"]]
    ankle_L, ankle_R = lmk3d[LM["L_ANKLE"]], lmk3d[LM["R_ANKLE"]]
    sh_L, sh_R = lmk3d[LM["L_SHOULDER"]], lmk3d[LM["R_SHOULDER"]]
    el_L, el_R = lmk3d[LM["L_ELBOW"]], lmk3d[LM["R_ELBOW"]]
    wr_L, wr_R = lmk3d[LM["L_WRIST"]], lmk3d[LM["R_WRIST"]]

    chest = midpoint(sh_L, sh_R)
    pelvis = midpoint(hip_L, hip_R)

    # Spine
    spine_curr = bone_dir(pelvis, chest)
    out["Spine"] = joint_from_bone(rest_pose_dirs["Spine"], spine_curr)

    # Left leg chain
    L_up_curr = bone_dir(hip_L, knee_L)
    L_lo_curr = bone_dir(knee_L, ankle_L)
    out["LeftUpLeg"] = joint_from_bone(rest_pose_dirs["LeftUpLeg"], L_up_curr)
    out["LeftLeg"]   = joint_from_bone(rest_pose_dirs["LeftLeg"], L_lo_curr)

    # Right leg chain
    R_up_curr = bone_dir(hip_R, knee_R)
    R_lo_curr = bone_dir(knee_R, ankle_R)
    out["RightUpLeg"] = joint_from_bone(rest_pose_dirs["RightUpLeg"], R_up_curr)
    out["RightLeg"]   = joint_from_bone(rest_pose_dirs["RightLeg"], R_lo_curr)

    # Shoulders/arms
    spine_top = chest
    L_sh_curr = bone_dir(spine_top, sh_L)
    R_sh_curr = bone_dir(spine_top, sh_R)
    out["LeftShoulder"]  = joint_from_bone(rest_pose_dirs["LeftShoulder"], L_sh_curr)
    out["RightShoulder"] = joint_from_bone(rest_pose_dirs["RightShoulder"], R_sh_curr)

    L_arm_curr = bone_dir(sh_L, el_L)
    L_fore_curr= bone_dir(el_L, wr_L)
    out["LeftArm"]    = joint_from_bone(rest_pose_dirs["LeftArm"], L_arm_curr)
    out["LeftForeArm"]= joint_from_bone(rest_pose_dirs["LeftForeArm"], L_fore_curr)

    R_arm_curr = bone_dir(sh_R, el_R)
    R_fore_curr= bone_dir(el_R, wr_R)
    out["RightArm"]    = joint_from_bone(rest_pose_dirs["RightArm"], R_arm_curr)
    out["RightForeArm"]= joint_from_bone(rest_pose_dirs["RightForeArm"], R_fore_curr)

    # Return joint quaternions in the server's canonical internal frame
    # (not converted to any target skeleton). The client (frontend)
    # will be responsible for retargeting from this canonical frame to
    # the Mixamo skeleton. We include a meta field to advertise the
    # server's model space so the client can decide whether to apply
    # additional corrections.
    meta = {"scale": float(scale), "model_space": "canonical"}
    return out, root_pos, root_rot, meta
