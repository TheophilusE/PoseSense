import time
import cv2
import numpy as np
import mediapipe as mp
from posemath import OneEuro
from canonical import landmarks_to_canonical

# Define resolution tiers
RESOLUTIONS = [
    (7680, 4320),  # 8K
    (3840, 2160),  # 4K
    (1920, 1080),  # Full HD
    (1280, 720),   # HD
]

def test_camera_resolution(index, width, height, warmup_frames=10):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None

    # Request resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Warm-up
    for _ in range(warmup_frames):
        cap.read()
        time.sleep(0.05)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return None

    actual_height, actual_width = frame.shape[:2]
    if actual_width >= width and actual_height >= height:
        return actual_width * actual_height  # Score by resolution
    return None

def select_best_camera(max_devices=5):
    best_score      = -1
    best_index      = None
    best_resolution = None

    for i in range(max_devices):
        for res in RESOLUTIONS:
            score = test_camera_resolution(i, *res)
            if score:
                print(f"Camera {i} supports {res[0]}x{res[1]} -> score: {score}")
                if score > best_score:
                    best_score      = score
                    best_index      = i
                    best_resolution = res
                break  # Stop testing lower resolutions once one works

    if best_index is not None:
        print(f"Best camera: index {best_index}, resolution {best_resolution}")
        cap = cv2.VideoCapture(best_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_resolution[1])
        return cap
    else:
        print("No suitable camera found.")
        return None

class PoseProcessor:
    def __init__(self, source: cv2.VideoCapture = None, fps: float =30.0):
        self.cap = source
        self.fps = fps
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Filters for stability
        self.filter_pos = OneEuro(freq=fps, min_cutoff=1.5, beta=0.03, dcutoff=1.0)

        # Define rest pose bone directions in model space (Mixamo T-pose)
        # These depend on your model's axis setup; adjust as needed.
        self.rest_dirs = {
            "Spine":        np.array([0, 1, 0]),
            "LeftUpLeg":    np.array([0, -1, 0]),
            "LeftLeg":      np.array([0, -1, 0]),
            "RightUpLeg":   np.array([0, -1, 0]),
            "RightLeg":     np.array([0, -1, 0]),
            "LeftShoulder": np.array([ -1, 0, 0]),
            "RightShoulder":np.array([  1, 0, 0]),
            "LeftArm":      np.array([ -1, 0, 0]),
            "LeftForeArm":  np.array([ -1, 0, 0]),
            "RightArm":     np.array([  1, 0, 0]),
            "RightForeArm": np.array([  1, 0, 0]),
        }

    def _mp_landmarks_to_world(self, results, w, h):
        if not results.pose_world_landmarks:
            return None
        pts = []
        for lm in results.pose_world_landmarks.landmark:
            # pose_world_landmarks already in meters-ish world space
            pts.append([lm.x, lm.y, lm.z])
        return np.array(pts, dtype=np.float64)

    def read_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        lmk_world = self._mp_landmarks_to_world(res, w, h)
        return lmk_world, frame

    def process(self, frame_idx):
        lmk, _ = self.read_frame()
        if lmk is None:
            return None
        # Filter all 3D points collectively for smooth motion
        flat = lmk.flatten()
        flat_s = self.filter_pos.filter(flat)
        lmk_s = flat_s.reshape(lmk.shape)

        joints, root_pos, root_rot, meta = landmarks_to_canonical(lmk_s)
        # Compose payload
        payload = {
            "type": "poseFrame",
            "timestamp": int(time.time() * 1000),
            "skeleton": "Canonical",
            "frame": int(frame_idx),
            "fps": float(self.fps),
            "root": {
                "position": root_pos.tolist(),
                "rotation": root_rot.tolist()
            },
            "joints": [{"name": k, "rotation": v.tolist()} for k, v in joints.items()],
            "meta": meta
        }
        return payload
