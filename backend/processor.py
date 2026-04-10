import os
import time
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
try:
    from .posemath import OneEuro
    from .canonical import landmarks_to_canonical
except ImportError:  # Allows running this file directly from backend/.
    from posemath import OneEuro
    from canonical import landmarks_to_canonical

# Define resolution tiers
RESOLUTIONS = [
    (7680, 4320),  # 8K
    (3840, 2160),  # 4K
    (1920, 1080),  # Full HD
    (1280, 720),   # HD
]


DEFAULT_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
DEFAULT_POSE_MODEL_PATH = Path(__file__).resolve().parent / "models" / "pose_landmarker_full.task"
_MODEL_DOWNLOAD_ATTEMPTED = False
_MODEL_DOWNLOAD_ERROR: Optional[str] = None


def _resolve_pose_model_path() -> str:
    global _MODEL_DOWNLOAD_ATTEMPTED
    global _MODEL_DOWNLOAD_ERROR

    env_model_path = os.getenv("POSESENSE_POSE_MODEL_PATH")
    if env_model_path:
        model_path = Path(env_model_path).expanduser().resolve()
        if model_path.exists():
            return str(model_path)
        raise RuntimeError(
            f"POSESENSE_POSE_MODEL_PATH is set but file does not exist: {model_path}"
        )

    if DEFAULT_POSE_MODEL_PATH.exists():
        return str(DEFAULT_POSE_MODEL_PATH)

    if not _MODEL_DOWNLOAD_ATTEMPTED:
        _MODEL_DOWNLOAD_ATTEMPTED = True
        DEFAULT_POSE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            urlretrieve(DEFAULT_POSE_MODEL_URL, DEFAULT_POSE_MODEL_PATH)
        except URLError as exc:
            _MODEL_DOWNLOAD_ERROR = f"{type(exc).__name__}: {exc.reason}"
        except Exception as exc:
            _MODEL_DOWNLOAD_ERROR = f"{type(exc).__name__}: {exc}"

    if DEFAULT_POSE_MODEL_PATH.exists():
        return str(DEFAULT_POSE_MODEL_PATH)

    error_suffix = f" Last download error: {_MODEL_DOWNLOAD_ERROR}" if _MODEL_DOWNLOAD_ERROR else ""
    raise RuntimeError(
        "Pose Landmarker model file not found. Set POSESENSE_POSE_MODEL_PATH to a local "
        "pose_landmarker .task file, or place the model at "
        f"{DEFAULT_POSE_MODEL_PATH}.{error_suffix}"
    )


def _create_pose_landmarker():
    model_path = _resolve_pose_model_path()
    try:
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        return mp.tasks.vision.PoseLandmarker.create_from_options(options)
    except AttributeError as exc:
        raise RuntimeError(
            "This MediaPipe build does not expose Tasks PoseLandmarker APIs. "
            "Install a recent mediapipe package that includes mp.tasks."
        ) from exc

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
        assert best_resolution is not None
        print(f"Best camera: index {best_index}, resolution {best_resolution}")
        cap = cv2.VideoCapture(best_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_resolution[1])
        return cap
    else:
        print("No suitable camera found.")
        return None

class PoseProcessor:
    def __init__(self, source: Optional[cv2.VideoCapture] = None, fps: float =30.0):
        self.cap = source if source is not None and source.isOpened() else None
        self.fps = fps
        self.pose = _create_pose_landmarker()
        self._last_ts_ms = 0

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

    def _mp_landmarks_to_world(self, results):
        world_landmarks = getattr(results, "pose_world_landmarks", None)
        if not world_landmarks:
            return None

        first_pose = world_landmarks[0]
        if not first_pose:
            return None

        pts = []
        for lm in first_pose:
            pts.append([lm.x, lm.y, lm.z])
        return np.array(pts, dtype=np.float64)

    def _next_timestamp_ms(self) -> int:
        ts_ms = int(time.time() * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        return ts_ms

    def read_frame(self):
        if self.cap is None:
            return None, None
        ok, frame = self.cap.read()
        if not ok:
            return None, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.pose.detect_for_video(mp_image, self._next_timestamp_ms())
        lmk_world = self._mp_landmarks_to_world(res)
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
