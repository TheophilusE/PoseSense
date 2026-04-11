import os
import time
from pathlib import Path
from typing import List, Optional
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

STREAM_WIDTH = int(os.getenv("POSESENSE_STREAM_WIDTH", "640"))
STREAM_JPEG_QUALITY = int(os.getenv("POSESENSE_STREAM_JPEG_QUALITY", "70"))
MAX_WORLD_LANDMARK_JUMP = float(os.getenv("POSESENSE_MAX_WORLD_LANDMARK_JUMP", "0.28"))


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
        self._latest_camera_jpeg: Optional[bytes] = None
        self._latest_pose_landmarks_2d: Optional[List[List[float]]] = None
        self._world_axis_initialized = False
        self._world_axis_flip = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        self._prev_world_landmarks: Optional[np.ndarray] = None

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

    def _reject_world_landmark_outliers(self, lmk: np.ndarray) -> np.ndarray:
        if lmk is None:
            return lmk

        if self._prev_world_landmarks is None or self._prev_world_landmarks.shape != lmk.shape:
            self._prev_world_landmarks = lmk.copy()
            return lmk

        if MAX_WORLD_LANDMARK_JUMP <= 0:
            self._prev_world_landmarks = lmk.copy()
            return lmk

        prev = self._prev_world_landmarks
        delta = lmk - prev
        dist = np.linalg.norm(delta, axis=1)
        mask = dist > MAX_WORLD_LANDMARK_JUMP

        if np.any(mask):
            lmk_clamped = lmk.copy()
            d = np.maximum(dist[mask], 1e-6)
            scale = (MAX_WORLD_LANDMARK_JUMP / d)[:, None]
            lmk_clamped[mask] = prev[mask] + delta[mask] * scale
            self._prev_world_landmarks = lmk_clamped.copy()
            return lmk_clamped

        self._prev_world_landmarks = lmk.copy()
        return lmk

    def _normalize_world_axes(self, lmk: np.ndarray) -> np.ndarray:
        """Normalize MediaPipe world landmarks to a stable upright frame.

        MediaPipe world axes can vary by backend/camera conventions. We lock a
        flip vector once from the first valid frame to keep temporal continuity:
        - Ensure shoulders are above pelvis in +Y (upright convention).
        - Ensure right hip is to the right of left hip in +X.
        """
        if lmk is None or lmk.shape[0] <= 24:
            return lmk

        if not self._world_axis_initialized:
            try:
                shoulder_y = float((lmk[11, 1] + lmk[12, 1]) * 0.5)
                pelvis_y = float((lmk[23, 1] + lmk[24, 1]) * 0.5)
                if shoulder_y < pelvis_y:
                    self._world_axis_flip[1] = -1.0

                # Keep lateral handedness stable (right side should have +X).
                if float(lmk[24, 0]) < float(lmk[23, 0]):
                    self._world_axis_flip[0] = -1.0
            except Exception:
                pass
            self._world_axis_initialized = True

        return lmk * self._world_axis_flip

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
        lmk = np.array(pts, dtype=np.float64)
        return self._normalize_world_axes(lmk)

    def _next_timestamp_ms(self) -> int:
        ts_ms = int(time.time() * 1000)
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        return ts_ms

    def _cache_camera_stream_data(self, frame: np.ndarray, results) -> None:
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return

        scale = min(1.0, STREAM_WIDTH / float(w)) if STREAM_WIDTH > 0 else 1.0
        if scale < 1.0:
            stream_frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            stream_frame = frame

        ok, jpeg = cv2.imencode(
            ".jpg",
            stream_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(STREAM_JPEG_QUALITY, 30, 95))],
        )
        if ok:
            self._latest_camera_jpeg = jpeg.tobytes()

        pose_landmarks = getattr(results, "pose_landmarks", None)
        if not pose_landmarks:
            self._latest_pose_landmarks_2d = None
            return

        first_pose = pose_landmarks[0]
        if not first_pose:
            self._latest_pose_landmarks_2d = None
            return

        self._latest_pose_landmarks_2d = [
            [float(lm.x), float(lm.y), float(getattr(lm, "visibility", 1.0))]
            for lm in first_pose
        ]

    def get_latest_camera_jpeg(self) -> Optional[bytes]:
        return self._latest_camera_jpeg

    def get_latest_pose_landmarks_2d(self) -> Optional[List[List[float]]]:
        return self._latest_pose_landmarks_2d

    def read_frame(self):
        if self.cap is None:
            return None, None
        ok, frame = self.cap.read()
        if not ok:
            return None, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.pose.detect_for_video(mp_image, self._next_timestamp_ms())
        self._cache_camera_stream_data(frame, res)
        lmk_world = self._mp_landmarks_to_world(res)
        if lmk_world is not None:
            lmk_world = self._reject_world_landmark_outliers(lmk_world)
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
        pose_landmarks_2d = self.get_latest_pose_landmarks_2d()
        if pose_landmarks_2d:
            meta["pose_landmarks_2d"] = pose_landmarks_2d

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
