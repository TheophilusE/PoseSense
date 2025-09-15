import asyncio
import time
import cv2
import mediapipe as mp
from typing import AsyncGenerator, Dict, Any

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


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
    best_score = -1
    best_index = None
    best_resolution = None

    for i in range(max_devices):
        for res in RESOLUTIONS:
            score = test_camera_resolution(i, *res)
            if score:
                print(f"Camera {i} supports {res[0]}x{res[1]} → score: {score}")
                if score > best_score:
                    best_score = score
                    best_index = i
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

cap = select_best_camera()

async def mediapipe_live_stream(delay: float = 1/30) -> AsyncGenerator[Dict[str, Any], None]:
    """Yield pose frames from webcam using MediaPipe Pose."""
    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.01)
            continue

        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        joints = {}
        if results.pose_landmarks:
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                joints[f"joint_{idx}"] = [lm.x, lm.y, lm.z]

        yield {"frame": None, "joints": joints}
        await asyncio.sleep(delay)
