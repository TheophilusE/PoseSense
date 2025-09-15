import asyncio
import cv2
from typing import AsyncGenerator, Dict, Any
from mmpose.apis import init_pose_model, inference_top_down_pose_model
from mmdet.apis import init_detector, inference_detector

# Paths to your configs and checkpoints
POSE_CONFIG     = 'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'
POSE_CHECKPOINT = 'checkpoints/hrnet_w32_coco_256x192.pth'
DET_CONFIG      = 'mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
DET_CHECKPOINT  = 'checkpoints/faster_rcnn_r50_fpn_1x_coco.pth'

# Init models
det_model = init_detector(DET_CONFIG, DET_CHECKPOINT, device='cuda:0')
pose_model = init_pose_model(POSE_CONFIG, POSE_CHECKPOINT, device='cuda:0')

# Open webcam
cap = cv2.VideoCapture(0)

async def mmpose_live_stream(delay: float = 1/30) -> AsyncGenerator[Dict[str, Any], None]:
    """Yield pose frames from webcam using MMPose."""
    while True:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.01)
            continue

        # Detect people
        det_results = inference_detector(det_model, frame)
        # Filter person class (COCO class 0)
        person_bboxes = [
            dict(bbox=list(map(float, box)))
            for box in det_results[0] if box[4] > 0.5
        ]

        if not person_bboxes:
            await asyncio.sleep(delay)
            continue

        # Pose estimation
        pose_results, _ = inference_top_down_pose_model(
            pose_model,
            frame,
            person_bboxes,
            bbox_format='xyxy',
            format='xyxy'
        )

        # Convert to joint dict
        joints = {}
        if pose_results:
            keypoints = pose_results[0]['keypoints']  # shape: (num_joints, 3)
            for idx, (x, y, conf) in enumerate(keypoints):
                joints[f"joint_{idx}"] = [float(x), float(y), float(conf)]

        yield {"frame": None, "joints": joints}
        await asyncio.sleep(delay)
