# 1) Create and activate a clean environment (conda or venv)
conda create -n mmpose3d python=3.10 -y
conda activate mmpose3d

# 2) Install PyTorch with CUDA (visit pytorch.org if you need a different CUDA version)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) Install OpenMMLab stack + MMPose
pip install -U openmim
mim install "mmengine>=0.10.0" "mmcv>=2.1.0" "mmdet>=3.3.0"
pip install "mmpose>=1.3.0"

# 4) Utilities for streaming and video
pip install opencv-python websockets numpy
