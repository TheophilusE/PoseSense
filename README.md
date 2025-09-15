### Step-by-Step MMPose Installation (pip-only)

#### Common Pitfalls

- **Missing PyTorch:** If you haven’t installed PyTorch yet, do so before installing MMPose:
  ```bash
  pip install torch torchvision torchaudio
  ```
- **CUDA Compatibility:** If you're using a GPU, make sure your PyTorch version matches your CUDA version. You can find the right wheel at [PyTorch.org](https://pytorch.org/get-started/locally/).


#### 1. Set up a virtual environment (recommended)
This keeps dependencies isolated.

```bash
python -m venv mmpose-env
source mmpose-env/bin/activate  # On Windows: mmpose-env\Scripts\activate
```

#### 2. Upgrade pip and install core dependencies
```bash
pip install --upgrade pip
pip install -U openmim
```

#### 3. Install OpenMMLab core libraries
```bash
mim install mmengine
mim install "mmcv>=2.0.1"
```

#### 4. (Optional) Install MMDetection if you want to use detection-based pose models
```bash
mim install "mmdet>=3.1.0"
```

#### 5. Install MMPose from source
This gives you full access to demos, configs, and training scripts.

```bash
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install -v -e .
```
> The `-e .` flag installs MMPose in "editable" mode, so any changes you make to the source code will reflect immediately.

or

```
mim install "mmpose>=1.1.0"
```

---

### Test Your Installation

Run this quick check:
```bash
python -c "import mmpose; print(mmpose.__version__)"
```

If it prints a version number (e.g. `1.1.0`), you're good to go!

---

Use opencv-python instead of opencv-python-headless if you want to preview the webcam feed.