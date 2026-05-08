# 🤖 Rover Vision

Real-time computer vision suite for rover operations — combining high-precision **ArUco marker detection** and **YOLO-based object detection** in a live Streamlit dashboard.

---

## 📁 Project Structure

```
rover_vision/
├── arUco_detector/
│   └── arUco_detector_cv.py   # Standalone ArUco detection pipeline
├── vision_web/
│   ├── app.py                 # Streamlit YOLO inference dashboard
│   ├── best.pt                # Custom-trained YOLO model weights (not committed — see below)
│   └── requirements.txt       # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/Rishiey-txt/rover_vision.git
cd rover_vision
```

### 2. Create & activate a virtual environment

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
```

### 3. Install dependencies

```bash
pip install -r vision_web/requirements.txt
```

> **Apple Silicon (M-series):** PyTorch MPS acceleration is detected automatically — no extra steps needed.

### 4. Add the model weights

`best.pt` is not committed to this repo (it is large). Place your trained YOLO `.pt` file at:

```
vision_web/best.pt
```

Or update `MODEL_PATH` in `vision_web/app.py` to point to your file.

---

## 🖥️ Running the YOLO Dashboard

```bash
streamlit run vision_web/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Dashboard Features
| Feature | Details |
|---|---|
| **Live camera feed** | OpenCV `VideoCapture` — choose any camera index |
| **Real-time YOLO inference** | Runs on MPS / CUDA / CPU automatically |
| **Confidence threshold** | Adjustable slider (0.10 – 0.95) |
| **Class filter** | Multiselect — filter to specific detected classes |
| **Inference device selector** | Switch between MPS / CUDA / CPU on the fly |
| **HUD overlay** | FPS · device · detection count burned into the frame |
| **Live metrics** | Current FPS · session detections · top class |

---

## 🎯 Running the ArUco Detector

The ArUco detector is a standalone OpenCV script with extensive CLI flags.

```bash
# Basic usage (webcam index 0)
python arUco_detector/arUco_detector_cv.py

# Long-range mode with contrast enhancement
python arUco_detector/arUco_detector_cv.py \
    --source 0 \
    --long-range \
    --pyramid-levels 2 \
    --pyramid-scale 1.5 \
    --preprocess clahe_sharpen

# From a video file, with distance estimation
python arUco_detector/arUco_detector_cv.py \
    --source path/to/video.mp4 \
    --marker-size-cm 14.0 \
    --fx 800 \
    --output output.mp4
```

### Key CLI Flags
| Flag | Default | Description |
|---|---|---|
| `--source` | `0` | Camera index or video file path |
| `--dict` | `DICT_4X4_250` | ArUco dictionary |
| `--long-range` | off | Enable multi-scale long-range mode |
| `--pyramid-levels` | `2` | Upsampling pyramid levels |
| `--pyramid-scale` | `1.5` | Scale multiplier per level |
| `--preprocess` | `none` | `none` / `clahe` / `clahe_sharpen` |
| `--min-frames` | `2` | Consecutive frames before accepting detection |
| `--refine` | off | Sub-pixel corner refinement |
| `--marker-size-cm` | — | Marker side length for distance estimation |
| `--fx` | — | Camera focal length in pixels |
| `--output` | — | Save annotated video to this path |

Press **`q`** to quit · **`r`** to reset temporal tracking.

---

## ⚙️ Hardware Acceleration

| Platform | Backend | Notes |
|---|---|---|
| Apple Silicon (M1/M2/M3/M4) | **MPS** | Auto-detected by PyTorch |
| NVIDIA GPU | **CUDA** | Auto-detected by PyTorch |
| Any machine | **CPU** | Fallback — always available |

The Streamlit app shows a banner if MPS is unavailable and falls back gracefully.

---

## 📦 Dependencies

See [`vision_web/requirements.txt`](vision_web/requirements.txt).

Core packages:
- `streamlit >= 1.32`
- `ultralytics >= 8.0` (YOLOv8/v11)
- `opencv-contrib-python >= 4.8` (ArUco + standard CV)
- `torch >= 2.2`
- `numpy >= 1.26`

---

## 🔒 Model Weights

`best.pt` is excluded from version control (`.gitignore`). To share weights:

- Use **[Git LFS](https://git-lfs.github.com/)** for large files in the repo, or
- Host on [Hugging Face Hub](https://huggingface.co/) / Google Drive and update the README with a download link.

---

## 📄 License

MIT — see [LICENSE](LICENSE).
