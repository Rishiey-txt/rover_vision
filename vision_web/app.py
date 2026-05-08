from __future__ import annotations

import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st
import torch
from ultralytics import YOLO


st.set_page_config(layout="wide", page_title="YOLO Vision", page_icon="🎯")

MODEL_PATH = "vision_web/best.pt"
IMGSZ = 640
HALF_PRECISION = True
SNAPSHOT_DIR = Path("snapshots")
FALLBACK_INFO_KEY = "device_fallback_info_shown"

def available_devices() -> list[str]:
    devices: list[str] = []
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")
    return devices

def select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> tuple[YOLO, str, dict[int, str] | list[str]]:
    device = select_device()
    model = YOLO(model_path)
    model.to(device)
    names = model.names
    return model, device, names

def class_name_to_id_map(names: dict[int, str] | list[str]) -> dict[str, int]:
    if isinstance(names, dict):
        return {name: idx for idx, name in names.items()}
    return {name: idx for idx, name in enumerate(names)}

def save_snapshot(frame) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = SNAPSHOT_DIR / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(snapshot_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return snapshot_path

def draw_hud(image, fps: float, device: str, detection_count: int) -> None:
    label = f"FPS: {fps:4.1f} | {device.upper()} | {detection_count} obj"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    thickness = 2
    pad_x = 16
    pad_y = 12
    x = 16
    y = 16

    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pill_width = text_width + (pad_x * 2)
    pill_height = text_height + (pad_y * 2) + baseline
    radius = pill_height // 2

    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (x + radius, y),
        (x + pill_width - radius, y + pill_height),
        (18, 24, 38),
        -1,
    )
    cv2.circle(overlay, (x + radius, y + radius), radius, (18, 24, 38), -1)
    cv2.circle(overlay, (x + pill_width - radius, y + radius), radius, (18, 24, 38), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0.0, image)

    cv2.putText(
        image,
        label,
        (x + pad_x, y + pad_y + text_height),
        font,
        font_scale,
        (134, 245, 187),
        thickness,
        cv2.LINE_AA,
    )

# ── Bootstrap ──────────────────────────────────────────────────────────────
model, active_device, model_names = load_model(MODEL_PATH)
class_map = class_name_to_id_map(model_names)
class_labels = list(class_map.keys())
device_options = available_devices()

# ── Session-state defaults ─────────────────────────────────────────────────
if "selected_device" not in st.session_state:
    st.session_state.selected_device = active_device

if active_device != "mps" and not st.session_state.get(FALLBACK_INFO_KEY, False):
    st.info(f"MPS is unavailable on this machine, so inference is running on `{active_device.upper()}`.")
    st.session_state[FALLBACK_INFO_KEY] = True

st.title("YOLO Vision")
st.caption("Real-time object detection using OpenCV VideoCapture.")

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Model")
    st.markdown(
        f"""
        <div style="
            padding: 0.9rem 1rem;
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.7);
            margin-bottom: 1rem;
        ">
            <div><strong>Model:</strong> {MODEL_PATH}</div>
            <div><strong>Classes:</strong> {len(class_labels)}</div>
            <div><strong>Device:</strong> {st.session_state.selected_device.upper()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    confidence = st.slider("Confidence", min_value=0.10, max_value=0.95, value=0.60, step=0.05)
    selected_inference_device = st.radio(
        "Inference Device",
        options=device_options,
        horizontal=True,
        key="selected_device",
        help="Switches YOLO inference between the backends available on this machine.",
    )
    selected_classes = st.multiselect("Class Filter", options=class_labels, default=[])

    st.markdown("### Camera Settings")
    camera_index = st.number_input("Camera Index", min_value=0, max_value=10, value=0, step=1)
    run_camera = st.checkbox("Start Camera", value=False)
    
    st.markdown("### Live Metrics")
    fps_placeholder = st.empty()
    detections_placeholder = st.empty()
    top_class_placeholder = st.empty()

frame_placeholder = st.empty()

if run_camera:
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        st.error(f"Cannot open camera with index {camera_index}")
    else:
        fps_deque = deque(maxlen=15)
        prev_time = time.perf_counter()
        total_session_detections = 0
        
        while cap.isOpened() and run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame from camera.")
                break
            
            if selected_inference_device != active_device:
                model.to(selected_inference_device)
                active_device = selected_inference_device
                
            classes_filter = [class_map[name] for name in selected_classes] if selected_classes else None
            
            results = model.predict(
                frame,
                conf=confidence,
                classes=classes_filter,
                device=selected_inference_device,
                half=HALF_PRECISION,
                imgsz=IMGSZ,
                verbose=False,
            )
            
            result = results[0]
            annotated = result.plot()
            
            detection_count = 0 if result.boxes is None else len(result.boxes)
            total_session_detections += detection_count
            
            top_class = "—"
            if result.boxes is not None and result.boxes.cls is not None:
                class_counter = Counter()
                for class_id in result.boxes.cls.tolist():
                    class_name = model_names[int(class_id)]
                    class_counter[class_name] += 1
                if class_counter:
                    top_class = class_counter.most_common(1)[0][0]
                    
            now = time.perf_counter()
            delta = max(now - prev_time, 1e-6)
            fps_deque.append(1.0 / delta)
            current_fps = sum(fps_deque) / len(fps_deque)
            prev_time = now
            
            draw_hud(annotated, current_fps, selected_inference_device, detection_count)
            
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
            
            fps_placeholder.metric("Current FPS", f"{current_fps:.1f}")
            detections_placeholder.metric("Total Session Detections", str(total_session_detections))
            top_class_placeholder.metric("Top Detected Class", top_class)
            
        cap.release()