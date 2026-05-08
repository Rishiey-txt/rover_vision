import cv2
import numpy as np
import argparse
import time
from typing import Tuple, List, Optional, Any, Dict
from collections import defaultdict

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-Precision ArUco Detection (with optional long-range mode)")
    parser.add_argument("--source", type=str, default="0", 
                        help="Video file path or camera index (default: 0)")
    parser.add_argument("--dict", type=str, default="DICT_4X4_250", 
                        choices=["DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
                                 "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
                                 "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
                                 "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000", "AUTO"],
                        help="ArUco dictionary to use (default: DICT_4X4_250 for lower false positive rate)")
    parser.add_argument("--dict-long-range", type=str, default=None,
                        choices=["DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
                                 "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
                                 "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
                                 "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000", "AUTO"],
                        help="Optional dictionary override used only in --long-range mode. "
                             "For distance, larger dictionaries like DICT_4X4_1000 or DICT_5X5_1000 can be more robust.")
    parser.add_argument("--output", type=str, default=None, 
                        help="Path to save output video")
    parser.add_argument("--refine", action="store_true", 
                        help="Enable sub-pixel corner refinement")
    parser.add_argument("--long-range", action="store_true",
                        help="Enable long-range optimizations (multi-scale, relaxed size thresholds, contrast enhancement)")
    parser.add_argument("--max-distance", type=float, default=None,
                        help="Optional hint (meters). Used only to adjust thresholds/labeling in --long-range mode.")
    parser.add_argument("--pyramid-levels", type=int, default=2,
                        help="Number of additional pyramid levels for multi-scale detection in --long-range mode (default: 2). "
                             "Total scales = 1 + levels.")
    parser.add_argument("--pyramid-scale", type=float, default=1.5,
                        help="Scale multiplier per pyramid level in --long-range mode (default: 1.5). "
                             "Example levels=2 => scales: 1.0, 1.5, 2.25")
    parser.add_argument("--max-pyramid-dim", type=int, default=1920,
                        help="Max width/height for upscaled pyramid images (default: 1920). Limits CPU usage.")
    parser.add_argument("--preprocess", type=str, default="none",
                        choices=["none", "clahe", "clahe_sharpen"],
                        help="Optional preprocessing to help distant/low-contrast markers (default: none). "
                             "In --long-range mode, default is clahe_sharpen unless explicitly set.")
    parser.add_argument("--min-area-ratio", type=float, default=0.005,
                        help="Minimum marker area as ratio of frame area (default: 0.005 = 0.5%%)")
    parser.add_argument("--min-area-ratio-long", type=float, default=0.0005,
                        help="Minimum marker area ratio for --long-range mode at scale=1.0 (default: 0.0005 = 0.05%%). "
                             "Effective threshold is further scaled based on pyramid scale.")
    parser.add_argument("--min-perimeter-rate", type=float, default=0.04,
                        help="DetectorParameters.minMarkerPerimeterRate for strict mode (default: 0.04).")
    parser.add_argument("--min-perimeter-rate-long", type=float, default=0.015,
                        help="DetectorParameters.minMarkerPerimeterRate for --long-range mode (default: 0.015). "
                             "Small markers at distance often need 0.01–0.02.")
    parser.add_argument("--min-frames", type=int, default=2,
                        help="Minimum consecutive frames to accept a detection (default: 2)")
    parser.add_argument("--max-aspect", type=float, default=1.3,
                        help="Maximum aspect ratio deviation from square (default: 1.3)")
    parser.add_argument("--marker-size-cm", type=float, default=None,
                        help="Optional real marker side length in centimeters (for distance estimation).")
    parser.add_argument("--fx", type=float, default=None,
                        help="Optional camera focal length in pixels (fx). If provided with --marker-size-cm, estimates distance.")
    return parser.parse_args()

def _build_detector_params(profile: str, min_perimeter_rate: float) -> "cv2.aruco.DetectorParameters":
    """
    Build DetectorParameters.

    Profiles:
    - strict: tuned for close-range high precision / low false positives
    - long_range: tuned for small markers at distance (more permissive, but relies on geometric + temporal filtering)
    """
    detector_params = cv2.aruco.DetectorParameters()

    # Marker size gates (key long-range knob is minMarkerPerimeterRate)
    detector_params.minMarkerPerimeterRate = float(min_perimeter_rate)
    detector_params.maxMarkerPerimeterRate = 4.0
    detector_params.minCornerDistanceRate = 0.05
    detector_params.minDistanceToBorder = 3

    # Polygon approximation accuracy:
    # - strict: slightly tighter corners
    # - long_range: allow rougher quads (small markers will quantize / blur)
    detector_params.polygonalApproxAccuracyRate = 0.04 if profile == "strict" else 0.06

    # Perspective removal (decoding):
    # Too large values increase compute; too small can harm decoding stability.
    detector_params.perspectiveRemovePixelPerCell = 8 if profile == "strict" else 6
    detector_params.perspectiveRemoveIgnoredMarginPerCell = 0.13

    # Border checking + thresholding robustness:
    # In long_range we tolerate lower contrast.
    detector_params.maxErroneousBitsInBorderRate = 0.30 if profile == "strict" else 0.35
    detector_params.minOtsuStdDev = 5.0 if profile == "strict" else 2.0

    # Corner refinement inside ArUco:
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector_params.cornerRefinementWinSize = 5 if profile == "strict" else 7
    detector_params.cornerRefinementMaxIterations = 30 if profile == "strict" else 50
    detector_params.cornerRefinementMinAccuracy = 0.1 if profile == "strict" else 0.03

    return detector_params

def setup_detectors(
    dict_choice: str,
    *,
    profile: str = "strict",
    min_perimeter_rate: float = 0.04,
) -> List[Tuple[str, cv2.aruco.ArucoDetector]]:
    """Initializes ArucoDetectors for the selected profile."""
    dict_map = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
        "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
        "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
        "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
        "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
        "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
        "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
        "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
        "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    }

    dict_names = list(dict_map.keys()) if dict_choice == "AUTO" else [dict_choice]

    detectors = []

    for name in dict_names:
        detector_params = _build_detector_params(profile=profile, min_perimeter_rate=min_perimeter_rate)

        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_map[name])
        detectors.append((name, cv2.aruco.ArucoDetector(aruco_dict, detector_params)))

    return detectors

def _preprocess_gray(gray: np.ndarray, mode: str) -> np.ndarray:
    """
    Preprocessing aimed at distant/low-contrast markers.
    - clahe: local contrast enhancement (helps against haze / lighting gradients)
    - clahe_sharpen: clahe + mild unsharp mask (helps tiny blurred markers)
    """
    if mode == "none":
        return gray

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = clahe.apply(gray)

    if mode == "clahe":
        return out

    # Mild unsharp mask
    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0, sigmaY=1.0)
    sharp = cv2.addWeighted(out, 1.5, blur, -0.5, 0)
    return sharp

def _build_pyramid_scales(levels: int, scale_step: float) -> List[float]:
    """Scales for multi-scale detection. We *upsample* to make small distant markers easier to detect."""
    scales = [1.0]
    s = 1.0
    for _ in range(max(0, int(levels))):
        s *= float(scale_step)
        scales.append(s)
    return scales

def _scale_image(gray: np.ndarray, scale: float, max_dim: int) -> Tuple[np.ndarray, float]:
    """
    Resize gray by scale, but cap max dimension to max_dim to keep runtime bounded.
    Returns resized image and the actual scale used (may be smaller than requested).
    """
    if scale == 1.0:
        return gray, 1.0
    h, w = gray.shape[:2]
    target_w = int(round(w * scale))
    target_h = int(round(h * scale))
    if max(target_w, target_h) > int(max_dim) and max_dim > 0:
        cap_scale = float(max_dim) / float(max(w, h))
        if cap_scale <= 1.0:
            return gray, 1.0
        scale = min(scale, cap_scale)
        target_w = int(round(w * scale))
        target_h = int(round(h * scale))
    resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA)
    return resized, float(scale)

def _unscale_corners(corner: np.ndarray, scale: float) -> np.ndarray:
    """Map corners detected in a scaled image back to original coordinates."""
    if scale == 1.0:
        return corner
    out = corner.copy()
    out[0, :, :] = out[0, :, :] / float(scale)
    return out

def _marker_pixel_size(corner: np.ndarray) -> float:
    """Estimate marker side length in pixels (average of 4 edges)."""
    pts = corner[0].astype(np.float32)
    edges = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        edges.append(float(np.linalg.norm(p2 - p1)))
    return float(np.mean(edges)) if edges else 0.0

def _estimate_distance_m(marker_size_cm: float, fx: float, corner: np.ndarray) -> Optional[float]:
    """
    Rough distance estimate (meters) from marker size + focal length:
      Z ~= fx * marker_size / pixel_size
    Assumes marker plane roughly fronto-parallel. Good enough for HUD/rough gating.
    """
    px = _marker_pixel_size(corner)
    if px <= 1e-6:
        return None
    marker_size_m = float(marker_size_cm) / 100.0
    return float(fx) * marker_size_m / px

def is_valid_marker(corner: np.ndarray, frame_shape: Tuple[int, ...], 
                    min_area_ratio: float, max_aspect: float) -> bool:
    """
    Filters out false positives based on geometric constraints.

    NOTE: In long-range mode we deliberately allow smaller projected areas; false positives are controlled by:
    - convexity + angle constraints here
    - confidence scoring (border quality)
    - temporal consistency (marker_history)
    """
    frame_h, frame_w = frame_shape[:2]
    frame_area = frame_h * frame_w

    # Calculate marker area
    area = cv2.contourArea(corner)
    area_ratio = area / (frame_area + 1e-9)
    if area_ratio < min_area_ratio:
        return False

    # Check aspect ratio (must be close to square)
    rect = cv2.minAreaRect(corner)
    width, height = rect[1]
    if width == 0 or height == 0:
        return False
    aspect = max(width, height) / min(width, height)
    # Scale-aware looseness: tiny, distant markers are more likely to be aliased/warped.
    # Keep close-range strictness, but allow some extra skew for very small candidates.
    eff_max_aspect = float(max_aspect)
    if area_ratio < 0.001:
        eff_max_aspect *= 1.25
    if aspect > eff_max_aspect:
        return False

    # Check convexity — real markers are convex quadrilaterals
    if not cv2.isContourConvex(corner.astype(np.float32)):
        return False

    # Check that all angles are reasonably close to 90 degrees
    # (perspective distortion can skew this, but extreme angles indicate false positive)
    pts = corner[0]
    angles = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        p3 = pts[(i + 2) % 4]

        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        angles.append(angle)

    # All angles should be between 30 and 150 degrees (reasonable perspective range)
    if any(a < 30 or a > 150 for a in angles):
        return False

    return True

def compute_confidence(corner: np.ndarray, gray: np.ndarray) -> float:
    """
    Compute a pseudo-confidence score based on marker quality metrics.

    Long-range tweak: Use a smaller warp size for small markers to reduce aliasing artifacts and
    avoid over-penalizing tiny/warped detections.
    """
    pts = corner[0].astype(np.int32)

    # Get perspective transform to canonical square
    px_size = _marker_pixel_size(corner)
    # For very small markers, use smaller normalization grid.
    dst_size = 70 if px_size < 40 else 100
    dst_pts = np.array([[0, 0], [dst_size, 0], [dst_size, dst_size], [0, dst_size]], dtype=np.float32)
    src_pts = pts.astype(np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(gray, M, (dst_size, dst_size))

    # Binarize
    _, binary = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Check border uniformity (black border should be uniformly dark)
    border = np.concatenate([
        binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]
    ])
    border_black_ratio = np.sum(border < 128) / len(border)

    # Confidence based on border quality
    # For tiny markers we expect more blur; slightly softer mapping.
    gain = 1.15 if px_size < 40 else 1.2
    confidence = min(1.0, border_black_ratio * gain)
    confidence = max(0.0, confidence)

    return round(confidence, 2)

def process_frame(frame: np.ndarray, 
                  detectors: List[Tuple[str, cv2.aruco.ArucoDetector]], 
                  long_range_detectors: Optional[List[Tuple[str, cv2.aruco.ArucoDetector]]],
                  refine_corners: bool,
                  min_area_ratio: float,
                  min_area_ratio_long: float,
                  max_aspect: float,
                  marker_history: defaultdict,
                  min_frames: int,
                  *,
                  long_range: bool = False,
                  pyramid_levels: int = 0,
                  pyramid_scale: float = 1.5,
                  max_pyramid_dim: int = 1920,
                  preprocess: str = "none",
                  marker_size_cm: Optional[float] = None,
                  fx: Optional[float] = None,
                  max_distance_hint_m: Optional[float] = None,
                  ) -> Tuple[np.ndarray, defaultdict]:
    """
    Detects ArUco markers with false-positive filtering and draws annotations.

    Long-range mode additions:
    - Optional preprocessing for contrast / sharpness
    - Multi-scale detection (image pyramid upsampling) to support small distant markers
    - Dynamic min area ratio per scale (lower at higher scale)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if long_range:
        gray = _preprocess_gray(gray, preprocess)

    all_corners = []
    all_ids = []
    all_confidences = []
    all_scales = []

    # Multi-scale detection (long-range):
    # We upsample the image so the marker occupies more pixels, improving the detector's perimeter/threshold tests.
    pyramid_scales = _build_pyramid_scales(pyramid_levels, pyramid_scale) if long_range else [1.0]

    strict_found_any = False
    for scale_req in pyramid_scales:
        scaled_gray, scale = _scale_image(gray, scale_req, max_pyramid_dim)
        if scale != scale_req and scale_req != 1.0:
            # scale capped; still proceed with actual scale
            pass

        # Dynamic min area ratio:
        # - strict uses user-provided min_area_ratio
        # - long_range uses lower base, and we further divide by scale^2 since area grows with scale^2
        def _handle_detection_set(dets: List[Tuple[str, cv2.aruco.ArucoDetector]], eff_min_area_ratio: float) -> bool:
            found_any = False
            for _name, detector in dets:
                corners, ids, _rejected = detector.detectMarkers(scaled_gray)
                if ids is None:
                    continue

                for corner, mid in zip(corners, ids):
                    corner_unscaled = _unscale_corners(corner, scale)
                    if not is_valid_marker(corner_unscaled, frame.shape, eff_min_area_ratio, max_aspect):
                        continue

                    conf = compute_confidence(corner_unscaled, gray)

                    if long_range and marker_size_cm is not None and fx is not None and max_distance_hint_m is not None:
                        est_d = _estimate_distance_m(marker_size_cm, fx, corner_unscaled)
                        if est_d is not None and est_d > (max_distance_hint_m * 1.25):
                            continue

                    all_corners.append(corner_unscaled)
                    all_ids.append(mid)
                    all_confidences.append(conf)
                    all_scales.append(scale)
                    found_any = True
            return found_any

        if scale == 1.0:
            # Strict pass first.
            if _handle_detection_set(detectors, float(min_area_ratio)):
                strict_found_any = True

            # In long-range mode, only run the permissive pass at 1.0x if strict found nothing.
            if long_range and (not strict_found_any) and long_range_detectors:
                _handle_detection_set(long_range_detectors, float(min_area_ratio_long))
        else:
            # Upscaled pyramid levels: only use long-range detectors to keep runtime reasonable.
            if long_range and long_range_detectors:
                eff_min_area_ratio = float(min_area_ratio_long) / max(1.0, (scale * scale))
                _handle_detection_set(long_range_detectors, eff_min_area_ratio)

    # If multi-scale / multi-detector produced duplicates of the same marker ID in this frame,
    # keep the best candidate only (prevents artificially boosting temporal counts and reduces flicker).
    if len(all_ids) > 1:
        best_by_id: Dict[int, int] = {}
        for idx, mid in enumerate(all_ids):
            m_id = int(mid[0])
            if m_id not in best_by_id:
                best_by_id[m_id] = idx
                continue
            prev_idx = best_by_id[m_id]
            # Prefer higher confidence; tie-breaker: larger pixel size.
            prev_score = (all_confidences[prev_idx], _marker_pixel_size(all_corners[prev_idx]))
            cur_score = (all_confidences[idx], _marker_pixel_size(all_corners[idx]))
            if cur_score > prev_score:
                best_by_id[m_id] = idx

        keep_indices = sorted(best_by_id.values())
        all_corners = [all_corners[i] for i in keep_indices]
        all_ids = [all_ids[i] for i in keep_indices]
        all_confidences = [all_confidences[i] for i in keep_indices]
        all_scales = [all_scales[i] for i in keep_indices]

    # Temporal consistency: track detections across frames
    current_detections = set()
    if len(all_corners) > 0:
        for i, (corner, mid, conf) in enumerate(zip(all_corners, all_ids, all_confidences)):
            # Use a looser key for temporal tracking — just the ID
            marker_id_key = int(mid[0])
            marker_history[marker_id_key] += 1
            current_detections.add(marker_id_key)

    # Decay history for markers not seen this frame
    for key in list(marker_history.keys()):
        if key not in current_detections:
            marker_history[key] = max(0, marker_history[key] - 1)

    # Draw only stable detections
    annotated = frame.copy()
    drawn_count = 0

    if len(all_corners) > 0:
        for i, (corner, mid, conf) in enumerate(zip(all_corners, all_ids, all_confidences)):
            marker_id = int(mid[0])

            # Require minimum consecutive detections
            if marker_history.get(marker_id, 0) < min_frames:
                continue

            # Optional sub-pixel refinement (extra pass) using a scale-aware window size.
            if refine_corners:
                # Build a small ROI-based window size: tiny markers need smaller windows; larger can use larger.
                px = _marker_pixel_size(corner)
                win = int(np.clip(round(px * 0.08), 2, 8))
                try:
                    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                    refined = corner[0].astype(np.float32).reshape(-1, 1, 2)
                    cv2.cornerSubPix(gray, refined, (win, win), (-1, -1), term)
                    corner = corner.copy()
                    corner[0] = refined.reshape(4, 2)
                except Exception:
                    # If cornerSubPix fails (e.g., near border), keep original.
                    pass

            pts = corner[0].astype(np.int32)

            # 1. Draw tight boundary (blue polygon, no fill)
            cv2.polylines(annotated, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

            # 2. Draw small corner markers for precision feel
            for pt in pts:
                cv2.circle(annotated, tuple(pt), 3, (0, 255, 255), -1)

            # 3. Label placement at top-left of polygon
            # Find top-leftmost point
            top_left = pts[np.argmin(pts[:, 0] + pts[:, 1])]
            label_x = int(top_left[0])
            label_y = int(top_left[1]) - 10

            label = f"ArUco {marker_id} {conf:.2f}"
            if marker_size_cm is not None and fx is not None:
                est_d = _estimate_distance_m(marker_size_cm, fx, corner)
                if est_d is not None:
                    label += f" {est_d:.2f}m"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Keep text within frame bounds
            label_x = max(0, min(label_x, annotated.shape[1] - text_w - 5))
            label_y = max(text_h + 10, label_y)

            # Text background (semi-transparent black)
            bg_top_left = (label_x, label_y - text_h - 6)
            bg_bottom_right = (label_x + text_w + 4, label_y + 2)

            overlay = annotated.copy()
            cv2.rectangle(overlay, bg_top_left, bg_bottom_right, (0, 0, 0), cv2.FILLED)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

            # Text (white)
            cv2.putText(annotated, label, (label_x + 2, label_y - 2), 
                       font, font_scale, (255, 255, 255), thickness)

            drawn_count += 1

    return annotated, marker_history

def main() -> None:
    args = parse_args()

    # Open Video Source
    source: Any = args.source
    if str(args.source).isdigit():
        source = int(args.source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source '{args.source}'")
        return

    # Set buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Try to set higher resolution for better detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Setup Video Writer if output is specified
    out_writer: Optional[cv2.VideoWriter] = None
    if args.output:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

    # Resolve preprocessing default: in long-range mode, default to clahe_sharpen unless user overrides.
    preprocess_mode = args.preprocess
    if args.long_range and args.preprocess == "none":
        preprocess_mode = "clahe_sharpen"

    detectors = setup_detectors(args.dict, profile="strict", min_perimeter_rate=args.min_perimeter_rate)
    long_range_detectors = None
    if args.long_range:
        # Use a more permissive detector profile; false positives are still controlled by geometric + temporal checks.
        dict_lr = args.dict_long_range if args.dict_long_range is not None else args.dict
        if dict_lr == "AUTO":
            # Default to a robust, large dictionary for long-range.
            dict_lr = "DICT_4X4_1000"
        long_range_detectors = setup_detectors(
            dict_lr,
            profile="long_range",
            min_perimeter_rate=args.min_perimeter_rate_long,
        )

    # Temporal tracking for false positive suppression
    marker_history = defaultdict(int)

    prev_time = time.time()
    fps_history = []

    print("Starting High-Precision ArUco detection...")
    print(f"Dictionary: {args.dict}")
    print(f"Min area ratio: {args.min_area_ratio}")
    if args.long_range:
        print("Long-range mode: ENABLED")
        print(f"  minMarkerPerimeterRate (long): {args.min_perimeter_rate_long}")
        print(f"  min area ratio (long base @1.0x): {args.min_area_ratio_long}")
        print(f"  pyramid: levels={args.pyramid_levels} scale_step={args.pyramid_scale} max_dim={args.max_pyramid_dim}")
        print(f"  preprocess: {preprocess_mode}")
    print(f"Min stable frames: {args.min_frames}")
    print("Press 'q' to quit, 'r' to reset tracking.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS with smoothing
        current_time = time.time()
        time_diff = current_time - prev_time
        instant_fps = 1.0 / time_diff if time_diff > 0 else 0.0
        fps_history.append(instant_fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        fps_val = sum(fps_history) / len(fps_history)
        prev_time = current_time

        # Process frame
        annotated_frame, marker_history = process_frame(
            frame,
            detectors,
            long_range_detectors,
            args.refine,
            args.min_area_ratio,
            args.min_area_ratio_long,
            args.max_aspect,
            marker_history,
            args.min_frames,
            long_range=args.long_range,
            pyramid_levels=args.pyramid_levels,
            pyramid_scale=args.pyramid_scale,
            max_pyramid_dim=args.max_pyramid_dim,
            preprocess=preprocess_mode,
            marker_size_cm=args.marker_size_cm,
            fx=args.fx,
            max_distance_hint_m=args.max_distance,
        )

        # Draw FPS counter (top-right, green)
        fps_text = f"FPS: {fps_val:.1f}"
        (text_w, text_h), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(annotated_frame, fps_text, 
                    (annotated_frame.shape[1] - text_w - 10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw active marker count (top-left)
        active_markers = sum(1 for v in marker_history.values() if v >= args.min_frames)
        count_text = f"Markers: {active_markers}"
        cv2.putText(annotated_frame, count_text, 
                    (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display
        cv2.imshow('High-Precision ArUco Detection', annotated_frame)

        if out_writer:
            out_writer.write(annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            marker_history.clear()
            print("Tracking reset.")

    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

if __name__ == "__main__":
    main()
