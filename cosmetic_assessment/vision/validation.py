from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from core.settings import (
    CREDIT_CARD_BOARD_MARKER_IDS,
    CREDIT_CARD_SQUARE_MARKER_CENTERS_MM,
    MARKER_SIZE_MM,
    RECTIFIED_PX_PER_MM,
)


def _marker_object_corners_px(marker_id: int) -> np.ndarray:
    cx_mm, cy_mm = CREDIT_CARD_SQUARE_MARKER_CENTERS_MM[marker_id]
    half = MARKER_SIZE_MM / 2.0

    corners_mm = np.array([
        [cx_mm - half, cy_mm - half],  # top-left
        [cx_mm + half, cy_mm - half],  # top-right
        [cx_mm + half, cy_mm + half],  # bottom-right
        [cx_mm - half, cy_mm + half],  # bottom-left
    ], dtype=np.float32)

    return corners_mm * RECTIFIED_PX_PER_MM


def _build_reprojection_sets(detections):
    src_points = []
    dst_points = []

    for det in detections:
        if det.marker_id not in CREDIT_CARD_BOARD_MARKER_IDS:
            continue
        if det.marker_id not in CREDIT_CARD_SQUARE_MARKER_CENTERS_MM:
            continue

        img_corners = np.array(det.corners, dtype=np.float32)
        obj_corners = _marker_object_corners_px(det.marker_id)

        src_points.extend(img_corners.tolist())
        dst_points.extend(obj_corners.tolist())

    if len(src_points) < 8:
        return None, None

    return (
        np.array(src_points, dtype=np.float32),
        np.array(dst_points, dtype=np.float32),
    )


def _compute_reprojection_error(src: np.ndarray, dst: np.ndarray, H: np.ndarray) -> float:
    pts = src.reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    error = np.linalg.norm(projected - dst, axis=1)
    return float(np.mean(error))


def evaluate_calibration(detections, marker_size_mm, homography: Optional[np.ndarray]):
    if not detections:
        return {
            "ok": False,
            "confidence": 0.0,
            "message": "No markers detected",
            "print_scale_ok": False,
            "print_scale_ratio": None,
            "reprojection_error_px": None,
        }

    sides = [d.side_px for d in detections if d.side_px > 0]
    if not sides:
        return {
            "ok": False,
            "confidence": 0.0,
            "message": "Invalid marker detection",
            "print_scale_ok": False,
            "print_scale_ratio": None,
            "reprojection_error_px": None,
        }

    avg_side_px = sum(sides) / len(sides)

    # Fallback local scale from raw marker side
    raw_mm_per_px = marker_size_mm / avg_side_px if avg_side_px > 0 else 0.0

    reprojection_error_px = None
    confidence = 0.35 + min(len(detections), 4) * 0.1

    if homography is not None:
        src, dst = _build_reprojection_sets(detections)
        if src is not None and dst is not None:
            reprojection_error_px = _compute_reprojection_error(src, dst, homography)

            # Lower reprojection error = better fit
            if reprojection_error_px <= 1.5:
                confidence += 0.40
            elif reprojection_error_px <= 3.0:
                confidence += 0.28
            elif reprojection_error_px <= 6.0:
                confidence += 0.15
            else:
                confidence -= 0.10
        else:
            confidence += 0.10

    # Print-scale sanity check:
    # If the board model is correct, the marker-derived raw scale should be
    # broadly compatible with the rectified board scale.
    rectified_mm_per_px = 1.0 / RECTIFIED_PX_PER_MM
    print_scale_ratio = (
        raw_mm_per_px / rectified_mm_per_px
        if rectified_mm_per_px > 0 else None
    )

    # This ratio is NOT expected to be 1.0 in perspective views, so only use it
    # as a weak warning signal. Big mismatches often indicate wrong print size or
    # wrong board profile.
    print_scale_ok = True
    if print_scale_ratio is not None:
        if print_scale_ratio < 0.45 or print_scale_ratio > 2.2:
            print_scale_ok = False
            confidence -= 0.15

    confidence = max(0.0, min(confidence, 1.0))

    if confidence > 0.85:
        msg = "Calibration OK"
    elif confidence > 0.60:
        msg = "Calibration usable"
    else:
        msg = "Calibration unreliable"

    if not print_scale_ok:
        msg += " | check print scale / board profile"

    return {
        "ok": confidence > 0.60,
        "confidence": confidence,
        "message": msg,
        "print_scale_ok": print_scale_ok,
        "print_scale_ratio": print_scale_ratio,
        "reprojection_error_px": reprojection_error_px,
    }