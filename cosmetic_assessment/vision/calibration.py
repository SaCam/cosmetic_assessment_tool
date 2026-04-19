from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from core.settings import (
    CREDIT_CARD_BOARD_MARKER_IDS,
    CREDIT_CARD_SQUARE_MARKER_CENTERS_MM,
    MARKER_SIZE_MM,
    RECTIFIED_PX_PER_MM,
)
from vision.fiducials import FiducialDetection

Point = Tuple[float, float]


@dataclass
class CalibrationResult:
    corrected_bgr: np.ndarray
    rectified: bool
    mm_per_px: float
    method: str
    homography: Optional[np.ndarray] = None


def _average_marker_mm_per_px(detections: list[FiducialDetection]) -> float:
    valid_sides = [d.side_px for d in detections if d.side_px > 0]
    if not valid_sides:
        return 0.0
    avg_side_px = sum(valid_sides) / len(valid_sides)
    return MARKER_SIZE_MM / avg_side_px


def _marker_object_corners_px(marker_id: int) -> np.ndarray:
    cx_mm, cy_mm = CREDIT_CARD_SQUARE_MARKER_CENTERS_MM[marker_id]
    half = MARKER_SIZE_MM / 2.0

    # same corner order as OpenCV ArUco:
    # top-left, top-right, bottom-right, bottom-left
    corners_mm = np.array([
        [cx_mm - half, cy_mm - half],
        [cx_mm + half, cy_mm - half],
        [cx_mm + half, cy_mm + half],
        [cx_mm - half, cy_mm + half],
    ], dtype=np.float32)

    return corners_mm * RECTIFIED_PX_PER_MM


def _build_square_board_homography(detections: list[FiducialDetection]) -> Optional[np.ndarray]:
    used = [d for d in detections if d.marker_id in CREDIT_CARD_BOARD_MARKER_IDS]
    if len(used) < 2:
        return None

    src_points = []
    dst_points = []

    for det in used:
        if det.marker_id not in CREDIT_CARD_SQUARE_MARKER_CENTERS_MM:
            continue

        img_corners = np.array(det.corners, dtype=np.float32)
        obj_corners = _marker_object_corners_px(det.marker_id)

        src_points.extend(img_corners.tolist())
        dst_points.extend(obj_corners.tolist())

    if len(src_points) < 8:
        return None

    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)

    H, mask = cv2.findHomography(src, dst, method=0)
    return H


def calibrate_image_from_fiducials(image_bgr: np.ndarray, detections: list[FiducialDetection]) -> CalibrationResult:
    if not detections:
        return CalibrationResult(
            corrected_bgr=image_bgr.copy(),
            rectified=False,
            mm_per_px=0.0,
            method="no_fiducials",
        )

    H = _build_square_board_homography(detections)
    mm_per_px = _average_marker_mm_per_px(detections)

    if H is not None:
        return CalibrationResult(
            corrected_bgr=image_bgr.copy(),
            rectified=False,
            mm_per_px=mm_per_px,
            method="square_board_homography_ready",
            homography=H,
        )

    return CalibrationResult(
        corrected_bgr=image_bgr.copy(),
        rectified=False,
        mm_per_px=mm_per_px,
        method="single_marker_scale",
    )


def transform_points_with_homography(points_xy: list[tuple[float, float]], H: Optional[np.ndarray]) -> list[tuple[float, float]]:
    if H is None or not points_xy:
        return points_xy

    pts = np.array(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, H)
    warped = warped.reshape(-1, 2)
    return [(float(x), float(y)) for x, y in warped]