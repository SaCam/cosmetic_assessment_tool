from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


@dataclass
class FiducialDetection:
    marker_id: int
    corners: list[Point]
    side_px: float


def _build_detector() -> cv2.aruco.ArucoDetector:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 35
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.minMarkerPerimeterRate = 0.01
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.04
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def detect_aruco_markers(image_bgr: np.ndarray) -> List[FiducialDetection]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    detector = _build_detector()

    corners, ids, _rejected = detector.detectMarkers(gray)

    detections: List[FiducialDetection] = []

    if ids is None or len(ids) == 0:
        return detections

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        pts = marker_corners[0]
        side_lengths = [float(np.linalg.norm(pts[i] - pts[(i + 1) % 4])) for i in range(4)]
        avg_side = sum(side_lengths) / 4.0
        detections.append(FiducialDetection(
            marker_id=int(marker_id),
            corners=[(float(x), float(y)) for x, y in pts],
            side_px=avg_side,
        ))

    detections.sort(key=lambda d: d.marker_id)
    return detections