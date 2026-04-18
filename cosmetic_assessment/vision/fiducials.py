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


def detect_aruco_markers(image_bgr: np.ndarray) -> List[FiducialDetection]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    corners, ids, _rejected = detector.detectMarkers(gray)

    detections: List[FiducialDetection] = []

    if ids is None or len(ids) == 0:
        return detections

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        pts = marker_corners[0]  # shape (4, 2)

        side_lengths = [
            float(np.linalg.norm(pts[i] - pts[(i + 1) % 4]))
            for i in range(4)
        ]
        avg_side = sum(side_lengths) / 4.0

        detections.append(
            FiducialDetection(
                marker_id=int(marker_id),
                corners=[(float(x), float(y)) for x, y in pts],
                side_px=avg_side,
            )
        )

    return detections

