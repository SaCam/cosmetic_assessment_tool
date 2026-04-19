from __future__ import annotations

import cv2
import numpy as np

from vision.fiducials import FiducialDetection


def draw_fiducial_detections(
    image_bgr: np.ndarray,
    detections: list[FiducialDetection],
) -> np.ndarray:
    output = image_bgr.copy()

    for det in detections:
        pts = np.array(det.corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(output, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        top_left = tuple(map(int, det.corners[0]))
        label = f"ID {det.marker_id} | {det.side_px:.1f}px"
        cv2.putText(
            output,
            label,
            (top_left[0], max(20, top_left[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return output