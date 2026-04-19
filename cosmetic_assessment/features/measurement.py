from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Optional

import cv2
import numpy as np

Point = Tuple[float, float]


def polyline_length_px(points: Iterable[Point]) -> float:
    pts = list(points)
    if len(pts) < 2:
        return 0.0

    total = 0.0
    for i in range(len(pts) - 1):
        p1 = np.array(pts[i], dtype=np.float32)
        p2 = np.array(pts[i + 1], dtype=np.float32)
        total += float(np.linalg.norm(p2 - p1))
    return total


def distance_px(p1: Point, p2: Point) -> float:
    a = np.array(p1, dtype=np.float32)
    b = np.array(p2, dtype=np.float32)
    return float(np.linalg.norm(b - a))


def polygon_area_px2(points: Sequence[Point]) -> float:
    if len(points) < 3:
        return 0.0
    pts = np.array(points, dtype=np.float32)
    return float(cv2.contourArea(pts))


def px_to_mm(length_px: float, mm_per_px: float) -> float:
    return float(length_px) * float(mm_per_px)


def px2_to_mm2(area_px2: float, mm_per_px: float) -> float:
    return float(area_px2) * float(mm_per_px ** 2)


def transform_points(points: Sequence[Point], homography: Optional[np.ndarray]) -> list[Point]:
    if homography is None or len(points) == 0:
        return [(float(x), float(y)) for x, y in points]

    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in warped]


def measure_polyline_mm(
    points: Sequence[Point],
    mm_per_px: float,
    homography: Optional[np.ndarray] = None,
    rectified_px_per_mm: Optional[float] = None,
) -> float:
    if len(points) < 2:
        return 0.0

    if homography is not None and rectified_px_per_mm:
        warped = transform_points(points, homography)
        return polyline_length_px(warped) / float(rectified_px_per_mm)

    return px_to_mm(polyline_length_px(points), mm_per_px)


def measure_distance_mm(
    p1: Point,
    p2: Point,
    mm_per_px: float,
    homography: Optional[np.ndarray] = None,
    rectified_px_per_mm: Optional[float] = None,
) -> float:
    return measure_polyline_mm(
        [p1, p2],
        mm_per_px=mm_per_px,
        homography=homography,
        rectified_px_per_mm=rectified_px_per_mm,
    )


def measure_polygon_area_mm2(
    points: Sequence[Point],
    mm_per_px: float,
    homography: Optional[np.ndarray] = None,
    rectified_px_per_mm: Optional[float] = None,
) -> float:
    if len(points) < 3:
        return 0.0

    if homography is not None and rectified_px_per_mm:
        warped = transform_points(points, homography)
        return polygon_area_px2(warped) / float(rectified_px_per_mm ** 2)

    return px2_to_mm2(polygon_area_px2(points), mm_per_px)