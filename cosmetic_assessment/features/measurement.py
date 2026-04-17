from typing import Iterable, Tuple
import numpy as np

Point = Tuple[float, float]

def polyline_length_px(points: Iterable[Point]) -> float:
    pts = list(points)
    if len(pts) < 2:
        return 0.0

    total = 0.0
    for i in range(len(pts) - 1):
        p1 = np.array(pts[i], dtype=float)
        p2 = np.array(pts[i + 1], dtype=float)
        total += np.linalg.norm(p2 - p1)
    return total


def px_to_mm(length_px: float, mm_per_px: float) -> float:
    return length_px * mm_per_px