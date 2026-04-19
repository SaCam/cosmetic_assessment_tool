import cv2
import numpy as np


def detect_scratch_candidates(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Contrast boost
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Light blur to suppress tiny texture noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Scratches can be dark or bright depending on lighting
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 3))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel_line)
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel_line)

    # Edge map helps with gouges / sharp scratch borders
    edges = cv2.Canny(blurred, 40, 120)

    # Combine responses
    combined = cv2.max(blackhat, tophat)
    combined = cv2.max(combined, edges)

    # Threshold
    _, thresh = cv2.threshold(combined, 18, 255, cv2.THRESH_BINARY)

    # Connect broken scratch segments
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    # Remove tiny noise
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel, iterations=1)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        length = max(w, h)
        thickness = max(1, min(w, h))
        aspect_ratio = length / thickness

        if length < 40:
            continue

        if aspect_ratio < 2.5:
            continue

        # Prefer long, thin contours
        score = (length * aspect_ratio) + (0.2 * area)
        candidates.append((score, cnt))

    candidates.sort(key=lambda item: item[0], reverse=True)

    return [cnt for score, cnt in candidates], thresh, combined