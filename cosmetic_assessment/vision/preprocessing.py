from __future__ import annotations

import cv2
import numpy as np


def gray_world_white_balance(image_bgr: np.ndarray) -> np.ndarray:
    img = image_bgr.astype(np.float32)
    mean_b, mean_g, mean_r = cv2.mean(img)[:3]
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    gains = np.array([
        mean_gray / max(mean_b, 1e-6),
        mean_gray / max(mean_g, 1e-6),
        mean_gray / max(mean_r, 1e-6),
    ], dtype=np.float32)
    balanced = img * gains.reshape(1, 1, 3)
    return np.clip(balanced, 0, 255).astype(np.uint8)


def enhance_phone_photo(image_bgr: np.ndarray) -> np.ndarray:
    balanced = gray_world_white_balance(image_bgr)
    denoised = cv2.fastNlMeansDenoisingColored(balanced, None, 3, 3, 7, 21)

    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    contrast = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(contrast, (0, 0), 1.0)
    sharpened = cv2.addWeighted(contrast, 1.15, blur, -0.15, 0)
    return sharpened