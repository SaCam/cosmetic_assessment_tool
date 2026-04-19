from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class ImageState:
    path: Optional[Path] = None
    source_bgr: Optional[np.ndarray] = None
    original_bgr: Optional[np.ndarray] = None
    display_bgr: Optional[np.ndarray] = None

    def is_loaded(self) -> bool:
        return self.original_bgr is not None

    def clear(self) -> None:
        self.path = None
        self.source_bgr = None
        self.original_bgr = None
        self.display_bgr = None

    def load_from_file(self, file_path: str) -> None:
        pil_img = Image.open(file_path)
        pil_img = ImageOps.exif_transpose(pil_img)
        rgb = np.array(pil_img.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        self.path = Path(file_path)
        self.source_bgr = bgr
        self.original_bgr = bgr.copy()
        self.display_bgr = bgr.copy()

    def set_working_image(self, image_bgr: np.ndarray) -> None:
        self.original_bgr = image_bgr.copy()
        self.display_bgr = image_bgr.copy()

    def reset_display(self) -> None:
        if self.original_bgr is not None:
            self.display_bgr = self.original_bgr.copy()