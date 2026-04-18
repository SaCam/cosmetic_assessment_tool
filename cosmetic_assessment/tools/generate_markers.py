from __future__ import annotations

from pathlib import Path

import cv2
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "tools" else SCRIPT_DIR
OUTPUT_DIR = BASE_DIR / "assets" / "markers"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = list(range(8))
MARKER_SIZE_PX = 1200
EXPORT_SIZES_MM = [15.0, 12.0]
DPI = 300


def mm_to_px(mm: float, dpi: int = DPI) -> int:
    return int(round(mm / 25.4 * dpi))


def main() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    for marker_id in MARKER_IDS:
        img = dictionary.generateImageMarker(marker_id, MARKER_SIZE_PX)
        cv2.imwrite(str(OUTPUT_DIR / f"aruco_4x4_50_id_{marker_id}.png"), img)
        pil_img = Image.fromarray(img).convert("L")
        for size_mm in EXPORT_SIZES_MM:
            px = mm_to_px(size_mm)
            resized = pil_img.resize((px, px), Image.Resampling.NEAREST)
            resized.save(
                OUTPUT_DIR / f"aruco_4x4_50_id_{marker_id}_{str(size_mm).replace('.', '_')}mm.png",
                dpi=(DPI, DPI),
            )
    print(f"Saved markers to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
