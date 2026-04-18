from __future__ import annotations

from pathlib import Path

import cv2

OUTPUT_DIR = Path("assets/markers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = list(range(8))
MARKER_SIZE_PX = 800


def main() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)

    for marker_id in MARKER_IDS:
        img = dictionary.generateImageMarker(marker_id, MARKER_SIZE_PX)
        out_path = OUTPUT_DIR / f"aruco_4x4_50_id_{marker_id}.png"
        cv2.imwrite(str(out_path), img)

    print(f"Saved {len(MARKER_IDS)} markers to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
