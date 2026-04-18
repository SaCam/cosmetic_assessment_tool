from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ==============================
# Settings
# ==============================
DICT = cv2.aruco.DICT_4X4_50
TOP_MARKER_IDS = [0, 1, 2, 3]
BOTTOM_MARKER_IDS = [4, 5, 6, 7]
MARKER_SIZE_PX = 800              # source resolution for generated markers
PRINT_MARKER_SIZE_MM = 30         # printed marker square size
PAGE_WIDTH_MM = 210               # A4 portrait
PAGE_HEIGHT_MM = 297
HALF_PAGE_HEIGHT_MM = PAGE_HEIGHT_MM / 2
DPI = 300
OUTER_MARGIN_MM = 10              # distance from page edge to marker block
HALF_GAP_MM = 5                   # extra clearance from the center cut line
LABEL_HEIGHT_MM = 7
OUTPUT_DIR = Path("assets/markers")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# When True, create a transparent PNG intended for printing on transparent film.
# Note: marker detection will be less robust on dark or busy surfaces because the
# "white" area around the marker becomes transparent.
SAVE_TRANSPARENT_TEMPLATE = True
SAVE_WHITE_REFERENCE_TEMPLATE = True


# ==============================
# Helpers
# ==============================
def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def create_marker_image(dictionary, marker_id: int, size_px: int) -> np.ndarray:
    return dictionary.generateImageMarker(marker_id, size_px)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "arial.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def marker_to_rgba(marker: np.ndarray, size_px: int) -> Image.Image:
    """
    Convert a generated marker to RGBA.

    Black pixels stay black and opaque.
    White pixels become transparent so the template can be printed on clear film.
    """
    marker_img = Image.fromarray(marker).convert("L")
    marker_img = marker_img.resize((size_px, size_px), Image.Resampling.NEAREST)

    arr = np.array(marker_img)
    rgba = np.zeros((size_px, size_px, 4), dtype=np.uint8)
    rgba[arr < 128] = [0, 0, 0, 255]     # black areas opaque
    rgba[arr >= 128] = [0, 0, 0, 0]      # white areas transparent
    return Image.fromarray(rgba, mode="RGBA")


def marker_to_white_bg(marker: np.ndarray, size_px: int) -> Image.Image:
    marker_img = Image.fromarray(marker).convert("L")
    marker_img = marker_img.resize((size_px, size_px), Image.Resampling.NEAREST)
    rgb = Image.new("RGB", (size_px, size_px), "white")
    rgb.paste(marker_img)
    return rgb


def draw_center_guides(draw: ImageDraw.ImageDraw, page_w: int, y0: int, half_h: int) -> None:
    cx = page_w // 2
    cy = y0 + (half_h // 2)
    guide = mm_to_px(10, DPI)
    color = (0, 0, 0, 90)
    draw.line((cx - guide, cy, cx + guide, cy), fill=color, width=2)
    draw.line((cx, cy - guide, cx, cy + guide), fill=color, width=2)


def paste_half_layout(
    page: Image.Image,
    draw: ImageDraw.ImageDraw,
    dictionary,
    marker_ids: Iterable[int],
    y_offset: int,
    half_height_px: int,
    marker_print_px: int,
    margin_px: int,
    gap_px: int,
    label_font: ImageFont.ImageFont,
    transparent: bool,
) -> None:
    page_w, _ = page.size
    marker_ids = list(marker_ids)
    if len(marker_ids) != 4:
        raise ValueError("Each half layout must contain exactly 4 marker IDs.")

    label_pad = mm_to_px(2, DPI)
    label_height_px = mm_to_px(LABEL_HEIGHT_MM, DPI)

    top_y = y_offset + margin_px
    bottom_y = y_offset + half_height_px - margin_px - gap_px - marker_print_px - label_height_px

    left_x = margin_px
    right_x = page_w - margin_px - marker_print_px

    positions = [
        (left_x, top_y),
        (right_x, top_y),
        (left_x, bottom_y),
        (right_x, bottom_y),
    ]

    for marker_id, (x, y) in zip(marker_ids, positions):
        marker = create_marker_image(dictionary, marker_id, MARKER_SIZE_PX)
        marker_image = (
            marker_to_rgba(marker, marker_print_px)
            if transparent
            else marker_to_white_bg(marker, marker_print_px)
        )
        page.paste(marker_image, (x, y), marker_image if transparent else None)

        label = f"ID {marker_id}"
        bbox = draw.textbbox((0, 0), label, font=label_font)
        label_w = bbox[2] - bbox[0]
        label_x = x + (marker_print_px - label_w) // 2
        label_y = y + marker_print_px + label_pad
        draw.text((label_x, label_y), label, fill=(0, 0, 0, 255), font=label_font)

    # Light center guide to help place the defect in the middle of the marker frame
    draw_center_guides(draw, page_w, y_offset, half_height_px)


def build_template(transparent: bool) -> Image.Image:
    dictionary = cv2.aruco.getPredefinedDictionary(DICT)

    page_w = mm_to_px(PAGE_WIDTH_MM, DPI)
    page_h = mm_to_px(PAGE_HEIGHT_MM, DPI)
    half_h = page_h // 2
    margin_px = mm_to_px(OUTER_MARGIN_MM, DPI)
    gap_px = mm_to_px(HALF_GAP_MM, DPI)
    marker_print_px = mm_to_px(PRINT_MARKER_SIZE_MM, DPI)

    mode = "RGBA" if transparent else "RGB"
    background = (255, 255, 255, 0) if transparent else "white"
    page = Image.new(mode, (page_w, page_h), background)
    draw = ImageDraw.Draw(page, mode)

    label_font = load_font(22)
    note_font = load_font(18)

    # Cut line between the two half-A4 templates
    cut_y = half_h
    cut_color = (0, 0, 0, 120) if transparent else (140, 140, 140)
    draw.line((margin_px, cut_y, page_w - margin_px, cut_y), fill=cut_color, width=2)
    draw.text((page_w - margin_px - 95, cut_y - 26), "CUT", fill=cut_color, font=note_font)

    paste_half_layout(
        page=page,
        draw=draw,
        dictionary=dictionary,
        marker_ids=TOP_MARKER_IDS,
        y_offset=0,
        half_height_px=half_h,
        marker_print_px=marker_print_px,
        margin_px=margin_px,
        gap_px=gap_px,
        label_font=label_font,
        transparent=transparent,
    )

    paste_half_layout(
        page=page,
        draw=draw,
        dictionary=dictionary,
        marker_ids=BOTTOM_MARKER_IDS,
        y_offset=half_h,
        half_height_px=half_h,
        marker_print_px=marker_print_px,
        margin_px=margin_px,
        gap_px=gap_px,
        label_font=label_font,
        transparent=transparent,
    )

    footer = (
        f"Aruco {PRINT_MARKER_SIZE_MM} mm | print at 100% actual size | "
        f"dictionary: DICT_4X4_50"
    )
    footer_y = page_h - mm_to_px(6, DPI)
    draw.text((margin_px, footer_y - 28), footer, fill=(0, 0, 0, 200) if transparent else "black", font=note_font)

    return page


# ==============================
# Main
# ==============================
def main() -> None:
    if SAVE_TRANSPARENT_TEMPLATE:
        transparent_template = build_template(transparent=True)
        transparent_path = OUTPUT_DIR / "aruco_marker_template_half_a4_x2_transparent.png"
        transparent_template.save(transparent_path, dpi=(DPI, DPI))
        print(f"Saved transparent PNG: {transparent_path.resolve()}")

    if SAVE_WHITE_REFERENCE_TEMPLATE:
        white_template = build_template(transparent=False)
        white_png_path = OUTPUT_DIR / "aruco_marker_template_half_a4_x2_white.png"
        white_pdf_path = OUTPUT_DIR / "aruco_marker_template_half_a4_x2_white.pdf"
        white_template.save(white_png_path, dpi=(DPI, DPI))
        white_template.save(white_pdf_path, resolution=DPI)
        print(f"Saved white reference PNG: {white_png_path.resolve()}")
        print(f"Saved white reference PDF: {white_pdf_path.resolve()}")


if __name__ == "__main__":
    main()
