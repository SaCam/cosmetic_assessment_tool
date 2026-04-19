from __future__ import annotations

from math import floor
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS_SQUARE = [0, 1, 2, 3]
MARKER_IDS_TRIANGLE = [0, 1, 2]
MARKER_SIZE_PX = 1200
DPI = 300

PAGE_WIDTH_MM = 210.0
PAGE_HEIGHT_MM = 297.0
CARD_WIDTH_MM = 85.60
CARD_HEIGHT_MM = 53.98
CORNER_RADIUS_MM = 3.18
PRINT_MARKER_SIZE_MM = 15.0
OUTER_MARGIN_MM = 4.0
CUT_GAP_MM = 5.0
SHEET_MARGIN_X_MM = 10.0
SHEET_MARGIN_Y_MM = 8.0
LABEL_FONT_PT = 8
NOTE_FONT_PT = 9
SAVE_TRANSPARENT_TEMPLATE = True
SAVE_WHITE_REFERENCE_TEMPLATE = True

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "tools" else SCRIPT_DIR
OUTPUT_DIR = BASE_DIR / "assets" / "templates"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def mm_to_px(mm: float, dpi: int = DPI) -> int:
    return int(round(mm / 25.4 * dpi))


def pt_to_px(pt: float, dpi: int = DPI) -> int:
    return int(round(pt * dpi / 72.0))


def create_marker_image(dictionary, marker_id: int, size_px: int) -> np.ndarray:
    return dictionary.generateImageMarker(marker_id, size_px)


def load_font(size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(candidate, size_px)
        except OSError:
            continue
    return ImageFont.load_default()


def marker_to_rgba(marker: np.ndarray, size_px: int) -> Image.Image:
    marker_img = Image.fromarray(marker).convert("L")
    marker_img = marker_img.resize((size_px, size_px), Image.Resampling.NEAREST)
    arr = np.array(marker_img)
    rgba = np.zeros((size_px, size_px, 4), dtype=np.uint8)
    rgba[arr < 128] = [0, 0, 0, 255]
    rgba[arr >= 128] = [0, 0, 0, 0]
    return Image.fromarray(rgba, mode="RGBA")


def marker_to_white_bg(marker: np.ndarray, size_px: int) -> Image.Image:
    marker_img = Image.fromarray(marker).convert("L")
    marker_img = marker_img.resize((size_px, size_px), Image.Resampling.NEAREST)
    rgb = Image.new("RGB", (size_px, size_px), "white")
    rgb.paste(marker_img)
    return rgb


def make_rounded_mask(size: tuple[int, int], radius_px: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius_px, fill=255)
    return mask


def place_marker(card: Image.Image, draw, dictionary, marker_id: int, center_x: int, center_y: int, marker_print_px: int, label_font, transparent: bool) -> None:
    marker = create_marker_image(dictionary, marker_id, MARKER_SIZE_PX)
    marker_image = marker_to_rgba(marker, marker_print_px) if transparent else marker_to_white_bg(marker, marker_print_px)
    x0 = int(round(center_x - marker_print_px / 2))
    y0 = int(round(center_y - marker_print_px / 2))
    card.paste(marker_image, (x0, y0), marker_image if transparent else None)

    label = f"ID {marker_id}"
    bbox = draw.textbbox((0, 0), label, font=label_font)
    label_w = bbox[2] - bbox[0]
    label_x = center_x - label_w // 2
    label_y = y0 + marker_print_px + mm_to_px(0.8)
    draw.text((label_x, label_y), label, fill=(0, 0, 0, 255) if transparent else "black", font=label_font)


def draw_crosshair(draw, cx: int, cy: int, transparent: bool) -> None:
    guide = mm_to_px(5.0)
    color = (0, 0, 0, 120) if transparent else (120, 120, 120)
    draw.line((cx - guide, cy, cx + guide, cy), fill=color, width=2)
    draw.line((cx, cy - guide, cx, cy + guide), fill=color, width=2)


def compute_square_positions(card_w: int, card_h: int, marker_px: int) -> Sequence[tuple[int, int]]:
    margin = mm_to_px(OUTER_MARGIN_MM)
    return [
        (margin + marker_px // 2, margin + marker_px // 2),
        (card_w - margin - marker_px // 2, margin + marker_px // 2),
        (margin + marker_px // 2, card_h - margin - marker_px // 2),
        (card_w - margin - marker_px // 2, card_h - margin - marker_px // 2),
    ]


def compute_triangle_positions(card_w: int, card_h: int, marker_px: int) -> Sequence[tuple[int, int]]:
    margin = mm_to_px(OUTER_MARGIN_MM)
    return [
        (margin + marker_px // 2, margin + marker_px // 2),
        (card_w - margin - marker_px // 2, margin + marker_px // 2),
        (card_w // 2, card_h - margin - marker_px // 2),
    ]


def build_card_template(layout: str, transparent: bool) -> Image.Image:
    if layout not in {"square", "triangle"}:
        raise ValueError("layout must be 'square' or 'triangle'")

    dictionary = cv2.aruco.getPredefinedDictionary(DICT)
    card_w = mm_to_px(CARD_WIDTH_MM)
    card_h = mm_to_px(CARD_HEIGHT_MM)
    radius_px = mm_to_px(CORNER_RADIUS_MM)
    marker_px = mm_to_px(PRINT_MARKER_SIZE_MM)

    mode = "RGBA" if transparent else "RGB"
    background = (255, 255, 255, 0) if transparent else "white"
    card = Image.new(mode, (card_w, card_h), background)
    draw = ImageDraw.Draw(card, mode)

    if not transparent:
        draw.rounded_rectangle((0, 0, card_w - 1, card_h - 1), radius=radius_px, fill="white", outline="black", width=2)
    else:
        draw.rounded_rectangle((0, 0, card_w - 1, card_h - 1), radius=radius_px, outline=(0, 0, 0, 120), width=2)

    positions = compute_square_positions(card_w, card_h, marker_px) if layout == "square" else compute_triangle_positions(card_w, card_h, marker_px)
    ids = MARKER_IDS_SQUARE if layout == "square" else MARKER_IDS_TRIANGLE

    label_font = load_font(pt_to_px(LABEL_FONT_PT))
    for marker_id, (cx, cy) in zip(ids, positions):
        place_marker(card, draw, dictionary, marker_id, cx, cy, marker_px, label_font, transparent)

    draw_crosshair(draw, card_w // 2, card_h // 2, transparent)

    if transparent:
        rounded = make_rounded_mask((card_w, card_h), radius_px)
        out = Image.new("RGBA", (card_w, card_h), (255, 255, 255, 0))
        out.paste(card, (0, 0), rounded)
        return out
    return card


def compute_sheet_grid() -> tuple[int, int, int, int, int, int]:
    page_w = mm_to_px(PAGE_WIDTH_MM)
    page_h = mm_to_px(PAGE_HEIGHT_MM)
    card_w = mm_to_px(CARD_WIDTH_MM)
    card_h = mm_to_px(CARD_HEIGHT_MM)
    gap_x = mm_to_px(CUT_GAP_MM)
    gap_y = mm_to_px(CUT_GAP_MM)
    margin_x = mm_to_px(SHEET_MARGIN_X_MM)
    margin_y = mm_to_px(SHEET_MARGIN_Y_MM)

    usable_w = page_w - 2 * margin_x
    usable_h = page_h - 2 * margin_y
    cols = max(1, floor((usable_w + gap_x) / (card_w + gap_x)))
    rows = max(1, floor((usable_h + gap_y) / (card_h + gap_y)))

    total_w = cols * card_w + (cols - 1) * gap_x
    total_h = rows * card_h + (rows - 1) * gap_y
    start_x = (page_w - total_w) // 2
    start_y = (page_h - total_h) // 2
    return cols, rows, start_x, start_y, gap_x, gap_y


def build_a4_sheet(layout: str, transparent: bool) -> Image.Image:
    mode = "RGBA" if transparent else "RGB"
    bg = (255, 255, 255, 0) if transparent else "white"
    page_w = mm_to_px(PAGE_WIDTH_MM)
    page_h = mm_to_px(PAGE_HEIGHT_MM)
    page = Image.new(mode, (page_w, page_h), bg)
    draw = ImageDraw.Draw(page, mode)
    note_font = load_font(pt_to_px(NOTE_FONT_PT))

    card = build_card_template(layout, transparent=transparent)
    card_w, card_h = card.size
    cols, rows, start_x, start_y, gap_x, gap_y = compute_sheet_grid()

    guide_color = (0, 0, 0, 90) if transparent else (150, 150, 150)
    for row in range(rows):
        for col in range(cols):
            x = start_x + col * (card_w + gap_x)
            y = start_y + row * (card_h + gap_y)
            page.paste(card, (x, y), card if transparent else None)
            draw.rectangle((x, y, x + card_w - 1, y + card_h - 1), outline=guide_color, width=1)

    footer = (
        f"A4 | {layout} | card {CARD_WIDTH_MM:.2f} x {CARD_HEIGHT_MM:.2f} mm | "
        f"marker {PRINT_MARKER_SIZE_MM:.1f} mm | print at 100% / actual size | "
        f"{cols * rows} cutouts"
    )
    footer_bbox = draw.textbbox((0, 0), footer, font=note_font)
    footer_x = (page_w - (footer_bbox[2] - footer_bbox[0])) // 2
    footer_y = page_h - mm_to_px(6.0)
    draw.text((footer_x, footer_y - (footer_bbox[3] - footer_bbox[1])), footer, fill=guide_color if transparent else "black", font=note_font)
    return page


def save_pdf(image: Image.Image, path: Path) -> None:
    image.convert("RGB").save(path, "PDF", resolution=DPI)


def save_outputs(layout: str) -> None:
    label = f"aruco_a4_credit_card_{layout}"
    if SAVE_WHITE_REFERENCE_TEMPLATE:
        white_sheet = build_a4_sheet(layout, transparent=False)
        white_sheet.save(OUTPUT_DIR / f"{label}_white.png", dpi=(DPI, DPI))
        save_pdf(white_sheet, OUTPUT_DIR / f"{label}_white.pdf")
    if SAVE_TRANSPARENT_TEMPLATE:
        transp_sheet = build_a4_sheet(layout, transparent=True)
        transp_sheet.save(OUTPUT_DIR / f"{label}_transparent.png", dpi=(DPI, DPI))

    # Also save a single true-size card reference for debugging / app docs.
    card_white = build_card_template(layout, transparent=False)
    card_white.save(OUTPUT_DIR / f"aruco_credit_card_{layout}_single_white.png", dpi=(DPI, DPI))


def main() -> None:
    save_outputs("square")
    save_outputs("triangle")
    print(f"Saved A4 sheets to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
