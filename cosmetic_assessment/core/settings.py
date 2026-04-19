from __future__ import annotations

import json
from pathlib import Path
from typing import Any

APP_TITLE = "Cosmetic Analysis Tool"
APP_VERSION = "0.5.0"

WINDOW_GEOMETRY = "1440x900"
WINDOW_MIN_SIZE = (1120, 760)

ARUCO_DICT_NAME = "DICT_4X4_50"

ACTIVE_BOARD_PROFILE = "credit_card_v1"

BOARD_PROFILES = {
    "credit_card_v1": {
        "name": "Credit card v1",
        "marker_size_mm": 14.65,
        "marker_ids": [0, 1, 2, 3],
        "marker_centers_mm": {
            0: (0.0, 0.0),       # top-left
            1: (60.35, 0.0),     # top-right
            2: (0.0, 30.35),     # bottom-left
            3: (60.35, 30.35),   # bottom-right
        },
        "rectified_px_per_mm": 20.0,
    },
}

BOARD = BOARD_PROFILES[ACTIVE_BOARD_PROFILE]

MARKER_SIZE_MM = BOARD["marker_size_mm"]
CREDIT_CARD_BOARD_MARKER_IDS = BOARD["marker_ids"]
CREDIT_CARD_SQUARE_MARKER_CENTERS_MM = BOARD["marker_centers_mm"]
RECTIFIED_PX_PER_MM = BOARD["rectified_px_per_mm"]

DEFAULT_SAVE_ROOT = "inspections"
CREATE_UNIQUE_SAVE_FOLDER = True

SCRATCH_MIN_SAMPLE_SPACING_MM = 3.0

EXPORT_WRITE_ORIGINAL = True
EXPORT_WRITE_OVERLAY = True
EXPORT_WRITE_JSON = True


# ============================================================================
# External paths that can be changed without updating the application code
# ============================================================================
# Set these to a shared folder, network drive, SharePoint-synced folder, etc.
# Leave empty (None or "") to use the built-in fallback behavior.
SAVE_PATH = "C:\\Users\\Sam\\Downloads"
SPEC_PATH = "C:\\Users\\Sam\\Downloads\\specs.json"
VERSION_CHECK_PATH = "C:\\Users\\Sam\\Downloads\\version.json"


SETTINGS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SETTINGS_DIR.parent


def _clean_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser()


def resolve_config_path(value: str | Path | None) -> Path | None:
    """Resolve a configured path.

    Relative paths are resolved from the project folder so the app behaves
    consistently whether it is started from the repo root or another folder.
    """
    path = _clean_path(value)
    if path is None:
        return None
    if path.is_absolute():
        return path
    return (PROJECT_DIR / path).resolve()


def read_json_file(path: str | Path | None) -> dict[str, Any] | None:
    resolved = resolve_config_path(path)
    if resolved is None or not resolved.exists() or not resolved.is_file():
        return None

    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def get_save_root() -> Path:
    configured = resolve_config_path(SAVE_PATH)
    if configured is not None:
        return configured
    return (PROJECT_DIR / DEFAULT_SAVE_ROOT).resolve()


def get_spec_path() -> Path | None:
    return resolve_config_path(SPEC_PATH)


def get_version_check_path() -> Path | None:
    return resolve_config_path(VERSION_CHECK_PATH)


def get_latest_version_info() -> dict[str, Any]:
    """Read an external version info file when configured.

    Supported formats:
    1) JSON file, for example:
       {
         "latest_version": "0.6.0",
         "minimum_version": "0.5.0",
         "download_path": "\\\\server\\tools\\CosmeticTool",
         "notes": "Scratch and scuff improvements"
       }
    2) Plain text file containing only the latest version string.
    """
    path = get_version_check_path()
    if path is None or not path.exists() or not path.is_file():
        return {
            "configured": path is not None,
            "path": str(path) if path else None,
            "latest_version": None,
            "minimum_version": None,
            "download_path": None,
            "notes": None,
            "status": "missing" if path else "not_configured",
        }

    if path.suffix.lower() == ".json":
        data = read_json_file(path)
        if data is None:
            return {
                "configured": True,
                "path": str(path),
                "latest_version": None,
                "minimum_version": None,
                "download_path": None,
                "notes": None,
                "status": "invalid",
            }

        return {
            "configured": True,
            "path": str(path),
            "latest_version": str(data.get("latest_version", "")).strip() or None,
            "minimum_version": str(data.get("minimum_version", "")).strip() or None,
            "download_path": str(data.get("download_path", "")).strip() or None,
            "notes": str(data.get("notes", "")).strip() or None,
            "status": "ok",
        }

    try:
        latest_version = path.read_text(encoding="utf-8").strip()
    except OSError:
        return {
            "configured": True,
            "path": str(path),
            "latest_version": None,
            "minimum_version": None,
            "download_path": None,
            "notes": None,
            "status": "invalid",
        }

    return {
        "configured": True,
        "path": str(path),
        "latest_version": latest_version or None,
        "minimum_version": None,
        "download_path": None,
        "notes": None,
        "status": "ok" if latest_version else "invalid",
    }
