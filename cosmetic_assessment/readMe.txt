fiducial_measure_app/
│
├─ app.py                  # Tkinter startup
├─ requirements.txt
│
├─ ui/
│  ├─ main_window.py       # menu, canvas, status bar
│  ├─ image_panel.py       # showing image + overlay rendering
│  └─ dialogs.py           # settings/calibration dialogs
│
├─ core/
│  ├─ models.py            # dataclasses for marker detections, measurements
│  ├─ image_store.py       # current image state
│  ├─ pipeline.py          # orchestrates processing steps
│  └─ settings.py          # app settings / defaults
│
├─ vision/
│  ├─ fiducials.py         # ArUco / ChArUco detection
│  ├─ calibration.py       # camera calibration / undistortion
│  ├─ geometry.py          # perspective transforms, distances, scale
│  ├─ overlays.py          # draw boxes, ids, dimensions
│  └─ preprocessing.py     # grayscale, thresholding, denoise, etc.
│
├─ features/
│  ├─ measure_object.py    # size estimation from fiducials
│  ├─ scratch_measure.py   # future scratch segmentation / measuring
│  └─ damage_classifier.py # future ML hooks
│
└─ assets/
   └─ markers/             # generated fiducial marker images / test images


External config paths
---------------------
You can now change these three paths in core/settings.py:
- SAVE_PATH
- SPEC_PATH
- VERSION_CHECK_PATH

Example:
SAVE_PATH = r"C:\CosmeticTool\Inspections"
SPEC_PATH = r"C:\CosmeticTool\config\specs.json"
VERSION_CHECK_PATH = r"C:\CosmeticTool\config\version_check.json"

Behavior:
- Save inspections now go directly to SAVE_PATH.
- Specs are loaded from SPEC_PATH when that JSON file exists.
- About reads VERSION_CHECK_PATH and shows whether the app is up to date.
- If a configured file is missing or invalid, the app safely falls back to built-in defaults.

Example files are included in config_examples/.
