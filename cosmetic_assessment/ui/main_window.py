
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from datetime import datetime
from uuid import uuid4

from core.image_store import ImageState
from features.defect_modes import DEFECT_MODES
from features.scratch_detection import detect_scratch_candidates
from features.specs import evaluate_results
from vision.fiducials import detect_aruco_markers
from vision.preprocessing import enhance_phone_photo
from core.settings import RECTIFIED_PX_PER_MM
from vision.calibration import calibrate_image_from_fiducials
from features.measurement import (
    measure_polyline_mm,
    measure_distance_mm,
    measure_polygon_area_mm2,
)
from vision.validation import evaluate_calibration

from core.settings import (
    APP_TITLE,
    APP_VERSION,
    WINDOW_GEOMETRY,
    WINDOW_MIN_SIZE,
    MARKER_SIZE_MM,
    DEFAULT_SAVE_ROOT,
    CREATE_UNIQUE_SAVE_FOLDER,
    get_latest_version_info,
    get_save_root,
    get_spec_path,
    get_version_check_path,
)



class MainWindow:
    DEFECT_TYPES = ("Scratch", "Scuff", "Dent")

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(WINDOW_GEOMETRY)
        self.root.minsize(*WINDOW_MIN_SIZE)

        self.image_state = ImageState()
        self.defect_modes = DEFECT_MODES
        self.tk_image = None

        self.status_var = tk.StringVar(value="Ready")
        self.verdict_var = tk.StringVar(value="Unreviewed")

        self.detections = []
        self.mm_per_px = None
        self.annotation_active = False
        self.annotation_step = None
        self.scratch_length_points = []
        self.scratch_width_points = []
        self.scratch_width_mm = None
        self.measure_points = []
        self.defect_types = ("Scratch", "Scuff", "Dent")
        self.current_defect_type = tk.StringVar(value="Scratch")
        self.instructions_var = tk.StringVar(value="Select a defect type and mark the defect.")
        self.current_measurement_mm = None
        self.current_area_mm2 = None
        self.current_diameter_mm = None
        self.last_saved_path = None
        self.scratch_candidates = []
        self.scratch_width_samples = []
        self.scratch_width_preview = None
        self.scratch_width_anchor = None
        self.mouse_image_point = None
        self.calibration_method = None
        self.calibration_homography = None
        self.scratch_width_can_place = True
        self.calibration_confidence = 0.0
        self.calibration_reprojection_error_px = None
        self.print_scale_ok = True
        self.print_scale_ratio = None

        self.saved_inspection = False

        self.render_scale = 1.0
        self.render_offset_x = 0
        self.render_offset_y = 0
        self.rendered_width = 0
        self.rendered_height = 0

        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None
        self.is_selecting_roi = False

        self.zoom_factor = 1.0
        self.min_zoom = 0.2
        self.max_zoom = 8.0

        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False
        self.pan_start_mouse = None
        self.pan_start_offset = None

        self.current_spec_name = tk.StringVar(value="none")
        self.spec_result = None
        self.version_info = get_latest_version_info()

        self._build_menu()
        self._build_layout()
        self._show_empty_canvas_message()
        self._refresh_ui_state()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image...", command=self.open_image)
        file_menu.add_command(label="Save Inspection...", command=self.save_inspection)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        actions_menu = tk.Menu(menubar, tearoff=0)
        actions_menu.add_command(label="Reset View", command=self.reset_view)
        actions_menu.add_command(label="Detect Fiducials", command=self.detect_fiducials)
        actions_menu.add_command(label="Select ROI", command=self.start_roi_selection)
        actions_menu.add_command(label="Finish Annotation", command=self.finish_annotation)
        actions_menu.add_command(label="Undo Point", command=self.undo_last_point)
        actions_menu.add_separator()
        actions_menu.add_command(label="Detect Scratch Candidates", command=self.detect_scratch)
        actions_menu.add_command(label="Save Inspection...", command=self.save_inspection)
        actions_menu.add_separator()
        actions_menu.add_command(label="Zoom In", command=self.zoom_in)
        actions_menu.add_command(label="Zoom Out", command=self.zoom_out)
        actions_menu.add_command(label="Reset Zoom", command=self.reset_zoom)
        actions_menu.add_separator()
        actions_menu.add_command(label="Clear Measurement", command=self.cancel_measurement)
        actions_menu.add_command(label="Clear Image", command=self.clear_image)
        menubar.add_cascade(label="Actions", menu=actions_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def _build_layout(self) -> None:
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.main_frame = tk.Frame(self.root, bg="#d9d9d9")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(0, weight=0)  # sidebar
        self.main_frame.columnconfigure(1, weight=1)  # image area

        # --- Sidebar container ---
        sidebar_container = tk.Frame(self.main_frame, bg="#f3f3f3", width=220)
        sidebar_container.grid(row=0, column=0, sticky="ns", padx=(8, 4), pady=8)
        sidebar_container.grid_propagate(False)

        # --- Canvas for scrolling ---
        self.sidebar_canvas = tk.Canvas(
            sidebar_container,
            bg="#f3f3f3",
            highlightthickness=0,
        )
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)

        # --- Scrollbar ---
        scrollbar = tk.Scrollbar(
            sidebar_container,
            orient="vertical",
            command=self.sidebar_canvas.yview,
        )
        scrollbar.pack(side="right", fill="y")

        self.sidebar_canvas.configure(yscrollcommand=scrollbar.set)

        # --- Actual sidebar content frame ---
        self.sidebar = tk.Frame(self.sidebar_canvas, bg="#f3f3f3")

        self.sidebar_window = self.sidebar_canvas.create_window(
            (0, 0),
            window=self.sidebar,
            anchor="nw"
        )

        def _on_sidebar_canvas_configure(event):
            self.sidebar_canvas.itemconfigure(self.sidebar_window, width=event.width)

        self.sidebar_canvas.bind("<Configure>", _on_sidebar_canvas_configure)

        # --- Make scrolling work ---
        def _on_sidebar_configure(event):
            self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

        self.sidebar.bind("<Configure>", _on_sidebar_configure)

        def _on_sidebar_mousewheel(event):
            self.sidebar_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_mousewheel_recursive(widget):
            widget.bind("<MouseWheel>", _on_sidebar_mousewheel)
            for child in widget.winfo_children():
                _bind_mousewheel_recursive(child)


        title_lbl = tk.Label(
            self.sidebar,
            text="Cosmetic Analysis Tool",
            font=("Arial", 13, "bold"),
            bg="#f3f3f3",
            anchor="w",
        )
        title_lbl.pack(fill="x", pady=(8, 6), padx=10)

        defect_lbl = tk.Label(
            self.sidebar,
            text="Defect type",
            font=("Arial", 10, "bold"),
            bg="#f3f3f3",
            anchor="w",
        )
        defect_lbl.pack(fill="x", padx=10)

        self.defect_combo = ttk.Combobox(
            self.sidebar,
            textvariable=self.current_defect_type,
            values=self.defect_types,
            state="readonly",
        )
        self.defect_combo.pack(fill="x", padx=10, pady=(4, 12))
        self.defect_combo.bind("<<ComboboxSelected>>", self.on_defect_type_changed)

        side_title = tk.Label(
            self.sidebar,
            text="System side",
            font=("Arial", 10, "bold"),
            bg="#f3f3f3",
            anchor="w",
        )
        side_title.pack(fill="x", padx=10, pady=(0, 0))

        self.side_outside = tk.Radiobutton(
            self.sidebar,
            text="Outside (visible when closed)",
            variable=self.current_spec_name,
            value="outside",
            command=self.on_side_changed,
            bg="#f3f3f3",
            anchor="w",
            justify="left",
        )
        self.side_outside.pack(fill="x", padx=10)

        self.side_inside = tk.Radiobutton(
            self.sidebar,
            text="Inside (everything else)",
            variable=self.current_spec_name,
            value="inside",
            command=self.on_side_changed,
            bg="#f3f3f3",
            anchor="w",
            justify="left",
        )
        self.side_inside.pack(fill="x", padx=10, pady=(0, 12))

        instructions_title = tk.Label(
            self.sidebar,
            text="Instructions",
            font=("Arial", 10, "bold"),
            bg="#f3f3f3",
            anchor="w",
        )
        instructions_title.pack(fill="x", padx=10)

        self.instructions_label = tk.Label(
            self.sidebar,
            textvariable=self.instructions_var,
            justify="left",
            wraplength=180,
            bg="#f3f3f3",
            anchor="nw",
        )
        self.instructions_label.pack(fill="x", padx=10, pady=(4, 12))

        self.btn_restart = tk.Button(
            self.sidebar,
            text="Clear Annotation",
            command=self.cancel_measurement,
        )
        self.btn_restart.pack(fill="x", padx=10, pady=(2, 4))

        self.btn_detect_fiducials = tk.Button(
            self.sidebar,
            text="Redetect Fiducials",
            command=self.detect_fiducials,
        )
        self.btn_detect_fiducials.pack(fill="x", padx=10, pady=(0, 4))

        self.btn_select_roi = tk.Button(
            self.sidebar,
            text="Select ROI",
            command=self.start_roi_selection,
        )
        self.btn_select_roi.pack(fill="x", padx=10, pady=(0, 4))

        self.btn_scratch_detect = tk.Button(
            self.sidebar,
            text="Detect Scratch in ROI",
            command=self.detect_scratch,
        )
        self.btn_scratch_detect.pack(fill="x", padx=10, pady=(0, 4))

        review_title = tk.Label(
            self.sidebar,
            text="Review & Save",
            font=("Arial", 10, "bold"),
            bg="#f3f3f3",
            anchor="w",
        )
        review_title.pack(fill="x", padx=10, pady=(12, 0))

        self.result_box = tk.Frame(
            self.sidebar,
            bg="white",
            highlightbackground="#d1d5db",
            highlightthickness=1,
            bd=0,
        )
        self.result_box.pack(fill="x", padx=10, pady=(4, 8))

        self.result_var = tk.StringVar(value="No result yet")
        self.result_label = tk.Label(
            self.result_box,
            textvariable=self.result_var,
            justify="left",
            wraplength=175,
            bg="white",
            fg="#111827",
            anchor="nw",
            font=("Arial", 10, "bold"),
            padx=8,
            pady=8,
        )
        self.result_label.pack(fill="x")

        spec_title = tk.Label(
            self.sidebar,
            text="Spec Check",
            font=("Arial", 10, "bold"),
            bg="#f3f3f3",
            anchor="w",
        )
        spec_title.pack(fill="x", padx=10, pady=(8, 0))

        self.spec_box = tk.Frame(
            self.sidebar,
            bg="white",
            highlightbackground="#d1d5db",
            highlightthickness=1,
            bd=0,
        )
        self.spec_box.pack(fill="x", padx=10, pady=(4, 8))

        self.spec_name_var = tk.StringVar(value="Applied spec: -")
        self.spec_name_label = tk.Label(
            self.spec_box,
            textvariable=self.spec_name_var,
            bg="white",
            fg="#111827",
            anchor="w",
            justify="left",
            font=("Arial", 9, "bold"),
            padx=8,
            pady=6,
        )
        self.spec_name_label.pack(fill="x")

        self.spec_suggestion_var = tk.StringVar(value="Suggested verdict: -")
        self.spec_suggestion_label = tk.Label(
            self.spec_box,
            textvariable=self.spec_suggestion_var,
            bg="white",
            fg="#111827",
            anchor="w",
            justify="left",
            wraplength=175,
            font=("Arial", 12, "bold"),
            padx=8,
            pady=4,
        )
        self.spec_suggestion_label.pack(fill="x")

        self.spec_reason_var = tk.StringVar(value="No spec evaluation yet")
        self.spec_reason_label = tk.Label(
            self.spec_box,
            textvariable=self.spec_reason_var,
            bg="white",
            fg="#374151",
            anchor="w",
            justify="left",
            wraplength=175,
            font=("Arial", 9),
            padx=8,
            pady=6,
        )
        self.spec_reason_label.pack(fill="x")

        self.verdict_ok = tk.Radiobutton(
            self.sidebar,
            text="OK (accepted)",
            variable=self.verdict_var,
            value="OK",
            command=self.on_verdict_changed,
            bg="#f3f3f3",
            anchor="w",
        )
        self.verdict_ok.pack(fill="x", padx=10)

        self.verdict_nok = tk.Radiobutton(
            self.sidebar,
            text="NOK (not accepted)",
            variable=self.verdict_var,
            value="NOK",
            command=self.on_verdict_changed,
            bg="#f3f3f3",
            anchor="w",
        )
        self.verdict_nok.pack(fill="x", padx=10)

        self.btn_save_inspection = tk.Button(
            self.sidebar,
            text="Save Inspection",
            command=self.save_inspection,
        )
        self.btn_save_inspection.pack(fill="x", padx=10, pady=(8, 4))

        _bind_mousewheel_recursive(self.sidebar)
        _bind_mousewheel_recursive(self.sidebar_canvas)

        # ---------- Image area ----------
        self.image_canvas = tk.Canvas(
            self.main_frame,
            bg="#1e1e1e",
            highlightthickness=0,
        )
        self.image_canvas.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)

        self.canvas_image_id = None

        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            padx=10,
            pady=6,
            bg="#efefef",
        )
        self.status_bar.grid(row=1, column=0, sticky="ew")

        self.calibration_label = tk.Label(
            self.sidebar,
            text="Calibration: -",
            bg="#f3f3f3",
            fg="#6b7280",
            anchor="w",
            justify="left",
            wraplength=180,
            font=("Arial", 8, "bold"),
        )
        self.calibration_label.pack(fill="x", padx=10, pady=(0, 8))

        self.root.bind("<Configure>", self._on_resize)

        self.image_canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.image_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.image_canvas.bind("<Button-3>", self.on_right_click)

        self.image_canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.image_canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.image_canvas.bind("<Button-5>", self.on_mouse_wheel)

        self.image_canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.image_canvas.bind("<B2-Motion>", self.on_pan_drag)
        self.image_canvas.bind("<ButtonRelease-2>", self.on_pan_end)

        self.image_canvas.bind("<Motion>", self.on_mouse_move)
        self.root.bind("<Escape>", self.cancel_measurement)

        self.on_defect_type_changed()

    def _make_card(self, parent: tk.Widget, title: str) -> tk.Frame:
        outer = tk.Frame(parent, bg="white", highlightbackground="#dde1e6", highlightthickness=1)
        outer.columnconfigure(0, weight=1)
        tk.Label(outer, text=title, font=("Arial", 11, "bold"), bg="white", fg="#111827", anchor="w").grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))
        content = tk.Frame(outer, bg="white")
        content.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 10))
        return content

    def _build_instruction_text(self) -> str:
        return self._current_mode().instruction_text(self.annotation_step)

    def on_defect_type_changed(self, event=None):
        mode = self._current_mode()

        self.measure_points = []
        self.annotation_active = True
        self.saved_inspection = False
        self.verdict_var.set("Unreviewed")

        self.current_measurement_mm = None
        self.current_area_mm2 = None
        self.current_diameter_mm = None
        self.scratch_width_mm = None

        self.spec_result = None

        self.annotation_step = mode.start_step()
        self.instructions_var.set(mode.instruction_text(self.annotation_step))
        self.status_var.set(f"{mode.name} annotation active")

        if self.image_state.is_loaded():
            self._redraw_overlay_base()

        self._refresh_ui_state()

    def on_verdict_changed(self) -> None:
        verdict = self.verdict_var.get()
        if verdict in ("OK", "NOK"):
            self.status_var.set(f"Operator verdict set to {verdict}. Ready to save.")
        else:
            self.status_var.set("Choose OK or NOK to continue.")
        self._refresh_ui_state()

    def _current_stage_text(self) -> str:
        if not self.image_state.is_loaded():
            return "Load an image to start"
        if not self.detections:
            return "Auto-detect fiducials"
        if self.annotation_active:
            return f"Marking {self.current_defect_type.get().lower()}"
        if self.current_measurement_mm is None and self.current_area_mm2 is None and self.current_diameter_mm is None:
            return f"Choose {self.current_defect_type.get().lower()} points"
        if self.verdict_var.get() not in ("OK", "NOK"):
            return "Review result and choose OK / NOK"
        if not self.saved_inspection:
            return "Ready to save inspection"
        return "Inspection saved"

    def _refresh_ui_state(self) -> None:
        has_image = self.image_state.is_loaded()
        has_scale = self.mm_per_px is not None
        has_points = len(self.measure_points) > 0

        # Optional buttons: only configure them if they exist in this layout
        if hasattr(self, "btn_detect_fiducials"):
            self.btn_detect_fiducials.config(state=tk.NORMAL if has_image else tk.DISABLED)

        if hasattr(self, "btn_select_roi"):
            self.btn_select_roi.config(state=tk.NORMAL if has_image else tk.DISABLED)

        if hasattr(self, "btn_scratch_detect"):
            self.btn_scratch_detect.config(state=tk.NORMAL if has_image else tk.DISABLED)

        if hasattr(self, "btn_restart"):
            self.btn_restart.config(state=tk.NORMAL if has_points else tk.DISABLED)

        # if hasattr(self, "btn_start_annotation"):
        #     can_annotate = has_image and has_scale
        #     self.btn_start_annotation.config(state=tk.NORMAL if can_annotate else tk.DISABLED)

        if hasattr(self, "btn_finish_annotation"):
            defect = self.current_defect_type.get() if hasattr(self, "current_defect_type") else "Scratch"

            mode = self._current_mode()
            min_points = mode.min_points(self.annotation_step)

            self.btn_finish_annotation.config(
                state=tk.NORMAL if len(self.measure_points) >= min_points else tk.DISABLED
            )
            
        self._update_result_summary()

        has_result = (
            self.current_measurement_mm is not None
            or self.current_area_mm2 is not None
            or self.current_diameter_mm is not None
        )

        if hasattr(self, "verdict_ok"):
            self.verdict_ok.config(state=tk.NORMAL if has_result else tk.DISABLED)
        if hasattr(self, "verdict_nok"):
            self.verdict_nok.config(state=tk.NORMAL if has_result else tk.DISABLED)

        if hasattr(self, "btn_save_inspection"):
            self.btn_save_inspection.config(
                state=tk.NORMAL if has_result and self.verdict_var.get() in ("OK", "NOK") else tk.DISABLED
            )
        
        self._evaluate_current_spec()

    def _set_step_state(self, key: str, done: bool) -> None:
        label = self.step_labels[key]
        clean_text = label.cget("text")[2:]
        label.config(text=("● " if done else "○ ") + clean_text, fg="#166534" if done else "#374151")

    def _show_empty_canvas_message(self) -> None:
        self.image_canvas.delete("all")
        self.image_canvas.create_text(30, 30, anchor="nw", text="No image loaded\n\nUse 'Open Image' to start the inspection workflow.", fill="white", font=("Arial", 16))

    def open_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            self.image_state.load_from_file(file_path)

            if self.image_state.source_bgr is not None:
                enhanced = enhance_phone_photo(self.image_state.source_bgr)
                self.image_state.set_working_image(enhanced)

            self._reset_analysis_state(keep_image=True)
            self._render_current_image()
            self.detect_fiducials()

            if self.mm_per_px is not None:
                self.on_defect_type_changed()
                self.status_var.set(
                    f"Image loaded: {Path(file_path).name}. Fiducials detected automatically. "
                    f"{self.current_defect_type.get()} mode is active."
                )
            else:
                self.status_var.set(
                    f"Image loaded: {Path(file_path).name}. Auto fiducial detection failed; use Redetect Fiducials."
                )

        except Exception as exc:
            messagebox.showerror("Error", f"Could not open image.\n\n{exc}")
            self.status_var.set("Failed to load image")

        self._refresh_ui_state()

    def _reset_analysis_state(self, keep_image: bool = False) -> None:
        self.detections = []
        self.mm_per_px = None
        self.measure_points = []
        self.current_measurement_mm = None
        self.current_area_mm2 = None
        self.current_diameter_mm = None
        self.last_saved_path = None
        self.scratch_candidates = []
        self.scratch_width_samples = []
        self.saved_inspection = False
        self.current_result = None
        self.verdict_var.set("Unreviewed")
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None
        self.is_selecting_roi = False
        self.annotation_active = False
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.scratch_width_mm = None
        self.scratch_length_points = []
        self.scratch_width_points = []
        self.scratch_width_mm = None
        self.spec_result = None
        self.current_spec_name.set("none")
        self.scratch_width_anchor = None
        self.scratch_width_preview = None
        self.mouse_image_point = None
        self.scratch_width_can_place = True
        if keep_image and self.image_state.original_bgr is not None:
            self.image_state.display_bgr = self.image_state.original_bgr.copy()

    def reset_view(self) -> None:
        if not self.image_state.is_loaded():
            return
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._redraw_overlay_base()
        self.status_var.set("View reset")
        self._refresh_ui_state()

    def detect_fiducials(self) -> None:
        if not self.image_state.is_loaded():
            messagebox.showinfo("No image", "Please open an image first.")
            return

        # Detect on the current working full image
        self.detections = detect_aruco_markers(self.image_state.original_bgr)
        calibration = calibrate_image_from_fiducials(self.image_state.original_bgr, self.detections)

        self.mm_per_px = calibration.mm_per_px if calibration.mm_per_px > 0 else None
        self.calibration_method = calibration.method
        self.calibration_homography = calibration.homography

        # IMPORTANT:
        # keep showing the full image, do NOT replace it with the rectified board image
        self.image_state.reset_display()

        self._redraw_overlay_base()

        if self.mm_per_px is not None:
            self.status_var.set(
                f"Detected {len(self.detections)} fiducial marker(s). "
                f"Calibration: {self.calibration_method}. Next: choose defect type and mark the defect."
            )
        else:
            self.status_var.set("No fiducial markers detected")

        validation = evaluate_calibration(
            self.detections,
            MARKER_SIZE_MM,
            self.calibration_homography,
        )

        self.calibration_confidence = validation["confidence"]
        self.calibration_reprojection_error_px = validation.get("reprojection_error_px")
        self.print_scale_ok = validation.get("print_scale_ok", True)
        self.print_scale_ratio = validation.get("print_scale_ratio")

        color = (
            "green" if self.calibration_confidence > 0.85 else
            "orange" if self.calibration_confidence > 0.60 else
            "red"
        )

        label_text = f"Calibration: {self.calibration_confidence:.2f}"
        if self.calibration_reprojection_error_px is not None:
            label_text += f" | err {self.calibration_reprojection_error_px:.2f}px"
        if not self.print_scale_ok:
            label_text += " | check print"

        self.calibration_label.config(
            text=label_text,
            fg=color
        )

        status_text = f"Detected {len(self.detections)} markers | {validation['message']}"
        if self.calibration_reprojection_error_px is not None:
            status_text += f" | reproj {self.calibration_reprojection_error_px:.2f}px"

        self.status_var.set(status_text)

        self._refresh_ui_state()

    def clear_image(self) -> None:
        self.image_state.clear()
        self.tk_image = None
        self.image_canvas.delete("all")
        self._show_empty_canvas_message()
        self._reset_analysis_state(keep_image=False)
        self.status_var.set("Image cleared")
        self._refresh_ui_state()

    def show_about(self) -> None:
        latest_version = self.version_info.get("latest_version")
        version_status = self.version_info.get("status")
        version_line = f"Current version: {APP_VERSION}"

        if version_status == "ok" and latest_version:
            if latest_version == APP_VERSION:
                version_line += f"\nLatest available: {latest_version} (up to date)"
            else:
                version_line += f"\nLatest available: {latest_version} (update available)"
        elif version_status == "missing":
            version_line += "\nVersion check file not found"
        elif version_status == "invalid":
            version_line += "\nVersion check file is invalid"
        else:
            version_line += "\nVersion check not configured"

        save_root = get_save_root()
        spec_path = get_spec_path()
        version_path = get_version_check_path()

        messagebox.showinfo(
            "About",
            f"{APP_TITLE}\n\n"
            f"{version_line}\n\n"
            "Current stage:\n"
            "- Guided operator workflow\n"
            "- Defect type selection\n"
            "- Fiducial detection and scale\n"
            "- ROI selection\n"
            "- Manual annotation for scratch / scuff / dent\n"
            "- Inspection export\n\n"
            f"Save root: {save_root}\n"
            f"Spec path: {spec_path if spec_path is not None else 'built-in defaults'}\n"
            f"Version check path: {version_path if version_path is not None else 'not configured'}",
        )

    def _render_current_image(self) -> None:
        image_bgr = self.image_state.display_bgr
        if image_bgr is None:
            self._show_empty_canvas_message()
            return

        canvas_width = max(self.image_canvas.winfo_width(), 300)
        canvas_height = max(self.image_canvas.winfo_height(), 300)
        img_h, img_w = image_bgr.shape[:2]
        base_scale = min(canvas_width / img_w, canvas_height / img_h)
        scale = base_scale * self.zoom_factor
        rendered_width = max(1, int(img_w * scale))
        rendered_height = max(1, int(img_h * scale))

        self.render_scale = scale
        self.rendered_width = rendered_width
        self.rendered_height = rendered_height

        base_offset_x = (canvas_width - rendered_width) // 2
        base_offset_y = (canvas_height - rendered_height) // 2
        self.render_offset_x = base_offset_x + self.pan_x
        self.render_offset_y = base_offset_y + self.pan_y

        rendered_bgr = cv2.resize(
            image_bgr,
            (rendered_width, rendered_height),
            interpolation=cv2.INTER_LINEAR,
        )
        self._draw_scaled_overlays(rendered_bgr)

        rgb = cv2.cvtColor(rendered_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.tk_image = ImageTk.PhotoImage(pil_img)

        self.image_canvas.delete("all")
        self.canvas_image_id = self.image_canvas.create_image(
            self.render_offset_x,
            self.render_offset_y,
            anchor="nw",
            image=self.tk_image,
        )

    def _on_resize(self, event: tk.Event) -> None:
        if event.widget == self.root:
            if self.image_state.display_bgr is not None:
                self._render_current_image()
            else:
                self._show_empty_canvas_message()

    def run(self) -> None:
        self.root.mainloop()

    def _min_points_required(self) -> int:
        return self._current_mode().min_points_required()

    def start_annotation(self, auto_started: bool = False) -> None:
        if not self.image_state.is_loaded():
            self.status_var.set("Load an image first")
            return
        if self.mm_per_px is None:
            self.status_var.set("Detect fiducials first")
            return

        self.measure_points = []
        self.current_measurement_mm = None
        self.current_area_mm2 = None
        self.current_diameter_mm = None
        self.current_result = None
        self.annotation_active = True
        self.saved_inspection = False
        self.verdict_var.set("Unreviewed")

        mode = self._current_mode()
        if auto_started:
            self.status_var.set(f"{mode.name} mode active. Mark the defect.")
        else:
            self.status_var.set(f"Annotation restarted for {mode.name.lower()}.")

        self._redraw_overlay_base()
        self._refresh_ui_state()

    def on_left_click(self, event):
        if not self.annotation_active:
            return

        point = self._label_to_image_coords(event.x, event.y)
        if point is None:
            return

        defect = self.current_defect_type.get()

        # ---- Scratch width sampling ----
        if defect == "Scratch" and self.annotation_step == "width":
            if not self.scratch_width_preview or self.mm_per_px is None:
                return

            anchor = self.scratch_width_preview["anchor"]

            if not self._can_place_scratch_width_sample(anchor):
                self.status_var.set(
                    "Move further along the scratch before placing another sample."
                )
                return

            p1 = self.scratch_width_preview["p1"]
            p2 = self.scratch_width_preview["p2"]
            width_px = self.scratch_width_preview["width_px"]

            sample = {
                "anchor": anchor,
                "p1": p1,
                "p2": p2,
                "width_px": width_px,
                "width_mm": self._distance_mm_between(p1, p2),
            }
            self.scratch_width_samples.append(sample)

            widths_mm = [s["width_mm"] for s in self.scratch_width_samples]
            self.scratch_width_mm = sum(widths_mm) / len(widths_mm)

            # reset temporary preview state so next sample can be placed
            self.scratch_width_anchor = None
            self.scratch_width_preview = None
            self.mouse_image_point = None
            self.scratch_width_can_place = True

            self.status_var.set(
                f"Width sample {len(self.scratch_width_samples)} added. "
                f"Avg width: {self.scratch_width_mm:.2f} mm. "
                "Move and left-click to add more, or right-click to finish."
            )

            self._redraw_overlay_base()
            self._refresh_ui_state()
            return

        # ---- Scuff: click first point again to close ----
        if defect == "Scuff" and self.annotation_step == "outline":
            if len(self.measure_points) >= 3 and self._is_near_point(point, self.measure_points[0]):
                self.finish_annotation()
                return

        self.measure_points.append(point)

        mode = self._current_mode()
        if mode.auto_finish(self.annotation_step, self.measure_points):
            self.finish_annotation()
            return

        self._redraw_overlay_base()
        self._refresh_ui_state()

    def on_right_click(self, event):
        if not self.annotation_active:
            return

        if (
            self.current_defect_type.get() == "Scratch"
            and self.annotation_step == "width"
        ):
            self.finish_scratch_width(allow_empty=True)
            return

        self.finish_annotation()

    def finish_annotation(self):
        if self.mm_per_px is None:
            self.status_var.set("Detect fiducials first")
            return

        mode = self._current_mode()

        result = mode.finish_step(
            step=self.annotation_step,
            points=self.measure_points,
            mm_per_px=self.mm_per_px,
            context=self._mode_context(),
        )

        if not result.get("ok", False):
            self.status_var.set(result.get("message", "Could not finish annotation"))
            return

        data = result.get("results", {})
        defect_type = self.current_defect_type.get()

        # ---- Scratch: handle step transition explicitly ----
        if defect_type == "Scratch":
            if self.annotation_step == "length":
                if "length_points" in data:
                    self.scratch_length_points = list(data["length_points"])
                    self.current_measurement_mm = self._distance_mm_from_points(
                        self.scratch_length_points
                    )

                # switch explicitly into width mode
                self.annotation_step = "width"
                self.annotation_active = True
                self.measure_points = []
                self.scratch_width_anchor = None
                self.scratch_width_preview = None
                self.mouse_image_point = None
                self.scratch_width_can_place = True

                self.instructions_var.set(
                    "Scratch flow:\n"
                    "- Click two points across the scratch width\n"
                    "- Right click to finish"
                )

                if self.current_measurement_mm is not None:
                    self.status_var.set(
                        f"Scratch length recorded: {self.current_measurement_mm:.2f} mm. Now mark width."
                    )
                else:
                    self.status_var.set("Scratch length recorded. Now mark width.")

                self._redraw_overlay_base()
                self._refresh_ui_state()
                return

            elif self.annotation_step == "width":
                if "width_points" in data:
                    self.scratch_width_points = list(data["width_points"])
                    self.scratch_width_mm = self._distance_mm_from_points(
                        self.scratch_width_points
                    )

                self.annotation_active = False
                self.instructions_var.set(
                    "Scratch analysis complete.\n"
                    "- Review the result\n"
                    "- Mark OK or NOK\n"
                    "- Save the inspection"
                )

                if self.current_measurement_mm is not None and self.scratch_width_mm is not None:
                    self.status_var.set(
                        f"Analysis complete. Scratch length: {self.current_measurement_mm:.2f} mm, "
                        f"width: {self.scratch_width_mm:.2f} mm. Review the result, mark OK or NOK, then save."
                    )
                else:
                    self.status_var.set("Scratch analysis complete.")

                self._redraw_overlay_base()
                self._refresh_ui_state()
                return

        # ---- Scuff ----
        elif defect_type == "Scuff":
            outline_points = data.get("outline_points", list(self.measure_points))
            self.current_area_mm2 = self._area_mm2_from_points(outline_points)
            self.annotation_active = False
            self.instructions_var.set(
                "Scuff analysis complete.\n"
                "- Review the result\n"
                "- Mark OK or NOK\n"
                "- Save the inspection"
            )
            if self.current_area_mm2 is not None:
                self.status_var.set(
                    f"Analysis complete. Scuff area: {self.current_area_mm2:.2f} mm². "
                    "Review the result, mark OK or NOK, then save."
                )
            else:
                self.status_var.set("Scuff analysis complete.")

        # ---- Dent ----
        elif defect_type == "Dent":
            diameter_points = data.get("diameter_points", list(self.measure_points))
            if len(diameter_points) >= 2:
                self.current_diameter_mm = self._distance_mm_between(
                    diameter_points[0],
                    diameter_points[1],
                )
            self.annotation_active = False
            self.instructions_var.set(
                "Dent analysis complete.\n"
                "- Review the result\n"
                "- Mark OK or NOK\n"
                "- Save the inspection"
            )
            if self.current_diameter_mm is not None:
                self.status_var.set(
                    f"Analysis complete. Dent diameter: {self.current_diameter_mm:.2f} mm. "
                    "Review the result, mark OK or NOK, then save."
                )
            else:
                self.status_var.set("Dent analysis complete.")

        self._redraw_overlay_base()
        self._refresh_ui_state()

    def undo_last_point(self) -> None:
        if not self.measure_points:
            self.status_var.set("No points to undo")
            return
        self.measure_points.pop()
        self.current_measurement_mm = None
        self.current_area_mm2 = None
        self.current_diameter_mm = None
        self.saved_inspection = False
        self._redraw_overlay_base()
        self.status_var.set("Last point removed")
        self._refresh_ui_state()

    def cancel_measurement(self, event=None, redraw_only: bool = False):
        self.measure_points = []
        self.current_measurement_mm = None
        self.current_area_mm2 = None
        self.current_diameter_mm = None
        self.current_result = None
        self.saved_inspection = False
        self.scratch_length_points = []
        self.scratch_width_points = []
        self.scratch_width_samples = []
        self.scratch_width_mm = None
        self.verdict_var.set("Unreviewed")
        self.spec_result = None
        self.scratch_width_anchor = None
        self.scratch_width_preview = None
        self.mouse_image_point = None
        self.scratch_width_can_place = True
        self.scratch_width_samples = []
        

        if hasattr(self, "scratch_width_mm"):
            self.scratch_width_mm = None
        if hasattr(self, "scratch_length_points"):
            self.scratch_length_points = []

        # Restart current defect flow automatically
        defect = self.current_defect_type.get()

        if self.image_state.is_loaded() and self.mm_per_px is not None:
            self.annotation_active = True

            if defect == "Scratch":
                self.annotation_step = "length"
                self.instructions_var.set(
                    "Scratch flow:\n"
                    "- Click along the scratch centerline\n"
                    "- Right click to finish"
                )
                if not redraw_only:
                    self.status_var.set("Scratch annotation restarted. Mark the scratch.")
            elif defect == "Scuff":
                self.annotation_step = "outline"
                self.instructions_var.set(
                    "Scuff flow:\n"
                    "- Click around the scuff boundary\n"
                    "- Close the loop by clicking the first point again"
                )
                if not redraw_only:
                    self.status_var.set("Scuff annotation restarted. Mark the scuff boundary.")
            elif defect == "Dent":
                self.annotation_step = "diameter"
                self.instructions_var.set(
                    "Dent flow:\n"
                    "- Click two points across the dent\n"
                    "- Measurement finishes automatically after the second point"
                )
                if not redraw_only:
                    self.status_var.set("Dent annotation restarted. Mark the dent.")
            else:
                self.annotation_active = False
                self.annotation_step = None
                if not redraw_only:
                    self.status_var.set("Annotation cleared")
        else:
            self.annotation_active = False
            self.annotation_step = None
            if not redraw_only:
                self.status_var.set("Annotation cleared")

        if self.image_state.is_loaded():
            self._redraw_overlay_base()

        self._refresh_ui_state()

    def _redraw_overlay_base(self):
        if self.image_state.original_bgr is None:
            return
        self.image_state.display_bgr = self.image_state.original_bgr.copy()
        self._render_current_image()

    def _draw_scaled_overlays(self, rendered_bgr):
        self._draw_scaled_fiducials(rendered_bgr)
        self._draw_scaled_roi(rendered_bgr)
        self._draw_scaled_scratch_candidates(rendered_bgr)
        self._draw_scaled_annotation(rendered_bgr)

    def _draw_scaled_fiducials(self, rendered_bgr):
        if not self.detections:
            return

        for det in self.detections:
            pts = np.array([self._image_to_rendered_coords(pt) for pt in det.corners], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(rendered_bgr, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            top_left = tuple(pts[0][0])
            label = f"ID {det.marker_id} | {det.side_px:.1f}px"
            cv2.putText(
                rendered_bgr,
                label,
                (int(top_left[0]), max(20, int(top_left[1]) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    def _draw_scaled_roi(self, rendered_bgr):
        if self.is_selecting_roi and self.roi_start is not None and self.roi_end is not None:
            p1 = self._image_to_rendered_coords(self.roi_start)
            p2 = self._image_to_rendered_coords(self.roi_end)
            cv2.rectangle(rendered_bgr, p1, p2, (255, 0, 0), 2)
            return

        if self.roi_rect is None:
            return
        x1, y1, x2, y2 = self.roi_rect
        p1 = self._image_to_rendered_coords((x1, y1))
        p2 = self._image_to_rendered_coords((x2, y2))
        cv2.rectangle(rendered_bgr, p1, p2, (255, 0, 0), 2)

    def _draw_scaled_scratch_candidates(self, rendered_bgr):
        if not self.scratch_candidates or self.current_defect_type.get() != "Scratch":
            return

        offset_x = 0
        offset_y = 0
        if self.roi_rect is not None:
            offset_x, offset_y, _, _ = self.roi_rect

        for i, cnt in enumerate(self.scratch_candidates):
            cnt_points = cnt.reshape(-1, 2)
            shifted = [(int(pt[0] + offset_x), int(pt[1] + offset_y)) for pt in cnt_points]
            scaled = np.array([self._image_to_rendered_coords(pt) for pt in shifted], dtype=np.int32).reshape((-1, 1, 2))
            color = (0, 255, 255) if i else (0, 0, 255)
            cv2.drawContours(rendered_bgr, [scaled], -1, color, 2 if i else 3)

    def _draw_scaled_annotation(self, rendered_bgr):
        is_scratch = self.current_defect_type.get() == "Scratch"

        if is_scratch and self.annotation_step == "length" and self.measure_points:
            self._draw_scaled_polyline(rendered_bgr, self.measure_points, (0, 0, 200), point_radius=4, line_thickness=2)

        if is_scratch and len(self.scratch_length_points) >= 2:
            self._draw_scaled_polyline(rendered_bgr, self.scratch_length_points, (0, 255, 255), point_radius=0, line_thickness=2)

        if is_scratch:
            for sample in self.scratch_width_samples:
                p1 = self._image_to_rendered_coords(sample["p1"])
                p2 = self._image_to_rendered_coords(sample["p2"])
                anchor = self._image_to_rendered_coords(sample["anchor"])
                cv2.line(rendered_bgr, p1, p2, (0, 255, 255), 1)
                cv2.circle(rendered_bgr, anchor, 3, (255, 255, 0), -1)

        if is_scratch and self.annotation_step == "width" and self.scratch_width_preview is not None:
            p1 = self._image_to_rendered_coords(self.scratch_width_preview["p1"])
            p2 = self._image_to_rendered_coords(self.scratch_width_preview["p2"])
            anchor = self._image_to_rendered_coords(self.scratch_width_preview["anchor"])
            color = (0, 255, 0) if self.scratch_width_can_place else (0, 0, 255)
            cv2.line(rendered_bgr, p1, p2, color, 1)
            cv2.circle(rendered_bgr, anchor, 4, color, -1)

        if not is_scratch and self.measure_points:
            defect = self.current_defect_type.get()
            if defect == "Scuff":
                self._draw_scaled_scuff(rendered_bgr)
            elif defect == "Dent":
                self._draw_scaled_dent(rendered_bgr)
            else:
                self._draw_scaled_polyline(rendered_bgr, self.measure_points, (0, 255, 0), point_radius=4, line_thickness=2)

    def _draw_scaled_scuff(self, rendered_bgr):
        points = self.measure_points
        if not points:
            return
        scaled_points = [self._image_to_rendered_coords(pt) for pt in points]
        for i, pt in enumerate(scaled_points):
            color = (0, 255, 255) if i == 0 else (0, 0, 255)
            radius = 6 if i == 0 else 4
            cv2.circle(rendered_bgr, pt, radius, color, -1)
        for i in range(len(scaled_points) - 1):
            cv2.line(rendered_bgr, scaled_points[i], scaled_points[i + 1], (0, 200, 255), 2)
        if not self.annotation_active and len(scaled_points) >= 3:
            cv2.line(rendered_bgr, scaled_points[-1], scaled_points[0], (0, 200, 255), 2)

    def _draw_scaled_dent(self, rendered_bgr):
        points = self.measure_points
        if not points:
            return
        scaled_points = [self._image_to_rendered_coords(pt) for pt in points]
        for pt in scaled_points:
            cv2.circle(rendered_bgr, pt, 5, (255, 0, 0), -1)
        if len(scaled_points) >= 2:
            cv2.line(rendered_bgr, scaled_points[0], scaled_points[1], (255, 0, 0), 2)

    def _draw_scaled_polyline(self, rendered_bgr, points, color, point_radius=4, line_thickness=2):
        scaled_points = [self._image_to_rendered_coords(pt) for pt in points]
        if point_radius > 0:
            for pt in scaled_points:
                cv2.circle(rendered_bgr, pt, point_radius, color, -1)
        for i in range(len(scaled_points) - 1):
            cv2.line(rendered_bgr, scaled_points[i], scaled_points[i + 1], color, line_thickness)

    def _label_to_image_coords(self, x: int, y: int):
        local_x = x - self.render_offset_x
        local_y = y - self.render_offset_y
        if local_x < 0 or local_y < 0 or local_x >= self.rendered_width or local_y >= self.rendered_height:
            return None
        return int(round(local_x / self.render_scale)), int(round(local_y / self.render_scale))

    def _image_to_rendered_coords(self, point):
        px, py = point
        return (
            int(round(px * self.render_scale)),
            int(round(py * self.render_scale)),
        )

    def _point_in_roi(self, point) -> bool:
        if self.roi_rect is None:
            return True
        x1, y1, x2, y2 = self.roi_rect
        return x1 <= point[0] <= x2 and y1 <= point[1] <= y2

    def start_roi_selection(self):
        if not self.image_state.is_loaded() or self.mm_per_px is None:
            self.status_var.set("Load image and detect fiducials first")
            return
        self.cancel_measurement(redraw_only=True)
        self.is_selecting_roi = True
        self.roi_start = None
        self.roi_end = None
        self.saved_inspection = False
        self.status_var.set("Click and drag to select the defect ROI")
        self._refresh_ui_state()

    def on_mouse_down(self, event):
        if self.is_selecting_roi:
            point = self._label_to_image_coords(event.x, event.y)
            if point is None:
                return
            self.roi_start = point
            self.roi_end = point
        else:
            self.on_left_click(event)

    def on_mouse_drag(self, event):
        if not self.is_selecting_roi or self.roi_start is None:
            return
        point = self._label_to_image_coords(event.x, event.y)
        if point is None:
            return
        self.roi_end = point
        self._draw_roi_overlay()

    def on_mouse_up(self, event):
        if not self.is_selecting_roi or self.roi_start is None:
            return
        point = self._label_to_image_coords(event.x, event.y)
        if point is None:
            return
        self.roi_end = point
        self.is_selecting_roi = False
        x1, y1 = self.roi_start
        x2, y2 = self.roi_end
        self.roi_rect = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        self._redraw_overlay_base()
        self.status_var.set("ROI selected. Next: start annotation.")
        self._refresh_ui_state()

    def _draw_roi_overlay(self):
        if self.image_state.original_bgr is None:
            return
        self.image_state.display_bgr = self.image_state.original_bgr.copy()
        self._render_current_image()

    def get_roi_image(self):
        if self.roi_rect is None:
            return None
        x1, y1, x2, y2 = self.roi_rect
        return self.image_state.original_bgr[y1:y2, x1:x2]

    def detect_scratch(self):
        if self.current_defect_type.get() != "Scratch":
            self.status_var.set("Scratch candidate detection is only available in Scratch mode")
            return
        roi = self.get_roi_image() if self.roi_rect is not None else self.image_state.original_bgr
        if roi is None or roi.size == 0:
            self.status_var.set("Invalid image region")
            self._refresh_ui_state()
            return
        contours, mask, combined = detect_scratch_candidates(roi)
        self.scratch_candidates = contours
        cv2.imwrite("debug_roi.png", roi)
        cv2.imwrite("debug_scratch_combined.png", combined)
        cv2.imwrite("debug_scratch_mask.png", mask)
        self._redraw_overlay_base()
        self.status_var.set(f"{len(contours)} scratch candidate(s) found" if contours else "No scratch candidates found")
        self._refresh_ui_state()

    def save_inspection(self):
        if self.image_state.original_bgr is None:
            self.status_var.set("No inspection to save")
            return

        has_result = (
            self.current_measurement_mm is not None
            or self.current_area_mm2 is not None
            or self.current_diameter_mm is not None
        )
        if not has_result:
            self.status_var.set("Finish an annotation before saving")
            return

        if self.current_spec_name.get() not in ("inside", "outside"):
            self.status_var.set("Choose whether the defect is inside or outside before saving")
            return

        if self.verdict_var.get() not in ("OK", "NOK"):
            self.status_var.set("Mark the result as OK or NOK before saving")
            return

        save_root = get_save_root()
        save_root.mkdir(parents=True, exist_ok=True)

        export_dir = self.build_inspection_folder(save_root, self.current_defect_type.get())
        export_dir.mkdir(parents=True, exist_ok=False)

        original_path = export_dir / "original.png"
        overlay_path = export_dir / "overlay.png"
        data_path = export_dir / "inspection.json"

        source_for_save = self.image_state.source_bgr if self.image_state.source_bgr is not None else self.image_state.original_bgr
        cv2.imwrite(str(original_path), source_for_save)
        overlay = self.image_state.display_bgr if self.image_state.display_bgr is not None else self.image_state.original_bgr
        cv2.imwrite(str(overlay_path), overlay)

        defect_type = self.current_defect_type.get()

        data = {
            "tool_version": APP_VERSION,
            "save_root": str(get_save_root()),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "defect_type": defect_type,
            "image_path": str(self.image_state.path) if self.image_state.path else None,
            "fiducials_detected": len(self.detections),
            "calibration_method": self.calibration_method,
            "mm_per_px": self.mm_per_px,
            "roi_rect": list(self.roi_rect) if self.roi_rect else None,
            "operator_verdict": self.verdict_var.get(),
            "system_side": self.current_spec_name.get(),
            "spec_name": self.current_spec_name.get(),
            "spec_path": str(get_spec_path()) if get_spec_path() is not None else None,
            "version_check_path": str(get_version_check_path()) if get_version_check_path() is not None else None,
            "result": self.spec_result,
            "saved_files": {
                "original_image": str(original_path),
                "overlay_image": str(overlay_path),
            },
            "results": {},
            "annotations": {},
        }

        mode = self._current_mode()
        export_data = mode.export_data(self._export_context())

        data["results"] = export_data.get("results", {})
        data["annotations"] = export_data.get("annotations", {})

        data_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        self.last_saved_path = export_dir
        self.saved_inspection = True
        self.status_var.set(f"Inspection saved to {export_dir}")
        self._refresh_ui_state()

    def zoom_in(self):
        self.zoom_factor = min(self.zoom_factor * 1.25, self.max_zoom)
        self._render_current_image()
        self.status_var.set(f"Zoom: {self.zoom_factor:.2f}x")
        self._refresh_ui_state()

    def zoom_out(self):
        self.zoom_factor = max(self.zoom_factor / 1.25, self.min_zoom)
        self._render_current_image()
        self.status_var.set(f"Zoom: {self.zoom_factor:.2f}x")
        self._refresh_ui_state()

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._render_current_image()
        self.status_var.set("Zoom reset")
        self._refresh_ui_state()

    def on_mouse_wheel(self, event):
        if event.num == 4 or getattr(event, "delta", 0) > 0:
            self.zoom_factor = min(self.zoom_factor * 1.1, self.max_zoom)
        elif event.num == 5 or getattr(event, "delta", 0) < 0:
            self.zoom_factor = max(self.zoom_factor / 1.1, self.min_zoom)
        self._render_current_image()
        self.status_var.set(f"Zoom: {self.zoom_factor:.2f}x")
        self._refresh_ui_state()

    def on_pan_start(self, event):
        self.is_panning = True
        self.pan_start_mouse = (event.x, event.y)
        self.pan_start_offset = (self.pan_x, self.pan_y)
        self.status_var.set("Panning...")
        self._refresh_ui_state()

    def on_pan_drag(self, event):
        if not self.is_panning or self.pan_start_mouse is None or self.pan_start_offset is None:
            return
        start_x, start_y = self.pan_start_mouse
        start_pan_x, start_pan_y = self.pan_start_offset
        self.pan_x = start_pan_x + (event.x - start_x)
        self.pan_y = start_pan_y + (event.y - start_y)
        self._render_current_image()

    def on_pan_end(self, event):
        self.is_panning = False
        self.pan_start_mouse = None
        self.pan_start_offset = None
        self.status_var.set(f"Zoom: {self.zoom_factor:.2f}x")
        self._refresh_ui_state()

    def _current_mode(self):
        return self.defect_modes[self.current_defect_type.get()]

    def _is_near_point(self, p1, p2, threshold=12):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return (dx * dx + dy * dy) ** 0.5 <= threshold
    
    def _update_result_summary(self) -> None:
        if self.current_measurement_mm is not None:
            if self.scratch_width_samples:
                widths = [s["width_mm"] for s in self.scratch_width_samples]

                avg = sum(widths) / len(widths)
                min_w = min(widths)
                max_w = max(widths)

                self.result_var.set(
                    f"Scratch length: {self.current_measurement_mm:.2f} mm\n"
                    f"Avg width: {avg:.2f} mm\n"
                    f"Min width: {min_w:.2f} mm\n"
                    f"Max width: {max_w:.2f} mm\n"
                    f"Samples: {len(widths)}"
                )
            else:
                self.result_var.set(f"Scratch length: {self.current_measurement_mm:.2f} mm")
        elif self.current_area_mm2 is not None:
            self.result_var.set(f"Scuff area: {self.current_area_mm2:.2f} mm²")
        elif self.current_diameter_mm is not None:
            self.result_var.set(f"Dent diameter: {self.current_diameter_mm:.2f} mm")
        elif self.measure_points:
            self.result_var.set(f"{len(self.measure_points)} point(s) placed")
        else:
            self.result_var.set("No result yet")

    def _mode_context(self):
        return {
            "scratch_length_mm": self.current_measurement_mm,
            "scratch_width_mm": self.scratch_width_mm,
        }

    def _export_context(self):
        return {
            "scratch_length_mm": self.current_measurement_mm,
            "scratch_width_mm": self.scratch_width_mm,
            "scratch_length_points": self.scratch_length_points,
            "scratch_width_points": self.scratch_width_points,
            "scratch_width_samples": self.scratch_width_samples,
            "area_mm2": self.current_area_mm2,
            "diameter_mm": self.current_diameter_mm,
            "measure_points": self.measure_points,
        }
    
    def _current_results_for_spec(self):
        defect_type = self.current_defect_type.get()

        if defect_type == "Scratch":
            return {
                "length_mm": self.current_measurement_mm,
                "width_mm": self.scratch_width_mm,
            }
        elif defect_type == "Scuff":
            return {
                "area_mm2": self.current_area_mm2,
            }
        elif defect_type == "Dent":
            return {
                "diameter_mm": self.current_diameter_mm,
            }

        return {}
    
    def _evaluate_current_spec(self):
        results = self._current_results_for_spec()
        defect_type = self.current_defect_type.get()

        side = self.current_spec_name.get()
        if side in ("inside", "outside"):
            self.spec_name_var.set(f"Applied spec: {side.capitalize()}")
        else:
            self.spec_name_var.set("Applied spec: -")

        if self.current_spec_name.get() not in ("inside", "outside"):
            self.spec_result = None
            self.spec_suggestion_var.set("Suggested verdict: -")
            self.spec_reason_var.set("Choose Inside or Outside first")
            self.spec_suggestion_label.config(fg="#111827", font=("Arial", 12, "bold"))
            self.spec_reason_label.config(fg="#6b7280")
            self.spec_box.config(highlightbackground="#d1d5db")
            return

        has_result = any(v is not None for v in results.values())
        if not has_result:
            self.spec_result = None
            self.spec_suggestion_var.set("Suggested verdict: -")
            self.spec_reason_var.set("No spec evaluation yet")
            self.spec_suggestion_label.config(fg="#111827", font=("Arial", 12, "bold"))
            self.spec_reason_label.config(fg="#6b7280")
            self.spec_box.config(highlightbackground="#d1d5db")
            return

        self.spec_result = evaluate_results(
            defect_type=defect_type,
            results=results,
            spec_name=self.current_spec_name.get(),
        )

        suggested = self.spec_result.get("pass")
        if suggested is True:
            self.spec_suggestion_var.set("Suggested verdict: OK")
            self.spec_suggestion_label.config(fg="#15803d", font=("Arial", 12, "bold"))
            self.spec_box.config(highlightbackground="#86efac")
        elif suggested is False:
            self.spec_suggestion_var.set("Suggested verdict: NOK")
            self.spec_suggestion_label.config(fg="#b91c1c", font=("Arial", 12, "bold"))
            self.spec_box.config(highlightbackground="#fca5a5")
        else:
            self.spec_suggestion_var.set("Suggested verdict: Unknown")
            self.spec_suggestion_label.config(fg="#111827", font=("Arial", 12, "bold"))
            self.spec_box.config(highlightbackground="#d1d5db")

        reasons = self.spec_result.get("reasons", [])
        limits = self.spec_result.get("limits", {})

        limit_lines = []
        for key, value in limits.items():
            pretty_key = key.replace("_", " ")
            limit_lines.append(f"{pretty_key}: {value}")

        if reasons:
            self.spec_reason_label.config(fg="#7f1d1d" if suggested is False else "#374151")
            self.spec_reason_var.set(
                "Limits:\n"
                + "\n".join(limit_lines)
                + "\n\nReason:\n"
                + "\n".join(reasons)
            )
        else:
            self.spec_reason_label.config(fg="#166534" if suggested is True else "#374151")
            self.spec_reason_var.set(
                "Limits:\n"
                + "\n".join(limit_lines)
                + "\n\nWithin spec"
            )

    def on_side_changed(self):
        side = self.current_spec_name.get()
        if side == "outside":
            self.status_var.set("System side set to Outside")
        elif side == "inside":
            self.status_var.set("System side set to Inside")
        else:
            self.status_var.set("Choose whether the defect is inside or outside")

        self._refresh_ui_state()

    def on_mouse_move(self, event):
        point = self._label_to_image_coords(event.x, event.y)
        if point is None:
            return

        self.mouse_image_point = point

        if (
            self.current_defect_type.get() == "Scratch"
            and self.annotation_active
            and self.annotation_step == "width"
            and len(self.scratch_length_points) >= 2
        ):
            anchor, p1, p2, width_px = self._build_scratch_width_preview(point)

            self.scratch_width_preview = {
                "anchor": anchor,
                "p1": p1,
                "p2": p2,
                "width_px": width_px,
            }

            self.scratch_width_can_place = self._can_place_scratch_width_sample(anchor)

            if self.scratch_width_can_place:
                self.status_var.set("Width sample can be placed")
            else:
                self.status_var.set("Too close to previous sample")

            self._redraw_overlay_base()

    def _build_scratch_width_preview(self, mouse_point):
        anchor, tangent = self._nearest_point_and_tangent_on_polyline(
            mouse_point,
            self.scratch_length_points,
        )

        tx, ty = tangent
        norm = (tx * tx + ty * ty) ** 0.5
        if norm == 0:
            return None, None, None, 0.0

        tx /= norm
        ty /= norm

        # perpendicular
        px = -ty
        py = tx

        mx, my = mouse_point
        ax, ay = anchor

        # signed distance from mouse to perpendicular center
        dx = mx - ax
        dy = my - ay
        half_width_px = abs(dx * px + dy * py)

        if half_width_px < 2:
            half_width_px = 2

        p1 = (
            int(round(ax - px * half_width_px)),
            int(round(ay - py * half_width_px)),
        )
        p2 = (
            int(round(ax + px * half_width_px)),
            int(round(ay + py * half_width_px)),
        )

        return (int(round(ax)), int(round(ay))), p1, p2, half_width_px * 2.0


    def _nearest_point_and_tangent_on_polyline(self, point, polyline):
        if len(polyline) < 2:
            return polyline[0], (1.0, 0.0)

        px, py = point
        best_dist2 = float("inf")
        best_point = polyline[0]
        best_tangent = (1.0, 0.0)

        for i in range(len(polyline) - 1):
            x1, y1 = polyline[i]
            x2, y2 = polyline[i + 1]
            dx = x2 - x1
            dy = y2 - y1
            seg_len2 = dx * dx + dy * dy
            if seg_len2 == 0:
                continue

            t = ((px - x1) * dx + (py - y1) * dy) / seg_len2
            t = max(0.0, min(1.0, t))

            qx = x1 + t * dx
            qy = y1 + t * dy

            dist2 = (px - qx) ** 2 + (py - qy) ** 2
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_point = (qx, qy)
                best_tangent = (dx, dy)

        return best_point, best_tangent


    def _project_point_to_line(self, point, anchor, direction):
        px, py = point
        ax, ay = anchor
        dx, dy = direction
        denom = dx * dx + dy * dy
        if denom == 0:
            return point

        t = ((px - ax) * dx + (py - ay) * dy) / denom
        qx = ax + t * dx
        qy = ay + t * dy
        return (int(round(qx)), int(round(qy)))
    
    def finish_scratch_width(self, allow_empty=True):
        if not self.scratch_width_samples:
            if not allow_empty:
                self.status_var.set("Add at least one width sample")
                return
            self.scratch_width_mm = None
        else:
            widths = [s["width_mm"] for s in self.scratch_width_samples]
            avg_width = sum(widths) / len(widths)
            min_width = min(widths)
            max_width = max(widths)
            self.scratch_width_mm = avg_width

        self.annotation_active = False
        self.annotation_step = None

        self.scratch_width_anchor = None
        self.scratch_width_preview = None
        self.mouse_image_point = None

        if self.scratch_width_mm is None:
            self.status_var.set(
                f"Analysis complete. Length: {self.current_measurement_mm:.2f} mm | Width not recorded"
            )
        else:
            self.status_var.set(
                f"Analysis complete. Length: {self.current_measurement_mm:.2f} mm | "
                f"Avg width: {avg_width:.2f} mm | Min: {min_width:.2f} mm | Max: {max_width:.2f} mm"
            )

        self.instructions_var.set(
            "Scratch analysis complete.\n"
            "- Review the result\n"
            "- Mark OK or NOK\n"
            "- Save the inspection"
        )

        self._redraw_overlay_base()
        self._refresh_ui_state()

    def _can_place_scratch_width_sample(self, anchor):
        if anchor is None:
            return False

        if not self.scratch_width_samples:
            return True

        last_anchor = self.scratch_width_samples[-1]["anchor"]

        min_spacing_px = 12

        dx = anchor[0] - last_anchor[0]
        dy = anchor[1] - last_anchor[1]
        return (dx * dx + dy * dy) ** 0.5 >= min_spacing_px
    
    @staticmethod
    def build_inspection_folder(base_folder: str | Path, defect_type: str) -> Path:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        suffix = uuid4().hex[:4]
        folder_name = f"{defect_type.lower()}_{ts}_{suffix}"
        return Path(base_folder) / folder_name
        
    def _on_board_change(self, value):
        from core import settings
        settings.ACTIVE_BOARD_PROFILE = value
        # reload board config (or restart app)


    def _distance_mm_from_points(self, points):
        return measure_polyline_mm(
            points=points,
            mm_per_px=self.mm_per_px if self.mm_per_px is not None else 0.0,
            homography=self.calibration_homography,
            rectified_px_per_mm=RECTIFIED_PX_PER_MM,
        )

    def _distance_mm_between(self, p1, p2):
        return measure_distance_mm(
            p1=p1,
            p2=p2,
            mm_per_px=self.mm_per_px if self.mm_per_px is not None else 0.0,
            homography=self.calibration_homography,
            rectified_px_per_mm=RECTIFIED_PX_PER_MM,
        )

    def _area_mm2_from_points(self, points):
        return measure_polygon_area_mm2(
            points=points,
            mm_per_px=self.mm_per_px if self.mm_per_px is not None else 0.0,
            homography=self.calibration_homography,
            rectified_px_per_mm=RECTIFIED_PX_PER_MM,
        )
    
    def clear_annotation(self):
        self.annotation_active = False
        self.annotation_step = None
        self.measure_points = []
        self.current_measurement_mm = None

        self.scratch_width_mm = None
        self.scratch_width_samples = []
        self.scratch_width_anchor = None
        self.scratch_width_preview = None
        self.mouse_image_point = None

        self.current_defect_mask = None
        self.current_defect_contour = None
        self.current_roi_rect = None

        self.status_var.set("Annotation cleared.")
        self.instructions_var.set(
            "Scratch flow:\n"
            "- Click along the scratch centerline\n"
            "- Right click to finish"
        )

        self._redraw_overlay_base()
        self._refresh_ui_state()