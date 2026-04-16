from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk
import numpy as np

from core.image_store import ImageState
from vision.fiducials import detect_aruco_markers
from vision.overlays import draw_fiducial_detections
from features.measurement import polyline_length_px, px_to_mm
from features.scratch_detection import detect_scratch_candidates


class MainWindow:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Fiducial Measure App")
        self.root.geometry("1100x750")
        self.root.minsize(800, 600)

        self.image_state = ImageState()
        self.tk_image = None

        self.detections = []
        self.mm_per_px = None
        self.measure_points = []

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

        self._build_menu()
        self._build_layout()

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image...", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        actions_menu = tk.Menu(menubar, tearoff=0)
        actions_menu.add_command(label="Reset View", command=self.reset_view)
        actions_menu.add_command(label="Detect Fiducials", command=self.detect_fiducials)
        actions_menu.add_separator()
        actions_menu.add_command(label="Clear Image", command=self.clear_image)
        actions_menu.add_command(label="Select ROI", command=self.start_roi_selection)
        actions_menu.add_command(label="Detect Scratch (ROI)", command=self.detect_scratch)
        actions_menu.add_separator()
        actions_menu.add_command(label="Zoom In", command=self.zoom_in)
        actions_menu.add_command(label="Zoom Out", command=self.zoom_out)
        actions_menu.add_command(label="Reset Zoom", command=self.reset_zoom)

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
        self.main_frame.columnconfigure(0, weight=1)

        self.image_canvas = tk.Canvas(
            self.main_frame,
            bg="#1e1e1e",
            highlightthickness=0,
        )
        self.image_canvas.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)

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
        self.root.bind("<Escape>", self.cancel_measurement)

    def open_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            self.image_state.load_from_file(file_path)
            self._render_current_image()
            self.status_var.set("Image loaded")

        except Exception as exc:
            messagebox.showerror("Error", f"Could not open image.\n\n{exc}")
            self.status_var.set("Failed to load image")

        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0

    def reset_view(self) -> None:
        if not self.image_state.is_loaded():
            return
        self.image_state.reset_display()
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._render_current_image()
        self.status_var.set("View reset")

    def detect_fiducials(self) -> None:
        if not self.image_state.is_loaded():
            messagebox.showinfo("No image", "Please open an image first.")
            return

        self.detections = detect_aruco_markers(self.image_state.original_bgr)

        marker_size_mm = 30.0  # must match your printed marker size
        if self.detections:
            self.mm_per_px = marker_size_mm / self.detections[0].side_px
        else:
            self.mm_per_px = None

        annotated = draw_fiducial_detections(self.image_state.original_bgr, self.detections)

        self.image_state.display_bgr = annotated
        self._render_current_image() 

        if self.mm_per_px is not None:
            self.status_var.set(
                f"Detected {len(self.detections)} fiducial marker(s) | scale: {self.mm_per_px:.4f} mm/px"
            )
        else:
            self.status_var.set("No fiducial markers detected")

    def clear_image(self) -> None:
        self.image_state.clear()
        self.tk_image = None
        self.image_canvas.delete("all")
        self.image_canvas.create_text(
            20,
            20,
            anchor="nw",
            text="No image loaded\n\nUse File > Open Image...",
            fill="white",
            font=("Arial", 16),
        )
        self.status_var.set("Image cleared")

    def show_about(self) -> None:
        messagebox.showinfo(
            "About",
            "Fiducial Measure App\n\n"
            "Milestone 1: load an image and detect ArUco fiducial markers.",
        )

    def _render_current_image(self) -> None:
        image_bgr = self.image_state.display_bgr
        if image_bgr is None:
            self.image_canvas.delete("all")
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

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img = pil_img.resize((rendered_width, rendered_height), Image.Resampling.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(pil_img)

        self.image_canvas.delete("all")
        self.canvas_image_id = self.image_canvas.create_image(
            self.render_offset_x,
            self.render_offset_y,
            anchor="nw",
            image=self.tk_image,
        )

    def _on_resize(self, event: tk.Event) -> None:
        if event.widget == self.root and self.image_state.display_bgr is not None:
            self._render_current_image()

    def run(self) -> None:
        self.root.mainloop()

    def on_left_click(self, event):
        if not self.image_state.is_loaded():
            return

        if self.mm_per_px is None:
            self.status_var.set("Detect markers first")
            return

        point = self._label_to_image_coords(event.x, event.y)
        if point is None:
            self.status_var.set("Click inside the image")
            return

        self.measure_points.append(point)
        self._redraw_measurement_overlay()

        if len(self.measure_points) == 1:
            self.status_var.set("Point 1 added")
        else:
            self.status_var.set(f"{len(self.measure_points)} points added")

    def on_right_click(self, event):
        if self.mm_per_px is None:
            return

        if len(self.measure_points) < 2:
            self.status_var.set("Add at least 2 points")
            return

        total_px = polyline_length_px(self.measure_points)
        total_mm = px_to_mm(total_px, self.mm_per_px)

        self.status_var.set(
            f"Scratch length: {total_mm:.2f} mm ({len(self.measure_points)} pts)"
        )

        self.measure_points = []

    def cancel_measurement(self, event=None):
        self.measure_points = []
        self.reset_view()
        self.status_var.set("Measurement cancelled")

    def _redraw_measurement_overlay(self):
        if self.image_state.original_bgr is None:
            return

        base = self.image_state.original_bgr.copy()

        # draw fiducials again
        if self.detections:
            base = draw_fiducial_detections(base, self.detections)

        # draw points
        for point in self.measure_points:
            cv2.circle(base, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

        # draw lines
        for i in range(len(self.measure_points) - 1):
            p1 = tuple(map(int, self.measure_points[i]))
            p2 = tuple(map(int, self.measure_points[i + 1]))
            cv2.line(base, p1, p2, (0, 0, 255), 2)

        self.image_state.display_bgr = base
        self._render_current_image()

    def _label_to_image_coords(self, x: int, y: int):
        local_x = x - self.render_offset_x
        local_y = y - self.render_offset_y

        if (
            local_x < 0
            or local_y < 0
            or local_x >= self.rendered_width
            or local_y >= self.rendered_height
        ):
            return None

        orig_x = local_x / self.render_scale
        orig_y = local_y / self.render_scale

        return int(round(orig_x)), int(round(orig_y))
    
    def start_roi_selection(self):
        if not self.image_state.is_loaded():
            return

        self.is_selecting_roi = True
        self.roi_start = None
        self.roi_end = None
        self.status_var.set("Click and drag to select ROI")

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

        self.roi_rect = (
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        )

        self._draw_roi_overlay()

        self.status_var.set("ROI selected")

    def _draw_roi_overlay(self):
        if self.image_state.original_bgr is None:
            return

        base = self.image_state.original_bgr.copy()

        if self.detections:
            base = draw_fiducial_detections(base, self.detections)

        if self.roi_start and self.roi_end:
            x1, y1 = self.roi_start
            x2, y2 = self.roi_end

            cv2.rectangle(
                base,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 0, 0),  # blue box
                2,
            )

        self.image_state.display_bgr = base
        self._render_current_image()

    def get_roi_image(self):
        if self.roi_rect is None:
            return None

        x1, y1, x2, y2 = self.roi_rect
        return self.image_state.original_bgr[y1:y2, x1:x2]
    
    def detect_scratch(self):
        if self.roi_rect is None:
            self.status_var.set("Select ROI first")
            return

        roi = self.get_roi_image()
        if roi is None or roi.size == 0:
            self.status_var.set("Invalid ROI")
            return

        contours, mask, combined = detect_scratch_candidates(roi)

        cv2.imwrite("debug_roi.png", roi)
        cv2.imwrite("debug_scratch_combined.png", combined)
        cv2.imwrite("debug_scratch_mask.png", mask)

        x1, y1, x2, y2 = self.roi_rect

        base = self.image_state.original_bgr.copy()

        # redraw fiducials if present
        if self.detections:
            base = draw_fiducial_detections(base, self.detections)

        # draw ROI box
        cv2.rectangle(base, (x1, y1), (x2, y2), (0, 0, 200), 2)

        # draw all candidate contours
        for i, cnt in enumerate(contours):
            cnt_shifted = cnt + np.array([[[x1, y1]]], dtype=cnt.dtype)

            color = (0, 255, 255)  # yellow for normal candidates
            thickness = 2

            if i == 0:
                color = (0, 0, 255)  # red for best candidate
                thickness = 3

            cv2.drawContours(base, [cnt_shifted], -1, color, thickness)

        self.image_state.display_bgr = base
        self._render_current_image()

        if contours:
            self.status_var.set(f"{len(contours)} scratch candidate(s) found")
        else:
            self.status_var.set("No scratch candidates found")

    def zoom_in(self):
        self.zoom_factor = min(self.zoom_factor * 1.25, self.max_zoom)
        self._render_current_image()
        self.status_var.set(f"Zoom: {self.zoom_factor:.2f}x")


    def zoom_out(self):
        self.zoom_factor = max(self.zoom_factor / 1.25, self.min_zoom)
        self._render_current_image()
        self.status_var.set(f"Zoom: {self.zoom_factor:.2f}x")


    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._render_current_image()
        self.status_var.set("Zoom reset")

    def on_mouse_wheel(self, event):
        if event.num == 4 or getattr(event, "delta", 0) > 0:
            self.zoom_factor = min(self.zoom_factor * 1.1, self.max_zoom)
        elif event.num == 5 or getattr(event, "delta", 0) < 0:
            self.zoom_factor = max(self.zoom_factor / 1.1, self.min_zoom)

        self._render_current_image()
        self.status_var.set(f"Zoom: {self.zoom_factor:.2f}x")

    def on_pan_start(self, event):
        self.is_panning = True
        self.pan_start_mouse = (event.x, event.y)
        self.pan_start_offset = (self.pan_x, self.pan_y)
        self.status_var.set("Panning...")


    def on_pan_drag(self, event):
        if not self.is_panning or self.pan_start_mouse is None or self.pan_start_offset is None:
            return

        start_x, start_y = self.pan_start_mouse
        start_pan_x, start_pan_y = self.pan_start_offset

        dx = event.x - start_x
        dy = event.y - start_y

        self.pan_x = start_pan_x + dx
        self.pan_y = start_pan_y + dy

        self._render_current_image()


    def on_pan_end(self, event):
        self.is_panning = False
        self.pan_start_mouse = None
        self.pan_start_offset = None
        self.status_var.set(f"Zoom: {self.zoom_factor:.2f}x")