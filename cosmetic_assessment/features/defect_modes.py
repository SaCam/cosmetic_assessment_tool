import cv2
import numpy as np


class BaseDefectMode:
    name = "Base"

    def start_step(self):
        return None

    def instruction_text(self, step=None):
        return "Mark the defect."

    def min_points(self, step):
        return 1

    def auto_finish(self, step, points):
        return False

    def draw(self, img, points, active):
        for pt in points:
            cv2.circle(img, pt, 4, (0, 255, 0), -1)

    def finish_step(self, step, points, mm_per_px, context=None):
        raise NotImplementedError("finish_step must be implemented by subclasses")
    
    def export_data(self, context):
        return {
            "results": {},
            "annotations": {},
        }


class ScratchMode(BaseDefectMode):
    name = "Scratch"

    def start_step(self):
        return "length"

    def instruction_text(self, step="length"):
        if step == "length":
            return (
                "Scratch flow:\n"
                "- Click along the scratch centerline\n"
                "- Right click to finish"
            )
        elif step == "width":
            return (
                "Scratch flow:\n"
                "- Click two points across the scratch width\n"
                "- Right click to finish"
            )
        return "Scratch flow"

    def min_points(self, step):
        if step == "length":
            return 2
        if step == "width":
            return 2
        return 1

    def draw(self, img, points, active):
        for pt in points:
            cv2.circle(img, pt, 4, (0, 0, 200), -1)

        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], (0, 0, 200), 2)

    def finish_step(self, step, points, mm_per_px, context=None):
        if step == "length":
            if len(points) < 2:
                return {
                    "ok": False,
                    "message": "Scratch length needs at least 2 points",
                }

            total_px = 0.0
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                total_px += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            length_mm = total_px * mm_per_px
            return {
                "ok": True,
                "complete": False,
                "next_step": "width",
                "results": {
                    "length_mm": length_mm,
                    "length_points": list(points),
                },
                "status": f"Scratch length recorded: {length_mm:.2f} mm. Now mark width.",
                "instructions": self.instruction_text("width"),
            }

        elif step == "width":
            if len(points) < 2:
                return {
                    "ok": False,
                    "message": "Scratch width needs 2 points",
                }

            x1, y1 = points[0]
            x2, y2 = points[1]
            width_px = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            width_mm = width_px * mm_per_px

            length_mm = None
            if context:
                length_mm = context.get("scratch_length_mm")

            return {
                "ok": True,
                "complete": True,
                "next_step": None,
                "results": {
                    "width_mm": width_mm,
                    "width_points": list(points),
                },
                "status": (
                    f"Analysis complete. Scratch length: {length_mm:.2f} mm, "
                    f"width: {width_mm:.2f} mm. "
                    "Review the result, mark OK or NOK, then save."
                    if length_mm is not None
                    else f"Analysis complete. Scratch width: {width_mm:.2f} mm. "
                         "Review the result, mark OK or NOK, then save."
                ),
                "instructions": (
                    "Scratch analysis complete.\n"
                    "- Review the result\n"
                    "- Mark OK (accepted) or NOK (not accepted)\n"
                    "- Save the inspection for ML"
                ),
            }

        return {
            "ok": False,
            "message": "Unknown scratch step",
        }
    
    def export_data(self, context):
        return {
            "results": {
                "length_mm": context.get("scratch_length_mm"),
                "width_mm": context.get("scratch_width_mm"),
            },
            "annotations": {
                "length_points": context.get("scratch_length_points", []),
                "width_points": context.get("scratch_width_points", []),
            },
        }


class ScuffMode(BaseDefectMode):
    name = "Scuff"

    def start_step(self):
        return "outline"

    def instruction_text(self, step="outline"):
        return (
            "Scuff flow:\n"
            "- Click around the scuff boundary\n"
            "- Close the loop by clicking the first point again"
        )

    def min_points(self, step):
        return 3

    def draw(self, img, points, active):
        for i, pt in enumerate(points):
            color = (0, 255, 255) if i == 0 else (0, 0, 255)
            radius = 7 if i == 0 else 5
            cv2.circle(img, pt, radius, color, -1)

        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], (0, 200, 255), 2)

        if not active and len(points) >= 3:
            cv2.line(img, points[-1], points[0], (0, 200, 255), 2)

    def finish_step(self, step, points, mm_per_px, context=None):
        if len(points) < 3:
            return {
                "ok": False,
                "message": "Scuff outline needs at least 3 points",
            }

        pts = np.array(points, dtype=np.float32)
        area_px2 = float(cv2.contourArea(pts))
        area_mm2 = area_px2 * (mm_per_px ** 2)

        return {
            "ok": True,
            "complete": True,
            "next_step": None,
            "results": {
                "area_mm2": area_mm2,
                "outline_points": list(points),
            },
            "status": (
                f"Analysis complete. Scuff area: {area_mm2:.2f} mm². "
                "Review the result, mark OK or NOK, then save."
            ),
            "instructions": (
                "Scuff analysis complete.\n"
                "- Review the result\n"
                "- Mark OK (accepted) or NOK (not accepted)\n"
                "- Save the inspection for ML"
            ),
        }
    
    def export_data(self, context):
        return {
            "results": {
                "area_mm2": context.get("area_mm2"),
            },
            "annotations": {
                "outline_points": context.get("measure_points", []),
            },
        }


class DentMode(BaseDefectMode):
    name = "Dent"

    def start_step(self):
        return "diameter"

    def instruction_text(self, step="diameter"):
        return (
            "Dent flow:\n"
            "- Click two points across the dent\n"
            "- Measurement finishes automatically"
        )

    def min_points(self, step):
        return 2

    def auto_finish(self, step, points):
        return len(points) >= 2

    def draw(self, img, points, active):
        for pt in points:
            cv2.circle(img, pt, 5, (255, 0, 0), -1)

        if len(points) >= 2:
            cv2.line(img, points[0], points[1], (255, 0, 0), 2)

    def finish_step(self, step, points, mm_per_px, context=None):
        if len(points) < 2:
            return {
                "ok": False,
                "message": "Dent measurement needs 2 points",
            }

        x1, y1 = points[0]
        x2, y2 = points[1]
        diameter_px = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        diameter_mm = diameter_px * mm_per_px

        return {
            "ok": True,
            "complete": True,
            "next_step": None,
            "results": {
                "diameter_mm": diameter_mm,
                "diameter_points": list(points),
            },
            "status": (
                f"Analysis complete. Dent diameter: {diameter_mm:.2f} mm. "
                "Review the result, mark OK or NOK, then save."
            ),
            "instructions": (
                "Dent analysis complete.\n"
                "- Review the result\n"
                "- Mark OK (accepted) or NOK (not accepted)\n"
                "- Save the inspection for ML"
            ),
        }
    
    def export_data(self, context):
        return {
            "results": {
                "diameter_mm": context.get("diameter_mm"),
            },
            "annotations": {
                "diameter_points": context.get("measure_points", []),
            },
        }


DEFECT_MODES = {
    "Scratch": ScratchMode(),
    "Scuff": ScuffMode(),
    "Dent": DentMode(),
}