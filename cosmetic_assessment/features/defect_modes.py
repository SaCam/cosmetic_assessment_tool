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

            return {
                "ok": True,
                "complete": False,
                "next_step": "width",
                "results": {
                    "length_points": list(points),
                },
                "status": "Scratch length recorded. Now mark width.",
                "instructions": self.instruction_text("width"),
            }

        elif step == "width":
            if len(points) < 2:
                return {
                    "ok": False,
                    "message": "Scratch width needs 2 points",
                }

            return {
                "ok": True,
                "complete": True,
                "next_step": None,
                "results": {
                    "width_points": list(points),
                },
                "status": "Scratch width recorded.",
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
        samples = context.get("scratch_width_samples", [])
        widths = [s.get("width_mm") for s in samples if s.get("width_mm") is not None]

        avg_width = sum(widths) / len(widths) if widths else context.get("scratch_width_mm")
        min_width = min(widths) if widths else None
        max_width = max(widths) if widths else None

        return {
            "results": {
                "length_mm": context.get("scratch_length_mm"),
                "width_avg_mm": avg_width,
                "width_min_mm": min_width,
                "width_max_mm": max_width,
                "width_sample_count": len(widths),
            },
            "annotations": {
                "length_points": context.get("scratch_length_points", []),
                "width_samples": samples,
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

        return {
            "ok": True,
            "complete": True,
            "next_step": None,
            "results": {
                "outline_points": list(points),
            },
            "status": "Scuff outline recorded. Review the result, mark OK or NOK, then save.",
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

        return {
            "ok": True,
            "complete": True,
            "next_step": None,
            "results": {
                "diameter_points": list(points),
            },
            "status": "Dent measurement recorded. Review the result, mark OK or NOK, then save.",
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