SPEC_LIBRARY = {
    "Scratch": {
        "outside": {
            "max_length_mm": 10.0,
            "max_width_mm": 0.5,
        },
        "inside": {
            "max_length_mm": 15.0,
            "max_width_mm": 0.8,
        },
    },
    "Scuff": {
        "outside": {
            "max_area_mm2": 25.0,
        },
        "inside": {
            "max_area_mm2": 40.0,
        },
    },
    "Dent": {
        "outside": {
            "max_diameter_mm": 3.0,
        },
        "inside": {
            "max_diameter_mm": 5.0,
        },
    },
}


def get_spec(defect_type: str, spec_name: str = "default"):
    defect_specs = SPEC_LIBRARY.get(defect_type, {})
    return defect_specs.get(spec_name)


def evaluate_scratch(results: dict, spec: dict):
    reasons = []

    length_mm = results.get("length_mm")
    width_mm = results.get("width_mm")

    if length_mm is not None and "max_length_mm" in spec:
        if length_mm > spec["max_length_mm"]:
            reasons.append(
                f"Length {length_mm:.2f} mm exceeds max {spec['max_length_mm']:.2f} mm"
            )

    if width_mm is not None and "max_width_mm" in spec:
        if width_mm > spec["max_width_mm"]:
            reasons.append(
                f"Width {width_mm:.2f} mm exceeds max {spec['max_width_mm']:.2f} mm"
            )

    return {
        "pass": len(reasons) == 0,
        "reasons": reasons,
        "limits": spec,
    }


def evaluate_scuff(results: dict, spec: dict):
    reasons = []

    area_mm2 = results.get("area_mm2")
    if area_mm2 is not None and "max_area_mm2" in spec:
        if area_mm2 > spec["max_area_mm2"]:
            reasons.append(
                f"Area {area_mm2:.2f} mm² exceeds max {spec['max_area_mm2']:.2f} mm²"
            )

    return {
        "pass": len(reasons) == 0,
        "reasons": reasons,
        "limits": spec,
    }


def evaluate_dent(results: dict, spec: dict):
    reasons = []

    diameter_mm = results.get("diameter_mm")
    if diameter_mm is not None and "max_diameter_mm" in spec:
        if diameter_mm > spec["max_diameter_mm"]:
            reasons.append(
                f"Diameter {diameter_mm:.2f} mm exceeds max {spec['max_diameter_mm']:.2f} mm"
            )

    return {
        "pass": len(reasons) == 0,
        "reasons": reasons,
        "limits": spec,
    }


def evaluate_results(defect_type: str, results: dict, spec_name: str = "default"):
    spec = get_spec(defect_type, spec_name)
    if spec is None:
        return {
            "spec_name": spec_name,
            "pass": None,
            "reasons": ["No spec found"],
            "limits": {},
        }

    if defect_type == "Scratch":
        verdict = evaluate_scratch(results, spec)
    elif defect_type == "Scuff":
        verdict = evaluate_scuff(results, spec)
    elif defect_type == "Dent":
        verdict = evaluate_dent(results, spec)
    else:
        return {
            "spec_name": spec_name,
            "pass": None,
            "reasons": [f"No evaluator for defect type '{defect_type}'"],
            "limits": spec,
        }

    verdict["spec_name"] = spec_name
    return verdict