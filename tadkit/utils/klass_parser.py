import inspect
import re
from numbers import Integral
from sklearn.utils._param_validation import Interval  # real sklearn Interval


def get_default_class_values(cls):
    sig = inspect.signature(cls.__init__)
    defaults = {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty and k != "self"
    }
    return defaults


def get_param_descriptions(cls):
    """
    Parses NumPy-style docstring of a class to extract parameter descriptions.
    """
    doc = inspect.getdoc(cls)
    if not doc:
        return {}

    param_descriptions = {}
    lines = doc.splitlines()
    in_params_section = False

    for i, line in enumerate(lines):
        # Detect start of 'Parameters' section
        if line.strip().lower() == "parameters":
            if i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}:
                in_params_section = True
                start_idx = i + 2
                break

    if not in_params_section:
        return {}

    # Now extract parameters
    i = start_idx
    while i < len(lines):
        line = lines[i]
        match = re.match(r"^(\w+)\s*:\s*([^,]+)(?:,\s*default=(.*))?", line.strip())
        if match:
            param_name = match.group(1)
            desc_lines = []
            i += 1
            # Capture indented description lines
            while i < len(lines) and (
                lines[i].startswith("    ") or lines[i].strip() == ""
            ):
                desc_lines.append(lines[i].strip())
                i += 1
            param_descriptions[param_name] = " ".join(desc_lines).strip()
        else:
            i += 1

    return param_descriptions


def parse_sklearn_constraints(parameter_constraints):
    """
    Translates sklearn style _parameter_constraints into structured metadata
    {
        param_name: {
            "type": int/float/"categorical",
            "bounds": {"min":..., "max":...},
            "options": [...],  # for categorical
        }
    }
    """
    metadata = {}

    for param_name, constraints in parameter_constraints.items():
        param_info = {
            "type": None,
            "bounds": {"min": None, "max": None},
            "options": None,
        }

        if not constraints:
            metadata[param_name] = param_info
            continue

        # Handle Interval from sklearn directly
        if isinstance(constraints[0], Interval):
            interval = constraints[0]
            param_info["type"] = int if interval.type == Integral else float
            param_info["bounds"]["min"] = interval.left
            param_info["bounds"]["max"] = interval.right
            metadata[param_name] = param_info
            continue

        for c in constraints:
            if isinstance(c, str):
                if c in ("integer", "number"):
                    param_info["type"] = int if c == "integer" else float
                elif c.startswith(">="):
                    val = float(c[2:])
                    param_info["bounds"]["min"] = (
                        int(val) if param_info["type"] == int else val
                    )
                elif c.startswith("<="):
                    val = float(c[2:])
                    param_info["bounds"]["max"] = (
                        int(val) if param_info["type"] == int else val
                    )
            elif isinstance(c, set):
                param_info["type"] = "categorical"
                param_info["options"] = sorted(list(c))

        metadata[param_name] = param_info

    return metadata


def enriched_metadata(cls):
    metadata = parse_sklearn_constraints(cls._parameter_constraints)
    defaults = get_default_class_values(cls)
    descriptions = get_param_descriptions(cls)

    for param, info in metadata.items():
        info["default"] = defaults.get(param)
        info["description"] = descriptions.get(param)
    return metadata


# # ===== Example usage =====
# _parameter_constraints = {
#     "window_size": ["integer", ">=1"],
#     "step": ["integer", ">=1"],
#     "tda_max_dim": ["in", {0, 1, 2}],
#     "n_centers_by_dim": ["integer", ">=1"],
#     "support_fraction": ["number", ">=0", "<=1", "optional"],
#     "random_state": ["integer", ">=0"],
#     "some_float_param": [Interval(Real, left=0.0, right=1.0, closed="both")],
# }

# parse_sklearn_constraints(_parameter_constraints)


# from tdaad.anomaly_detectors import TopologicalAnomalyDetector

# get_default_class_values(TopologicalAnomalyDetector)
# get_param_descriptions(TopologicalAnomalyDetector)

# enriched_metadata(TopologicalAnomalyDetector)
