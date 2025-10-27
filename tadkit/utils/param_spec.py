from typing import Dict, Any

import inspect
import re
from numbers import Integral
from sklearn.utils._param_validation import Interval, StrOptions


def get_default_class_values(cls) -> Dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    return {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty and k != "self"
    }


def get_param_descriptions(cls) -> Dict[str, str]:
    doc = inspect.getdoc(cls)
    if not doc:
        return {}

    param_descriptions = {}
    lines = doc.splitlines()
    in_params_section = False
    start_idx = -1

    for i, line in enumerate(lines):
        if line.strip().lower() == "parameters":
            if i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}:
                in_params_section = True
                start_idx = i + 2
                break

    if not in_params_section:
        return {}

    i = start_idx
    while i < len(lines):
        line = lines[i].strip()
        match = re.match(r"^(\w+)\s*:\s*([^,]+)(?:,\s*default=.*)?", line)
        if match:
            param_name = match.group(1)
            desc_lines = []
            i += 1
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
    Parse _parameter_constraints into a structured format for rendering.
    """

    def map_type(c):
        if isinstance(c, Interval):
            return float if c.type.__name__ == "Real" else int
        if isinstance(c, str):
            if c == "integer":
                return int
            if c in {"number", "float"}:
                return float
            if c == "boolean":
                return "categorical"
            if c == "random_state":
                return int
            if c in {"auto", "warn", "legacy"}:
                return str
        if isinstance(c, type):
            if issubclass(c, Integral):
                return int
            if c.__name__ == "Real":
                return float
            if c in (int, float, str, bool):
                return c
        return None

    def parse_bound(s):
        if not isinstance(s, str):
            return None
        if s.startswith(">="):
            return ("min", float(s[2:]))
        if s.startswith("<="):
            return ("max", float(s[2:]))
        return None

    param_spec = {}

    for param_name, constraints in parameter_constraints.items():
        types = set()
        options = set()
        bounds = {"min": None, "max": None, "closed": "both"}

        if None in constraints:
            types.add(type(None))

        for c in constraints:
            if isinstance(c, Interval):
                t = map_type(c)
                if t:
                    types.add(t)
                bounds["min"], bounds["max"] = c.left, c.right
                bounds["closed"] = c.closed  # <—— new
            elif isinstance(c, StrOptions):
                options.update(c.options)
                types.add("categorical")
            elif isinstance(c, str):
                t = map_type(c)
                if t:
                    types.add(t)
                b = parse_bound(c)
                if b:
                    key, val = b
                    val = int(val) if val.is_integer() else val
                    bounds[key] = val
                elif t is str:
                    options.add(c)
                elif c == "boolean":
                    options.update([True, False])
            elif isinstance(c, type):
                t = map_type(c)
                if t:
                    types.add(t)
            elif isinstance(c, set):
                options.update(c)
                types.add("categorical")
            elif c is None:
                types.add(type(None))

        # Special handling for known tricky params
        if param_name in {"verbose", "n_jobs"}:
            if param_name == "verbose":
                param_spec[param_name] = {
                    "type": "multi",
                    "options": [0, 1, 2, 3],
                    "bounds": bounds,
                }
            elif param_name == "n_jobs":
                # Keep int and None for selection
                param_spec[param_name] = {
                    "type": "multi",
                    "options": [None, -1],  # allow adding numeric range if needed
                    "bounds": bounds,
                }
            continue

        # Finalize type
        if options:
            selected_type = "multi" if len(types) > 1 else "categorical"
        else:
            for t in [float, int, str, bool, "categorical"]:
                if t in types:
                    selected_type = t
                    break
            else:
                selected_type = list(types)[0] if types else None

        param_spec[param_name] = {
            "type": selected_type,
            "bounds": bounds,
            "options": sorted(options, key=str) if options else None,
            "allow_none": type(None) in types,
        }

    return param_spec


def params_from_class(cls) -> dict:
    """
    Return parameter info from class __init__ and _parameter_constraints.
    """
    defaults = get_default_class_values(cls)
    descriptions = get_param_descriptions(cls)
    constraints = getattr(cls, "_parameter_constraints", {})

    params = {}

    for name, constraint in constraints.items():
        info = parse_sklearn_constraints({name: constraint})[name]
        info["default"] = defaults.get(name)
        info["description"] = descriptions.get(name)

        # Multi-type parameters: None, integers, or special strings
        if info.get("options") and (
            len(info["options"]) > 1 or None in info["options"]
        ):
            info["type"] = "multi"
            # Default to None if allowed
            if None in info["options"]:
                info["default"] = None

        # Special case: metric_params (dict) → treat as text input for now
        if name == "metric_params":
            info["type"] = dict
            info["default"] = defaults.get(name)

        params[name] = info

    return params
