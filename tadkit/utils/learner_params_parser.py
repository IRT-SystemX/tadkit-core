import inspect
import re
from typing import Any, Dict
from numbers import Integral
from sklearn.utils._param_validation import Interval
from sklearn.utils._param_validation import StrOptions


def get_default_class_values(cls) -> Dict[str, Any]:
    """
    Extracts default parameter values from a class's __init__ method.
    """
    sig = inspect.signature(cls.__init__)
    return {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty and k != "self"
    }


def get_param_descriptions(cls) -> Dict[str, str]:
    """
    Parses a NumPy-style docstring of a class to extract parameter descriptions.
    """
    doc = inspect.getdoc(cls)
    if not doc:
        return {}

    param_descriptions = {}
    lines = doc.splitlines()
    in_params_section = False
    start_idx = -1

    # Locate "Parameters" section
    for i, line in enumerate(lines):
        if line.strip().lower() == "parameters":
            if i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"}:
                in_params_section = True
                start_idx = i + 2
                break

    if not in_params_section:
        return {}

    # Parse parameter block
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
    Parses sklearn-style _parameter_constraints into structured metadata.
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

    metadata = {}

    for param_name, constraints in parameter_constraints.items():
        types = set()
        options = set()
        bounds = {"min": None, "max": None}

        for c in constraints:
            # Handle Intervals
            if isinstance(c, Interval):
                t = map_type(c)
                if t:
                    types.add(t)
                bounds["min"], bounds["max"] = c.left, c.right

            # Handle StrOptions (new)
            elif isinstance(c, StrOptions):
                options.update(c.options)
                types.add("categorical")

            # Handle strings
            elif isinstance(c, str):
                t = map_type(c)
                if t:
                    types.add(t)
                b = parse_bound(c)
                if b:
                    key, val = b
                    val = int(val) if val.is_integer() else val
                    bounds[key] = val
                elif t == str:
                    options.add(c)
                elif c == "boolean":
                    options.update([True, False])

            # Handle type objects (e.g., Integral)
            elif isinstance(c, type):
                t = map_type(c)
                if t:
                    types.add(t)

            # Handle categorical sets
            elif isinstance(c, set):
                options.update(c)
                types.add("categorical")

            elif c is None:
                types.add(type(None))

        # Finalize
        if param_name == "verbose":
            metadata[param_name] = {
                "type": "categorical",
                "bounds": bounds,
                "options": [0, 1, 2, 3],
            }
        elif param_name == "n_jobs" and int in types and type(None) in types:
            metadata[param_name] = {
                "type": "categorical",
                "bounds": bounds,
                "options": [None],
            }
        elif options:
            metadata[param_name] = {
                "type": "categorical",
                "bounds": bounds,
                "options": sorted(options, key=str),
            }
        else:
            # Prefer supported types in priority order
            for t in [float, int, str, bool, "categorical"]:
                if t in types:
                    selected = t
                    break
            else:
                selected = list(types)[0] if types else None

            metadata[param_name] = {
                "type": selected,
                "bounds": bounds,
                "options": None,
            }

    return metadata


def enriched_metadata(cls) -> Dict[str, Dict[str, Any]]:
    """
    Combines constraints, defaults, and docstring descriptions for a sklearn-like class.
    """
    defaults = get_default_class_values(cls)
    descriptions = get_param_descriptions(cls)
    constraints = getattr(cls, "_parameter_constraints", {})
    metadata = parse_sklearn_constraints(constraints)

    enriched = {}
    for param in metadata.keys():
        info = metadata.get(
            param,
            {
                "type": None,
                "bounds": {"min": None, "max": None},
                "options": None,
            },
        )
        info["default"] = defaults.get(param)
        info["description"] = descriptions.get(param)
        enriched[param] = info

    return enriched
