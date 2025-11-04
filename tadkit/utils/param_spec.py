from typing import Dict, Any
import inspect
from numbers import Integral

import math
from sklearn.utils._param_validation import Interval, StrOptions

UI_INF = 1e6


def get_default_class_values(cls) -> Dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    return {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty and k != "self"
    }


def get_param_descriptions(cls) -> Dict[str, str]:
    import re

    doc = inspect.getdoc(cls)
    if not doc:
        return {}

    param_descriptions = {}
    lines = doc.splitlines()
    in_params_section = False

    # locate "Parameters" section
    for i, line in enumerate(lines):
        if (
            line.strip().lower() in {"parameters", "attributes"}
            and i + 1 < len(lines)
            and set(lines[i + 1].strip()) == {"-"}
        ):
            start_idx = i + 2
            break
    else:
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


def parse_sklearn_constraints(parameter_constraints) -> Dict[str, Dict[str, Any]]:
    """Convert sklearn-style constraints to structured type info."""

    def map_type(c):
        if isinstance(c, Interval):
            return float if c.type.__name__ == "Real" else int
        if isinstance(c, str):
            if c == "integer":
                return int
            if c in {"number", "float"}:
                return float
            if c == "boolean":
                return bool
            if c == "random_state":
                return int
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

        if not constraints:
            param_spec[param_name] = {
                "type": None,
                "bounds": bounds,
                "options": None,
                "allow_none": True,
            }
            continue

        if None in constraints:
            types.add(type(None))

        for c in constraints:
            if isinstance(c, Interval):
                t = map_type(c)
                if t:
                    types.add(t)
                bounds["min"], bounds["max"], bounds["closed"] = (
                    c.left,
                    c.right,
                    c.closed,
                )
            elif isinstance(c, StrOptions):
                options.update(c.options)
                types.add(str)
            elif isinstance(c, str):
                t = map_type(c)
                if t:
                    types.add(t)
                b = parse_bound(c)
                if b:
                    key, val = b
                    val = int(val) if val.is_integer() else val
                    bounds[key] = val
            elif isinstance(c, type):
                t = map_type(c)
                if t:
                    types.add(t)
            elif isinstance(c, set):
                options.update(c)
                types.add(str)

        # Determine main type
        if options:
            selected_type = "categorical"
        else:
            selected_type = next(iter(types)) if types else None

        param_spec[param_name] = {
            "type": selected_type,
            "bounds": bounds,
            "options": sorted(options, key=str) if options else None,
            "allow_none": type(None) in types,
        }

    return param_spec


# --- Composite type resolver ---
def anchor_type_to_default(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    If parameter has multiple possible types or categories,
    restrict to the one matching the default’s type.
    """
    default = entry.get("default")
    if default is inspect._empty:
        return entry

    default_type = type(default)
    type_field = entry.get("type")
    options = entry.get("options")

    # if the default is None, keep only None
    if default is None:
        entry["type"] = type(None)
        entry["options"] = None
        entry["allow_none"] = True
        return entry

    # if type_field represents a composite (list, tuple, "multi", or set)
    if isinstance(type_field, (list, tuple, set)) or type_field in (
        "multi",
        "categorical",
    ):
        entry["type"] = default_type

    # if there are options but default doesn’t match option type
    if options:
        if not isinstance(default, str) and isinstance(options[0], str):
            # default is not str, drop options entirely
            entry["options"] = None
        elif isinstance(default, str) and all(
            isinstance(o, (int, float)) for o in options
        ):
            entry["options"] = None

    # if options and allow_none but default not None -> drop allow_none
    if entry.get("allow_none") and default is not None:
        entry["allow_none"] = False

    return entry


# --- Widget inference ---
def determine_widget(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Infer a UI widget and arguments from parameter metadata."""
    t = entry.get("type")
    options = entry.get("options")
    bounds = entry.get("bounds", {"min": None, "max": None})
    default = entry.get("default")
    allow_none = entry.get("allow_none", False)

    widget, widget_args = None, {}

    # 1️⃣ Categorical / enum-like
    if options:
        widget = "select"
        opts = list(options)
        if allow_none:
            opts = [None] + opts
        widget_args = {
            "options": opts,
            "default": default if default in opts else opts[0],
        }

    # 2️⃣ Numeric sliders
    elif t in (int, float):
        min_val, max_val = bounds.get("min"), bounds.get("max")
        # clean infinities
        if isinstance(min_val, (int, float)) and not math.isfinite(min_val):
            min_val = -UI_INF if t is int else 0.0
        if isinstance(max_val, (int, float)) and not math.isfinite(max_val):
            max_val = UI_INF if t is int else 10.0
        step = 1 if t is int else 0.1
        widget = "slider"
        widget_args = {
            "min": min_val,
            "max": max_val,
            "step": step,
            "default": default if default is not None else (0 if t is int else 0.0),
        }

    # 3️⃣ Booleans
    elif t is bool:
        widget = "checkbox"
        widget_args["default"] = bool(default)

    # 4️⃣ Callable / dict / list
    elif t in ("callable", dict, list):
        widget = "text"
        widget_args["default"] = str(default) if default is not None else ""

    # 5️⃣ Strings or None
    elif t in (str, type(None)):
        widget = "text"
        widget_args["default"] = "" if default is None else str(default)

    # 6️⃣ Fallback
    else:
        widget = "text"
        widget_args["default"] = str(default) if default is not None else ""

    entry["widget"] = widget
    entry["widget_args"] = widget_args
    return entry


# --- New unified builder ---
def params_from_class(cls) -> Dict[str, Dict[str, Any]]:
    """
    Combine default values, docstrings, and sklearn parameter constraints
    into a unified param specification dictionary.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        param_name -> {
            'default': Any,
            'type': type or str,
            'bounds': {'min':..., 'max':..., 'closed':...},
            'options': list[str] or None,
            'allow_none': bool,
            'description': str
        }
    """
    defaults = get_default_class_values(cls)
    docs = get_param_descriptions(cls)
    constraints = getattr(cls, "_parameter_constraints", {})

    parsed_constraints = parse_sklearn_constraints(constraints)

    all_params = set(defaults) | set(parsed_constraints) | set(docs)
    spec = {}

    for name in all_params:
        if name.endswith("_"):
            continue
        default = defaults.get(name, inspect._empty)
        entry = {
            "default": default,
            "description": docs.get(name, ""),
        }

        if name in parsed_constraints:
            entry.update(parsed_constraints[name])
        else:
            entry.update(
                {
                    "type": type(defaults[name]).__name__
                    if name in defaults and defaults[name] is not None
                    else None,
                    "bounds": {"min": None, "max": None, "closed": "both"},
                    "options": None,
                    "allow_none": defaults.get(name) is None,
                }
            )

        inferred_type = type(default)
        entry.update(parsed_constraints.get(name, {}))
        entry["type"] = inferred_type
        entry["allow_none"] = entry.get("allow_none", False) or default is None

        entry = anchor_type_to_default(entry)
        entry = determine_widget(entry)

        spec[name] = entry

    return spec
