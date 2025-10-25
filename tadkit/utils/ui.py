import json
import ipywidgets as widgets
from IPython.display import display
import streamlit as st


def format_bounds_text(min_val, max_val):
    if min_val is None and max_val is None:
        return ""
    if min_val is None:
        return f"≤ {max_val}"
    if max_val is None:
        return f"≥ {min_val}"
    return f"{min_val}-{max_val}"


def sanitize_default(default, min_val, max_val, ptype, closed="both", allow_none=False):
    """
    Adjusts a default value to respect open/closed bounds and optional None.
    """

    def to_type(val):
        if val is None:
            return None
        try:
            return ptype(val)
        except (ValueError, TypeError):
            return None

    # If None is allowed, preserve it
    if allow_none and default is None:
        return None

    default_val = to_type(default)
    min_val = to_type(min_val)
    max_val = to_type(max_val)

    # If default is None and not allowed, choose a safe in-range value
    if default_val is None:
        if min_val is not None:
            # If lower bound is open, step slightly above
            if closed in ("right", "neither"):
                return min_val + (1e-12 if ptype == float else 1)
            return min_val
        return 0 if ptype == int else 0.0

    # Clamp respecting open/closed interval
    if min_val is not None:
        if closed in ("both", "left") and default_val < min_val:
            default_val = min_val
        elif closed in ("right", "neither") and default_val <= min_val:
            default_val = min_val + (1e-12 if ptype == float else 1)

    if max_val is not None:
        if closed in ("both", "right") and default_val > max_val:
            default_val = max_val
        elif closed in ("left", "neither") and default_val >= max_val:
            default_val = max_val - (1e-12 if ptype == float else 1)

    return default_val


def render_widgets_from_params(params: dict, frontend="ipywidgets"):
    """
    Render widgets from params spec returned by params_from_class().
    Returns (widgets_dict, get_values_fn).
    """

    def safe_parse_dict(raw_val):
        if not raw_val:
            return None
        try:
            return json.loads(raw_val)
        except json.JSONDecodeError:
            return None

    # Helper for numeric widgets
    def render_numeric(
        label,
        default,
        min_val,
        max_val,
        ptype,
        description="",
        closed="both",
        allow_none=False,
    ):
        if allow_none and default is None:
            # Preserve it as None — don't sanitize into numeric
            default = None
        else:
            default = sanitize_default(
                default, min_val, max_val, ptype, closed, allow_none
            )
        if frontend == "st":
            return st.number_input(
                label,
                value=default,
                min_value=min_val,
                max_value=max_val,
                step=1 if ptype == int else 0.01,
                format="%d" if ptype == int else "%.3f",
                help=description,
            )
        else:
            if ptype == int:
                return widgets.BoundedIntText(
                    value=default,
                    min=min_val if min_val is not None else -1_000_000,
                    max=max_val if max_val is not None else 1_000_000,
                    description=label,
                    style={"description_width": "initial"},
                    tooltip=description,
                )
            else:
                return widgets.BoundedFloatText(
                    value=default,
                    min=min_val if min_val is not None else -1e6,
                    max=max_val if max_val is not None else 1e6,
                    description=label,
                    style={"description_width": "initial"},
                    tooltip=description,
                )

    user_inputs = {}

    for param_name, info in params.items():
        label = param_name.replace("_", " ").capitalize()
        ptype = info["type"]
        default = info.get("default")
        description = info.get("description", "")
        bounds = info.get("bounds", {})
        min_val = bounds.get("min")
        max_val = bounds.get("max")
        closed = bounds.get("closed", "both")
        options = info.get("options") or []
        allow_none = info.get("allow_none", False) or None in options

        # --- Multi-type parameter ---
        if ptype == "multi":
            if frontend == "st":
                val = st.selectbox(
                    label,
                    options,
                    index=options.index(default) if default in options else 0,
                    help=description,
                )
            else:
                val = widgets.Dropdown(
                    options=options,
                    value=default if default in options else options[0],
                    description=label,
                    style={"description_width": "initial"},
                    tooltip=description,
                )
            user_inputs[param_name] = val
            continue

        # --- Categorical ---
        if ptype == "categorical":
            if not options:
                options = ["(none)"]
            default_val = default if default in options else options[0]
            if frontend == "st":
                val = st.selectbox(
                    label, options, index=options.index(default_val), help=description
                )
            else:
                val = widgets.Dropdown(
                    options=options,
                    value=default_val,
                    description=label,
                    style={"description_width": "initial"},
                    tooltip=description,
                )
            user_inputs[param_name] = val
            continue

        # --- Numeric ---
        # --- Optional numeric: render None selector instead of numeric input ---
        if ptype in (int, float):
            if allow_none and default is None:
                if frontend == "st":
                    val = st.selectbox(label, [None], index=0, help=description)
                else:
                    val = widgets.Dropdown(
                        options=[None],
                        value=None,
                        description=label,
                        style={"description_width": "initial"},
                        tooltip=description,
                    )
                user_inputs[param_name] = val
                continue
            val = render_numeric(
                label, default, min_val, max_val, ptype, description, closed, allow_none
            )
            user_inputs[param_name] = val
            continue

        # --- Dict / complex ---
        if ptype == dict:
            json_str = json.dumps(default or {}, indent=2)
            if frontend == "st":
                val = st.text_area(label, value=json_str, help=description, height=100)
            else:
                val = widgets.Textarea(
                    value=json_str,
                    description=label,
                    layout=widgets.Layout(width="100%", height="80px"),
                    tooltip=description,
                )
            user_inputs[param_name] = val
            continue

        # --- Text fallback ---
        if frontend == "st":
            val = st.text_input(label, value=str(default or ""), help=description)
        else:
            val = widgets.Text(
                value=str(default or ""),
                description=label,
                style={"description_width": "initial"},
                tooltip=description,
            )
        user_inputs[param_name] = val

    if frontend == "ipywidgets":
        display(widgets.VBox(list(user_inputs.values())))

    # --- Separate retrieval function ---
    def get_values():
        values = {}
        for k, w in user_inputs.items():
            raw = w.value if frontend == "ipywidgets" else w
            if params[k]["type"] == dict:
                values[k] = safe_parse_dict(raw)
            else:
                values[k] = raw
        return values

    return user_inputs, get_values
