import json
import ipywidgets as widgets
from IPython.display import display
import streamlit as st


# -------------------------------
# 1. Value Sanitization Utility
# -------------------------------
def sanitize_default(default, min_val, max_val, ptype, closed="both", allow_none=False):
    """Clamp and adjust defaults based on type and bounds."""

    def to_type(val):
        if val is None:
            return None
        try:
            return ptype(val)
        except (ValueError, TypeError):
            return None

    if allow_none and default is None:
        return None

    default_val = to_type(default)
    min_val = to_type(min_val)
    max_val = to_type(max_val)

    if default_val is None:
        if min_val is not None:
            return (
                min_val + (1e-12 if ptype == float else 1)
                if closed in ("right", "neither")
                else min_val
            )
        return 0 if ptype == int else 0.0

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


# -------------------------------
# 2. Widget Factory
# -------------------------------
class WidgetFactory:
    """Factory for creating UI widgets or no-UI representations."""

    def __init__(self, frontend="ipywidgets"):
        self.frontend = frontend

    def make_numeric(
        self, label, default, min_val, max_val, ptype, description, closed, allow_none
    ):
        """Create a numeric input widget (Streamlit / ipywidgets / no_ui)."""

        # ✅ Preserve None if allowed — don't sanitize it
        if allow_none and default is None:
            if self.frontend == "st":
                return st.selectbox(label, [None], index=0, help=description)
            elif self.frontend == "ipywidgets":
                return widgets.Dropdown(
                    options=[None],
                    value=None,
                    description=label,
                    style={"description_width": "initial"},
                    tooltip=description,
                )
            elif self.frontend == "no_ui":
                return {
                    "type": "number",
                    "label": label,
                    "default": None,
                    "description": description,
                    "nullable": True,
                }

        # 🧠 Only sanitize if we actually have a numeric default
        default = sanitize_default(default, min_val, max_val, ptype, closed, allow_none)

        if self.frontend == "st":
            step = 1 if ptype == int else 0.01
            fmt = "%d" if ptype == int else "%.6f"

            # ✅ Ensure type consistency for all numeric arguments
            def cast(val):
                return None if val is None else ptype(val)

            kwargs = {
                "label": label,
                "value": cast(default),
                "step": cast(step),
                "format": fmt,
                "help": description,
            }
            if min_val is not None:
                kwargs["min_value"] = cast(min_val)
            if max_val is not None:
                kwargs["max_value"] = cast(max_val)
            return st.number_input(**kwargs)

        elif self.frontend == "ipywidgets":
            cls = widgets.BoundedIntText if ptype == int else widgets.BoundedFloatText
            return cls(
                value=default,
                min=min_val if min_val is not None else -1e6,
                max=max_val if max_val is not None else 1e6,
                description=label,
                style={"description_width": "initial"},
                tooltip=description,
            )

        elif self.frontend == "no_ui":
            return {
                "type": "number",
                "label": label,
                "default": default,
                "min": min_val,
                "max": max_val,
                "description": description,
                "nullable": allow_none,
            }

    def make_dropdown(self, label, options, default, description):
        """Create a dropdown for ipywidgets, streamlit, or no_ui safely."""
        if not options:
            options = ["(none)"]

        # Ensure valid default
        default_val = default if default in options else options[0]

        if self.frontend == "st":
            return st.selectbox(
                label,
                options,
                index=options.index(default_val),
                help=description,
            )

        elif self.frontend == "ipywidgets":
            return widgets.Dropdown(
                options=options,
                value=default_val,
                description=label,
                style={"description_width": "initial"},
                tooltip=description,
            )

        elif self.frontend == "no_ui":
            return {
                "type": "dropdown",
                "label": label,
                "options": options,
                "default": default_val,
                "description": description,
            }

    def make_text(self, label, default, description):
        if self.frontend == "st":
            return st.text_input(label, value=str(default or ""), help=description)
        elif self.frontend == "ipywidgets":
            return widgets.Text(
                value=str(default or ""),
                description=label,
                style={"description_width": "initial"},
                tooltip=description,
            )
        elif self.frontend == "no_ui":
            return {
                "type": "text",
                "label": label,
                "default": default,
                "description": description,
            }

    def make_dict(self, label, default, description):
        json_str = json.dumps(default or {}, indent=2)
        if self.frontend == "st":
            return st.text_area(label, value=json_str, help=description, height=100)
        elif self.frontend == "ipywidgets":
            return widgets.Textarea(
                value=json_str,
                description=label,
                layout=widgets.Layout(width="100%", height="80px"),
                tooltip=description,
            )
        elif self.frontend == "no_ui":
            return {
                "type": "dict",
                "label": label,
                "default": default,
                "description": description,
            }


# -------------------------------
# 3. Main Rendering Function
# -------------------------------


def render_widgets_from_params(params: dict, frontend="ipywidgets"):
    """Render or return parameter specs as widgets, streamlit inputs, or dicts."""
    factory = WidgetFactory(frontend)
    user_inputs = {}

    for name, info in params.items():
        label = name.replace("_", " ").capitalize()
        ptype = info.get("type")
        default = info.get("default")
        description = info.get("description", "")
        bounds = info.get("bounds", {}) or {}
        min_val, max_val, closed = (
            bounds.get("min"),
            bounds.get("max"),
            bounds.get("closed", "both"),
        )
        options = info.get("options") or []
        allow_none = bool(info.get("allow_none", False) or (None in options))

        # Numeric types
        if ptype in (int, float):
            user_inputs[name] = factory.make_numeric(
                label, default, min_val, max_val, ptype, description, closed, allow_none
            )

        # Dict types
        elif ptype == dict:
            user_inputs[name] = factory.make_dict(label, default, description)

        # Categorical or multi-select
        elif ptype in ("categorical", "multi"):
            default_val = (
                default if default in options else (options[0] if options else None)
            )
            user_inputs[name] = factory.make_dropdown(
                label, options, default_val, description
            )

        # Special NoneType with only None allowed
        elif ptype is type(None) and allow_none:
            # Use a dropdown with string "None" because ipywidgets cannot display Python None
            user_inputs[name] = factory.make_dropdown(
                label, options=[None], default=None, description=description
            )

        # Fallback text
        else:
            user_inputs[name] = factory.make_text(label, default, description)

    # Render for ipywidgets
    if frontend == "ipywidgets":
        display(widgets.VBox(list(user_inputs.values())))

    # Function to retrieve values
    def get_values():
        if frontend == "no_ui":
            return {k: v for k, v in user_inputs.items()}

        values = {}
        for k, w in user_inputs.items():
            val = w.value if frontend == "ipywidgets" else w

            # Convert string "None" back to Python None
            if (
                params[k]["type"] is None
                and params[k].get("allow_none", False)
                and val == "None"
            ):
                val = None

            # Deserialize dict if needed
            if params[k]["type"] == dict and isinstance(val, str):
                val = json.loads(val)

            values[k] = val

        return values

    return user_inputs, get_values
