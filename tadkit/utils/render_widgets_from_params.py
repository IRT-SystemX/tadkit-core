from tadkit.utils.ui import WidgetFactory


import ipywidgets as widgets
from IPython.display import display


import json


def render_widgets_from_params(params: dict, frontend="ipywidgets"):
    """
    Render or return parameter specs as widgets, Streamlit inputs, or dicts.

    Adapted to new param spec including `widget` and `widget_args`.
    """
    factory = WidgetFactory(frontend)
    user_inputs = {}

    for name, info in params.items():
        label = name.replace("_", " ").capitalize()
        widget_type = info.get("widget")
        widget_args = info.get("widget_args", {})
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
        allow_none = bool(info.get("allow_none", False))

        # 1️⃣ Use widget hint if available
        if widget_type == "slider":
            user_inputs[name] = factory.make_numeric(
                label,
                widget_args.get("default", default),
                widget_args.get("min", min_val),
                widget_args.get("max", max_val),
                ptype or float,
                description,
                closed,
                allow_none,
            )

        elif widget_type == "select":
            opts = widget_args.get("options", options)
            user_inputs[name] = factory.make_dropdown(
                label,
                opts,
                widget_args.get("default", default),
                description,
            )

        elif widget_type == "checkbox":
            user_inputs[name] = factory.make_dropdown(
                label,
                [True, False],
                widget_args.get("default", bool(default)),
                description,
            )

        elif widget_type == "text":
            user_inputs[name] = factory.make_text(
                label,
                widget_args.get("default", default),
                description,
            )

        else:
            # 2️⃣ Fallback to type-based inference
            if ptype in (int, float):
                user_inputs[name] = factory.make_numeric(
                    label,
                    default,
                    min_val,
                    max_val,
                    ptype,
                    description,
                    closed,
                    allow_none,
                )
            elif ptype == dict:
                user_inputs[name] = factory.make_dict(label, default, description)
            elif ptype in ("categorical", "multi"):
                user_inputs[name] = factory.make_dropdown(
                    label, options, default, description
                )
            elif ptype is type(None) and allow_none:
                user_inputs[name] = factory.make_dropdown(
                    label, [None], None, description
                )
            else:
                user_inputs[name] = factory.make_text(label, default, description)

    # 3️⃣ Render for ipywidgets
    if frontend == "ipywidgets":
        display(widgets.VBox(list(user_inputs.values())))

    # 4️⃣ Unified get_values() method
    def get_values():
        if frontend == "no_ui":
            return {k: v for k, v in user_inputs.items()}

        values = {}
        for k, w in user_inputs.items():
            val = w.value if frontend == "ipywidgets" else w

            # Convert string "None" → Python None
            if (
                params[k].get("type") == type(None)
                and params[k].get("allow_none", False)
                and (val == "None" or val == "")
            ):
                val = None

            # Parse dict if entered as JSON text
            if params[k].get("type") == dict and isinstance(val, str):
                try:
                    val = json.loads(val)
                except json.JSONDecodeError:
                    pass

            values[k] = val

        return values

    return user_inputs, get_values
