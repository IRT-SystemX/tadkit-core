def format_bounds_text(min_val, max_val):
    if min_val is None and max_val is None:
        return ""
    if min_val is None:
        return f"≤ {max_val}"
    if max_val is None:
        return f"≥ {min_val}"
    return f"{min_val}-{max_val}"


def sanitize_default(default, min_val, max_val, ptype):
    # Helper to convert safely or return None
    def to_type(val):
        if val is None:
            return None
        try:
            return ptype(val)
        except (ValueError, TypeError):
            return None

    default_val = to_type(default)
    min_val = to_type(min_val)
    max_val = to_type(max_val)

    if default_val is None:
        if min_val is not None:
            return min_val
        return 0 if ptype == int else 0.0

    if min_val is not None and default_val < min_val:
        return min_val
    if max_val is not None and default_val > max_val:
        return max_val

    return default_val


def render_widgets_from_metadata(metadata, frontend="ipywidgets"):
    """
    Dynamically render widgets from metadata using either:
    - Streamlit (frontend='st')
    - Ipywidgets (frontend='ipywidgets')

    Returns a dict of {param_name: widget or value}.
    """
    user_inputs = {}

    # Try imports only if needed
    if frontend == "st":
        import streamlit as st
    else:
        import ipywidgets as widgets
        from IPython.display import display

    # --- numeric input subfunction ---
    def render_numeric_input(label, description, default, min_val, max_val, ptype):
        if frontend == "st":
            default = sanitize_default(default, min_val, max_val, ptype)
            if ptype == int:
                return st.number_input(
                    label,
                    min_value=int(min_val) if min_val is not None else None,
                    max_value=int(max_val) if max_val is not None else None,
                    value=int(default),
                    step=1,
                    format="%d",
                    help=description,
                )
            else:
                return st.number_input(
                    label,
                    min_value=float(min_val) if min_val is not None else None,
                    max_value=float(max_val) if max_val is not None else None,
                    value=float(default),
                    format="%.3f",
                    help=description,
                )

        else:  # ipywidgets mode
            default = sanitize_default(default, min_val, max_val, ptype)

            if ptype == int:
                widget = widgets.BoundedIntText(
                    value=default,
                    min=min_val if min_val is not None else -1_000_000,
                    max=max_val if max_val is not None else 1_000_000,
                    description=label,
                    style={"description_width": "initial"},
                    tooltip=description,
                )
            else:
                widget = widgets.BoundedFloatText(
                    value=default,
                    min=min_val if min_val is not None else -1e6,
                    max=max_val if max_val is not None else 1e6,
                    description=label,
                    style={"description_width": "initial"},
                    tooltip=description,
                )
            return widget

    # --- main widget builder ---
    for param_name, param_info in metadata.items():
        base_label = param_name.replace("_", " ").capitalize()
        ptype = param_info["type"]
        default = param_info.get("default")
        description = param_info.get("description", "")
        bounds = param_info.get("bounds", {})
        min_val = bounds.get("min")
        max_val = bounds.get("max")

        bounds_text = format_bounds_text(min_val, max_val)
        label = f"{base_label} ({bounds_text})" if bounds_text else base_label

        # Boolean categorical
        if ptype == "categorical" and set(param_info.get("options", [])) == {
            True,
            False,
        }:
            if frontend == "st":
                val = st.checkbox(label, value=default or False, help=description)
            else:
                val = widgets.Checkbox(
                    value=bool(default), description=label, tooltip=description
                )
            user_inputs[param_name] = val
            continue

        # General categorical
        if ptype == "categorical":
            options = param_info.get("options", [])
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

        # Numeric
        if ptype in (int, float):
            val = render_numeric_input(
                label, description, default, min_val, max_val, ptype
            )
            user_inputs[param_name] = val
            continue

        # Text fallback
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

    # Display (ipywidgets only)
    if frontend == "ipywidgets":
        display(widgets.VBox(list(user_inputs.values())))

    return user_inputs
