import streamlit as st


def format_bounds_text(minimum, maximum):
    if minimum is not None and maximum is not None:
        return f"{minimum} to {maximum}"
    elif minimum is not None:
        return f"≥ {minimum}"
    elif maximum is not None:
        return f"≤ {maximum}"
    else:
        return "any value"


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


def render_numeric_input(label, description, default, min_val, max_val, ptype):
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
    else:  # float
        return st.number_input(
            label,
            min_value=float(min_val) if min_val is not None else None,
            max_value=float(max_val) if max_val is not None else None,
            value=float(default),
            format="%.3f",
            help=description,
        )


def render_widgets_from_metadata(metadata):
    user_inputs = {}

    for param_name, param_info in metadata.items():
        base_label = param_name.replace("_", " ").capitalize()
        ptype = param_info["type"]
        default = param_info.get("default")
        description = param_info.get("description", "")
        bounds = param_info.get("bounds", {})
        min_val = bounds.get("min")
        max_val = bounds.get("max")

        bounds_text = format_bounds_text(min_val, max_val)
        label = f"{base_label} ({bounds_text})"

        # Handle categorical booleans as checkbox
        if ptype == "categorical" and set(param_info.get("options", [])) == {
            True,
            False,
        }:
            default = default if default is not None else False
            value = st.checkbox(label, value=default, help=description)
            user_inputs[param_name] = value
            continue

        # Handle other categorical as selectbox
        if ptype == "categorical":
            options = param_info.get("options", [])
            if not options:
                value = None
            else:
                if len(options) > 5:
                    choices_display = ", ".join(map(str, options[:5])) + ", ..."
                else:
                    choices_display = ", ".join(map(str, options))

                label = f"{base_label} (choices: {choices_display})"
                default_idx = options.index(default) if default in options else 0
                value = st.selectbox(
                    label, options, index=default_idx, help=description
                )

            user_inputs[param_name] = value
            continue

        # Handle numeric types
        if ptype in (int, float):
            value = render_numeric_input(
                label, description, default, min_val, max_val, ptype
            )
            user_inputs[param_name] = value
            continue

        # Fallback to text input for unknown types
        value = st.text_input(
            label, value=str(default) if default is not None else "", help=description
        )
        user_inputs[param_name] = value

    return user_inputs
