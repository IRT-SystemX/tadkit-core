import streamlit as st


def _format_bounds_text(min_val, max_val):
    parts = []
    if min_val is not None:
        parts.append(f"≥ {min_val}")
    else:
        parts.append("no lower bound")
    if max_val is not None:
        parts.append(f"≤ {max_val}")
    else:
        parts.append("no upper bound")
    return ", ".join(parts)


def render_widgets_from_metadata(metadata):
    user_inputs = {}

    for param_name, param_info in metadata.items():
        base_label = param_name.replace("_", " ").capitalize()
        default = param_info.get("default")
        description = param_info.get("description", None)
        min_val = param_info["bounds"]["min"]
        max_val = param_info["bounds"]["max"]
        bounds_text = (
            _format_bounds_text(min_val, max_val)
            if (min_val is not None or max_val is not None)
            else "any value"
        )

        label = f"{base_label} ({bounds_text})"

        # Categorical params
        if param_info["type"] == "categorical":
            options = param_info["options"]
            # Show up to 5 choices, then "..." if many
            if len(options) > 5:
                choices_display = ", ".join(map(str, options[:5])) + ", ..."
            else:
                choices_display = ", ".join(map(str, options))

            label = f"{base_label} (choices: {choices_display})"

            default_idx = options.index(default)
            value = st.selectbox(label, options, index=default_idx, help=description)

            user_inputs[param_name] = value
            continue

        param_type = param_info["type"]

        if param_type == int:
            if min_val is not None and max_val is not None:
                value = st.slider(
                    label,
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=default,
                    step=1,
                    help=description,
                )
            else:
                value = st.number_input(
                    label, value=default, step=1, format="%d", help=description
                )
                st.caption(f"Enter an integer value. Expected range: {bounds_text}.")
            user_inputs[param_name] = value
            continue

        if param_type == float:
            if min_val is not None and max_val is not None:
                value = st.slider(
                    label,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=default,
                    step=0.01,
                    format="%.3f",
                    help=description,
                )
            else:
                value = st.number_input(
                    label, value=default, format="%.3f", help=description
                )
                st.caption(f"Enter a numeric value. Expected range: {bounds_text}.")
            user_inputs[param_name] = value
            continue

        # Fallback text input for unknown types
        user_inputs[param_name] = st.text_input(label)

    return user_inputs


# # ===== Example usage =====

# from tdaad.anomaly_detectors import TopologicalAnomalyDetector
# from klass_parser import enriched_metadata

# metadata = enriched_metadata(TopologicalAnomalyDetector)
# user_params = render_widgets_from_metadata(metadata)

# st.write("User parameters:")
# st.json(user_params)
