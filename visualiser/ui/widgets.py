from typing import Any, Dict

import streamlit as st


# ------------------------- Widget creators -------------------------
def widget_factory(value_type: str):
    return {
        "range": _create_range_widget,
        "real_range": _create_range_widget,
        "choice": _create_choice_widget,
        "boolean": _create_bool_choice_widget,
    }.get(value_type, _create_bool_choice_widget)


def parameter_widget_selection(
    name: str, params_description: Dict[str, Any]
) -> Dict[str, Any]:
    with st.expander(name):
        return {
            param_name: widget_factory(desc["value_type"])(param_name, desc)
            for param_name, desc in params_description.items()
        }


def _create_range_widget(param_name: str, desc: Dict[str, Any]):
    return st.slider(
        label=param_name,
        value=desc.get("default"),
        min_value=desc.get("start"),
        max_value=desc.get("stop"),
        step=desc.get("step"),
        format="%i"
        if isinstance(desc.get("step"), int)
        else f"%0.{len(str(desc.get('step')).split('.')[-1])}f",
    )


def _create_choice_widget(param_name: str, desc: Dict[str, Any]):
    return st.radio(label=param_name, options=desc.get("set"))


def _create_bool_choice_widget(param_name: str, desc: Dict[str, Any]):
    return st.toggle(param_name)
