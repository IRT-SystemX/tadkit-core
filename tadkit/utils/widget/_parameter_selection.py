from typing import Any, Dict

import ipywidgets as widgets
from ipywidgets import HBox, Label, GridBox


def parameter_widget_selection(
    tad_object_name: str,
    params_description: Dict[str, Any],
) -> Dict[str, Any]:
    print(f"  >>> Parameters for learner \033[1m{tad_object_name}\033[0m")

    widget_dict = {}
    boxes = []
    for param_name, param_description in params_description.items():
        widget = {
            "range": _create_range_widget,
            "real_range": _create_range_widget,
            "log_range": _create_logrange_widget,
            "choice": _create_choice_widget,
            "boolean": _create_bool_choice_widget,
        }.get(param_description.get("value_type"), _create_choice_widget)(
            param_description
        )

        widget.description = param_name

        widget_dict[param_name] = widget
        boxes.append(HBox([widget, Label(param_description.get("description"))]))
    display(GridBox(children=boxes))

    return widget_dict


def _create_range_widget(param_description: Dict[str, Any]):
    value = param_description.get("default")
    min_value = param_description.get("start")
    max_value = param_description.get("stop")
    step = param_description.get("step")

    choose_int_slide: bool = all(
        [
            elt is None or isinstance(elt, int)
            for elt in (value, min_value, max_value, step)
        ]
    )

    if choose_int_slide:
        return widgets.IntSlider(
            value=value,
            min=min_value,
            max=max_value,
            step=step,
        )
    return widgets.FloatSlider(
        value=value,
        min=min_value,
        max=max_value,
        step=step,
        readout_format=f".{len(str(step).replace('.', ''))}f",
    )


def _create_logrange_widget(param_description: Dict[str, Any]):
    return widgets.FloatLogSlider(
        value=param_description.get("default"),
        min=param_description.get("log_start"),
        max=param_description.get("log_stop"),
        step=param_description.get("log_step"),
    )


def _create_choice_widget(param_description: Dict[str, Any]):
    choice_widget = widgets.Select(
        options=param_description.get("set"),
    )
    return choice_widget


def _create_bool_choice_widget(param_description: Dict[str, Any]):
    bool_choice_widget = widgets.Checkbox(
        value=param_description.get("default"),
    )
    return bool_choice_widget
