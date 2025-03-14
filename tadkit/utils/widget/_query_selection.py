from typing import Any, Dict

import ipywidgets as widgets
from ipywidgets import HBox, Label
import pandas as pd


def query_widget_selection(
        formalizer_name: str,
        query_description: Dict[str, Any],
) -> Dict[str, Any]:
    display(formalizer_name)

    widget_dict = {}
    for param_name, param_description in query_description.items():
        widget = {
            "space": _create_space_widget,
            "time_interval": _create_timeinterval_widget,
            "time": _create_time_widget,
            "bool": _create_bool_choice_widget,
        }.get(param_description.get("family"), _create_time_widget)(param_description)

        widget.description = param_name

        widget_dict[param_name] = widget
        display(HBox([widget, Label(param_description.get("description"))]))

    return widget_dict


def _create_bool_choice_widget(param_description: Dict[str, Any]):
    bool_choice_widget = widgets.Checkbox(
        value=param_description.get("default"),
    )
    return bool_choice_widget


def _create_space_widget(param_description: Dict[str, Any]):
    space_widget = widgets.SelectMultiple(
        options=param_description.get("set"),
        value=tuple(param_description.get("set")),
    )
    return space_widget


def _create_timeinterval_widget(param_description: Dict[str, Any]):
    start_date = param_description.get("start")
    end_date = param_description.get("stop")
    dates = pd.date_range(start_date, end_date, freq='D')
    options = [(date.strftime(" %d %b %Y "), date) for date in dates]
    index = (0, len(options) - 1)
    selection_range_slider = widgets.SelectionRangeSlider(
        options=options,
        index=index,
        orientation="horizontal",
        layout={"width": "500px"},
    )
    return selection_range_slider


def _create_time_widget(param_description: Dict[str, Any]):
    start_value = param_description.get("default") if hasattr(param_description, "default") else param_description.get(
        "start")
    time_widget = widgets.BoundedIntText(
        value=start_value,
        min=param_description.get("start"),
        max=param_description.get("stop"),
    )
    return time_widget
