from typing import Dict, Any

import numpy as np


def domain_mapper(param_description):
    return {
        "range": _create_integer_range_distribution,
        "real_range": _create_real_range_distribution,
        "log_range": _create_logrange_distribution,
        "choice": lambda description: description.get("set"),
        "boolean": lambda description: [True, False],
    }.get(param_description.get("value_type"), "boolean")


def numerical_domain_mapper(param_description):
    return {
        "range": _create_integer_range_distribution,
        "real_range": _create_real_range_distribution,
        "log_range": _create_logrange_distribution,
        "choice": lambda description: np.arange(len(description.get("set"))),
        "boolean": lambda description: [0, 1],
    }.get(param_description.get("value_type"), "boolean")


def param_distributions(
    params_description: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        param_name: domain_mapper(param_description)(param_description)
        for param_name, param_description in params_description.items()
    }


def _create_integer_range_distribution(param_description: Dict[str, Any]):
    return np.arange(
        start=param_description.get("start"),
        stop=param_description.get("stop") if "stop" in param_description else -1,
        step=param_description.get("step"),
    )


def _create_real_range_distribution(param_description: Dict[str, Any]):
    num = int(
        (param_description.get("stop") - param_description.get("start"))
        / param_description.get("step")
    )
    return np.linspace(
        start=param_description.get("start"),
        stop=param_description.get("stop"),
        num=num,
    )


def _create_logrange_distribution(param_description: Dict[str, Any]):
    return np.logspace(
        start=param_description.get("log_start"),
        stop=param_description.get("log_stop"),
    )
