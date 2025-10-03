import hashlib
import json
from typing import Any, Dict
from joblib import Parallel, delayed

import pandas as pd
import streamlit as st


# ------------------------- Helper Functions -------------------------
def prepare_learner_configs(learner_names, learner_params, available_learners):
    learner_configs = []
    for name in learner_names:
        cls = available_learners[name]
        class_path = f"{cls.__module__}.{cls.__name__}"
        param_items = tuple(sorted(learner_params.get(name, {}).items()))
        learner_configs.append((name, class_path, param_items))

    cache_key = hashlib.md5(
        json.dumps(learner_configs, sort_keys=True, default=str).encode()
    ).hexdigest()

    return tuple(learner_configs), cache_key


@st.cache_data
def compute_anomalies_parallel(X, _learner_configs, cache_key):
    def import_class(class_path):
        module_path, class_name = class_path.rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)

    def fit_and_score(name, class_path, param_items):
        LearnerClass = import_class(class_path)
        params = dict(param_items)
        learner = LearnerClass(**params)
        learner.fit(X)
        scores = learner.score_samples(X)
        return name, str(learner), scores

    results = Parallel(n_jobs=min(len(_learner_configs), 4))(
        delayed(fit_and_score)(name, class_path, param_items)
        for name, class_path, param_items in _learner_configs
    )

    return pd.DataFrame(
        {display_name: scores for _, display_name, scores in results}, index=X.index
    )


@st.cache_data
def convert_for_download(df: pd.DataFrame):
    return df.to_csv().encode("utf-8")


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
