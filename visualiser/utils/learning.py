import hashlib
import json
from joblib import Parallel, delayed

import pandas as pd
import streamlit as st


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
        try:
            learner.fit(X)
            scores = learner.score_samples(X)
        except Exception as e:
            st.warning(f"Failed learner {name}: {e}")
            scores = pd.Series([0] * len(X), index=X.index)
        return name, str(learner), scores

    results = Parallel(n_jobs=min(len(_learner_configs), 4))(
        delayed(fit_and_score)(name, class_path, param_items)
        for name, class_path, param_items in _learner_configs
    )

    return pd.DataFrame(
        {display_name: scores for _, display_name, scores in results}, index=X.index
    )
