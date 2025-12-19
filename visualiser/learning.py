import hashlib
import json
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import streamlit as st


def prepare_learner_configs(learner_classes, learner_params):
    learner_configs = []
    for learner_class in learner_classes:
        learner_name = learner_class.__name__
        param_dict = learner_params[learner_name]()
        param_items = tuple(sorted(param_dict.items()))
        learner_configs.append((learner_class.__name__, learner_class, param_items))

    cache_key = hashlib.md5(
        json.dumps(learner_configs, sort_keys=True, default=str).encode()
    ).hexdigest()

    return tuple(learner_configs), cache_key


def normalize(a):
    return (
        (a - a.min()) / (a.max() - a.min()) if a.max() != a.min() else np.zeros_like(a)
    )


def fit_and_score(X, name, learner_class, param_items):
    """Fit a learner and return results or error."""
    try:
        learner = learner_class(**dict(param_items))
        learner.fit(X)
        anomaly_scores = learner.score_samples(X)
        scores = pd.Series(normalize(anomaly_scores), index=X.index)
        predictions = pd.Series(learner.predict(X), index=X.index)
        error_msg = None
    except Exception as e:
        scores = pd.Series(0, index=X.index)
        predictions = pd.Series(0, index=X.index)
        learner = None
        error_msg = f"[{name}] Failed: {e}"
    return {
        "name": name,
        "learner_str": str(learner),
        "scores": scores,
        "predictions": predictions,
        "error": error_msg,
    }


@st.cache_data(show_spinner="Computing anomalies...")
def compute_anomalies_parallel(X, learner_configs, cache_key=None, n_jobs=4):
    """
    Runs anomaly detection learners in parallel, returns scores, predictions, and any errors.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        learner_configs (list): List of (name, class_path, param_items) tuples.
        cache_key (Any): Extra value used to control Streamlit caching behavior.
        n_jobs (int): Maximum number of parallel jobs. Default is 4.

    Returns:
        scores_df (pd.DataFrame): Anomaly scores per learner.
        preds_df (pd.DataFrame): Binary predictions per learner.
        errors (list): List of error messages for failed learners.
    """
    if not learner_configs:
        return (
            pd.DataFrame(index=X.index),
            pd.DataFrame(index=X.index),
            ["No learners provided."],
        )

    # Ensure n_jobs doesn't exceed number of learners
    n_jobs = min(n_jobs, len(learner_configs))

    # Run learners in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_and_score)(X, name, learner_class, param_items)
        for name, learner_class, param_items in learner_configs
    )

    # Construct DataFrames
    scores_df = pd.DataFrame({r["name"]: r["scores"] for r in results}, index=X.index)
    preds_df = pd.DataFrame(
        {r["name"]: r["predictions"] for r in results}, index=X.index
    )

    # Optional: sort columns alphabetically
    scores_df = scores_df.reindex(sorted(scores_df.columns), axis=1)
    preds_df = preds_df.reindex(sorted(preds_df.columns), axis=1)

    # Collect errors
    errors = [r["error"] for r in results if r["error"]]

    return scores_df, preds_df, errors
