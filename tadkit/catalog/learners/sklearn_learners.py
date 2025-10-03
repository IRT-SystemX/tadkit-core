import numpy as np

from sklearn.neighbors import KernelDensity
from sklearn.ensemble import IsolationForest

from tadkit.utils.klass_parser import enriched_metadata


# ---------- AD Learner: IsolationForest ----------
IsolationForestLearner = IsolationForest
IsolationForestLearner.required_properties = []
metadata = enriched_metadata(IsolationForestLearner)
IsolationForestLearner.metadata = metadata

# ---------- AD Learner: KernelDensityLearner ----------
KernelDensityLearner = KernelDensity
KernelDensityLearner.required_properties = []

KernelDensity.oldfit = KernelDensity.fit


def fit(self, X, y=None, sample_weight=None):
    self.oldfit(X=X, y=y)
    contamination = 0.1
    self.offset_ = np.percentile(self.score_samples(X), 100.0 * contamination)
    return self


KernelDensityLearner.fit = fit


def predict(self, X):
    decision_func = self.score_samples(X) - self.offset_
    is_inlier = np.ones_like(decision_func, dtype=int)
    is_inlier[decision_func < 0] = -1
    return is_inlier


KernelDensityLearner.predict = predict
metadata = enriched_metadata(KernelDensityLearner)
[
    metadata.pop(key)
    for key in ["atol", "rtol", "breadth_first", "leaf_size", "metric_params"]
]
KernelDensityLearner.metadata = metadata
