import numpy as np

from sklearn.neighbors import KernelDensity
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.pipeline import Pipeline


IsolationForestLearner = IsolationForest
IsolationForestLearner.required_properties = []
IsolationForestLearner.params_description = {
    "n_estimators": {
        "description": "The number of base estimators in the ensemble" + ":" + str(
            IsolationForestLearner._parameter_constraints["n_estimators"][0]),
        "value_type": "range",
        "start": 1, "stop": 1000, "step": 10,
        "default": 10
    }
}

KernelDensityLearner = KernelDensity
KernelDensityLearner.required_properties = []
KernelDensityLearner.params_description = {
    "kernel": {
        "description": str(KernelDensity._parameter_constraints["kernel"][0]),
        "value_type": "choice",
        "set": list(KernelDensity._parameter_constraints["kernel"][0].options),
        "default": "gaussian"
    }
}

KernelDensity.oldfit = KernelDensity.fit


def fit(self, X, y=None, sample_weight=None):
    self.oldfit(X=X, y=y)
    contamination = .1
    self.offset_ = np.percentile(self.score_samples(X), 100.0 * contamination)
    return self


KernelDensityLearner.fit = fit


def predict(self, X):
    decision_func = self.score_samples(X) - self.offset_
    is_inlier = np.ones_like(decision_func, dtype=int)
    is_inlier[decision_func < 0] = -1
    return is_inlier


KernelDensityLearner.predict = predict


class ScaledKernelDensityLearner(Pipeline):
    """Learner class wrapped from scikit-learn's KernelDensity class, with a scaler preprocessor."""

    required_properties = []
    params_description = {
        'scaling': {
            'description': 'Scaling method',
            'family': 'scaling',
            'value_type': 'choice', 'set': ['quantile_normal', 'standard'],
        }
    }

    def __init__(self, scaling="standard"):
        self.scaling = scaling
        if scaling == 'standard':
            scaler = StandardScaler()
        elif scaling == 'quantile_normal':
            scaler = QuantileTransformer(output_distribution='normal')
        else:
            raise ValueError('Unavailable scaling')
        super().__init__([('scaler', scaler), ('learner', KernelDensity())])
