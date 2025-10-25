import numpy as np

from sklearn.neighbors import KernelDensity


# ---------- AD Learner: KernelDensityLearner ----------
KernelDensityLearner = KernelDensity

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

# @todo: check if monkeypatch really works with api, and see if something can be done anyway
