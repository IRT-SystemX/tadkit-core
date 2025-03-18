import numpy as np


def get_wrapped_kcplearner():
    """Return the TADlearner wrapped from Kernel Change Point Detection's KcpLearner method.

    The function is intended for use if the dependency is available.
    """

    from kcpdi.kcp_ss_learner import KcpLearner

    KcpLearner.params_description["kernel"] = {
        "description": "One of (linear, cosine, rbf)",
        "value_type": "choice",
        "set": ["linear", "cosine", "rbf"],
        "default": "linear",
    }
    KcpLearner.params_description["max_n_time_points"] = {
        "description": "max_n_time_points",
        "value_type": "range",
        "start": 1000,
        "stop": 10000,
        "step": 500,
        "default": 1000,
    }
    KcpLearner.params_description["expected_frac_anomaly"] = {
        "description": "expected_frac_anomaly",
        "value_type": "real_range",
        "start": 0.00001,
        "stop": 0.5,
        "step": 0.0001,
        "default": 0.00001,
    }

    KcpLearner.oldfit = KcpLearner.fit

    # we rebind fit to set the offset_ parameter so as to be able to implement predict like other methods do
    def fit(self, X, y=None, sample_weight=None):
        self.oldfit(X=X, y=y)
        contamination = self.expected_frac_anomaly
        self.offset_ = np.percentile(self.score_samples(X), 100.0 * contamination)
        return self

    KcpLearner.fit = fit

    def predict(self, X):
        decision_func = self.score_samples(X) - self.offset_
        is_inlier = np.ones_like(decision_func, dtype=int)
        is_inlier[decision_func < 0] = -1
        return is_inlier

    KcpLearner.predict = predict

    return KcpLearner
