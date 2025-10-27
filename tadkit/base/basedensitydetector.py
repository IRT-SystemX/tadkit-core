import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin


class BaseDensityOutlierDetector(BaseEstimator, OutlierMixin):
    """
    Base class for density-based outlier detection.

    Subclasses must implement:
        - _fit_density(X)
        - _score_density(X)
    """

    contamination: float
    offset_: float | None

    def __init__(self, contamination: float = 0.1):
        """
        Parameters
        ----------
        contamination : float, default=0.1
            The proportion of outliers in the dataset. Must be in (0, 0.5).
        """
        if not 0.0 < contamination < 0.5:
            raise ValueError("contamination must be between 0 and 0.5")
        self.contamination = contamination
        self.offset_ = None

    def fit(self, X: np.ndarray, y=None):
        """Fit density estimator and compute threshold."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        self._fit_density(X)
        scores = self._score_density(X)
        self.offset_ = np.percentile(scores, 100.0 * self.contamination)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        scores = self._score_density(X)
        return scores - self.offset_

    def predict(self, X: np.ndarray) -> np.ndarray:
        decision = self.decision_function(X)
        labels = np.ones_like(decision, dtype=int)
        labels[decision < 0] = -1
        return labels

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        return self._score_density(X)

    def _fit_density(self, X: np.ndarray):
        raise NotImplementedError("_fit_density must be implemented in subclass")

    def _score_density(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("_score_density must be implemented in subclass")
