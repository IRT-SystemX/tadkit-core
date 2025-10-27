from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin


class BaseDensityOutlierDetector(BaseEstimator, OutlierMixin):
    """
    Base class for density-based outlier detection.

    Subclasses must implement:
        - _fit_density(X)
        - _score_density(X)

            Accepts pandas DataFrame/Series but works internally with NumPy arrays.
            Returns results with same index as input if input is pandas.
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

    def fit(self, X: Any, y=None):
        self._is_pandas_input = isinstance(X, (pd.DataFrame, pd.Series))
        self._input_index = X.index if self._is_pandas_input else None

        X_np = X.values if self._is_pandas_input else np.asarray(X)
        if X_np.ndim != 2:
            raise ValueError("X must be 2D array or DataFrame")

        self._fit_density(X_np)
        scores = self._score_density(X_np)
        self.offset_ = np.percentile(scores, 100.0 * self.contamination)
        return self

    def decision_function(self, X: Any) -> np.ndarray | pd.Series:
        X_np, index = self._convert_input(X)
        decision = self._score_density(X_np) - self.offset_
        return pd.Series(decision, index=index) if index is not None else decision

    def predict(self, X: Any) -> np.ndarray | pd.Series:
        decision = self.decision_function(X)
        labels = np.ones_like(decision, dtype=int)
        labels[decision < 0] = -1
        if isinstance(decision, pd.Series):
            labels = pd.Series(labels, index=decision.index)
        return labels

    def score_samples(self, X: Any) -> np.ndarray | pd.Series:
        X_np, index = self._convert_input(X)
        scores = self._score_density(X_np)
        return pd.Series(scores, index=index) if index is not None else scores

    def _convert_input(self, X: Any):
        """Helper to handle pandas input."""
        is_pandas = isinstance(X, (pd.DataFrame, pd.Series))
        index = X.index if is_pandas else None
        X_np = X.values if is_pandas else np.asarray(X)
        if X_np.ndim != 2:
            raise ValueError("X must be 2D array or DataFrame")
        return X_np, index

    def _fit_density(self, X: np.ndarray):
        raise NotImplementedError("_fit_density must be implemented in subclass")

    def _score_density(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("_score_density must be implemented in subclass")
