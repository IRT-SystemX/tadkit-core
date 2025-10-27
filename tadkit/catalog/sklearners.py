from typing import Callable

import numpy as np

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

from tadkit.base.basedensitydetector import BaseDensityOutlierDetector


class KDEOutlierDetector(BaseDensityOutlierDetector):
    """
    Density-based outlier detection using KernelDensity.
    """

    _parameter_constraints = KernelDensity._parameter_constraints.copy()

    def __init__(self, **kwargs):
        """
        All parameters are passed directly to sklearn.neighbors.KernelDensity.

        Parameters
        ----------
        **kwargs : dict
            All keyword arguments are passed to KernelDensity.
        """
        super().__init__(contamination=kwargs.pop("contamination", 0.1))
        self.kde_params = kwargs

    def _fit_density(self, X: np.ndarray):
        self.kde_ = KernelDensity(**self.kde_params)
        self.kde_.fit(X)

    def _score_density(self, X: np.ndarray) -> np.ndarray:
        return self.kde_.score_samples(X)


class GMMOutlierDetector(BaseDensityOutlierDetector):
    """
    Density-based outlier detection using GaussianMixture.
    """

    _parameter_constraints = GaussianMixture._parameter_constraints.copy()

    def __init__(self, **kwargs):
        """
        All parameters are passed directly to sklearn.mixture.GaussianMixture.

        Parameters
        ----------
        **kwargs : dict
            All keyword arguments are passed to GaussianMixture.
        """
        super().__init__(contamination=kwargs.pop("contamination", 0.1))
        self.gmm_params = kwargs

    def _fit_density(self, X: np.ndarray):
        self.gmm_ = GaussianMixture(**self.gmm_params)
        self.gmm_.fit(X)

    def _score_density(self, X: np.ndarray) -> np.ndarray:
        return self.gmm_.score_samples(X)


class CustomScoreOutlierDetector(BaseDensityOutlierDetector):
    """
    Parameters
    ----------
    score_func : callable
        Function X -> scores (higher = inliers). Must accept 2D array and return 1D array.
    contamination : float, default=0.1
        Proportion of outliers. Must be in (0, 0.5).
    """

    score_func: Callable[[np.ndarray], np.ndarray]

    def __init__(
        self, score_func: Callable[[np.ndarray], np.ndarray], contamination: float = 0.1
    ):
        super().__init__(contamination=contamination)
        if not callable(score_func):
            raise ValueError("score_func must be callable")
        self.score_func = score_func

    def _fit_density(self, X: np.ndarray):
        # Nothing to fit
        pass

    def _score_density(self, X: np.ndarray) -> np.ndarray:
        scores = self.score_func(X)
        scores = np.asarray(scores)
        if scores.ndim != 1:
            raise ValueError("score_func must return a 1D array")
        return scores
