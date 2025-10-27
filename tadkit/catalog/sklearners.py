from typing import Callable

import numpy as np

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

from tadkit.base.basedensitydetector import BaseDensityOutlierDetector


class KDEOutlierDetector(BaseDensityOutlierDetector):
    bandwidth: float
    kernel: str

    def __init__(
        self,
        bandwidth: float = 1.0,
        kernel: str = "gaussian",
        contamination: float = 0.1,
    ):
        """
        Parameters
        ----------
        bandwidth : float, default=1.0
            Bandwidth parameter for KernelDensity. Must be > 0.
        kernel : str, default='gaussian'
            Kernel type for KernelDensity. Choices: 'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'poly'
        contamination : float, default=0.1
            The proportion of outliers. Must be in (0, 0.5).
        """
        super().__init__(contamination=contamination)
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if kernel not in [
            "gaussian",
            "tophat",
            "epanechnikov",
            "exponential",
            "linear",
            "poly",
        ]:
            raise ValueError(
                f"kernel must be one of 'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'poly', got {kernel}"
            )
        self.bandwidth = bandwidth
        self.kernel = kernel

    def _fit_density(self, X: np.ndarray):
        self.kde_ = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
        self.kde_.fit(X)

    def _score_density(self, X: np.ndarray) -> np.ndarray:
        return self.kde_.score_samples(X)


class GMMOutlierDetector(BaseDensityOutlierDetector):
    n_components: int
    covariance_type: str

    def __init__(
        self,
        n_components: int = 2,
        covariance_type: str = "full",
        contamination: float = 0.1,
    ):
        """
        Parameters
        ----------
        n_components : int, default=2
            Number of mixture components. Must be > 0.
        covariance_type : str, default='full'
            Covariance type for GaussianMixture. Choices: 'full', 'tied', 'diag', 'spherical'
        contamination : float, default=0.1
            Proportion of outliers. Must be in (0, 0.5).
        """
        super().__init__(contamination=contamination)
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if covariance_type not in ["full", "tied", "diag", "spherical"]:
            raise ValueError(
                f"covariance_type must be one of 'full', 'tied', 'diag', 'spherical', got {covariance_type}"
            )
        self.n_components = n_components
        self.covariance_type = covariance_type

    def _fit_density(self, X: np.ndarray):
        self.gmm_ = GaussianMixture(
            n_components=self.n_components, covariance_type=self.covariance_type
        )
        self.gmm_.fit(X)

    def _score_density(self, X: np.ndarray) -> np.ndarray:
        return self.gmm_.score_samples(X)


class CustomScoreOutlierDetector(BaseDensityOutlierDetector):
    score_func: Callable[[np.ndarray], np.ndarray]

    def __init__(
        self, score_func: Callable[[np.ndarray], np.ndarray], contamination: float = 0.1
    ):
        """
        Parameters
        ----------
        score_func : callable
            Function X -> scores (higher = inliers). Must accept 2D array and return 1D array.
        contamination : float, default=0.1
            Proportion of outliers. Must be in (0, 0.5).
        """
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
