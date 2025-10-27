from typing import Callable

import numpy as np

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

from tadkit.base.basedensitydetector import BaseDensityOutlierDetector


class KDEOutlierDetector(BaseDensityOutlierDetector):
    """
    Density-based outlier detection using KernelDensity.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth of the kernel.
    algorithm : {'kd_tree', 'ball_tree', 'auto'}, default='auto'
        The tree algorithm to use.
    kernel : str, default='gaussian'
        The kernel to use. Valid kernels are
        ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'].
    metric : str, default='euclidean'
        The distance metric to use.
    atol : float, default=0
        The desired absolute tolerance of the result.
    rtol : float, default=0
        The desired relative tolerance of the result.
    breadth_first : bool, default=True
        If true, use a breadth-first approach to the problem.
    leaf_size : int, default=40
        Leaf size passed to BallTree or KDTree.
    metric_params : dict, default=None
        Additional parameters for the metric function.
    contamination : float, default=0.1
        Proportion of outliers in the data set.
    """

    _parameter_constraints = KernelDensity._parameter_constraints.copy()

    def __init__(
        self,
        bandwidth=1.0,
        algorithm="auto",
        kernel="gaussian",
        metric="euclidean",
        atol=0,
        rtol=0,
        breadth_first=True,
        leaf_size=40,
        metric_params=None,
        contamination: float = 0.1,
    ):
        super().__init__(contamination=contamination)

        # store KDE parameters explicitly
        self.bandwidth = bandwidth
        self.algorithm = algorithm
        self.kernel = kernel
        self.metric = metric
        self.atol = atol
        self.rtol = rtol
        self.breadth_first = breadth_first
        self.leaf_size = leaf_size
        self.metric_params = metric_params

    def _fit_density(self, X: np.ndarray):
        self.kde_ = KernelDensity(
            bandwidth=self.bandwidth,
            algorithm=self.algorithm,
            kernel=self.kernel,
            metric=self.metric,
            atol=self.atol,
            rtol=self.rtol,
            breadth_first=self.breadth_first,
            leaf_size=self.leaf_size,
            metric_params=self.metric_params,
        )
        self.kde_.fit(X)

    def _score_density(self, X: np.ndarray) -> np.ndarray:
        return self.kde_.score_samples(X)


class GMMOutlierDetector(BaseDensityOutlierDetector):
    """
    Density-based outlier detection using GaussianMixture.

    Parameters
    ----------
    n_components : int, default=1
        The number of mixture components.
    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        Type of covariance parameters to use.
    tol : float, default=1e-3
        Convergence threshold.
    reg_covar : float, default=1e-6
        Non-negative regularization added to the diagonal of covariance matrices.
    max_iter : int, default=100
        The number of EM iterations to perform.
    n_init : int, default=1
        The number of initializations to perform. The best result is kept.
    init_params : {'kmeans', 'random'}, default='kmeans'
        Method used to initialize the weights, means, and precisions.
    weights_init : array-like of shape (n_components,), default=None
        The user-provided initial weights.
    means_init : array-like of shape (n_components, n_features), default=None
        The user-provided initial means.
    precisions_init : array-like, default=None
        The user-provided initial precisions.
    random_state : int, RandomState instance, default=None
        Controls the random seed.
    warm_start : bool, default=False
        If True, reuse the solution of the last fitting.
    verbose : int, default=0
        Enable verbose output.
    verbose_interval : int, default=10
        Number of iteration steps between printing progress.
    contamination : float, default=0.1
        Proportion of outliers in the dataset.
    """

    _parameter_constraints = GaussianMixture._parameter_constraints.copy()

    def __init__(
        self,
        n_components=1,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weights_init=None,
        means_init=None,
        precisions_init=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
        contamination: float = 0.1,
    ):
        super().__init__(contamination=contamination)

        # Store GMM parameters explicitly
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    def _fit_density(self, X: np.ndarray):
        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            tol=self.tol,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            n_init=self.n_init,
            init_params=self.init_params,
            weights_init=self.weights_init,
            means_init=self.means_init,
            precisions_init=self.precisions_init,
            random_state=self.random_state,
            warm_start=self.warm_start,
            verbose=self.verbose,
            verbose_interval=self.verbose_interval,
        )
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
