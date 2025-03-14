from typing import Sequence, Optional, Protocol, runtime_checkable

from .typing import ParamsDescription, Array


@runtime_checkable
class TADLearner(Protocol):
    """Abstract class of Time Anomaly Detection Learner (model).

    Avoid explicit inheritance from this class. Better to simply do it implicitly.

    Methods:
        fit: Fit the learner on input data.
        score_samples: The measure of normality of an observation according to the fitted model.
            The lower, the more abnormal.
        predict: Predict if a particular sample is an outlier or not. For each observation, tells
            whether or not (+1 or -1) it should be considered as an inlier according to the fitted model.

    Class attributes:
        params_description: Description of the arguments of the __init__ method. See examples in the catalog.
        required_properties: Get the properties that the input data must satisfies. See examples in the catalog.

    Example:
        >>> assert isinstance(MyLearner, TADLearner)
        >>> MyLearner.required_properties  # The required property of input data
        >>> MyLearner.params_description  # The description of the params
        >>> params = ...  # Params to initiate learner
        >>> learner = MyLearner(**params)
        >>> learner.fit(X)  # X, y must satisfy MyLearner.required_properties
        >>> score_sample_pred = learner.score_samples(X_test)
    """

    required_properties: Sequence[str] = []
    params_description: ParamsDescription = {}

    def fit(self, X: Array, y: Optional[Array] = None) -> 'TADLearner':
        ...

    def score_samples(self, X: Array) -> Array:
        """
        The measure of normality of an observation according to the fitted model.
        Scikit-learn compatible.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more abnormal.
        """
        ...

    def predict(self, X: Array) -> Array:
        """
        Predict if a particular sample is an outlier or not.
        Scikit-learn compatible.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        is_inlier : ndarray of shape (n_samples,)
            For each observation, tells whether or not (+1 or -1) it should
            be considered as an inlier according to the fitted model.
        """
        ...
