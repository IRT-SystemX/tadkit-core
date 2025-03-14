import numpy as np
from deel.puncc.api.prediction import BasePredictor
from deel.puncc.anomaly_detection import SplitCAD
from modAL.models import ActiveLearner, Committee
from modAL.disagreement import max_disagreement_sampling


class ADPredictor(BasePredictor):
    def fit(self, X, **kwargs):
        self.model.fit(X)

    def predict(self, X, **kwargs):
        return -self.model.score_samples(X)


class ActiveSampler:
    """
    Class for active learning with a committee of learners.

    Attributes:
        learners (dict): A dictionary of learners where the keys are learner names and the values are estimator objects.
        random_state (int, optional): Random state for reproducibility. Defaults to 42.
        size_committee (int): The number of learners in the committee.
        committee (Committee): A committee of learners.
        cads (dict): A dictionary of conformal anomaly detectors where the keys are learner names.
    """

    def __init__(self, learners: dict, random_state=42):
        self._calibrated = False
        self.random_state = random_state
        self.learners = learners
        self.size_committee = len(learners)
        self.learner_list = []
        for estimator in self.learners.values():
            active_learner = ActiveLearner(estimator=estimator)
            self.learner_list.append(active_learner)
        self.committee = Committee(self.learner_list, query_strategy=max_disagreement_sampling)
        self.committee.classes_ = [0, 1]
        self.cads = {}

        for i, learner in enumerate(self.learner_list):
            key = list(self.learners.keys())[i]
            self.cads[key] = SplitCAD(ADPredictor(learner.estimator, is_trained=True), train=False,
                                      random_state=self.random_state)

    def fit(self, *, X_fit=None, y_fit=None, X_calib=None, y_calib=None):
        """
        Method to fit and/or calibrate the committee of models.
        If X_fit is provided, the committee is fitted.
        If X_calib is provided, the committee is calibrated.

        Args:
            X_fit: array-like, shape (n_samples, n_features), optional
                The input samples for fitting the committee.
            y_fit: array-like, shape (n_samples,), optional
                No need to be provided, just there for compatibility.
            X_calib: array-like, shape (n_samples, n_features), optional
                The input samples for calibrating the committee.
            y_calib: array-like, shape (n_samples,), optional
                Will be needed only for conformal FNR control.

        Raises:
        - RuntimeError: If neither X_fit nor X_calib is provided.

        Returns:
        None
        """
        if X_fit is None and X_calib is None:
            raise RuntimeError("Either X_fit or X_calib should be provided.")
        if X_fit is not None:
            print("Fitting the committee ...")
            self.committee.fit(X_fit, y_fit)
            for i, learner in enumerate(self.committee):
                key = list(self.learners.keys())[i]
                self.cads[key] = SplitCAD(ADPredictor(learner.estimator, is_trained=True), train=False,
                                          random_state=self.random_state)
        if X_calib is not None:
            print("Calibrating the committee ...")
            self._calibrate(X_calib, y_calib)
        print("Done.")

    def predict(self, X, alphas=None):
        """
        Method to predict anomalies using the committee of models and conformalized versions if alphas is proovided.

        Args:
            X: Input data to be predicted.
            alphas: List of maximum error values for conformal anomaly detection.

        Returns:
            ads_results: Anomaly predictions by each of the models in the committee. Shape (n_samples, n_models).
            cads_results: Conformal anomaly predictions by each of the models
            in the committee. Shape (n_alphas, n_samples, n_models).
        """
        if alphas is None:
            alphas = []
        ads_results = self.committee.vote(X)
        cads_results = []
        for alpha in alphas:
            # cads_results.append(np.array([cad.predict(X, alpha=alpha / len(alphas)) for cad in self.cads.values()]).T)
            cads_results.append(np.array([cad.predict(X, alpha=alpha) for cad in self.cads.values()]).T)
        return ads_results, np.array(cads_results)

    def query(self, X, n_instances, alphas=None, query_strategy="conformal_fpr_uncertainty"):
        """
        Query method for active sampling.

        Args:
            X: array-like, shape (n_samples, n_features)
                The input samples.
            n_instances: int
                The number of instances to query.
            alphas: list, optional (default=[])
                The list of alpha values for conformal prediction.
            query_strategy: str, optional (default="conformal")
                The query strategy to use.

        Returns:
            array: Indices of selected instances.

        Raises:
        RuntimeError: If the models are not calibrated before querying.

        """
        if alphas is None:
            alphas = []
        if query_strategy in ["conformal_fpr_uncertainty", "conformal_fpr_disagreement"] and not self._calibrated:
            raise RuntimeError("The models should be calibrated before querying. Need to call fit with X_calib.")
        if query_strategy == "conformal_fpr_uncertainty":
            return self._conformal_uncertainty_committee(X, alphas, n_instances)
        self.committee = Committee(list(self.committee), query_strategy=query_strategy)
        self.committee.classes_ = [0, 1]
        return query_strategy(self.committee, X, n_instances=n_instances)[0]

    def _calibrate(self, X_calib, y_calib=None):
        """
        Calibrates the CAD models using the provided calibration data.

        Args:
            X_calib: numpy array: The calibration data.
            y_calib: numpy array, optional. The calibration labels. Default is None.

        Returns:
        None
        """
        for name, cad in self.cads.items():
            cad.fit(z_fit=np.empty_like(X_calib), z_calib=X_calib)  # TODO change puncc to take only X_calib
        self._calibrated = True

    def _conformal_uncertainty_committee(self, X, alphas, n_instances):
        """
        Perform conformal uncertainty sampling coupled with committee selection.

        Args:
            X (array-like): Input data.
            alphas (list): List of two elements representing the significance levels.
            n_instances (int): Number of instances to select.

        Returns:
            array: Indices of selected instances.

        Raises:
            ValueError: If alphas is None or its length is not equal to 2.
        """
        if alphas is None or len(alphas) != 2:
            raise ValueError("Alphas should be a list of two elements.")
        _, cads_results = self.predict(X, alphas)
        cads_uncertainty_votes = np.logical_xor(cads_results[0], cads_results[1])
        # count the votes for each instance
        votes = np.sum(cads_uncertainty_votes, axis=1)
        # select the most uncertain instances indices
        uncertain_instances = np.argsort(votes)[-n_instances:]
        return uncertain_instances
