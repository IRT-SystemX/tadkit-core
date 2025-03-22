from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor

from tadkit.base.tadlearner import TADLearner
from tadkit.utils.tadlearner_factory import tadlearner_factory


class TestClassicModels:
    @staticmethod
    def _test(Model):
        Learner = tadlearner_factory(Model, [], {})
        Learner()
        # assert not isinstance(Model, TADLearner)
        assert isinstance(Learner, TADLearner)

    def test_1(self):
        self._test(OneClassSVM)

    def test_2(self):
        self._test(SGDOneClassSVM)

    def test_3(self):
        self._test(EllipticEnvelope)

    def test_4(self):
        self._test(LocalOutlierFactor)

    def test_5(self):
        self._test(IsolationForest)
