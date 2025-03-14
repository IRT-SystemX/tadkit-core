from ._confiance_components import (
    DataReconstructionADLearner,
    DiLAnoDetectmLearner,
    KcpLearner,
    TopologicalAnomalyDetector,
)

from ._sklearn_learners import (
    IsolationForestLearner,
    KernelDensityLearner,
    ScaledKernelDensityLearner,
)

learner_classes = {
    "cnndrad": DataReconstructionADLearner,
    "sbad": DiLAnoDetectmLearner,
    "kcpd": KcpLearner,
    "tdaad": TopologicalAnomalyDetector,
    "isolation-forest": IsolationForestLearner,
    "kernel-density": KernelDensityLearner,
    "scaled-kernel-density": ScaledKernelDensityLearner,
}

import inspect
installed_learner_classes = {}
for learner_name, learner_class in learner_classes.items():
    try:
        if inspect.isclass(learner_class):
            installed_learner_classes[learner_name] = learner_class
    except ModuleNotFoundError:
        pass

from .print_learner_catalog import print_catalog_classes

print_catalog_classes()


__all__ = [
    "DataReconstructionADLearner",
    "DiLAnoDetectmLearner",
    "KcpLearner",
    "TopologicalAnomalyDetector",
    "IsolationForestLearner",
    "KernelDensityLearner",
    "ScaledKernelDensityLearner",
    "learner_classes",
    "installed_learner_classes",
    "match_formalizer_learners",
]
