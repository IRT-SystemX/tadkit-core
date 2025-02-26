
import inspect

from tadkit.utils.print_learner_catalog import print_catalog_classes
from tadkit.catalog.learners._confiance_components import (
    DataReconstructionADLearner,
    DiLAnoDetectmLearner,
    KcpLearner,
    TopologicalAnomalyDetector,
)

from tadkit.catalog.learners._sklearn_learners import (
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

installed_learner_classes = {}
for learner_name, learner_class in learner_classes.items():
    try:
        if inspect.isclass(learner_class):
            installed_learner_classes[learner_name] = learner_class
    except ModuleNotFoundError:
        pass

print_catalog_classes(learner_classes)


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
