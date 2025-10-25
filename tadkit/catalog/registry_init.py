# tadkit/learners/registry_init.py
from tadkit.base.registry import registry


from tadkit.catalog.learners.sklearn_learners import (
    KernelDensityLearner,
)

registry.register_learner(
    name="IsolationForest",
    learner="sklearn.ensemble.IsolationForest",
    condition=lambda fmt: fmt.backend == "pandas",
)

registry.register_learner(
    name="KernelDensityLearner",
    learner=KernelDensityLearner,
    condition=lambda _: True,
)

registry.register_learner(
    name="TopologicalAnomalyDetector",
    learner="tdaad.anomaly_detectors.TopologicalAnomalyDetector",
    condition=lambda fmt: "multiple_time_series" in fmt.available_properties,
    optional=True,
)

registry.register_learner(  # @todo: default parameter adjustments from _cnndrad_wrapper.py
    name="DataReconstructionAD",
    learner="cnndrad.DataReconstructionAD",
    condition=lambda fmt: "univariate_time_series" in fmt.available_properties,
    optional=True,
)

registry.register_learner(  # @todo: default parameter adjustments from _kcpdi_wrapper.py
    name="KcpLearner",
    learner="kcpdi.kcp_ss_learner.KcpLearner",
    condition=lambda _: True,
    optional=True,
)

registry.register_learner(  # @todo: default parameter adjustments from _sbad_wrapper.py
    name="DataReconstructionAD",
    learner="sbad_fnn.models.DiLAnoDetectm",
    condition=lambda fmt: "multiple_time_series" in fmt.available_properties,
    optional=True,
)


# from tadkit.catalog.learners._confiance_components import (
#     DataReconstructionADLearner,
#     DiLAnoDetectmLearner,
#     KcpLearner,
# )

# learner_classes = {
#     "cnndrad": DataReconstructionADLearner,
#     "sbad": DiLAnoDetectmLearner,
#     "kcpd": KcpLearner,
#     # "tdaad": TopologicalAnomalyDetector,
#     "isolation-forest": IsolationForestLearner,
#     "kernel-density": KernelDensityLearner,
# }
