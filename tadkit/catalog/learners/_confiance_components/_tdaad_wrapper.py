from tadkit.utils.learner_params_parser import enriched_metadata


def get_wrapped_topolad_pp():
    """Returns the TADlearner wrapped from the TopologicalAnomalyDetector method of the tdaad framework.

    The function is intended for use if the dependency is available.
    This plus plus version is meant to remove the explicit heritage (through factory).
    """

    from tdaad.anomaly_detectors import TopologicalAnomalyDetector

    metadata = enriched_metadata(TopologicalAnomalyDetector)
    [metadata.pop(key) for key in ["contamination", "support_fraction"]]
    TopologicalAnomalyDetector.metadata = metadata

    return TopologicalAnomalyDetector
