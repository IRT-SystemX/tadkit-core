def get_wrapped_topolad_pp():
    """Returns the TADlearner wrapped from the TopologicalAnomalyDetector method of the tdaad framework.

    The function is intended for use if the dependency is available.
    This plus plus version is meant to remove the explicit heritage (through factory).
    """

    from tdaad.anomaly_detectors import TopologicalAnomalyDetector

    # We look at constraints and infer distribution as basis for inquiry.
    params_description = {
        param: description[0]
        for param, description in TopologicalAnomalyDetector._parameter_constraints.items()
    }
    params_description.pop("store_precision")
    params_description.pop("assume_centered")
    params_description.pop("random_state")
    params_description.pop("contamination")
    params_description["support_fraction"] = {
        "description": "Support fraction for the MinCovDet estimation"
        + ":"
        + str(params_description["support_fraction"]),
        "value_type": "real_range",
        "start": 0.01,
        "stop": 1.0,
        "step": 0.1,
        "default": 0.5,
    }
    params_description["window_size"] = {
        "description": "Window size for the time-delay embedding"
        + ":"
        + str(params_description["window_size"]),
        "value_type": "range",
        "start": 10,
        "stop": 1000,
        "step": 10,
        "default": 100,
    }
    params_description["step"] = {
        "description": "Step size for the time-delay embedding"
        + ":"
        + str(params_description["step"]),
        "value_type": "range",
        "start": 10,
        "stop": 100,
        "step": 10,
        "default": 10,
    }
    params_description["tda_max_dim"] = {
        "description": "Compute persistence in all homology dimension including this tda_max_dim"
        + ":"
        + str(params_description["tda_max_dim"]),
        "value_type": "range",
        "start": 0,
        "stop": 3,
        "step": 1,
        "default": 1,
    }
    params_description["n_centers_by_dim"] = {
        "description": "Size of the vectorization per homology dimension"
        + ":"
        + str(params_description["n_centers_by_dim"]),
        "value_type": "range",
        "start": 2,
        "stop": 20,
        "step": 1,
        "default": 2,
    }
    TopologicalAnomalyDetector.params_description = params_description

    return TopologicalAnomalyDetector
