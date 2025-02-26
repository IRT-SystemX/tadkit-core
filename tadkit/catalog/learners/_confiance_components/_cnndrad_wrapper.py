import numpy as np


def get_wrapped_datareconstructionad():
    """Return the TADlearner wrapped from cnndrad's DataReconstructionAD method.

    The function is intended for use if the dependency is available.
    """

    from cnndrad import DataReconstructionAD

    DataReconstructionAD.required_properties = [
        "fixed_time_step",
        "univariate_time_series",
    ]
    DataReconstructionAD.params_description = {
        "window_size": {
            "description": "Size of the sliding window applied on data samples.",
            "value_type": "range",
            "start": 10,
            "step": 10,
            "stop": 1000,
            "default": 10,
        },
        "window_stride": {
            "description": "Stride of the sliding window applied on data samples.",
            "value_type": "range",
            "start": 10,
            "stop": 100,
            "step": 10,
            "default": 10,
        },
    }

    DataReconstructionAD.__oldinit__ = DataReconstructionAD.__init__

    def __init__(
        self,
        window_size=100,
        window_stride=1,
        reconstruct=[True] * 3,
        model_name="CNN_1D_3x3Conv",
        metric="mae",
        batch_size=32,
        epochs=100,
        validation_split=0.2,
        work_dir="./",
        device="/gpu:0",
        **kwargs,
    ) -> None:
        DataReconstructionAD.__oldinit__(
            self,
            window_size=window_size,
            window_stride=window_stride,
            reconstruct=reconstruct,
            model_name=model_name,
            metric=metric,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            work_dir=work_dir,
            device=device,
            **kwargs,
        )

    DataReconstructionAD.__init__ = __init__

    def predict(self, X):
        decision_func = self.score_samples(X)
        is_inlier = np.ones_like(decision_func, dtype=int)
        is_inlier[decision_func < -1] = -1
        return is_inlier

    DataReconstructionAD.predict = predict

    return DataReconstructionAD
