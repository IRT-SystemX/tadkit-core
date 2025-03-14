from typing import List


def get_wrapped_dilanodetectm():
    """Returns the TADlearner wrapped from the DiLAnoDetectm method
    of the Sparsity-Based Anomaly Detection framework.

    The function is intended for use if the dependency is available.
    """

    from sbad_fnn.models import DiLAnoDetectm

    class DiLAnoDetectmWrapper(DiLAnoDetectm):

        def __init__(
                self,
                nb_atoms: int,
                nb_blocks: int,
                kernel_size: int,
                nb_scales: int,
                overlap_perc: float,
                layers_sizes: List[int] = None,
                activation: str = 'id',
                wvlt_name: str = 'db2',
                same: bool = False,
                share_weights: bool = False,
                soft: bool = False,
                shrink: bool = True,
                b: float = 1.5,
                min_nbsegments: int = 50,
                min_lag: int = 64,
        ) -> None:
            if layers_sizes is None:
                layers_sizes = [39, 3]
            super().__init__(
                nb_atoms=nb_atoms,
                nb_blocks=nb_blocks,
                kernel_size=kernel_size,
                layers_sizes=layers_sizes,
                nb_scales=nb_scales,
                overlap_perc=overlap_perc,
                activation=activation,
                wvlt_name=wvlt_name,
                same=same,
                share_weights=share_weights,
                soft=soft,
                shrink=shrink,
                b=b,
                min_nbsegments=min_nbsegments,
                min_lag=min_lag,
            )  # to be removed when lists are supported by widgets

        def fit(self, x):
            x = [[x.swapaxes(0, 1)]]
            return super().fit(x)

        def score_samples(self, x):
            x = [[x.swapaxes(0, 1)]]
            return - super().score_samples(x)[0][0][0][:len(x)]

    DiLAnoDetectmWrapper.required_properties = ["multiple_time_series"]
    DiLAnoDetectmWrapper.params_description = {
        "nb_atoms": {
            "description": "Number of patterns to learn.",
            "value_type": "range",
            "start": 1, "step": 1, "stop": 100,  # @martin: stop put to 100 without any info
            "default": 2,
        },
        "nb_blocks": {
            "description": "Number of unrolled gradient steps blocks.",
            "value_type": "range",
            "start": 1, "step": 1, "stop": 100,  # @martin: stop put to 100 without any info
            "default": 1,
        },
        "kernel_size": {
            "description": "Convolutional patterns size, must be odd.",
            "value_type": "range",
            "start": 1, "step": 2, "stop": 100,  # @martin: stop put to 100 without any info
            "default": 5,
        },
        # "layers_sizes": {
        #    "description": "Spectral autoencoder layers sizes.",
        #    "value_type": "array",
        #    "length": 2,
        #    "elements": {
        #        "value_type": "range", "start": 1, "step": 1,
        #    },
        #    "default": (39, 3),
        # },
        "nb_scales": {
            "description": "Number of wavelet scales for decomposing time series.",
            "value_type": "range",
            "start": 1, "step": 1, "stop": 100,  # @martin: stop put to 100 without any info
            "default": 1,
        },
        "overlap_perc": {
            "description": "Overlapping ratio between sliding windows.",
            "value_type": "range",
            "start": 0, "step": 0.01, "stop": 1,
            "default": 0.98,
        },
        "same": {
            "description": "Whether the blocks are identical or have different weights.",
            "value_type": "bool_choice",
            "default": False,
        },
        "share_weights": {
            "description": "Whether within a given block, the conv and conv_transpose layers share weights or not.",
            "value_type": "bool_choice",
            "default": False,
        },
        "soft": {
            "description": "Whether the shrinking is soft.",
            "value_type": "bool_choice",
            "default": False,
        },
        "shrink": {
            "description": "Whether a shrinking activation function is applied to each block's output.",
            "value_type": "bool_choice",
            "default": True,
        },
        "min_nbsegments": {
            "description": "Minimum number of segments.",
            "value_type": "range",
            "start": 1, "step": 1, "stop": 100,  # @martin: stop put to 100 without any info
            "default": 1,
        },
        "min_lag": {
            "description": "Minimum lag.",
            "value_type": "range",
            "start": 1, "step": 1, "stop": 100,  # @martin: stop put to 100 without any info
            "default": 2,
        },
    }

    return DiLAnoDetectmWrapper
