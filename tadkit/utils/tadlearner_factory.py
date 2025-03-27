from typing import TypeVar, Sequence, Type, Optional

from tadkit.base.tadlearner import TADLearner
from tadkit.base.typing import KWParams, ParamsDescription

Estimator = TypeVar("Estimator")


def tadlearner_factory(
    Model: Type[Estimator],
    required_properties: Sequence[str],
    params_description: ParamsDescription,
    name: Optional[str] = None,
) -> Type[TADLearner]:
    """Wrap a sklearn anomaly detection model to a TADLearner.

    Args:
        Model: sklearn type model- to wrapp, possessing get_params, fit and score_samples methods.
        required_properties: The properties that the input data must satisfies.
        params_description: Description of the kwargs of the __init__ method that is exposed.
        name: The Name of the class. Default name if None.

    Returns:
        A subclass of TADLearner wrapping Model with given required_properties, params_description
            and __name__.
    """

    if name is None:
        name = Model.__name__ + "Learner"

    def __init__(self, **params: KWParams):
        Model.__init__(self, **params)

    Model.name = name
    Model.required_properties = required_properties
    Model.params_description = params_description
    return Model
