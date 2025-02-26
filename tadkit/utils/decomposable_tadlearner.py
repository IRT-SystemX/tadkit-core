from typing import TypeVar, Type, Sequence, Optional

from tadkit.base import TADLearner, KWParams, ParamsDescription, Array

Transformer = TypeVar("Transformer")


class DecomposableTADLearner(TADLearner):
    """Abstract class for combining a preprocessor and a AD Learner."""

    def __init__(self, **params: KWParams):
        self.preprocesser = self.Preprocesser(**params)
        self.learner = self.Learner()

    def fit(self, X: Array, y: Optional[Array] = None) -> "DecomposableTADLearner":
        if hasattr(self.preprocesser, "fit_transform"):
            new_X = self.preprocesser.fit_transform(X, y)
        else:
            if hasattr(self.preprocesser, "fit"):
                self.preprocesser.fit(X)
            new_X = self.embed(X)
        self.learner.fit(new_X)
        return self

    def embed(self, X: Array):
        if hasattr(self.preprocesser, "transform"):
            new_X = self.preprocesser.transform(X)
        else:
            new_X = self.preprocesser(X)
        return new_X

    def score_samples(self, X: Array) -> Array:
        if hasattr(self.preprocesser, "transform"):
            X = self.preprocesser.transform(X)
        else:
            X = self.preprocesser(X)
        return self.learner.score_samples(X)


def decomposable_tadlearner_factory(
    Preprocesser: Type[Transformer],
    Learner: Type[TADLearner],
    required_properties: Sequence[str],
    preprocesser_params_description: ParamsDescription,
    name: Optional[str] = None,
) -> Type[TADLearner]:
    """Create a TADLearner class joining a class of preprocessing an a TADLearner subclass.

    Args:
        Preprocesser: The preprocessing, must have a fit_transform or transform method.
        Learner: The processing subclass of TADLearner
        required_properties: The properties that the input data must satisfies.
        preprocesser_params_description: Description of the **kwargs of the __init__ method of the
            preprocessor.
        name: The Name of the class. Default name if None.

    Returns:
        A subclass of TADLearner joining Preprocesser Learner with:
            * given required_properties and __name__.
            * Updating the Learner params_description by preprocesser_params_description.
            * Fit method that fit and transform the preprocessing and fit the learner.
            * score_samples that transform the preprocessing and scare_samples the learner.
    """

    if name is None:
        name = Preprocesser.__name__ + Learner.__name__
    params_description = dict(Learner.params_description)
    params_description.update(preprocesser_params_description)
    return type(
        name,
        (DecomposableTADLearner, Learner),
        {
            "Preprocesser": Preprocesser,
            "Learner": Learner,
            "required_properties": required_properties,
            "params_description": preprocesser_params_description,
            "__name__": name,
        },
    )
