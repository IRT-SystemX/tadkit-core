import inspect
from typing import Type, Optional
from tadkit.base.tadlearner import TADLearner
from tadkit.base.typing import ArrayLike


class DecomposableTADLearner(TADLearner):
    """Abstract base class combining a Preprocessor and a TADLearner."""

    Preprocessor: Type
    Learner: Type[TADLearner]

    def __init__(self, **params):
        """Instantiate preprocessor and learner using relevant subset of kwargs."""
        pre_sig = inspect.signature(self.Preprocessor.__init__)
        learner_sig = inspect.signature(self.Learner.__init__)

        pre_params = {k: v for k, v in params.items() if k in pre_sig.parameters}
        learner_params = {k: v for k, v in params.items() if k in learner_sig.parameters}

        self.preprocessor = self.Preprocessor(**pre_params)
        self.learner = self.Learner(**learner_params)

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "DecomposableTADLearner":
        if hasattr(self.preprocessor, "fit_transform"):
            X_transformed = self.preprocessor.fit_transform(X, y)
        else:
            if hasattr(self.preprocessor, "fit"):
                self.preprocessor.fit(X, y)
            X_transformed = self.embed(X)
        self.learner.fit(X_transformed, y)
        return self

    def embed(self, X: ArrayLike) -> ArrayLike:
        if hasattr(self.preprocessor, "transform"):
            return self.preprocessor.transform(X)
        return self.preprocessor(X)

    def score_samples(self, X: ArrayLike) -> ArrayLike:
        X_transformed = self.embed(X)
        return self.learner.score_samples(X_transformed)


def decomposable_tadlearner_factory(
    Preprocessor: Type,
    Learner: Type[TADLearner],
    name: Optional[str] = None,
) -> Type[TADLearner]:
    """Create a TADLearner class combining a preprocessor and a learner with a proper __init__ signature."""

    cls_name = name or f"{Preprocessor.__name__}{Learner.__name__}"

    class Wrapped(DecomposableTADLearner):
        __doc__ = f"{cls_name}: DecomposableTADLearner combining {Preprocessor.__name__} and {Learner.__name__}"
        __name__ = cls_name
        __qualname__ = cls_name

    Wrapped.Preprocessor = Preprocessor
    Wrapped.Learner = Learner

    pre_sig = inspect.signature(Preprocessor.__init__)
    learner_sig = inspect.signature(Learner.__init__)

    combined_params = [
        p.replace(kind=inspect.Parameter.KEYWORD_ONLY) if p.kind == inspect.Parameter.VAR_KEYWORD else p
        for p in list(pre_sig.parameters.values())[1:] + list(learner_sig.parameters.values())[1:]
    ]
    combined_sig = inspect.Signature(combined_params)

    def __init__(self, *args, **kwargs):
        bound = combined_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        super(Wrapped, self).__init__(**bound.arguments)

    __init__.__signature__ = combined_sig
    __init__.__doc__ = f"{cls_name} __init__ combines {Preprocessor.__name__} and {Learner.__name__} parameters"
    Wrapped.__init__ = __init__

    return Wrapped
