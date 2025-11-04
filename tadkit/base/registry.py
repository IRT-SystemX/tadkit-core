# tadkit/learners/registry.py
import importlib
from typing import Any, Callable, Dict, List, Type, Union
import inspect

from tadkit.base.tadlearner import TADLearner

HEADER = "\033[95m"
FAIL = "\033[91m"
FAIL_PROP = "\033[93m"
ENDC = "\033[0m"


class Registry:
    """Registry for dynamically tracking and matching learner classes."""

    def __init__(self):
        # name -> {"class": object or callable, "condition": callable}
        self._learners: Dict[str, Dict[str, Any]] = {}

    # -----------------------
    # Learner registration
    # -----------------------
    def register_learner(
        self,
        name: str,
        learner: Union[Type, str],
        condition: Callable[[Any], bool],
        optional: bool = False,
    ):
        """
        Register a learner with a compatibility condition.

        Parameters
        ----------
        name : str
            Display name for the learner.
        learner : class or str
            The learner class OR an import path ("module.submodule.ClassName").
        condition : callable(formatter) -> bool
            Determines whether this learner is compatible.
        optional : bool
            If True, missing imports are ignored instead of raising.
        """
        cls = None
        if isinstance(learner, str):
            # Try dynamic import
            module_name, cls_name = learner.rsplit(".", 1)
            try:
                mod = importlib.import_module(module_name)
                cls = getattr(mod, cls_name)
            except (ModuleNotFoundError, AttributeError) as e:
                if optional:
                    print(f"[registry] Skipping optional learner '{name}': {e}")
                    return
                raise
        else:
            cls = learner

        self._learners[name] = {"class": cls, "condition": condition}

    # -----------------------
    # Matching
    # -----------------------
    def match_learners(self, formatter: Any) -> List[Type]:
        """Return learner classes compatible with the given formatter."""
        matches = []
        for name, info in self._learners.items():
            try:
                if info["condition"](formatter):
                    matches.append(info["class"])
            except Exception as e:
                print(f"[registry] Learner '{name}' failed match check: {e}")
        return matches

    def list_learners(self) -> List[str]:
        return list(self._learners.keys())

    # -----------------------
    # Utils
    # -----------------------
    @staticmethod
    def _validate_default_init(learner_class, learner_name):
        """
        Validate that the `cls` satisfies the `protocol`'s __init__ method
        with the required default values.
        """
        class_init = getattr(learner_class, "__init__", None)
        if not class_init:
            print(f"{FAIL_PROP}{learner_name} must have an __init__ method.{ENDC}")

        # Check __init__ signatures
        class_sig = inspect.signature(class_init)
        # Check default values
        for param_name, param in class_sig.parameters.items():
            if param_name == "self":
                continue
            if param.default is param.empty:
                print(
                    f"{FAIL_PROP}{learner_name}.__init__ parameter '{param_name}' must have default value.{ENDC}"
                )
        return

    @staticmethod
    def print_compliance_miss(learner):
        missing = []
        for attr in TADLearner.__annotations__:
            if not hasattr(learner, attr):
                missing.append(attr)
        print("Missing attributes:", missing)

        for name in dir(TADLearner):
            if callable(getattr(TADLearner, name, None)) and not name.startswith("__"):
                if not hasattr(learner, name):
                    print(f"Missing method: {name}")

    def _print_class(self, learner_name, learner_class, detailed=False):
        print(f"Class {HEADER}{learner_name=}{ENDC} is registered in TADKit.")
        try:
            if inspect.isclass(learner_class):
                print(f"{learner_name} is operational in this environment.")
                if isinstance(learner_class, TADLearner):
                    print(f"{learner_name} is implicit child of TADLearner.")
                else:
                    print(
                        f"{FAIL_PROP}{learner_name} doesn't implicitly inherit from TADLearner:{ENDC}"
                    )
                    self.print_compliance_miss(learner_class)
            Registry._validate_default_init(learner_class, learner_name)
        except ModuleNotFoundError as err:
            print(f"{FAIL}{learner_name} returns {err=}.{ENDC}")
            return
        try:
            if detailed:
                printed_params_description = {
                    name: str(param_description)
                    for name, param_description in learner_class.metadata.items()
                }
                print(f"{learner_name} has {printed_params_description=}.")
        except AttributeError as err:
            print(
                f"{FAIL_PROP}{learner_name} with signature {learner_class=} returns {err=}.{ENDC}"
            )
        try:
            if detailed:
                print(f"{learner_name} has {learner_class.required_properties=}.")
        except AttributeError as err:
            print(
                f"{FAIL_PROP}{learner_name} with signature {learner_class=} returns {err=}.{ENDC}"
            )

    def print_catalog_classes(self, detailed=False):
        print("[TADkit registered Catalog]")
        for learner_name, learner in self._learners.items():
            self._print_class(
                learner_name, learner_class=learner["class"], detailed=detailed
            )


# global instance
registry = Registry()
