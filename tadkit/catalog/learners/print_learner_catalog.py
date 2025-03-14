import inspect

from tadkit.base.tadlearner import TADLearner
from tadkit.catalog.learners import learner_classes

HEADER = '\033[95m'
FAIL = '\033[91m'
FAIL_PROP = '\033[93m'
ENDC = '\033[0m'


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
            print(f"{FAIL_PROP}{learner_name}.__init__ parameter '{param_name}' must have default value.{ENDC}")
    return


def _print_class(learner_name, detailed=False):
    if learner_name not in learner_classes:
        print(f"target {HEADER}{learner_name=}{ENDC} not registered in TADKit.")
        return
    learner_class = learner_classes[learner_name]
    print(f"Class {HEADER}{learner_name=}{ENDC} is registered in TADKit.")
    try:
        if inspect.isclass(learner_class):
            print(f"{learner_name} is operational in this environment.")
            if isinstance(learner_class, TADLearner):
                print(f"{learner_name} is implicit child of TADLearner.")
            else:
                print(f"{FAIL_PROP}{learner_name} somewhat somehow doesn't implicitly inherit from TADLearner.{ENDC}")
        _validate_default_init(learner_class, learner_name)
    except ModuleNotFoundError as err:
        print(f"{FAIL}{learner_name} returns {err=}.{ENDC}")
        return
    try:
        if detailed:
            printed_params_description = {name: str(param_description) for name, param_description in
                                          learner_class.params_description.items()}
            print(f"{learner_name} has {printed_params_description=}.")
    except AttributeError as err:
        print(f"{FAIL_PROP}{learner_name} with signature {learner_class=} returns {err=}.{ENDC}")
    try:
        if detailed:
            print(f"{learner_name} has {learner_class.required_properties=}.")
    except AttributeError as err:
        print(f"{FAIL_PROP}{learner_name} with signature {learner_class=} returns {err=}.{ENDC}")


def print_catalog_classes(detailed=False):
    print("[TADKit-Catalog]")
    for learner_name in learner_classes.keys():
        _print_class(learner_name, detailed=detailed)
