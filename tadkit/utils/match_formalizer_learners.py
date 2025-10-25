from typing import Dict

from tadkit.base.formatter import Formatter
from tadkit.base.tadlearner import TADLearner


def match_formalizer_learners(
    formalizer: Formatter,
    learners: Dict[str, TADLearner],
):
    matching_learners = {}
    for learner_name, learner_class in learners.items():
        if set(formalizer.available_properties).issuperset(
            learner_class.required_properties
        ):
            matching_learners[learner_name] = learner_class
        else:
            print(
                f"Discarding {learner_name=},"
                f" has {learner_class.required_properties=} that doesn't match {formalizer.available_properties=}"
            )
    return matching_learners
