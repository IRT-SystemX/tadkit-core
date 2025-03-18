from typing import Sequence

from ... import Formalizer, TADLearner


def match_formalizer_learners(
    formalizer: Formalizer,
    learners: Sequence[TADLearner],
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
