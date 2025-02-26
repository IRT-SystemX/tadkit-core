from tadkit.base.tadlearner import TADLearner
from tadkit.catalog.learners import installed_learner_classes


def test_implicit_inheritance_of_catalog_learners():
    for name, learner in installed_learner_classes.items():
        print(f"Asserting {name=} is implicit child of TADLearner.")
        assert isinstance(learner, TADLearner)


def test_catalog_learners_defaults():
    for name, learner in installed_learner_classes.items():
        learner()
