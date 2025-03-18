"""Useful functions for formalizer and learner selection widgets."""

from typing import Sequence

import ipywidgets as widgets

from tadkit.catalog.learners.match_formalizer_learners import match_formalizer_learners
from ... import Formalizer, TADLearner


def select_matching_available_learners(
    formalizer: Formalizer,
    available_learners: Sequence[TADLearner],
):
    matching_available_learners = match_formalizer_learners(
        formalizer, available_learners
    ).keys()

    learner_widget = widgets.SelectMultiple(
        description="Learners:",
        options=matching_available_learners,
        value=tuple(matching_available_learners),
    )

    display(learner_widget)
    return learner_widget
