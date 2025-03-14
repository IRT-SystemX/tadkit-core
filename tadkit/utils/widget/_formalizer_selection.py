from typing import Any, Dict

import ipywidgets as widgets

from ... import Formalizer


def select_formalizer(formalizers: Dict[str, Formalizer]) -> Dict[str, Any]:
    """Utility function for dataset selection using widgets."""

    dataset_widget = widgets.Select(
        description="Formalizer:",
        options=formalizers.keys(),
    )

    return dataset_widget
