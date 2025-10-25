from abc import ABC, abstractmethod
from typing import Any, Dict, Union, List
import numpy as np
import pandas as pd

ParamsDescription = Dict[str, Any]
ArrayLike = Union[np.ndarray, pd.DataFrame]


class Formatter(ABC):
    """
    Abstract base class for all formalizers.
    Provides array-agnostic interface for ML pipelines.
    """

    def __init__(self):
        self.available_properties_: List[str] = []
        self.query_description_: ParamsDescription = {}

    # -----------------------
    # Available properties
    # -----------------------
    @property
    def available_properties(self) -> List[str]:
        return self.available_properties_

    def add_property(self, name: str):
        if name not in self.available_properties_:
            self.available_properties_.append(name)

    def remove_property(self, name: str):
        while name in self.available_properties_:
            self.available_properties_.remove(name)

    # -----------------------
    # Query description
    # -----------------------
    @property
    def query_description(self) -> ParamsDescription:
        return self.query_description_

    def add_query_description(self, name: str, param_info: Dict[str, Any]):
        self.query_description_[name] = param_info

    def default_query(self) -> Dict[str, Any]:
        """
        Return default query parameters based on query_description.
        """
        defaults = {}
        for k, desc in self.query_description_.items():
            defaults[k] = desc.get("default")
        return defaults

    # -----------------------
    # Abstract method
    # -----------------------
    @abstractmethod
    def format(self, **query) -> ArrayLike:
        """
        Transform raw data into standard array-like format.
        Return type depends on backend (numpy array, pandas DataFrame, etc.)
        """
        ...
