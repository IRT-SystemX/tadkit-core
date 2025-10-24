from abc import ABC, abstractmethod
from typing import Any, Dict, Union, List
import numpy as np
import pandas as pd

ParamsDescription = Dict[str, Any]
ArrayLike = Union[np.ndarray, pd.DataFrame]


class Formalizer(ABC):
    """
    Generic formalizer for array-like data.
    Can handle NumPy arrays, pandas DataFrames, or any similar array-like backend.
    """

    def __init__(self, data: ArrayLike = None, backend: str = "numpy"):
        self.data = data
        self.backend = backend  # "numpy" or "pandas"
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

    def add_query_param(self, name: str, param_info: Dict[str, Any]):
        self.query_description_[name] = param_info

    def get_default_query(self) -> Dict[str, Any]:
        return {k: v.get("default") for k, v in self.query_description_.items()}

    # -----------------------
    # Abstract method
    # -----------------------
    @abstractmethod
    def formalize(self, **query) -> ArrayLike:
        """
        Transform raw data into standard array-like format.
        Return type depends on backend (numpy array, pandas DataFrame, etc.)
        """
        ...

    # -----------------------
    # Helper for backend conversion
    # -----------------------
    def to_backend(self, data: Any) -> ArrayLike:
        """Convert any array-like data to the specified backend."""
        if self.backend == "pandas":
            return pd.DataFrame(data)
        return np.array(data)
