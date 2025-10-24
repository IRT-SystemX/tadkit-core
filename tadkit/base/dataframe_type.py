from enum import Enum
import pandas as pd


class DataFrameType(Enum):
    """Data type of datasets: long (asynchronous) vs wide (synchronous)."""

    ASYNCHRONOUS = ("asynchronous", ["timestamp", "sensor", "data"])
    SYNCHRONOUS = ("synchronous", None)  # columns vary per dataset

    def __init__(self, value, required_columns):
        self._value_ = value
        self.required_columns = required_columns

    @staticmethod
    def from_text(name: str):
        """Convert a string to a DataFrameType enum."""
        lookup = {t.value: t for t in DataFrameType}
        try:
            return lookup[name.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown DataFrameType '{name}'. Must be one of: "
                f"{', '.join(lookup.keys())}"
            )

    @staticmethod
    def infer_from_df(df: pd.DataFrame) -> "DataFrameType":
        if {"sensor", "data"}.issubset(df.columns):
            return DataFrameType.ASYNCHRONOUS
        else:
            return DataFrameType.SYNCHRONOUS
