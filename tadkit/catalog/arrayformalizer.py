from typing import Sequence, Optional, Union
import numpy as np
import pandas as pd

from tadkit.base.formalizer import Formalizer
from tadkit.base.dataframe_type import DataFrameType
from tadkit.base.typing import ParamsDescription


def index_has_fixed_time_step(index: Union[pd.DatetimeIndex, np.ndarray]) -> bool:
    """Check if index has constant time intervals."""
    if len(index) < 2:
        return True
    candidate_step = index[1] - index[0]
    return np.all(index[1:] == index[:-1] + candidate_step)


class ArrayToPandasFormalizer(Formalizer):
    """Generic formalizer for NumPy arrays or Pandas DataFrames."""

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        dataframe_type: Optional[str] = None,
        timestamps: Optional[Sequence] = None,
        columns: Optional[Sequence[str]] = None,
    ):
        """
        Args:
            data: pandas DataFrame or numpy array
            dataframe_type: "synchronous" or "asynchronous"
            timestamps: required if data is a np.ndarray
            columns: column names if data is numpy array
        """
        self.available_properties_: list[str] = []

        # Convert numpy array to DataFrame if needed
        if isinstance(data, np.ndarray):
            if timestamps is None:
                raise ValueError(
                    "NumPy array input requires a 'timestamps' argument to be used for time series."
                )
            if columns is None:
                columns = [f"X{i}" for i in range(data.shape[1])]
            self.data_df_ = pd.DataFrame(data, columns=columns)
            self.data_df_.index = pd.to_datetime(timestamps)

        elif isinstance(data, pd.DataFrame):
            self.data_df_ = data.copy()
            if "timestamp" in self.data_df_.columns:
                self.data_df_.set_index("timestamp", inplace=True)
            self.data_df_.index = pd.to_datetime(self.data_df_.index)

        else:
            raise TypeError("Data must be a pandas DataFrame or a NumPy ndarray.")

        self.dataframe_type_ = (
            DataFrameType.from_text(dataframe_type)
            if dataframe_type
            else DataFrameType.infer_from_df(self.data_df_)
        )
        self._prepare_dataframe()
        self.query_description_: ParamsDescription = {}
        self._fill_query_description()

    @property
    def available_properties(self) -> Sequence[str]:
        return self.available_properties_

    def add_available_properties(self, value: str):
        if value not in self.available_properties_:
            self.available_properties_.insert(0, value)

    def remove_available_properties(self, value: str):
        while value in self.available_properties_:
            self.available_properties_.remove(value)

    @property
    def query_description(self) -> ParamsDescription:
        return self.query_description_

    def add_query_description(self, param_name: str, param_description: dict):
        self.query_description_[param_name] = param_description

    def _prepare_dataframe(self):
        """Ensure timestamp index and mark fixed_time_step property."""
        if self.dataframe_type_ == DataFrameType.ASYNCHRONOUS:
            required = {"sensor", "data"}
            if not required.issubset(self.data_df_.columns):
                raise KeyError(
                    f"Missing required columns for asynchronous data: {required}"
                )
            self.data_df_["timestamp"] = pd.to_datetime(
                self.data_df_["timestamp"]
                if "timestamp" in self.data_df_.columns
                else self.data_df_.index
            )
            self.data_df_.set_index("timestamp", inplace=True)
            self.add_available_properties("asynchronous")
        else:
            if not np.issubdtype(self.data_df_.index.dtype, np.datetime64):
                self.data_df_.index = pd.to_datetime(self.data_df_.index)
            self.add_available_properties("synchronous")

        # Fixed time step detection
        if self.dataframe_type_ == DataFrameType.ASYNCHRONOUS:
            fixed = all(
                index_has_fixed_time_step(g.index)
                for _, g in self.data_df_.groupby("sensor")
            )
        else:
            fixed = index_has_fixed_time_step(self.data_df_.index)

        if fixed:
            self.add_available_properties("fixed_time_step")

    def get_space_set(self) -> list:
        if self.dataframe_type_ == DataFrameType.ASYNCHRONOUS:
            return list(self.data_df_["sensor"].unique())
        else:
            exclude = {"id", "filename", "minio", "timestamp"}
            return [c for c in self.data_df_.columns if c not in exclude]

    def get_timestamps(self) -> pd.DatetimeIndex:
        return self.data_df_.index

    def _fill_query_description(self):
        timestamps = self.get_timestamps()
        self.add_query_description(
            "target_period",
            {
                "description": "Time period for your query.",
                "family": "time_interval",
                "start": timestamps[0],
                "stop": timestamps[-1],
                "default": (timestamps[0], timestamps[-1]),
            },
        )
        space_set = self.get_space_set()
        self.add_query_description(
            "target_space",
            {
                "description": "List of sensors used for your query.",
                "family": "space",
                "set": space_set,
                "default": space_set,
            },
        )
        self.add_query_description(
            "resampling",
            {"description": "Resample data?", "family": "bool", "default": False},
        )
        self.add_query_description(
            "resampling_resolution",
            {
                "description": "Resampling interval in seconds.",
                "family": "time",
                "start": 60,
                "default": 120,
                "stop": 3600,
            },
        )

    def formalize(self, **query) -> pd.DataFrame:
        default_query = self.get_default_query()
        default_query.update(query)
        target_space: list = default_query["target_space"]
        time_start, time_stop = default_query["target_period"]
        resampling: bool = default_query["resampling"]
        resampling_resolution: int = default_query["resampling_resolution"]

        # Univariate vs multivariate
        if len(target_space) > 1:
            self.remove_available_properties("univariate_time_series")
            self.add_available_properties("multiple_time_series")
        else:
            self.remove_available_properties("multiple_time_series")
            self.add_available_properties("univariate_time_series")

        df = self.data_df_.copy()

        if self.dataframe_type_ == DataFrameType.ASYNCHRONOUS:
            df = df[df["sensor"].isin(target_space)]
            df = df.loc[time_start:time_stop]

            if resampling:
                df = (
                    df.groupby("sensor")
                    .apply(
                        lambda g: g["data"]
                        .resample(f"{resampling_resolution}s", origin=time_start)
                        .first()
                        .interpolate(method="piecewise_polynomial")
                    )
                    .reset_index()
                )

            df = df.pivot(index="timestamp", columns="sensor", values="data")

        else:  # synchronous
            df = df.loc[time_start:time_stop, target_space]
            if resampling:
                df = df.resample(f"{resampling_resolution}s", origin=time_start).first()
                df.interpolate(method="piecewise_polynomial", inplace=True)

        df.dropna(inplace=True)
        return df
