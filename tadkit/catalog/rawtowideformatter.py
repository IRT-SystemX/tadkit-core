from typing import Sequence, Optional, Union

import numpy as np
import pandas as pd
from tadkit.base.formatter import Formatter
from tadkit.base.dataframe_type import DataFrameType


def _resample_array(array, timestamps, freq_s):
    """Resample a NumPy array using linear interpolation (seconds-based)."""
    old_ts = timestamps.astype("datetime64[ns]").astype("int64") / 1e9  # seconds
    new_ts = np.arange(old_ts[0], old_ts[-1] + freq_s, freq_s)
    new_array = np.zeros((len(new_ts), array.shape[1]))
    for i in range(array.shape[1]):
        new_array[:, i] = np.interp(new_ts, old_ts, array[:, i])
    new_timestamps = new_ts.astype("datetime64[s]")
    return new_array, new_timestamps


def _asynchronous_to_synchronous(df):
    """Convert a long asynchronous DataFrame to wide synchronous format."""
    if not {"sensor", "data"}.issubset(df.columns):
        raise ValueError("Asynchronous data must have 'sensor', and 'data' columns.")

    target = df["timestamp"] if "timestamp" in df.columns else df.index
    df["timestamp"] = pd.to_datetime(target)
    df_pivot = df.pivot_table(index="timestamp", columns="sensor", values="data")
    df_pivot.sort_index(inplace=True)
    df_pivot.interpolate(inplace=True)
    return df_pivot


class RawToWideFormatter(Formatter):
    """
    A Formatter that supports both pandas DataFrame and NumPy array outputs.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Input data.
    backend : str
        'pandas' or 'numpy'.
    timestamps : np.ndarray, optional
        Required if data is a NumPy array.
    columns : list[str], optional
        Column names for NumPy arrays.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        timestamps: Optional[Sequence] = None,
        columns: Optional[Sequence[str]] = None,
        backend: str = "numpy",
    ):
        super().__init__()
        self.backend = backend

        # --- Handle pandas input ---
        if isinstance(data, pd.DataFrame):
            self.data_df_ = data.copy()
            if DataFrameType.infer_from_df(data) == DataFrameType.ASYNCHRONOUS:
                self.data_df_ = _asynchronous_to_synchronous(self.data_df_)
                self.add_property("converted_to_synchronous")
            if "timestamp" in self.data_df_.columns:
                self.data_df_.set_index("timestamp", inplace=True)
            self.data_df_.index = pd.to_datetime(self.data_df_.index, utc=True)
            self.data_df_.index = self.data_df_.index.tz_convert(None)
            self.timestamps = self.data_df_.index.to_numpy(dtype="datetime64[ns]")
            self.data_array = self.data_df_.values
            self.columns = self.data_df_.columns.tolist()
        # --- Handle numpy input ---
        elif isinstance(data, np.ndarray):
            if timestamps is None:
                raise ValueError("NumPy input requires a 'timestamps' array.")
            self.timestamps = (
                pd.to_datetime(timestamps, utc=True)
                .tz_convert(None)
                .to_numpy(dtype="datetime64[ns]")
            )
            self.data_array = data
            self.columns = (
                columns
                if columns is not None
                else [f"X{i}" for i in range(data.shape[1])]
            )

            # if pandas backend, create DataFrame from array
            if self.backend == "pandas":
                self.data_df_ = pd.DataFrame(
                    data, columns=self.columns, index=self.timestamps
                )
        else:
            raise TypeError("Data must be a pandas DataFrame or a NumPy ndarray.")

        """Set available properties and default query description."""
        self.add_property("synchronous")
        self.add_property(f"{self.backend}_backend")

        diffs = np.diff(self.timestamps.astype("datetime64[s]"))
        if len(set(diffs)) == 1:
            self.add_property("fixed_time_step")

        """Set default query description."""
        self.add_query_description(
            "target_period",
            {
                "description": "Time period to select",
                "family": "time_interval",
                "start": self.timestamps[0],
                "stop": self.timestamps[-1],
                "default": (self.timestamps[0], self.timestamps[-1]),
            },
        )
        self.add_query_description(
            "target_space",
            {
                "description": "Columns or sensors to select",
                "family": "space",
                "set": self.columns,
                "default": self.columns,
            },
        )
        self.add_query_description(
            "resample",
            {"description": "Resample data?", "family": "bool", "default": False},
        )
        self.add_query_description(
            "resample_freq",
            {
                "description": "Resampling frequency in seconds.",
                "family": "time",
                "start": 60,
                "default": 120,
                "stop": 3600,
            },
        )

    def format(
        self,
        target_period=None,
        target_space=None,
        resample=False,
        resample_freq: float = 1.0,
    ):
        """
        Slice and optionally resample the data, using backend-specific resampling.
        """
        # reset time-series-type properties
        self.remove_property("univariate_time_series")
        self.remove_property("multiple_time_series")

        # --- target period ---
        start, stop = (
            target_period
            if target_period is not None
            else (self.timestamps[0], self.timestamps[-1])
        )
        mask = (self.timestamps >= start) & (self.timestamps <= stop)

        # --- target space ---
        selected_space = target_space if target_space is not None else self.columns
        idx = [self.columns.index(c) for c in selected_space]

        # --- slice data ---
        array_slice = self.data_array[mask][:, idx]

        # --- univariate/multivariate properties ---
        if len(selected_space) > 1:
            self.add_property("multiple_time_series")
        else:
            self.add_property("univariate_time_series")

        if self.backend == "numpy":
            if resample:
                array_slice, _ = _resample_array(
                    array_slice, self.timestamps[mask], resample_freq
                )
            return array_slice

        else:  # pandas backend
            df_slice = pd.DataFrame(
                array_slice, columns=selected_space, index=self.timestamps[mask]
            )
            if resample:
                df_slice = df_slice.resample(f"{int(resample_freq)}s").interpolate()
            return df_slice
