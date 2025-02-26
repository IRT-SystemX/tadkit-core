from typing import Sequence
import warnings

import numpy as np
import pandas as pd

from tadkit.base.dataframe_type import DataFrameType
from tadkit.base.formalizer import Formalizer
from tadkit.base.typing import ParamsDescription, KWParams


def index_has_fixed_time_step(index):
    candidate_time_step = index[1] - index[0]
    return (index[1:] == index[:-1] + candidate_time_step).all()


class PandasFormalizer(Formalizer):
    """Transforms Data from Confiance DataProvider into standard Data for ML pipelines.
    This particular class returns pandas DataFrames.

    Methods:
        formalize: Take a data query with a DataPlatformType and return associated data.

    Properties:
        query_description: Get the description of a data query.
        available_properties: Get the properties that the formalized data satisfies.
    """

    def __init__(self, data_df=None, dataframe_type=""):
        self.data_df = data_df
        self.dataframe_type = dataframe_type
        self.available_properties_ = []
        self._fit()

    @property
    def available_properties(self) -> Sequence[str]:
        return self.available_properties_

    @available_properties.setter
    def available_properties(self, value):
        self.available_properties_ = value

    def add_available_properties(self, value):
        if value not in self.available_properties_:
            self.available_properties_.insert(0, value)

    def remove_available_properties(self, value):
        while value in self.available_properties_:
            self.available_properties_.remove(value)

    @property
    def query_description(self) -> ParamsDescription:
        return self.query_description_

    @query_description.setter
    def query_description(self, value):
        self.query_description_ = value

    def add_query_description(self, param_name, param_description):
        self.query_description_[param_name] = param_description

    def get_space_set(self):
        if "sensor" in self.data_df_:
            space_set = list(np.unique(self.data_df_["sensor"]))
        else:
            space_set = self.data_df_.columns
        return space_set

    def get_timestamps(self):
        return self.data_df_.index

    def _fill_query_description(self):
        self.query_description = {}
        self.add_query_description(
            "target_period",
            {
                "description": "Time period for your query.",
                "family": "time_interval",
                "start": self.get_timestamps()[0],
                "stop": self.get_timestamps()[-1],
                "default": (self.get_timestamps()[0], self.get_timestamps()[-1]),
            },
        )
        self.add_query_description(
            "target_space",
            {
                "description": "List of sensors used for your query.",
                "family": "space",
                "set": self.get_space_set(),
                "default": self.get_space_set(),
            },
        )
        self.add_query_description(
            "resampling",
            {
                "description": "Resampling of the target query",
                "family": "bool",
                "default": False,
            },
        )
        self.add_query_description(
            "resampling_resolution",
            {
                "description": "If resampling, resampling resolution in seconds.",
                "family": "time",
                "start": 60,
                "default": 120,
                "stop": 3600,
            },
        )

    def _fit(self):
        self.data_df_ = self.data_df.copy()
        if "timestamp" in self.data_df_.columns:
            self.data_df_ = self.data_df_.set_index("timestamp")
        self.data_df_.index = pd.to_datetime(self.data_df_.index)
        self.dataframe_type_ = DataFrameType.from_text(self.dataframe_type)

        fixed_time_step = {
            DataFrameType.ASYNCHRONOUS: lambda df: np.all(
                [
                    index_has_fixed_time_step(data.index)
                    for _, data in df.groupby("sensor")
                ]
            ),
            DataFrameType.SYNCHRONOUS: lambda df: index_has_fixed_time_step(df.index),
        }.get(self.dataframe_type_, DataFrameType.ASYNCHRONOUS)
        if fixed_time_step(self.data_df_):
            self.add_available_properties("fixed_time_step")

        self.add_available_properties("pandas")  # @todo: what do I mean with that?

        self._fill_query_description()
        return self

    def formalize(self, **query: KWParams):

        default_query = self.default_query()
        default_query.update(query)
        resampling = default_query["resampling"]
        if self.dataframe_type_ == DataFrameType.ASYNCHRONOUS and not resampling:
            warnings.warn(
                f"This data is of type {DataFrameType.ASYNCHRONOUS}, if you do not resample it"
                f"it probably will end up badly down the road."
            )
        resampling_resolution = default_query["resampling_resolution"]
        target_space = default_query["target_space"]
        if len(target_space) > 1:
            self.remove_available_properties("univariate_time_series")
            self.add_available_properties("multiple_time_series")
        elif len(target_space) == 1:
            self.remove_available_properties("multiple_time_series")
            self.add_available_properties("univariate_time_series")
        time_start, time_stop = (
            default_query["target_period"][0],
            default_query["target_period"][1],
        )

        timeseries_set = []
        df_handler = {
            DataFrameType.ASYNCHRONOUS: lambda df: df.groupby("sensor").data,
            DataFrameType.SYNCHRONOUS: lambda df: df.items(),
        }.get(self.dataframe_type_, DataFrameType.ASYNCHRONOUS)
        for name, sensor in df_handler(self.data_df_):
            if name not in target_space:
                warnings.warn(
                    f"Searching for sensor {name=}, not found in {target_space=}."
                )
                continue
            if resampling:
                clean_sensor = (
                    sensor.resample(f"{resampling_resolution}s", origin=time_start)
                    .first()
                    .interpolate(method="piecewise_polynomial")
                )
            else:
                clean_sensor = sensor
            clean_sensor.name = name
            cleancut_sensor = clean_sensor[
                (clean_sensor.index >= time_start) & (clean_sensor.index <= time_stop)
            ]
            timeseries_set.append(cleancut_sensor)

        if not len(timeseries_set):
            return []
        raw_df = pd.concat(timeseries_set, axis=1)
        raw_df.dropna(inplace=True)
        return raw_df
