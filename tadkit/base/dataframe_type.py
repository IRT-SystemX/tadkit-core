from enum import Enum


class DataFrameType(Enum):
    """Data type of Dataplatform datasets."""

    ASYNCHRONOUS = "asynchronous"
    """All data have their own "timestamp" x-axis, sensor names are found in the "sensor" column and data is found
    in the "data" column of the provided dataframe.
    """
    SYNCHRONOUS = "synchronous"
    """All data share a common "timestamp" x-axis, sensor names are in columns of the provided dataframe
    (except for "id", "filename", "minio", "timestamp" and possible others... tbd more precisely with DataPlatform.)
    """

    @staticmethod
    def from_text(name: str):
        """Converts the string-representation to an Enum-object."""
        if name.lower() == DataFrameType.ASYNCHRONOUS.value:
            return DataFrameType.ASYNCHRONOUS
        if name.lower() == DataFrameType.SYNCHRONOUS.value:
            return DataFrameType.SYNCHRONOUS
        raise ValueError(
            "In order to process input data you must know the format"
            "(DataFrameType.ASYNCHRONOUS or DataFrameType.SYNCHRONOUS)."
        )
