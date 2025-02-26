from typing import Sequence, TypeVar, Union, Dict
from numbers import Number
from datetime import datetime

Param = Union[
    Number, str, datetime, Sequence[Number], Sequence[str], Sequence[datetime]
]
KWParams = Dict[str, Param]
ParamsDescription = Dict[str, Dict[str, Param]]
Array = TypeVar("Array")
Dataset = TypeVar("Dataset")
