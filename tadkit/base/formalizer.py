import abc
from typing import Sequence, Union

from .typing import KWParams, ParamsDescription, Array


class Formalizer(abc.ABC):
    """Abstract class of data formalizer (provider).
    Transforms Data from Confiance DataProvider into standard Data for ML pipelines.

    Methods:
        formalize: Take a data query and return associated data.
        no_data_leakage: Check if no leakage from a first data query to a second.

    Properties:
        query_description: Get the description of a data query.
        available_properties: Get the properties that the formalized data satisfies.

    Example of usage:
        >>> assert issubclass(MyFormalizer, Formalizer)
        >>> formalizer = MyFormalizer(**args_init)
        >>> formalizer.available_properties  # The provided property of the formalized data
        >>> formalizer.query_description  # The description of the queries
        >>> query_train = ...  # Query to create data, following the query description
        >>> query_test = ...
        >>> X_test = formalizer.formalize(query_test)
        >>> X_train = formalizer.formalize(query_train)
    """

    @property
    @abc.abstractmethod
    def available_properties(self) -> Sequence[str]:
        return []

    @property
    @abc.abstractmethod
    def query_description(self) -> ParamsDescription:
        return {}

    def default_query(self):
        # NB: this hints at queries having a default value for all parameters.
        return {name: param["default"] for name, param in self.query_description.items()}

    @abc.abstractmethod
    def formalize(self, **query: KWParams) -> Union[Array, Sequence[Array]]:
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, subclass):
        if not (
                hasattr(subclass, 'formalize') and callable(subclass.formalize)
                and hasattr(subclass, 'available_properties')
                and not callable(subclass.available_properties)
                and hasattr(subclass, 'query_description') and not callable(subclass.query_description)
        ):
            return False
        if cls is Formalizer:
            return True
        return NotImplemented
