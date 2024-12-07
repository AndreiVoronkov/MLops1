from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AvailableModelTypesRequest(_message.Message):
    __slots__ = ["show"]
    SHOW_FIELD_NUMBER: _ClassVar[int]
    show: bool
    def __init__(self, show: bool = ...) -> None: ...

class AvailableModelTypesResponse(_message.Message):
    __slots__ = ["list_of_models"]
    class one_model_definition(_message.Message):
        __slots__ = ["model_name", "model_type"]
        MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
        MODEL_TYPE_FIELD_NUMBER: _ClassVar[int]
        model_name: str
        model_type: str
        def __init__(self, model_name: _Optional[str] = ..., model_type: _Optional[str] = ...) -> None: ...
    LIST_OF_MODELS_FIELD_NUMBER: _ClassVar[int]
    list_of_models: _containers.RepeatedCompositeFieldContainer[AvailableModelTypesResponse.one_model_definition]
    def __init__(self, list_of_models: _Optional[_Iterable[_Union[AvailableModelTypesResponse.one_model_definition, _Mapping]]] = ...) -> None: ...

class InitModelRequest(_message.Message):
    __slots__ = ["type_model", "user_model_name", "params_string", "params_float"]
    class ParamsStringEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    class ParamsFloatEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    TYPE_MODEL_FIELD_NUMBER: _ClassVar[int]
    USER_MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    PARAMS_STRING_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FLOAT_FIELD_NUMBER: _ClassVar[int]
    type_model: str
    user_model_name: str
    params_string: _containers.ScalarMap[str, str]
    params_float: _containers.ScalarMap[str, float]
    def __init__(self, type_model: _Optional[str] = ..., user_model_name: _Optional[str] = ..., params_string: _Optional[_Mapping[str, str]] = ..., params_float: _Optional[_Mapping[str, float]] = ...) -> None: ...

class InitModelResponse(_message.Message):
    __slots__ = ["user_model_name", "type_model"]
    USER_MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_MODEL_FIELD_NUMBER: _ClassVar[int]
    user_model_name: str
    type_model: str
    def __init__(self, user_model_name: _Optional[str] = ..., type_model: _Optional[str] = ...) -> None: ...

class FitModelRequest(_message.Message):
    __slots__ = ["X", "y", "user_model_name"]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    USER_MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    X: _containers.RepeatedScalarFieldContainer[float]
    y: _containers.RepeatedScalarFieldContainer[float]
    user_model_name: str
    def __init__(self, X: _Optional[_Iterable[float]] = ..., y: _Optional[_Iterable[float]] = ..., user_model_name: _Optional[str] = ...) -> None: ...

class FitModelResponse(_message.Message):
    __slots__ = ["status"]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: str
    def __init__(self, status: _Optional[str] = ...) -> None: ...

class PredictModelRequest(_message.Message):
    __slots__ = ["X", "n_feats", "user_model_name"]
    X_FIELD_NUMBER: _ClassVar[int]
    N_FEATS_FIELD_NUMBER: _ClassVar[int]
    USER_MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    X: _containers.RepeatedScalarFieldContainer[float]
    n_feats: int
    user_model_name: str
    def __init__(self, X: _Optional[_Iterable[float]] = ..., n_feats: _Optional[int] = ..., user_model_name: _Optional[str] = ...) -> None: ...

class PredictModelResponse(_message.Message):
    __slots__ = ["y"]
    Y_FIELD_NUMBER: _ClassVar[int]
    y: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, y: _Optional[_Iterable[float]] = ...) -> None: ...

class DeleteModelRequest(_message.Message):
    __slots__ = ["user_model_name"]
    USER_MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
    user_model_name: str
    def __init__(self, user_model_name: _Optional[str] = ...) -> None: ...

class DeleteModelResponse(_message.Message):
    __slots__ = ["status"]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: str
    def __init__(self, status: _Optional[str] = ...) -> None: ...
