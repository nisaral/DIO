from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class RegisterRequest(_message.Message):
    __slots__ = ("worker_id", "address", "models", "tier", "vram_gb")
    WORKER_ID_FIELD_NUMBER: _ClassVar[int]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    MODELS_FIELD_NUMBER: _ClassVar[int]
    TIER_FIELD_NUMBER: _ClassVar[int]
    VRAM_GB_FIELD_NUMBER: _ClassVar[int]
    worker_id: str
    address: str
    models: _containers.RepeatedScalarFieldContainer[str]
    tier: str
    vram_gb: int
    def __init__(self, worker_id: _Optional[str] = ..., address: _Optional[str] = ..., models: _Optional[_Iterable[str]] = ..., tier: _Optional[str] = ..., vram_gb: _Optional[int] = ...) -> None: ...

class RegisterResponse(_message.Message):
    __slots__ = ("success",)
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    success: bool
    def __init__(self, success: bool = ...) -> None: ...

class InferenceRequest(_message.Message):
    __slots__ = ("model_id", "data", "tier")
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    TIER_FIELD_NUMBER: _ClassVar[int]
    model_id: str
    data: bytes
    tier: str
    def __init__(self, model_id: _Optional[str] = ..., data: _Optional[bytes] = ..., tier: _Optional[str] = ...) -> None: ...

class InferenceResponse(_message.Message):
    __slots__ = ("output", "latency_ms", "tokens_used", "ttft_ms")
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    TOKENS_USED_FIELD_NUMBER: _ClassVar[int]
    TTFT_MS_FIELD_NUMBER: _ClassVar[int]
    output: bytes
    latency_ms: float
    tokens_used: int
    ttft_ms: float
    def __init__(self, output: _Optional[bytes] = ..., latency_ms: _Optional[float] = ..., tokens_used: _Optional[int] = ..., ttft_ms: _Optional[float] = ...) -> None: ...
