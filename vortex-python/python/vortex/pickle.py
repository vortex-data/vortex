# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

import io
import pickle as _pickle
from ast import literal_eval
from collections.abc import Callable, Iterable, Sequence
from typing import BinaryIO, TypeAlias, TypeGuard, cast

from typing_extensions import override

from vortex._lib.arrays import Array  # pyright: ignore[reportMissingModuleSource]
from vortex._lib.serde import (  # pyright: ignore[reportMissingModuleSource]
    decode_ipc_array_buffers,
    encode_ipc_array_buffers,
)
from vortex._lib.session import Session  # pyright: ignore[reportMissingModuleSource]

_ARRAY_PERSISTENT_ID = "vortex.array"
_ARRAY_PERSISTENT_ID_VERSION = 1

_BufferSequence: TypeAlias = Sequence[bytes | memoryview]
_ArrayPersistentId: TypeAlias = tuple[str, int, _BufferSequence, _BufferSequence]
_BufferCallback: TypeAlias = Callable[[_pickle.PickleBuffer], object | None]
_OutOfBandBuffers: TypeAlias = Iterable[bytes | bytearray | memoryview | _pickle.PickleBuffer]


def _is_buffer_sequence(obj: object) -> TypeGuard[_BufferSequence]:
    return isinstance(obj, Sequence) and all(isinstance(buffer, bytes | memoryview) for buffer in obj)


def _parse_array_persistent_id(pid: object) -> _ArrayPersistentId:
    parsed_pid: object = pid
    if isinstance(parsed_pid, str):
        try:
            parsed_pid = cast(object, literal_eval(parsed_pid))
        except (SyntaxError, ValueError) as err:
            raise _pickle.UnpicklingError(f"unsupported persistent id: {pid!r}") from err

    if not isinstance(parsed_pid, tuple):
        raise _pickle.UnpicklingError(f"unsupported persistent id: {pid!r}")

    parsed_tuple = cast(tuple[object, ...], parsed_pid)
    if len(parsed_tuple) != 4:
        raise _pickle.UnpicklingError(f"unsupported persistent id: {pid!r}")

    tag, version, array_buffers, dtype_buffers = parsed_tuple
    if tag != _ARRAY_PERSISTENT_ID or version != _ARRAY_PERSISTENT_ID_VERSION:
        raise _pickle.UnpicklingError(f"unsupported persistent id: {pid!r}")
    if not _is_buffer_sequence(array_buffers) or not _is_buffer_sequence(dtype_buffers):
        raise _pickle.UnpicklingError(f"unsupported persistent id: {pid!r}")

    return (_ARRAY_PERSISTENT_ID, _ARRAY_PERSISTENT_ID_VERSION, array_buffers, dtype_buffers)


class Pickler(_pickle.Pickler):
    """Pickler that serializes Vortex arrays using an explicit session."""

    def __init__(
        self,
        file: BinaryIO,
        *,
        session: Session,
        protocol: int | None = None,
        fix_imports: bool = True,
        buffer_callback: _BufferCallback | None = None,
    ) -> None:
        super().__init__(
            file,
            protocol=protocol,
            fix_imports=fix_imports,
            buffer_callback=buffer_callback,
        )
        self._session: Session = session

    @override
    def persistent_id(self, obj: object) -> object | None:
        if isinstance(obj, Array):
            array_buffers, dtype_buffers = encode_ipc_array_buffers(obj, session=self._session)
            return (_ARRAY_PERSISTENT_ID, _ARRAY_PERSISTENT_ID_VERSION, array_buffers, dtype_buffers)
        return None


class Unpickler(_pickle.Unpickler):
    """Unpickler that deserializes Vortex arrays using an explicit session."""

    def __init__(
        self,
        file: BinaryIO,
        *,
        session: Session,
        fix_imports: bool = True,
        encoding: str = "ASCII",
        errors: str = "strict",
        buffers: _OutOfBandBuffers | None = None,
    ) -> None:
        super().__init__(
            file,
            fix_imports=fix_imports,
            encoding=encoding,
            errors=errors,
            buffers=buffers,
        )
        self._session: Session = session

    @override
    def persistent_load(self, pid: object) -> object:
        _, _, array_buffers, dtype_buffers = _parse_array_persistent_id(pid)
        return decode_ipc_array_buffers(array_buffers, dtype_buffers, session=self._session)


def dump(
    obj: object,
    file: BinaryIO,
    *,
    session: Session,
    protocol: int | None = None,
    fix_imports: bool = True,
    buffer_callback: _BufferCallback | None = None,
) -> None:
    Pickler(
        file,
        session=session,
        protocol=protocol,
        fix_imports=fix_imports,
        buffer_callback=buffer_callback,
    ).dump(obj)


def dumps(
    obj: object,
    *,
    session: Session,
    protocol: int | None = None,
    fix_imports: bool = True,
    buffer_callback: _BufferCallback | None = None,
) -> bytes:
    file = io.BytesIO()
    dump(
        obj,
        file,
        session=session,
        protocol=protocol,
        fix_imports=fix_imports,
        buffer_callback=buffer_callback,
    )
    return file.getvalue()


def load(
    file: BinaryIO,
    *,
    session: Session,
    fix_imports: bool = True,
    encoding: str = "ASCII",
    errors: str = "strict",
    buffers: _OutOfBandBuffers | None = None,
) -> object:
    return cast(
        object,
        Unpickler(
            file,
            session=session,
            fix_imports=fix_imports,
            encoding=encoding,
            errors=errors,
            buffers=buffers,
        ).load(),
    )


def loads(
    data: bytes | bytearray | memoryview,
    *,
    session: Session,
    fix_imports: bool = True,
    encoding: str = "ASCII",
    errors: str = "strict",
    buffers: _OutOfBandBuffers | None = None,
) -> object:
    return load(
        io.BytesIO(data),
        session=session,
        fix_imports=fix_imports,
        encoding=encoding,
        errors=errors,
        buffers=buffers,
    )


VortexPickler = Pickler
VortexUnpickler = Unpickler

__all__ = [
    "Pickler",
    "Unpickler",
    "VortexPickler",
    "VortexUnpickler",
    "dump",
    "dumps",
    "load",
    "loads",
]
