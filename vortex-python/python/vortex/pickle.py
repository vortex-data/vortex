# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

import io
import pickle as _pickle
from ast import literal_eval
from typing import Any

from vortex._lib.arrays import Array  # pyright: ignore[reportMissingModuleSource]
from vortex._lib.serde import (  # pyright: ignore[reportMissingModuleSource]
    decode_ipc_array_buffers,
    encode_ipc_array_buffers,
)
from vortex._lib.session import Session  # pyright: ignore[reportMissingModuleSource]

_ARRAY_PERSISTENT_ID = "vortex.array"
_ARRAY_PERSISTENT_ID_VERSION = 1


class Pickler(_pickle.Pickler):
    """Pickler that serializes Vortex arrays using an explicit session."""

    def __init__(
        self,
        file: Any,  # pyright: ignore[reportExplicitAny]
        *,
        session: Session,
        protocol: int | None = None,
        fix_imports: bool = True,
        buffer_callback: Any | None = None,  # pyright: ignore[reportExplicitAny]
    ) -> None:
        super().__init__(
            file,
            protocol=protocol,
            fix_imports=fix_imports,
            buffer_callback=buffer_callback,
        )
        self._session = session

    def persistent_id(self, obj: object) -> object | None:
        if isinstance(obj, Array):
            array_buffers, dtype_buffers = encode_ipc_array_buffers(obj, session=self._session)
            return (_ARRAY_PERSISTENT_ID, _ARRAY_PERSISTENT_ID_VERSION, array_buffers, dtype_buffers)
        return None


class Unpickler(_pickle.Unpickler):
    """Unpickler that deserializes Vortex arrays using an explicit session."""

    def __init__(
        self,
        file: Any,  # pyright: ignore[reportExplicitAny]
        *,
        session: Session,
        fix_imports: bool = True,
        encoding: str = "ASCII",
        errors: str = "strict",
        buffers: Any | None = None,  # pyright: ignore[reportExplicitAny]
    ) -> None:
        super().__init__(
            file,
            fix_imports=fix_imports,
            encoding=encoding,
            errors=errors,
            buffers=buffers,
        )
        self._session = session

    def persistent_load(self, pid: object) -> object:
        if isinstance(pid, str):
            try:
                pid = literal_eval(pid)
            except (SyntaxError, ValueError) as err:
                raise _pickle.UnpicklingError(f"unsupported persistent id: {pid!r}") from err

        if not isinstance(pid, tuple) or len(pid) != 4:
            raise _pickle.UnpicklingError(f"unsupported persistent id: {pid!r}")

        tag, version, array_buffers, dtype_buffers = pid
        if tag != _ARRAY_PERSISTENT_ID or version != _ARRAY_PERSISTENT_ID_VERSION:
            raise _pickle.UnpicklingError(f"unsupported persistent id: {pid!r}")

        return decode_ipc_array_buffers(array_buffers, dtype_buffers, session=self._session)


def dump(
    obj: object,
    file: Any,  # pyright: ignore[reportExplicitAny]
    *,
    session: Session,
    protocol: int | None = None,
    fix_imports: bool = True,
    buffer_callback: Any | None = None,  # pyright: ignore[reportExplicitAny]
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
    buffer_callback: Any | None = None,  # pyright: ignore[reportExplicitAny]
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
    file: Any,  # pyright: ignore[reportExplicitAny]
    *,
    session: Session,
    fix_imports: bool = True,
    encoding: str = "ASCII",
    errors: str = "strict",
    buffers: Any | None = None,  # pyright: ignore[reportExplicitAny]
) -> object:
    return Unpickler(
        file,
        session=session,
        fix_imports=fix_imports,
        encoding=encoding,
        errors=errors,
        buffers=buffers,
    ).load()


def loads(
    data: bytes | bytearray | memoryview,
    *,
    session: Session,
    fix_imports: bool = True,
    encoding: str = "ASCII",
    errors: str = "strict",
    buffers: Any | None = None,  # pyright: ignore[reportExplicitAny]
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
