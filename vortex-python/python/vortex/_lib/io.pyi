#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

from typing import Protocol

from ..type_aliases import IntoArrayIterator
from .arrays import Array
from .expr import Expr
from .store import ObjectStore

class _BinaryOutput(Protocol):
    def write(self, buffer: bytes, /) -> int: ...
    def flush(self) -> None: ...

def read_url(
    url: str,
    *,
    store: ObjectStore | None = None,
    projection: list[str] | list[int] | None = None,
    row_filter: Expr | None = None,
    indices: Array | None = None,
    row_range: tuple[int, int] | None = None,
) -> Array: ...
def write(
    iter: IntoArrayIterator,
    path: str,
    *,
    store: ObjectStore | None = None,
    edition: str | None = None,
) -> None: ...
def write_to(
    iter: IntoArrayIterator,
    output: _BinaryOutput,
    *,
    edition: str | None = None,
) -> None: ...

class VortexWriteOptions:
    @staticmethod
    def default() -> VortexWriteOptions: ...
    @staticmethod
    def compact() -> VortexWriteOptions: ...
    def with_edition(self, edition: str) -> VortexWriteOptions: ...
    @staticmethod
    def write(
        iter: IntoArrayIterator,
        path: str,
        *,
        store: ObjectStore | None = None,
    ) -> VortexWriteOptions: ...
    def write_to(self, iter: IntoArrayIterator, output: _BinaryOutput) -> None: ...
