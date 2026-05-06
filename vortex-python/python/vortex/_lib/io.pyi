#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

from ..type_aliases import IntoArrayIterator
from .arrays import Array
from .expr import Expr
from .session import Session
from .store import ObjectStore

def read_url(
    url: str,
    *,
    store: ObjectStore | None = None,
    projection: list[str] | list[int] | None = None,
    row_filter: Expr | None = None,
    indices: Array | None = None,
    row_range: tuple[int, int] | None = None,
    session: Session,
) -> Array: ...
def write(
    iter: IntoArrayIterator,
    path: str,
    *,
    store: ObjectStore | None = None,
    session: Session,
) -> None: ...

class VortexWriteOptions:
    @staticmethod
    def default() -> VortexWriteOptions: ...
    @staticmethod
    def compact() -> VortexWriteOptions: ...
    @staticmethod
    def write(
        iter: IntoArrayIterator,
        path: str,
        *,
        store: ObjectStore | None = None,
        session: Session,
    ) -> VortexWriteOptions: ...
