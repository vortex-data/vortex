#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

from ..type_aliases import IntoArrayIterator
from .arrays import Array
from .expr import Expr
from .store import (
    AzureStore,
    GCSStore,
    HTTPStore,
    S3Store,
)

def read_url(
    url: str,
    *,
    store=None,
    projection: list[str] | list[int] | None = None,
    row_filter: Expr | None = None,
    indices: Array | None = None,
) -> Array: ...
def write(iter: IntoArrayIterator, path: str, *, store: AzureStore | HTTPStore | GCSStore | S3Store | None) -> None: ...

class VortexWriteOptions:
    @staticmethod
    def default() -> VortexWriteOptions: ...
    @staticmethod
    def compact() -> VortexWriteOptions: ...
    @staticmethod
    def write(
        iter: IntoArrayIterator, path: str, *, store: AzureStore | HTTPStore | GCSStore | S3Store | None
    ) -> VortexWriteOptions: ...
