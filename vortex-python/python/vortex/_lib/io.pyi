#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

from .expr import Expr
from .arrays import Array
from .file import IntoArrayIterator

def read_url(
    url: str,
    *,
    projection: list[str] | list[int] | None = None,
    row_filter: Expr | None = None,
    indices: Array | None = None,
) -> Array: ...
def write(iter: IntoArrayIterator, path: str) -> None: ...
