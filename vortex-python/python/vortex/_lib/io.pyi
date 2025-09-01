#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

from ..type_aliases import IntoArrayIterator
from .arrays import Array
from .expr import Expr

def read_url(
    url: str,
    *,
    projection: list[str] | list[int] | None = None,
    row_filter: Expr | None = None,
    indices: Array | None = None,
) -> Array: ...
def write(iter: IntoArrayIterator, path: str) -> None: ...
