#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

from .arrays import Array
from .session import Session

def compress(array: Array, *, session: Session) -> Array: ...
