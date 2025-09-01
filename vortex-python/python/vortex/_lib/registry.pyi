#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

from typing import final

from .arrays import Array
from .serde import ArrayContext

@final
class Registry:
    def register(self, cls: type[Array]) -> None: ...
    def array_ctx(self, encodings: list[type[Array] | str]) -> ArrayContext: ...
