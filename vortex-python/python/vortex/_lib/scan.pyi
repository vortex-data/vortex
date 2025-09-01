#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

from typing import final

from vortex import ArrayIterator

from .scalar import Scalar

@final
class RepeatedScan:
    def execute(
        self,
        *,
        start: int | None = None,
        stop: int | None = None,
    ) -> ArrayIterator: ...
    def scalar_at(self, index: int) -> Scalar: ...
