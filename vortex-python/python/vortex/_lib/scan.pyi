#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

from typing import final

from vortex import Scalar
from vortex.type_aliases import IntoProjection

@final
class VortexScan:
    def __len__(self) -> int: ...
    def scalar_at(
        self,
        idx: int,
        projection: IntoProjection = None,
    ) -> Scalar: ...
    def scalar_at_prepared(self, idx: int, concurrency: int = 4) -> Scalar: ...

def open(path: str, projection: IntoProjection = None) -> VortexScan: ...
