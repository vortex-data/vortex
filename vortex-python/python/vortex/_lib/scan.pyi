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

def open(path: str) -> VortexScan: ...
