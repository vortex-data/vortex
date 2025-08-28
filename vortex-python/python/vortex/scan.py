# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

from typing import final

from . import Scalar
from ._lib import scan as _scan  # pyright: ignore[reportMissingModuleSource]
from .type_aliases import IntoProjection


@final
class VortexScan:
    def __init__(self, scan: _scan.VortexScan):
        self._scan = scan

    @classmethod
    def from_file(cls, path: str) -> VortexScan:
        return cls(_scan.open(path))

    def __len__(self) -> int:
        return self._scan.__len__()

    def scalar_at(
        self,
        idx: int,
        *,
        projection: IntoProjection = None,
    ) -> Scalar:
        return self._scan.scalar_at(idx, projection)
