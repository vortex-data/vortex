#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

from .arrays import Array

def register(cls: type[Array]) -> None: ...
