#  SPDX-License-Identifier: Apache-2.0
#  SPDX-FileCopyrightText: Copyright the Vortex contributors

from .arrays import Array


def register(self, cls: type[Array]) -> None: ...
