# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Register Python extension types with the process-wide Vortex registry."""

from vortex._lib.registry import register  # pyright: ignore[reportMissingModuleSource]

__all__ = ["register"]
