# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tencent Cloud COS object store, backed by OpenDAL.

The class is re-exported from the native extension module.
"""

from __future__ import annotations

from vortex._lib import CosStore as CosStore

__all__ = ["CosStore"]
