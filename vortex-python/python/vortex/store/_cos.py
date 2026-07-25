# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tencent Cloud COS object store, backed by OpenDAL.

This store is only available when Vortex is built with the ``opendal`` feature.
The class is re-exported from the native extension module; if the feature is
not enabled, instantiating :class:`CosStore` raises :class:`ImportError`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vortex._lib import CosStore

try:
    from vortex._lib import CosStore as CosStore  # type: ignore[attr-defined, no-redef]
except ImportError:

    class CosStore:  # type: ignore[no-redef]
        """Placeholder; the real implementation requires the ``opendal`` feature."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "CosStore requires Vortex to be built with the 'opendal' feature; "
                "build with `maturin build --features opendal` or `maturin develop --features opendal`."
            )


__all__ = ["CosStore"]
