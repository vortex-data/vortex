# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from typing import TypeAlias

import pyarrow as pa

from ._lib.arrays import Array  # pyright: ignore[reportMissingModuleSource]
from ._lib.expr import Expr  # pyright: ignore[reportMissingModuleSource]
from ._lib.iter import ArrayIterator  # pyright: ignore[reportMissingModuleSource]

# TypeAliases do not support __doc__.
IntoProjection: TypeAlias = Expr | list[str] | None
IntoArrayIterator: TypeAlias = Array | ArrayIterator | pa.Table | pa.RecordBatchReader

# If you make an intersphinx reference to pyarrow.RecordBatchReader in the return type of a function
# *and also* use the IntoProjection type alias in a parameter type, Sphinx thinks the type alias
# does not exist.
#
# Indirecting the intersphinx reference by way of a type alias avoids this bug.
#
# I failed to produce a small enough test case to report this bug.
RecordBatchReader: TypeAlias = pa.RecordBatchReader
