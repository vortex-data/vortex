"""
Vortex is an Apache Arrow-compatible toolkit for working with compressed array data.
"""

from __future__ import annotations
from vortex import arrays
from vortex import compress
from vortex import dataset
from vortex import dtype
from vortex import expr
from vortex import file
from vortex import io
from vortex import iter
from vortex import registry
from vortex import scalar
from vortex import serde

__all__: list = ["arrays", "compress", "dataset", "dtype", "expr", "file", "io", "iter", "registry", "scalar", "serde"]
