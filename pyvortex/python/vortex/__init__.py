from . import dataset, encoding
from ._lib import __doc__ as module_docs
from ._lib import dtype, expr, io, scalar

__doc__ = module_docs
del module_docs
array = encoding.array
compress = encoding.compress

__all__ = ["array", dtype, expr, io, encoding, scalar, dataset]
