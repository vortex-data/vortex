"""
Vortex is an Apache Arrow-compatible toolkit for working with compressed array data.
"""
from __future__ import annotations
from compress import compress
from file import open
from vortex.arrays import AlpArray
from vortex.arrays import AlpRdArray
from vortex.arrays import Array
from vortex.arrays import BoolArray
from vortex.arrays import ByteBoolArray
from vortex.arrays import ChunkedArray
from vortex.arrays import ConstantArray
from vortex.arrays import DateTimePartsArray
from vortex.arrays import DictArray
from vortex.arrays import ExtensionArray
from vortex.arrays import FastLanesBitPackedArray
from vortex.arrays import FastLanesDeltaArray
from vortex.arrays import FastLanesFoRArray
from vortex.arrays import FsstArray
from vortex.arrays import ListArray
from vortex.arrays import NativeArray
from vortex.arrays import NullArray
from vortex.arrays import PrimitiveArray
from vortex.arrays import RunEndArray
from vortex.arrays import SparseArray
from vortex.arrays import StructArray
from vortex.arrays import VarBinArray
from vortex.arrays import VarBinViewArray
from vortex.arrays import ZigZagArray
from vortex.convert import PyArray
from vortex.convert import array
from vortex.dtype import BinaryDType
from vortex.dtype import BoolDType
from vortex.dtype import DType
from vortex.dtype import DecimalDType
from vortex.dtype import ExtensionDType
from vortex.dtype import ListDType
from vortex.dtype import NullDType
from vortex.dtype import PType
from vortex.dtype import PrimitiveDType
from vortex.dtype import StructDType
from vortex.dtype import Utf8DType
from vortex.dtype import binary
from vortex.dtype import bool_
from vortex.dtype import ext
from vortex.dtype import float_
from vortex.dtype import int_
from vortex.dtype import list_
from vortex.dtype import null
from vortex.dtype import struct
from vortex.dtype import uint
from vortex.dtype import utf8
from vortex.scalar import BinaryScalar
from vortex.scalar import BoolScalar
from vortex.scalar import ExtensionScalar
from vortex.scalar import ListScalar
from vortex.scalar import NullScalar
from vortex.scalar import PrimitiveScalar
from vortex.scalar import Scalar
from vortex.scalar import StructScalar
from vortex.scalar import Utf8Scalar
from vortex.scalar import scalar
from . import arrays
from . import convert
from . import dataset
from . import dtype
from . import expr
from . import file
from . import io
from . import iter
from . import serde
from . import vortex
__all__: list = ['array', 'compress', 'Array', 'PyArray', 'DType', 'PType', 'NullDType', 'BoolDType', 'DecimalDType', 'PrimitiveDType', 'Utf8DType', 'BinaryDType', 'StructDType', 'ListDType', 'ExtensionDType', 'null', 'bool_', 'int_', 'uint', 'float_', 'utf8', 'binary', 'struct', 'list_', 'ext', 'ConstantArray', 'ChunkedArray', 'NullArray', 'BoolArray', 'ByteBoolArray', 'PrimitiveArray', 'VarBinArray', 'VarBinViewArray', 'StructArray', 'ListArray', 'ExtensionArray', 'AlpArray', 'AlpRdArray', 'DateTimePartsArray', 'DictArray', 'FsstArray', 'RunEndArray', 'SparseArray', 'ZigZagArray', 'FastLanesBitPackedArray', 'FastLanesDeltaArray', 'FastLanesFoRArray', 'scalar', 'Scalar', 'NullScalar', 'BoolScalar', 'PrimitiveScalar', 'Utf8Scalar', 'BinaryScalar', 'StructScalar', 'ListScalar', 'ExtensionScalar', 'Registry', 'ArrayContext', 'ArrayParts', 'VortexFile', 'open', 'ArrayIterator']
class ArrayContext:
    """
    An ArrayContext captures an ordered set of encodings.
    
    In a serialized array, encodings are identified by a positional index into such an
    :class:`~vortex.ArrayContext`.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __len__(self):
        """
        Return len(self).
        """
    def __str__(self):
        """
        Return str(self).
        """
class ArrayIterator:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def from_iter(dtype, iter):
        """
        Create a :class:`vortex.ArrayIterator` from an iterator of :class:`vortex.Array`.
        """
    def __iter__(self):
        """
        Implement iter(self).
        """
    def __next__(self):
        """
        Implement next(self).
        """
    def read_all(self):
        """
        Read all chunks into a single :class:`vortex.Array`. If there are multiple chunks,
        this will be a :class:`vortex.ChunkedArray`, otherwise it will be a single array.
        """
    def to_arrow(self):
        """
        Convert the :class:`vortex.ArrayIterator` into a :class:`pyarrow.RecordBatchReader`.
        """
class ArrayParts:
    """
    ArrayParts is a parsed representation of a serialized array.
    
    It can be decoded into a full array using the `decode` method.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def parse(data):
        """
        Parse a serialized array into its parts.
        """
    def decode(self, ctx, dtype, len):
        """
        Decode the array parts into a full array.
        
        # Returns
        
        The decoded array.
        """
class Registry:
    """
    A register of known array and layout encodings.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def array_ctx(self, encodings):
        """
        Create an :class:`~vortex.ArrayContext` containing the given encodings.
        """
    def register(self, cls):
        """
        Register an array encoding implemented by subclassing `PyArray`.
        
        It's not currently possible to register a layout encoding from Python.
        """
class VortexFile:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __len__(self):
        """
        Return len(self).
        """
    def scan(self, projection = None, *, expr = None, indices = None, batch_size = None):
        """
        Scan the Vortex file returning a :class:`vortex.ArrayIterator`.
        
        Parameters
        ----------
        projection : :class:`vortex.Expr` | None
            The projection expression to read, or else read all columns.
        expr : :class:`vortex.Expr` | None
            The predicate used to filter rows. The filter columns do not need to be in the projection.
        indices : :class:`vortex.Array` | None
            The indices of the rows to read. Must be sorted and non-null.
        batch_size : :class:`int` | None
            The number of rows to read per chunk.
        
        Examples
        --------
        
        Scan a file with a structured column and nulls at multiple levels and in multiple columns.
        
            >>> import vortex as vx
            >>> import vortex.expr as ve
            >>> a = vx.array([
            ...     {'name': 'Joseph', 'age': 25},
            ...     {'name': None, 'age': 31},
            ...     {'name': 'Angela', 'age': None},
            ...     {'name': 'Mikhail', 'age': 57},
            ...     {'name': None, 'age': None},
            ... ])
            >>> vx.io.write(a, "a.vortex")
            >>> vxf = vx.open("a.vortex")
            >>> vxf.scan().read_all().to_arrow_array()
            <pyarrow.lib.StructArray object at ...>
            -- is_valid: all not null
            -- child 0 type: int64
              [
                25,
                31,
                null,
                57,
                null
              ]
            -- child 1 type: string_view
              [
                "Joseph",
                null,
                "Angela",
                "Mikhail",
                null
              ]
        
        Read just the age column:
        
            >>> vxf.scan(['age']).read_all().to_arrow_array()
            <pyarrow.lib.StructArray object at ...>
            -- is_valid: all not null
            -- child 0 type: int64
              [
                25,
                31,
                null,
                57,
                null
              ]
        
        
        Keep rows with an age above 35. This will read O(N_KEPT) rows, when the file format allows.
        
            >>> vxf.scan(expr=ve.column("age") > 35).read_all().to_arrow_array()
            <pyarrow.lib.StructArray object at ...>
            -- is_valid: all not null
            -- child 0 type: int64
              [
                57
              ]
            -- child 1 type: string_view
              [
                "Mikhail"
              ]
        """
    def to_arrow(self, projection = None, *, expr = None, batch_size = None):
        """
        Scan the Vortex file as a :class:`pyarrow.RecordBatchReader`.
        """
    def to_dataset(self):
        """
        Scan the Vortex file using the :class:`pyarrow.dataset.Dataset` API.
        """
registry: Registry  # value = <vortex.Registry object>
