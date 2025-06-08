from __future__ import annotations
import numpy
import pandas
import pyarrow.lib
import typing
__all__: list = ['Array', 'NativeArray', 'PythonArray', 'NullArray', 'BoolArray', 'PrimitiveArray', 'VarBinArray', 'VarBinViewArray', 'StructArray', 'ListArray', 'ExtensionArray', 'ConstantArray', 'ChunkedArray', 'ByteBoolArray', 'AlpArray', 'AlpRdArray', 'DateTimePartsArray', 'DictArray', 'FsstArray', 'RunEndArray', 'SparseArray', 'ZigZagArray', 'FastLanesBitPackedArray', 'FastLanesDeltaArray', 'FastLanesFoRArray']
class AlpArray(NativeArray):
    """
    Concrete class for arrays with `vortex.alp` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class AlpRdArray(NativeArray):
    """
    Concrete class for arrays with `vortex.alprd` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class Array:
    """
    An array of zero or more *rows* each with the same set of *columns*.
    
    Examples
    --------
    
    Arrays support all the standard comparison operations:
    
        >>> import vortex as vx
        >>> a = vx.array(['dog', None, 'cat', 'mouse', 'fish'])
        >>> b = vx.array(['doug', 'jennifer', 'casper', 'mouse', 'faust'])
        >>> (a < b).to_arrow_array()
        <pyarrow.lib.BooleanArray object at ...>
        [
           true,
           null,
           false,
           false,
           false
        ]
        >>> (a <= b).to_arrow_array()
        <pyarrow.lib.BooleanArray object at ...>
        [
           true,
           null,
           false,
           true,
           false
        ]
        >>> (a == b).to_arrow_array()
        <pyarrow.lib.BooleanArray object at ...>
        [
           false,
           null,
           false,
           true,
           false
        ]
        >>> (a != b).to_arrow_array()
        <pyarrow.lib.BooleanArray object at ...>
        [
           true,
           null,
           true,
           false,
           true
        ]
        >>> (a >= b).to_arrow_array()
        <pyarrow.lib.BooleanArray object at ...>
        [
           false,
           null,
           true,
           true,
           true
        ]
        >>> (a > b).to_arrow_array()
        <pyarrow.lib.BooleanArray object at ...>
        [
           false,
           null,
           true,
           false,
           true
        ]
    """
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @staticmethod
    def from_arrow(obj):
        """
        Convert a PyArrow object into a Vortex array.
        
        One of :class:`pyarrow.Array`, :class:`pyarrow.ChunkedArray`, or :class:`pyarrow.Table`.
        
        Returns
        -------
        :class:`~vortex.Array`
        """
    def __eq__(self, value):
        """
        Return self==value.
        """
    def __ge__(self, value):
        """
        Return self>=value.
        """
    def __gt__(self, value):
        """
        Return self>value.
        """
    def __le__(self, value):
        """
        Return self<=value.
        """
    def __len__(self):
        """
        Return len(self).
        """
    def __lt__(self, value):
        """
        Return self<value.
        """
    def __ne__(self, value):
        """
        Return self!=value.
        """
    def __str__(self):
        """
        Return str(self).
        """
    def filter(self, mask):
        """
        Filter an Array by another Boolean array.
        
        Parameters
        ----------
        filter : :class:`~vortex.Array`
            Keep all the rows in ``self`` for which the correspondingly indexed row in `filter` is True.
        
        Returns
        -------
        :class:`~vortex.Array`
        
        Examples
        --------
        
        Keep only the single digit positive integers.
        
            >>> import vortex as vx
            >>> a = vx.array([0, 42, 1_000, -23, 10, 9, 5])
            >>> filter = vx.array([True, False, False, False, False, True, True])
            >>> a.filter(filter).to_arrow_array()
            <pyarrow.lib.Int64Array object at ...>
            [
              0,
              9,
              5
            ]
        """
    def scalar_at(self, index):
        """
        Retrieve a row by its index.
        
        Parameters
        ----------
        index : :class:`int`
            The index of interest. Must be greater than or equal to zero and less than the length of
            this array.
        
        Returns
        -------
        :class:`vortex.Scalar`
        
        Examples
        --------
        
        Retrieve the last element from an array of integers:
        
            >>> import vortex as vx
            >>> vx.array([10, 42, 999, 1992]).scalar_at(3).as_py()
            1992
        
        Retrieve the third element from an array of strings:
        
            >>> array = vx.array(["hello", "goodbye", "it", "is"])
            >>> array.scalar_at(2).as_py()
            'it'
        
        Retrieve an element from an array of structures:
        
            >>> array = vx.array([
            ...     {'name': 'Joseph', 'age': 25},
            ...     {'name': 'Narendra', 'age': 31},
            ...     {'name': 'Angela', 'age': 33},
            ...     None,
            ...     {'name': 'Mikhail', 'age': 57},
            ... ])
            >>> array.scalar_at(2).as_py()
            {'age': 33, 'name': 'Angela'}
        
        Retrieve a missing element from an array of structures:
        
            >>> array.scalar_at(3).as_py() is None
            True
        
        Out of bounds accesses are prohibited:
        
            >>> vx.array([10, 42, 999, 1992]).scalar_at(10)
            Traceback (most recent call last):
            ...
            ValueError: index 10 out of bounds from 0 to 4
            ...
        
        Unlike Python, negative indices are not supported:
        
            >>> vx.array([10, 42, 999, 1992]).scalar_at(-2)
            Traceback (most recent call last):
            ...
            OverflowError: can't convert negative int to unsigned
        """
    def serialize(self, ctx):
        ...
    def slice(self, start, end):
        """
        Slice this array.
        
        Parameters
        ----------
        start : :class:`int`
            The start index of the range to keep, inclusive.
        
        end : :class:`int`
            The end index, exclusive.
        
        Returns
        -------
        :class:`~vortex.Array`
        
        Examples
        --------
        
        Keep only the second through third elements:
        
            >>> import vortex as vx
            >>> a = vx.array(['a', 'b', 'c', 'd'])
            >>> a.slice(1, 3).to_arrow_array()
            <pyarrow.lib.StringArray object at ...>
            [
              "b",
              "c"
            ]
        
        Keep none of the elements:
        
            >>> a = vx.array(['a', 'b', 'c', 'd'])
            >>> a.slice(3, 3).to_arrow_array()
            <pyarrow.lib.StringViewArray object at ...>
            []
        
        Unlike Python, it is an error to slice outside the bounds of the array:
        
            >>> a = vx.array(['a', 'b', 'c', 'd'])
            >>> a.slice(2, 10).to_arrow_array()
            Traceback (most recent call last):
            ...
            ValueError: index 10 out of bounds from 0 to 4
        
        Or to slice with a negative value:
        
            >>> a = vx.array(['a', 'b', 'c', 'd'])
            >>> a.slice(-2, -1).to_arrow_array()
            Traceback (most recent call last):
            ...
            OverflowError: can't convert negative int to unsigned
        """
    def take(self, indices):
        """
        Filter, permute, and/or repeat elements by their index.
        
        Parameters
        ----------
        indices : :class:`~vortex.Array`
            An array of indices to keep.
        
        Returns
        -------
        :class:`~vortex.Array`
        
        Examples
        --------
        
        Keep only the first and third elements:
        
            >>> a = vx.array(['a', 'b', 'c', 'd'])
            >>> indices = vx.array([0, 2])
            >>> a.take(indices).to_arrow_array()
            <pyarrow.lib.StringArray object at ...>
            [
              "a",
              "c"
            ]
        
        Permute and repeat the first and second elements:
        
            >>> a = vx.array(['a', 'b', 'c', 'd'])
            >>> indices = vx.array([0, 1, 1, 0])
            >>> a.take(indices).to_arrow_array()
            <pyarrow.lib.StringArray object at ...>
            [
              "a",
              "b",
              "b",
              "a"
            ]
        """
    def to_arrow_array(self):
        """
        Convert this array to a PyArrow array.
        
        Convert this array to an Arrow array.
        
        .. seealso::
            :meth:`.to_arrow_table`
        
        Returns
        -------
        :class:`pyarrow.Array`
        
        Examples
        --------
        
        Round-trip an Arrow array through a Vortex array:
        
            >>> import vortex as vx
            >>> vx.array([1, 2, 3]).to_arrow_array()
            <pyarrow.lib.Int64Array object at ...>
            [
              1,
              2,
              3
            ]
        """
    def to_arrow_table(self) -> pyarrow.lib.Table:
        """
        Construct an Arrow table from this Vortex array.
        
            .. seealso::
                :meth:`.to_arrow_array`
        
            Warning
            -------
        
            Only struct-typed arrays can be converted to Arrow tables.
        
            Returns
            -------
        
            :class:`pyarrow.Table`
        
            Examples
            --------
        
            >>> array = vortex.array([
            ...     {'name': 'Joseph', 'age': 25},
            ...     {'name': 'Narendra', 'age': 31},
            ...     {'name': 'Angela', 'age': 33},
            ...     {'name': 'Mikhail', 'age': 57},
            ... ])
            >>> array.to_arrow_table()
            pyarrow.Table
            age: int64
            name: string_view
            ----
            age: [[25,31,33,57]]
            name: [["Joseph","Narendra","Angela","Mikhail"]]
        
            
        """
    def to_numpy(self, *, zero_copy_only: bool = True) -> numpy.ndarray:
        """
        Construct a NumPy array from this Vortex array.
        
            This is an alias for :code:`self.to_arrow_array().to_numpy(zero_copy_only)`
        
            Parameters
            ----------
            zero_copy_only : :class:`bool`
                When :obj:`True`, this method will raise an error unless a NumPy array can be created without
                copying the data. This is only possible when the array is a primitive array without nulls.
        
            Returns
            -------
            :class:`numpy.ndarray`
        
            Examples
            --------
        
            Construct an immutable ndarray from a Vortex array:
        
            >>> array = vortex.array([1, 0, 0, 1])
            >>> array.to_numpy()
            array([1, 0, 0, 1])
        
            
        """
    def to_pandas_df(self) -> pandas.DataFrame:
        """
        Construct a Pandas dataframe from this Vortex array.
        
            Warning
            -------
        
            Only struct-typed arrays can be converted to Pandas dataframes.
        
            Returns
            -------
            :class:`pandas.DataFrame`
        
            Examples
            --------
        
            Construct a dataframe from a Vortex array:
        
            >>> array = vortex.array([
            ...     {'name': 'Joseph', 'age': 25},
            ...     {'name': 'Narendra', 'age': 31},
            ...     {'name': 'Angela', 'age': 33},
            ...     {'name': 'Mikhail', 'age': 57},
            ... ])
            >>> array.to_pandas_df()
               age      name
            0   25    Joseph
            1   31  Narendra
            2   33    Angela
            3   57   Mikhail
        
            
        """
    def to_polars_dataframe(self):
        """
        Construct a Polars dataframe from this Vortex array.
        
            .. seealso::
                :meth:`.to_polars_series`
        
            Warning
            -------
        
            Only struct-typed arrays can be converted to Polars dataframes.
        
            Returns
            -------
        
            ..
                Polars excludes the DataFrame class from their Intersphinx index https://github.com/pola-rs/polars/issues/7027
        
            `polars.DataFrame <https://docs.pola.rs/api/python/stable/reference/dataframe/index.html>`__
        
            Examples
            --------
        
            >>> array = vortex.array([
            ...     {'name': 'Joseph', 'age': 25},
            ...     {'name': 'Narendra', 'age': 31},
            ...     {'name': 'Angela', 'age': 33},
            ...     {'name': 'Mikhail', 'age': 57},
            ... ])
            >>> array.to_polars_dataframe()
            shape: (4, 2)
            ┌─────┬──────────┐
            │ age ┆ name     │
            │ --- ┆ ---      │
            │ i64 ┆ str      │
            ╞═════╪══════════╡
            │ 25  ┆ Joseph   │
            │ 31  ┆ Narendra │
            │ 33  ┆ Angela   │
            │ 57  ┆ Mikhail  │
            └─────┴──────────┘
        
            
        """
    def to_polars_series(self):
        """
        Construct a Polars series from this Vortex array.
        
            .. seealso::
                :meth:`.to_polars_dataframe`
        
            Returns
            -------
        
            ..
                Polars excludes the Series class from their Intersphinx index https://github.com/pola-rs/polars/issues/7027
        
            `polars.Series <https://docs.pola.rs/api/python/stable/reference/series/index.html>`__
        
            Examples
            --------
        
            Convert a numeric array with nulls to a Polars Series:
        
            >>> vortex.array([1, None, 2, 3]).to_polars_series()  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [i64]
            [
                1
                null
                2
                3
            ]
        
            Convert a UTF-8 string array to a Polars Series:
        
            >>> vortex.array(['hello, ', 'is', 'it', 'me?']).to_polars_series()  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [str]
            [
                "hello, "
                "is"
                "it"
                "me?"
            ]
        
            Convert a struct array to a Polars Series:
        
            >>> array = vortex.array([
            ...     {'name': 'Joseph', 'age': 25},
            ...     {'name': 'Narendra', 'age': 31},
            ...     {'name': 'Angela', 'age': 33},
            ...     {'name': 'Mikhail', 'age': 57},
            ... ])
            >>> array.to_polars_series()  # doctest: +NORMALIZE_WHITESPACE
            shape: (4,)
            Series: '' [struct[2]]
            [
                {25,"Joseph"}
                {31,"Narendra"}
                {33,"Angela"}
                {57,"Mikhail"}
            ]
        
            
        """
    def to_pylist(self) -> list[typing.Any]:
        """
        Deeply copy an Array into a Python list.
        
            Returns
            -------
            :class:`list`
        
            Examples
            --------
        
            >>> array = vortex.array([
            ...     {'name': 'Joseph', 'age': 25},
            ...     {'name': 'Narendra', 'age': 31},
            ...     {'name': 'Angela', 'age': 33},
            ... ])
            >>> array.to_pylist()
            [{'age': 25, 'name': 'Joseph'}, {'age': 31, 'name': 'Narendra'}, {'age': 33, 'name': 'Angela'}]
        
            
        """
    def tree_display(self):
        """
        Internal technical details about the encoding of this Array.
        
        Warnings
        --------
        The format of the returned string may change without notice.
        
        Returns
        -------
        :class:`.str`
        
        Examples
        --------
        
        Uncompressed arrays have straightforward encodings:
        
            >>> import vortex as vx
            >>> arr = vx.array([1, 2, None, 3])
            >>> print(arr.tree_display())
            root: vortex.primitive(i64?, len=4) nbytes=33 B (100.00%)
              metadata: EmptyMetadata
              buffer (align=8): 32 B (96.97%)
              validity: vortex.bool(bool, len=4) nbytes=1 B (3.03%)
                metadata: BoolMetadata { offset: 0 }
                buffer (align=1): 1 B (100.00%)
            <BLANKLINE>
        
        Compressed arrays often have more complex, deeply nested encoding trees.
        """
class BoolArray(NativeArray):
    """
    Concrete class for arrays with `vortex.bool` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ByteBoolArray(NativeArray):
    """
    Concrete class for arrays with `vortex.bytebool` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ChunkedArray(NativeArray):
    """
    Concrete class for arrays with `vortex.chunked` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def chunks(self):
        ...
class ConstantArray(NativeArray):
    """
    Concrete class for arrays with `vortex.constant` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def scalar(self):
        """
        Return the scalar value of the constant array.
        """
class DateTimePartsArray(NativeArray):
    """
    Concrete class for arrays with `vortex.datetimeparts` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class DictArray(NativeArray):
    """
    Concrete class for arrays with `vortex.dict` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ExtensionArray(NativeArray):
    """
    Concrete class for arrays with `vortex.ext` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class FastLanesBitPackedArray(NativeArray):
    """
    Concrete class for arrays with `fastlanes.bitpacked` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class FastLanesDeltaArray(NativeArray):
    """
    Concrete class for arrays with `fastlanes.delta` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class FastLanesFoRArray(NativeArray):
    """
    Concrete class for arrays with `fastlanes.for` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class FsstArray(NativeArray):
    """
    Concrete class for arrays with `vortex.fsst` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ListArray(NativeArray):
    """
    Concrete class for arrays with `vortex.list` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class NativeArray(Array):
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
class NullArray(NativeArray):
    """
    Concrete class for arrays with `vortex.null` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class PrimitiveArray(NativeArray):
    """
    Concrete class for arrays with `vortex.primitive` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class PythonArray(Array):
    """
    Base class for implementing a Vortex encoding in Python.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class RunEndArray(NativeArray):
    """
    Concrete class for arrays with `vortex.runend` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class SparseArray(NativeArray):
    """
    Concrete class for arrays with `vortex.sparse` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class StructArray(NativeArray):
    """
    Concrete class for arrays with `vortex.struct` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def field(self, name):
        """
        Returns the given field of the struct array.
        """
class VarBinArray(NativeArray):
    """
    Concrete class for arrays with `vortex.varbin` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class VarBinViewArray(NativeArray):
    """
    Concrete class for arrays with `vortex.varbinview` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
class ZigZagArray(NativeArray):
    """
    Concrete class for arrays with `vortex.zigzag` encoding.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
