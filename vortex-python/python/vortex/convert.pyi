from __future__ import annotations
import abc as abc
import numpy
import pandas as pandas
import pyarrow as pyarrow
import typing
from typing import Any
import vortex
from vortex import ArrayContext
from vortex import ArrayParts
import vortex.arrays
from vortex.arrays import Array
import vortex.dtype
from vortex.dtype import DType

__all__ = [
    "Any",
    "Array",
    "ArrayContext",
    "ArrayParts",
    "DType",
    "PyArray",
    "abc",
    "array",
    "arrow_table_from_struct_array",
    "empty_arrow_table",
    "pandas",
    "pyarrow",
]

class PyArray(vortex.arrays.Array):
    """
    Abstract base class for Python-based Vortex arrays.
    """

    __abstractmethods__: typing.ClassVar[frozenset]  # value = frozenset({'decode', 'dtype', '__len__'})
    _abc_impl: typing.ClassVar[_abc._abc_data]  # value = <_abc._abc_data object>
    @classmethod
    def decode(
        cls, parts: vortex.ArrayParts, ctx: vortex.ArrayContext, dtype: vortex.dtype.DType, len: int
    ) -> vortex.arrays.Array:
        """
        Decode an array from its component parts.

                :class:`ArrayParts` contains the metadata, buffers and child :class:`ArrayParts` that represent the
                current array. Implementations of this function should validate this information, and then construct
                a new array.

        """
    def __len__(self) -> int:
        """
        The logical length of the array.
        """
    @property
    def dtype(self) -> vortex.dtype.DType:
        """
        The data type of the array.
        """

def _Array_to_arrow_table(self: vortex.arrays.Array) -> pyarrow.lib.Table:
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

def _Array_to_numpy(self: vortex.arrays.Array, *, zero_copy_only: bool = True) -> numpy.ndarray:
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

def _Array_to_pandas_df(self: vortex.arrays.Array) -> pandas.DataFrame:
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

def _Array_to_polars_dataframe(self: vortex.arrays.Array):
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

def _Array_to_polars_series(self: vortex.arrays.Array):
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

def _Array_to_pylist(self: vortex.arrays.Array) -> list[typing.Any]:
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

def array(obj: pyarrow.lib.Array | list | typing.Any) -> vortex.arrays.Array:
    """
    The main entry point for creating Vortex arrays from other Python objects.

        This function is also available as ``vortex.array``.

        Parameters
        ----------
        obj : :class:`pyarrow.Array`, :class:`list`, :class:`pandas.DataFrame`
            The elements of this array or list become the elements of the Vortex array.

        Returns
        -------
        :class:`vortex.Array`

        Examples
        --------

        A Vortex array containing the first three integers:

        >>> vortex.array([1, 2, 3]).to_arrow_array()
        <pyarrow.lib.Int64Array object at ...>
        [
          1,
          2,
          3
        ]

        The same Vortex array with a null value in the third position:

        >>> vortex.array([1, 2, None, 3]).to_arrow_array()
        <pyarrow.lib.Int64Array object at ...>
        [
          1,
          2,
          null,
          3
        ]

        Initialize a Vortex array from an Arrow array:

        >>> arrow = pyarrow.array(['Hello', 'it', 'is', 'me'], type=pyarrow.string_view())
        >>> vortex.array(arrow).to_arrow_array()
        <pyarrow.lib.StringViewArray object at ...>
        [
          "Hello",
          "it",
          "is",
          "me"
        ]

        Initialize a Vortex array from a Pandas dataframe:

        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "Name": ["Braund", "Allen", "Bonnell"],
        ...     "Age": [22, 35, 58],
        ... })
        >>> vortex.array(df).to_arrow_array()
        <pyarrow.lib.ChunkedArray object at ...>
        [
          -- is_valid: all not null
          -- child 0 type: string_view
            [
              "Braund",
              "Allen",
              "Bonnell"
            ]
          -- child 1 type: int64
            [
              22,
              35,
              58
            ]
        ]


    """

def arrow_table_from_struct_array(array: pyarrow.lib.StructArray | pyarrow.lib.ChunkedArray) -> pyarrow.lib.Table: ...
def empty_arrow_table(schema: pyarrow.lib.Schema) -> pyarrow.lib.Table: ...

__ArrowDtype_type_patched: property  # value = <property object>
_old_ArrowDtype_type: property  # value = <property object>
