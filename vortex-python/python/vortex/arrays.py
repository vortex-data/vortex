# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
from __future__ import annotations

import abc
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pyarrow
from typing_extensions import override

import vortex._lib.arrays as _arrays  # pyright: ignore[reportMissingModuleSource]
from vortex._lib.dtype import DType  # pyright: ignore[reportMissingModuleSource]
from vortex._lib.serde import ArrayContext, ArrayParts  # pyright: ignore[reportMissingModuleSource]

try:
    import pandas
except ImportError:
    pass
else:
    # HACK: monkey-patch a fixed implementation of the pd.ArrowDtype.type property accessor.
    # See https://github.com/pandas-dev/pandas/issues/60068 for more details
    _old_ArrowDtype_type: Callable[[pandas.ArrowDtype], type] = pandas.ArrowDtype.type.fget  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

    @property
    def __ArrowDtype_type_patched(self: pandas.ArrowDtype):
        if pyarrow.types.is_string_view(self.pyarrow_dtype):
            return str
        if pyarrow.types.is_binary_view(self.pyarrow_dtype):
            return bytes
        assert _old_ArrowDtype_type is not None
        return _old_ArrowDtype_type(self)

    setattr(pandas.ArrowDtype, "type", __ArrowDtype_type_patched)


if TYPE_CHECKING:
    import numpy

Array = _arrays.Array


def empty_arrow_table(schema: pyarrow.Schema) -> pyarrow.Table:
    return pyarrow.Table.from_arrays([pyarrow.array([], type=t) for t in schema], schema=schema)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType, reportUnknownArgumentType]


def arrow_table_from_struct_array(
    array: pyarrow.StructArray | pyarrow.ChunkedArray[pyarrow.StructScalar],
) -> pyarrow.Table:
    if len(array) == 0:
        return empty_arrow_table(pyarrow.schema(array.type))
    return pyarrow.Table.from_struct_array(array)


def _Array_to_arrow_table(self: _arrays.Array) -> pyarrow.Table:
    """Construct an Arrow table from this Vortex array.

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
    array = self.to_arrow_array()
    assert isinstance(array, pyarrow.StructArray | pyarrow.ChunkedArray)
    return arrow_table_from_struct_array(array)


Array.to_arrow_table = _Array_to_arrow_table


def _Array_to_pandas_df(self: _arrays.Array) -> pandas.DataFrame:
    """Construct a Pandas dataframe from this Vortex array.

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
    import pandas

    return self.to_arrow_table().to_pandas(types_mapper=pandas.ArrowDtype)  # pyright: ignore[reportUnknownMemberType]


Array.to_pandas_df = _Array_to_pandas_df


def _Array_to_polars_dataframe(
    self: _arrays.Array,
):  # -> 'polars.DataFrame':  # breaks docs due to Polars issue #7027
    """Construct a Polars dataframe from this Vortex array.

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
    import polars

    return polars.from_arrow(self.to_arrow_table())  # pyright: ignore[reportUnknownMemberType]


setattr(Array, "to_polars_dataframe", _Array_to_polars_dataframe)


def _Array_to_polars_series(self: _arrays.Array):  # -> 'polars.Series':  # breaks docs due to Polars issue #7027
    """Construct a Polars series from this Vortex array.

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
    import polars

    return polars.from_arrow(self.to_arrow_array())  # pyright: ignore[reportUnknownMemberType]


setattr(Array, "to_polars_series", _Array_to_polars_series)


def _Array_to_numpy(self: _arrays.Array, *, zero_copy_only: bool = True) -> numpy.ndarray:
    """Construct a NumPy array from this Vortex array.

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
    return self.to_arrow_array().to_numpy(zero_copy_only=zero_copy_only)


Array.to_numpy = _Array_to_numpy


def _Array_to_pylist(self: _arrays.Array) -> list[Any]:  # pyright: ignore[reportExplicitAny]
    """Deeply copy an Array into a Python list.

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
    return self.to_arrow_table().to_pylist()


Array.to_pylist = _Array_to_pylist


def array(
    obj: pyarrow.Array[pyarrow.Scalar[Any]]  # pyright: ignore[reportExplicitAny]
    | pyarrow.ChunkedArray[pyarrow.Scalar[Any]]  # pyright: ignore[reportExplicitAny]
    | pyarrow.Table
    | list[Any]  # pyright: ignore[reportExplicitAny]
    | pandas.DataFrame,
) -> Array:
    """The main entry point for creating Vortex arrays from other Python objects.

    This function is also available as ``vortex.array``.

    Parameters
    ----------
    obj : :class:`pyarrow.Array`, :class:`pyarrow.ChunkedArray`, :class:`pyarrow.Table`, :class:`list`,
          :class:`pandas.DataFrame`
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

    if isinstance(obj, list):
        return Array.from_arrow(pyarrow.array(obj))
    try:
        import pandas

        if isinstance(obj, pandas.DataFrame):
            return Array.from_arrow(pyarrow.Table.from_pandas(obj))
    except ImportError:
        # if we cannot import pandas, it cannot be a pandas DataFrame
        assert isinstance(obj, pyarrow.Array | pyarrow.ChunkedArray | pyarrow.Table)
    return Array.from_arrow(obj)


class PyArray(Array, metaclass=abc.ABCMeta):
    """Abstract base class for Python-based Vortex arrays."""

    @property
    @override
    @abc.abstractmethod
    def id(self) -> str:
        """The id of the array."""

    @override
    @abc.abstractmethod
    def __len__(self) -> int:
        """The logical length of the array."""

    @property
    @override
    @abc.abstractmethod
    def dtype(self) -> DType:
        """The data type of the array."""

    @classmethod
    @abc.abstractmethod
    def decode(cls, parts: ArrayParts, ctx: ArrayContext, dtype: DType, len: int) -> Array:
        """Decode an array from its component parts.

        :class:`ArrayParts` contains the metadata, buffers and child :class:`ArrayParts` that represent the
        current array. Implementations of this function should validate this information, and then construct
        a new array.
        """
