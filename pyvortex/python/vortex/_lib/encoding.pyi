from typing import Any, final

import pyarrow as pa

import vortex as vx

def _encode(obj: Any) -> pa.Array: ...
def compress(array: Array) -> Array:
    """Attempt to compress a vortex array.

    Parameters
    ----------
    array : :class:`~vortex.encoding.Array`
        The array.

    Examples
    --------

    Compress a very sparse array of integers:

    >>> import vortex as vx
    >>> a = vx.array([42 for _ in range(1000)])
    >>> str(vx.compress(a))
    'vortex.constant(0x09)(i64, len=1000)'

    Compress an array of increasing integers:

    >>> a = vx.array(list(range(1000)))
    >>> str(vx.compress(a))
    'fastlanes.bitpacked(0x16)(i64, len=1000)'

    Compress an array of increasing floating-point numbers and a few nulls:

    >>> a = vx.array([
    ...     float(x) if x % 20 != 0 else None
    ...     for x in range(1000)
    ... ])
    >>> str(vx.compress(a))
    'vortex.alp(0x11)(f64?, len=1000)'
    """
@final
class Array:
    """An array of zero or more *rows* each with the same set of *columns*.

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

    def to_arrow_array(self) -> pa.Array:
        """Convert this array to a PyArrow array.

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
    def __lt__(self, other: Array) -> Array: ...
    def __le__(self, other: Array) -> Array: ...
    def __gt__(self, other: Array) -> Array: ...
    def __ge__(self, other: Array) -> Array: ...
    def __len__(self) -> int: ...
    @property
    def encoding(self) -> str:
        """Returns the encoding ID of this array."""

    @property
    def nbytes(self) -> int:
        """Returns the number of bytes used by this array."""

    @property
    def dtype(self) -> vx.DType:
        """Returns the data type of this array.

        Returns
        -------
        :class:`vortex.dtype.DType`

        Examples
        --------

        By default, :func:`~vortex.encoding.array` uses the largest available bit-width:

            >>> import vortex as vx
            >>> vx.array([1, 2, 3]).dtype
            int(64, False)

        Including a :obj:`None` forces a nullable type:

            >>> vx.array([1, None, 2, 3]).dtype
            int(64, True)

        A UTF-8 string array:

            >>> vx.array(['hello, ', 'is', 'it', 'me?']).dtype
            utf8(False)

        """

    def filter(self, mask: Array) -> Array:
        """Filter an Array by another Boolean array.

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

    def fill_forward(self) -> Array:
        """Fill forward non-null values over runs of nulls.

        Leading nulls are replaced with the "zero" for that type. For integral and floating-point
        types, this is zero. For the Boolean type, this is `:obj:`False`.

        Fill forward sensor values over intermediate missing values. Note that leading nulls are
        replaced with 0.0:

        >>> import vortex as vx
        >>> a = vx.array([
        ...      None,  None, 30.29, 30.30, 30.30,  None,  None, 30.27, 30.25,
        ...     30.22,  None,  None,  None,  None, 30.12, 30.11, 30.11, 30.11,
        ...     30.10, 30.08,  None, 30.21, 30.03, 30.03, 30.05, 30.07, 30.07,
        ... ])
        >>> a.fill_forward().to_arrow_array()
        <pyarrow.lib.DoubleArray object at ...>
        [
          0,
          0,
          30.29,
          30.3,
          30.3,
          30.3,
          30.3,
          30.27,
          30.25,
          30.22,
          ...
          30.11,
          30.1,
          30.08,
          30.08,
          30.21,
          30.03,
          30.03,
          30.05,
          30.07,
          30.07
        ]
        """

    def scalar_at(self, index: int) -> Any:
        """Retrieve a row by its index.

        Parameters
        ----------
        index : :class:`int`
            The index of interest. Must be greater than or equal to zero and less than the length of
            this array.

        Returns
        -------
        one of :class:`int`, :class:`float`, :class:`bool`, :class:`vortex.scalar.Buffer`, :class:`vortex.scalar.BufferString`, :class:`vortex.scalar.VortexList`, :class:`vortex.scalar.VortexStruct`
            If this array contains numbers or Booleans, this array returns the corresponding
            primitive Python type, i.e. int, float, and bool. For structures and variable-length
            data types, a zero-copy view of the underlying data is returned.

        Examples
        --------

        Retrieve the last element from an array of integers:

        >>> import vortex as vx
        >>> vx.array([10, 42, 999, 1992]).scalar_at(3)
        1992

        Retrieve the third element from an array of strings:

        >>> array = vx.array(["hello", "goodbye", "it", "is"])
        >>> array.scalar_at(2)
        <vortex.BufferString ...>

        Vortex, by default, returns a view into the array's data. This avoids copying the data,
        which can be expensive if done repeatedly. :meth:`.BufferString.into_python` forcibly copies
        the scalar data into a Python data structure.

        >>> array.scalar_at(2).into_python()
        'it'

        Retrieve an element from an array of structures:

        >>> array = vx.array([
        ...     {'name': 'Joseph', 'age': 25},
        ...     {'name': 'Narendra', 'age': 31},
        ...     {'name': 'Angela', 'age': 33},
        ...     None,
        ...     {'name': 'Mikhail', 'age': 57},
        ... ])
        >>> array.scalar_at(2).into_python()
        {'age': 33, 'name': <vortex.BufferString ...>}

        Notice that :meth:`.VortexStruct.into_python` only copies one "layer" of data into
        Python. If we want to ensure the entire structure is recurisvely copied into Python we can
        specify ``recursive=True``:

        >>> array.scalar_at(2).into_python(recursive=True)
        {'age': 33, 'name': 'Angela'}

        Retrieve a missing element from an array of structures:

        >>> array.scalar_at(3) is None
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

    def take(self, indices: Array) -> Array:
        """Filter, permute, and/or repeat elements by their index.

        Parameters
        ----------
        indices : :class:`~vortex.encoding.Array`
            An array of indices to keep.

        Returns
        -------
        :class:`~vortex.encoding.Array`

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

    def slice(self, start: int, end: int) -> Array:
        """Slice this array.

        Parameters
        ----------
        start : :class:`int`
            The start index of the range to keep, inclusive.

        end : :class:`int`
            The end index, exclusive.

        Returns
        -------
        :class:`~vortex.encoding.Array`

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
            <pyarrow.lib.StringArray object at ...>
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

    def tree_display(self) -> str:
        """Internal technical details about the encoding of this Array.

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
            root: vortex.primitive(0x03)(i64?, len=4) nbytes=36 B (100.00%)
              metadata: PrimitiveMetadata { validity: Array }
              buffer (align=8): 32 B
              validity: vortex.bool(0x02)(bool, len=4) nbytes=3 B (8.33%)
                metadata: BoolMetadata { validity: NonNullable, first_byte_bit_offset: 0 }
                buffer (align=1): 1 B
            <BLANKLINE>

        Compressed arrays often have more complex, deeply nested encoding trees.
        """
