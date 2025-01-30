from typing import final

import pyarrow as pa

import vortex as vx

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
