from typing import Literal, final

import pyarrow as pa

@final
class DType:
    """A Vortex data type."""

    def maybe_columns(self) -> list[str] | None:
        """Return the names of the columns in a struct data type."""

    @classmethod
    def from_arrow(cls, arrow_dtype: pa.DataType, *, non_nullable: bool = False) -> DType:
        """Construct a Vortex data type from an Arrow data type."""

def null() -> DType:
    """Construct the data type for a column containing only the null value.

    Returns
    -------
    :class:`vortex.DType`

    Examples
    --------

    A data type permitting only :obj:`None`.

        >>> import vortex as vx
        >>> vx.null()
        null()
    """

def bool_(*, nullable: bool = False) -> DType:
    """Construct a Boolean data type.

    Parameters
    ----------
    nullable : :class:`bool`
        When :obj:`True`, :obj:`None` is a permissible value.

    Returns
    -------
    :class:`vortex.DType`

    Examples
    --------

    A data type permitting :obj:`None`, :obj:`True`, and :obj:`False`.

        >>> import vortex as vx
        >>> vx.bool_(nullable=True)
        bool(nullable=True)

    A data type permitting just :obj:`True` and :obj:`False`.

        >>> vx.bool_()
        bool(nullable=False)
    """

def int_(width: Literal[8, 16, 32, 64] = 64, *, nullable: bool = False) -> DType:
    """Construct a signed integral data type.

    Parameters
    ----------
    width : Literal[8, 16, 32, 64].
        The bit width determines the span of valid values. If :obj:`None`, 64 is used.

    nullable : :class:`bool`
        When :obj:`True`, :obj:`None` is a permissible value.

    Returns
    -------
    :class:`vortex.DType`

    Examples
    --------

    A data type permitting :obj:`None` and the integers from -128 to 127, inclusive:

        >>> import vortex as vx
        >>> vx.int_(8, nullable=True)
        int(8, nullable=True)

    A data type permitting just the integers from -2,147,483,648 to 2,147,483,647, inclusive:

        >>> vx.int_(32)
        int(32, nullable=False)
    """

def uint(width: Literal[8, 16, 32, 64] = 64, *, nullable: bool = False) -> DType:
    """Construct an unsigned integral data type.

    Parameters
    ----------
    width : Literal[8, 16, 32, 64].
        The bit width determines the span of valid values. If :obj:`None`, 64 is used.

    nullable : :class:`bool`
        When :obj:`True`, :obj:`None` is a permissible value.

    Returns
    -------
    :class:`vortex.DType`

    Examples
    --------

    A data type permitting :obj:`None` and the integers from 0 to 255, inclusive:

        >>> import vortex as vx
        >>> vx.uint(8, nullable=True)
        uint(8, True)

    A data type permitting just the integers from 0 to 4,294,967,296 inclusive:

        >>> vx.uint(32, nullable=False)
        uint(32, False)
    """

def float_(width: Literal[16, 32, 64] = 64, *, nullable: bool = False) -> DType:
    """Construct an IEEE 754 binary floating-point data type.

    Parameters
    ----------
    width : Literal[16, 32, 64].
        The bit width determines the range and precision of the floating-point values. If
        :obj:`None`, 64 is used.

    nullable : :class:`bool`
        When :obj:`True`, :obj:`None` is a permissible value.

    Returns
    -------
    :class:`vortex.DType`

    Examples
    --------

    A data type permitting :obj:`None` as well as IEEE 754 binary16 floating-point values. Values
    larger than 65,520 or less than -65,520 will respectively round to positive and negative
    infinity.

        >>> import vortex as vx
        >>> vx.float_(16, nullable=False)
        float(16, nullable=False)
    """

def utf8(*, nullable: bool = False) -> DType:
    """Construct a UTF-8-encoded string data type.

    Parameters
    ----------
    nullable : :class:`bool`
        When :obj:`True`, :obj:`None` is a permissible value.

    Returns
    -------
    :class:`vortex.DType`

    Examples
    --------

    A data type permitting any UTF-8-encoded string, such as :code:`"Hello World"`, but not
    permitting :obj:`None`.

        >>> import vortex as vx
        >>> vx.utf8(nullable=False)
        utf8(nullable=False)
    """

def binary(*, nullable: bool = False) -> DType:
    """Construct a binary data type.

    Parameters
    ----------
    nullable : :class:`bool`
        When :obj:`True`, :obj:`None` is a permissible value.

    Returns
    -------
    :class:`vortex.DType`

    Examples
    --------

    A data type permitting any string of bytes but not permitting :obj:`None`.

        >>> import vortex as vx
        >>> vx.binary(nullable=False)
        binary(nullable=False)
    """

def struct(fields: dict[str, DType] | None = None, *, nullable: bool = False) -> DType:
    """Construct a struct data type.

    Parameters
    ----------
    fields : :class:`dict`
        A mapping from field names to data types.
    nullable : :class:`bool`
        When :obj:`True`, :obj:`None` is a permissible value.

    Returns
    -------
    :class:`vortex.DType`

    Examples
    --------

    A data type permitting a struct with two fields, :code:`"name"` and :code:`"age"`, where :code:`"name"` is a UTF-8-encoded string and :code:`"age"` is a 32-bit signed integer:

    >>> import vortex as vx
    >>> vx.struct({"name": vx.utf8(), "age": vx.int_(32)})
    struct({"name": utf8(nullable=False), "age": int(32, nullable=False)}, nullable=False)
    """
