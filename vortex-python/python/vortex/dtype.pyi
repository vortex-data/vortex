from __future__ import annotations
import typing

__all__: list = [
    "DType",
    "PType",
    "NullDType",
    "BoolDType",
    "PrimitiveDType",
    "DecimalDType",
    "Utf8DType",
    "BinaryDType",
    "StructDType",
    "ListDType",
    "ExtensionDType",
    "null",
    "bool_",
    "int_",
    "decimal",
    "uint",
    "float_",
    "utf8",
    "binary",
    "struct",
    "list_",
    "ext",
]

class BinaryDType(DType):
    """
    Concrete class for utf8 dtypes.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """

class BoolDType(DType):
    """
    Concrete class for boolean dtypes.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """

class DType:
    """
    Base class for all Vortex data types.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @classmethod
    def from_arrow(cls, arrow_dtype, *, non_nullable=False):
        """
        Construct a Vortex data type from an Arrow data type.
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
    def __hash__(self):
        """
        Return hash(self).
        """
    def __le__(self, value):
        """
        Return self<=value.
        """
    def __lt__(self, value):
        """
        Return self<value.
        """
    def __ne__(self, value):
        """
        Return self!=value.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
    def __str__(self):
        """
        Return str(self).
        """
    def to_arrow_schema(self): ...
    def to_arrow_type(self): ...

class DecimalDType(DType):
    """
    Concrete class for primitive dtypes.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """

class ExtensionDType(DType):
    """
    Concrete class for extension dtypes.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """

class ListDType(DType):
    """
    Concrete class for list dtypes.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """

class NullDType(DType):
    """
    Concrete class for null dtypes.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """

class PType:
    """
    Enum for primitive types.
    """

    F16: typing.ClassVar[PType]  # value = PType.F16
    F32: typing.ClassVar[PType]  # value = PType.F32
    F64: typing.ClassVar[PType]  # value = PType.F64
    I16: typing.ClassVar[PType]  # value = PType.I16
    I32: typing.ClassVar[PType]  # value = PType.I32
    I64: typing.ClassVar[PType]  # value = PType.I64
    I8: typing.ClassVar[PType]  # value = PType.I8
    U16: typing.ClassVar[PType]  # value = PType.U16
    U32: typing.ClassVar[PType]  # value = PType.U32
    U64: typing.ClassVar[PType]  # value = PType.U64
    U8: typing.ClassVar[PType]  # value = PType.U8
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
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
    def __int__(self):
        """
        int(self)
        """
    def __le__(self, value):
        """
        Return self<=value.
        """
    def __lt__(self, value):
        """
        Return self<value.
        """
    def __ne__(self, value):
        """
        Return self!=value.
        """
    def __repr__(self):
        """
        Return repr(self).
        """
    def __str__(self):
        """
        Return str(self).
        """

class PrimitiveDType(DType):
    """
    Concrete class for primitive dtypes.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """

class StructDType(DType):
    """
    Concrete class for struct dtypes.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def fields(self):
        """
        Returns the field DTypes of the struct.
        """
    def names(self):
        """
        Returns the names of the struct fields.
        """

class Utf8DType(DType):
    """
    Concrete class for utf8 dtypes.
    """
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """

def binary(*, nullable=False):
    """
    Construct a binary data type.

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

def bool_(*, nullable=False):
    """
    Construct a Boolean data type.

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

def decimal(*, precision=10, scale=0, nullable=False):
    """
    Construct a decimal data type.

    Parameters
    ----------
    precision : :class:`int`
        The number of significant digits representable by an instance of this type.

    scale : :class:`int`
        The number of digits after the decimal point that are represented. If negative, the
        number of trailing zeros in the whole number portion.

    nullable : :class:`bool`
        When :obj:`True`, :obj:`None` is a permissible value.

    Returns
    -------
    :class:`vortex.DType`

    Examples
    --------

    A data type permitting :obj:`None` and the integers from -128 to 127, inclusive:

        >>> import vortex as vx
        >>> vx.decimal(precision=13, scale=2, nullable=True)
        decimal(precision=13, scale=2, nullable=True)

    A data type representing fixed-width decimal numbers with `precision` significant figures and
    `scale` digits after the decimal point. If `scale` is a negative value, then it is taken
    to be the number of trailing zeros before the decimal point.

        >>> vx.decimal(precision = 10, scale = 3)
        decimal(precision=10, scale=3, nullable=False)
    """

def ext(id, storage, *, metadata=None):
    """
    Construct an extension data type.

    Parameters
    ----------
    id : :class:`str`
        The extension identifier.
    storage : :class:`DType`
        The underlying storage type.
    metadata : :class:`bytes`
       The extension type metadata.

    Returns
    -------
    :class:`vortex.DType`
    """

def float_(width=64, *, nullable=False):
    """
    Construct an IEEE 754 binary floating-point data type.

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

def int_(width=64, *, nullable=False):
    """
    Construct a signed integral data type.

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

def list_(element, *, nullable=False):
    """
    Construct a list data type.

    Parameters
    ----------
    element : :class:`DType`
        The type of the list element.
    nullable : :class:`bool`
        When :obj:`True`, :obj:`None` is a permissible value (this is not element nullability).

    Returns
    -------
    :class:`vortex.DType`

    Examples
    --------

    A data type permitting a list of 32-bit signed integers, but not permitting :obj:`None`.

        >>> import vortex as vx
        >>> vx.list_(vx.int_(32), nullable=False)
        list(int(32, nullable=False), nullable=False)
    """

def null():
    """
    Construct the data type for a column containing only the null value.

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

def struct(fields=None, *, nullable=False):
    """
    Construct a struct data type.

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

def uint(width=64, *, nullable=False):
    """
    Construct an unsigned integral data type.

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
        uint(8, nullable=True)

    A data type permitting just the integers from 0 to 4,294,967,296 inclusive:

        >>> vx.uint(32, nullable=False)
        uint(32, nullable=False)
    """

def utf8(*, nullable=False):
    """
    Construct a UTF-8-encoded string data type.

    Parameters
    ----------
    nullable : :class:`bool`
        When :obj:`True`, :obj:`None` is a permissible value.

    Returns
    -------
    :class:`vortex.DType`

    Examples
    ---------

    A data type permitting any UTF-8-encoded string, such as :code:`"Hello World"`, but not
    permitting :obj:`None`.

        >>> import vortex as vx
        >>> vx.utf8(nullable=False)
        utf8(nullable=False)
    """
