class DType:
    """A Vortex data type."""

def null() -> DType:
    """A null DType."""

def bool_(*, nullable: bool = False) -> DType:
    """Construct a Boolean data type.

    Parameters
    ----------
    nullable : :class:`bool`
        When :obj:`True`, :obj:`None` is a permissible value.

    Returns
    -------
    :class:`vortex.dtype.DType`

    Examples
    --------

    A data type permitting :obj:`None`, :obj:`True`, and :obj:`False`.

        >>> import vortex as vx
        >>> vx.bool_(nullable=True)
        bool(True)

    A data type permitting just :obj:`True` and :obj:`False`.

        >>> import vortex as vx
        >>> vx.bool_()
        bool(False)
    """
