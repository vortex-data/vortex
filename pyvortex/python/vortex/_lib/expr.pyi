from typing import Any, final

import vortex as vx

@final
class Expr:
    """An expression describes how to filter rows when reading an array from a file.

    .. seealso::
       :func:`.column`

    Examples
    ========

    All the examples read the following file.

    >>> import vortex as vx
    >>> a = vx.array([
    ...     {'name': 'Joseph', 'age': 25},
    ...     {'name': None, 'age': 31},
    ...     {'name': 'Angela', 'age': None},
    ...     {'name': 'Mikhail', 'age': 57},
    ...     {'name': None, 'age': None},
    ... ])
    >>> vx.io.write_path(a, "a.vortex")

    Read only those rows whose age column is greater than 35:

    >>> e = vx.io.read_path("a.vortex", row_filter = vx.expr.column("age") > 35)
    >>> e.to_arrow_array()
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

    Read only those rows whose age column lies in (21, 33]. Notice that we must use parentheses
    because of the Python precedence rules for ``&``:

    >>> age = vx.expr.column("age")
    >>> e = vx.io.read_path("a.vortex", row_filter = (age > 21) & (age <= 33))
    >>> e.to_arrow_array()
    <pyarrow.lib.StructArray object at ...>
    -- is_valid: all not null
    -- child 0 type: int64
      [
        25,
        31
      ]
    -- child 1 type: string_view
      [
        "Joseph",
        null
      ]

    Read only those rows whose name is `Joseph`:

    >>> name = vx.expr.column("name")
    >>> e = vx.io.read_path("a.vortex", row_filter = name == "Joseph")
    >>> e.to_arrow_array()
    <pyarrow.lib.StructArray object at ...>
    -- is_valid: all not null
    -- child 0 type: int64
      [
        25
      ]
    -- child 1 type: string_view
      [
        "Joseph"
      ]

    Read all the rows whose name is _not_ `Joseph`

    >>> name = vx.expr.column("name")
    >>> e = vx.io.read_path("a.vortex", row_filter = name != "Joseph")
    >>> e.to_arrow_array()
    <pyarrow.lib.StructArray object at ...>
    -- is_valid: all not null
    -- child 0 type: int64
      [
        null,
        57
      ]
    -- child 1 type: string_view
      [
        "Angela",
        "Mikhail"
      ]

    Read rows whose name is `Angela` or whose age is between 20 and 30, inclusive. Notice that the
    Angela row is included even though its age is null. Under SQL / Kleene semantics, `true or
    null` is `true`.

    >>> name = vx.expr.column("name")
    >>> e = vx.io.read_path("a.vortex", row_filter = (name == "Angela") | ((age >= 20) & (age <= 30)))
    >>> e.to_arrow_array()
    <pyarrow.lib.StructArray object at ...>
    -- is_valid: all not null
    -- child 0 type: int64
      [
        25,
        null
      ]
    -- child 1 type: string_view
      [
        "Joseph",
        "Angela"
      ]
    """

    def __eq__(self, other: Expr) -> bool: ...

def column(name: str) -> Expr:
    """Create an expression that refers to a column by its name.

    Parameters
    ----------
    name : :class:`str`
        The name of the column.

    Returns
    -------
    :class:`vortex.Expr`

    Examples
    --------

    >>> import vortex.expr as ve
    >>> ve.column("age")
    column("age")  # noqa: F821

    """

def literal(dtype: vx.DType, value: Any) -> Expr:
    """Create an expression that represents a literal value.

    Parameters
    ----------
    dtype : :class:`vortex.DType`
        The data type of the literal value.
    value : :class:`Any`
        The literal value.

    Returns
    -------
    :class:`vortex.Expr`

    Examples
    --------

    >>> import vortex.expr as ve
    >>> ve.literal(ve.int_(), 42)
    literal(int(), 42)

    """
