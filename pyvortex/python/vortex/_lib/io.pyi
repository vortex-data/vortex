import vortex as vx

def read_url(
    url: str, *, projection=None, row_filter: vx.expr.Expr | None = None, indices: vx.Array | None = None
) -> vx.Array:
    """Read a vortex struct array from a URL.

    .. seealso::
        :func:`.read_path`

    Parameters
    ----------
    url : :class:`str`
        The URL to read from.
    projection : :class:`list` [ :class:`str` ``|`` :class:`int` ]
        The columns to read identified either by their index or name.
    row_filter : :class:`.Expr`
        Keep only the rows for which this expression evaluates to true.

    Examples
    --------

    Read an array from an HTTPS URL:

    >>> a = vx.io.read_url("https://example.com/dataset.vortex")  # doctest: +SKIP

    Read an array from an S3 URL:

    >>> a = vx.io.read_url("s3://bucket/path/to/dataset.vortex")  # doctest: +SKIP

    Read an array from an Azure Blob File System URL:

    >>> a = vx.io.read_url("abfss://my_file_system@my_account.dfs.core.windows.net/path/to/dataset.vortex")  # doctest: +SKIP

    Read an array from an Azure Blob Stroage URL:

    >>> a = vx.io.read_url("https://my_account.blob.core.windows.net/my_container/path/to/dataset.vortex")  # doctest: +SKIP

    Read an array from a Google Stroage URL:

    >>> a = vx.io.read_url("gs://bucket/path/to/dataset.vortex")  # doctest: +SKIP

    Read an array from a local file URL:

    >>> a = vx.io.read_url("file:/path/to/dataset.vortex")  # doctest: +SKIP

    """

def read_path() -> vx.Array:
    """Read a vortex struct array from the local filesystem.

    Parameters
    ----------
    path : :class:`str`
        The file path to read from.
    projection : :class:`list` [ :class:`str` ``|`` :class:`int` ]
        The columns to read identified either by their index or name.
    row_filter : :class:`.Expr`
        Keep only the rows for which this expression evaluates to true.

    Examples
    --------

    Read an array with a structured column and nulls at multiple levels and in multiple columns.

    >>> import vortex as vx
    >>> a = vx.array([
    ...     {'name': 'Joseph', 'age': 25},
    ...     {'name': None, 'age': 31},
    ...     {'name': 'Angela', 'age': None},
    ...     {'name': 'Mikhail', 'age': 57},
    ...     {'name': None, 'age': None},
    ... ])
    >>> vx.io.write_path(a, "a.vortex")
    >>> b = vx.io.read_path("a.vortex")
    >>> b.to_arrow_array()
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

    >>> c = vx.io.read_path("a.vortex", projection = ["age"])
    >>> c.to_arrow_array()
    <pyarrow.lib.ChunkedArray object at ...>
    [
      -- is_valid: all not null
      -- child 0 type: int64
        [
          25,
          31,
          null,
          57,
          null
        ]
    ]


    Keep rows with an age above 35. This will read O(N_KEPT) rows, when the file format allows.

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

    TODO(DK): Repeating a column in a projection does not work

    Read the age column by name, twice, and the name column by index, once:

    >>> # e = vx.io.read_path("a.vortex", projection = ["age", 1, "age"])
    >>> # e.to_arrow_array()

    TODO(DK): Top-level nullness does not work.

    >>> a = vx.array([
    ...     {'name': 'Joseph', 'age': 25},
    ...     {'name': None, 'age': 31},
    ...     {'name': 'Angela', 'age': None},
    ...     None,
    ...     {'name': 'Mikhail', 'age': 57},
    ...     {'name': None, 'age': None},
    ... ])
    >>> vx.io.write_path(a, "a.vortex")
    >>> # b = vx.io.read_path("a.vortex")
    >>> # b.to_arrow_array()

    """

def write_path(array: vx.Array, path: str, *, compress: bool = True):
    """
    Write a vortex struct array to the local filesystem.

    Parameters
    ----------
    array : :class:`~vortex.Array`
        The array. Must be an array of structures.

    path : :class:`str`
        The file path.

    compress : :class:`bool`
        Compress the array before writing, defaults to ``True``.

    Examples
    --------

    Write the array `a` to the local file `a.vortex`.

    >>> import vortex as vx
    >>> a = vx.array([
    ...     {'x': 1},
    ...     {'x': 2},
    ...     {'x': 10},
    ...     {'x': 11},
    ...     {'x': None},
    ... ])
    >>> vx.io.write_path(a, "a.vortex")

    """
