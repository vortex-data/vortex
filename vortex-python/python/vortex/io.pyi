from __future__ import annotations
__all__: list = ['read_url', 'write']
def read_url(url, *, projection = None, row_filter = None, indices = None):
    """
    Read a vortex struct array from a URL.
    
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
    
        >>> import vortex as vx
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
def write(iter, path):
    """
    Write a vortex struct array to the local filesystem.
    
    Parameters
    ----------
    array : :class:`~vortex.Array`
        The array. Must be an array of structures.
    
    f : :class:`str`
        The file path.
    
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
        >>> vx.io.write(a, "a.vortex")
    """
