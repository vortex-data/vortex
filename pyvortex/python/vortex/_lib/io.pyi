import vortex as vx

def read_url(
    url: str,
    *,
    projection=None,
    row_filter: vx.expr.Expr | None = None,
    indices: vx.Array | None = None,
) -> vx.Array: ...
def read_path(
    path: str,
    *,
    projection: list[str | int] | None = None,
    row_filter: vx.expr.Expr | None = None,
    indices: vx.Array | None = None,
) -> vx.Array: ...
def write_path(array: vx.Array, path: str, *, compress: bool = True): ...
