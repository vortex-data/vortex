import vortex as vx
import vortex.expr

def read_url(
    url: str,
    *,
    projection=None,
    row_filter: vortex.expr.Expr | None = None,
    indices: vx.Array | None = None,
) -> vx.Array: ...
def read_path(
    path: str,
    *,
    projection: list[str | int] | None = None,
    row_filter: vortex.expr.Expr | None = None,
    indices: vx.Array | None = None,
) -> vx.Array: ...
def write_path(array: vx.Array, path: str): ...
