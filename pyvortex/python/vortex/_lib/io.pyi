import vortex as vx
import vortex.expr

IntoArrayIterator = vx.Array | vx.ArrayIterator

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
def write(iter: IntoArrayIterator, path: str): ...
