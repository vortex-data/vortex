import vortex as vx
import vortex.expr

def read_url(
    url: str,
    *,
    projection=None,
    row_filter: vortex.expr.Expr | None = None,
    indices: vx.Array | None = None,
) -> vx.Array: ...
def write(iter: vx.file.IntoArrayIterator, path: str): ...
