from datetime import date, datetime
from typing import TypeAlias

from vortex.expr import Expr, column, ident, literal

IntoExpr: TypeAlias = Expr | int | str | date | datetime | None

__all__ = ["Expr", "column", "ident", "literal"]
