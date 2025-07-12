# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from datetime import date, datetime
from typing import TypeAlias

from vortex._lib.expr import Expr, column, literal, root

IntoExpr: TypeAlias = Expr | int | str | date | datetime | None

__all__ = ["Expr", "column", "literal", "root"]
