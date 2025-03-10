import json

import polars as pl

import vortex.expr as ve


def polars_to_vortex(expr: pl.Expr) -> ve.Expr:
    """Convert a Polars expression to a Vortex expression."""
    return _polars_to_vortex(json.loads(expr.meta.write_json()))


def _polars_to_vortex(expr: dict) -> ve.Expr:
    """Convert a Polars expression to a Vortex expression."""
    raise NotImplementedError
