# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Uncertainty: bootstrap over columns, since chunks within one column are correlated."""

from __future__ import annotations

import numpy as np
import polars as pl


def saved_ci(p: pl.DataFrame, reps: int = 400, seed: int = 0, level: float = 0.9) -> tuple[float, float]:
    """A confidence interval for byte-weighted saving, resampling whole (source, column) groups."""
    g = p.group_by(["source", "column"]).agg(pl.col("prod_cost").sum(), pl.col("best_cost").sum())
    if g.height < 2:
        return (float("nan"), float("nan"))
    prod, best = g["prod_cost"].to_numpy(), g["best_cost"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(prod), size=(reps, len(prod)))
    s = 1 - best[idx].sum(axis=1) / prod[idx].sum(axis=1)
    lo, hi = np.quantile(s, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
