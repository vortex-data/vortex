# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""The cost model, presets and the per-site oracle.

cost = bytes / bandwidth + decode seconds. Compress time is carried along but never scored.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import polars as pl

from .load import KEY, PRODUCTION


@dataclass(frozen=True)
class Preset:
    name: str
    bandwidth: float
    meaning: str = ""

    @property
    def bw_label(self) -> str:
        return fmt_bandwidth(self.bandwidth)


@dataclass(frozen=True)
class Presets:
    presets: tuple[Preset, ...]
    primary: str
    sweep: tuple[float, ...]

    def __iter__(self):
        return iter(self.presets)

    def get(self, name: str) -> Preset:
        for p in self.presets:
            if p.name == name:
                return p
        raise KeyError(f"no preset {name!r}; have {[p.name for p in self.presets]}")

    @property
    def main(self) -> Preset:
        return self.get(self.primary)


def fmt_bandwidth(bw: float) -> str:
    return f"{bw / 1e9:g} GB/s" if bw >= 1e9 else f"{bw / 1e6:g} MB/s"


def load_presets(path: str | Path | None = None) -> Presets:
    text = (
        Path(path).read_text() if path else resources.files("vx_analyze").joinpath("presets.toml").read_text()
    )
    data = tomllib.loads(text)
    presets = tuple(Preset(p["name"], float(p["bandwidth"]), p.get("meaning", "")) for p in data["preset"])
    return Presets(
        presets, data.get("primary", presets[0].name), tuple(float(b) for b in data.get("sweep", []))
    )


def cost_expr(bandwidth: float) -> pl.Expr:
    """Seconds to move and fully decode one candidate at `bandwidth` bytes per second."""
    return pl.col("bytes") / bandwidth + pl.col("decode_ns") / 1e9


def scorable(rows: pl.DataFrame, exclude: tuple[str, ...] = ()) -> pl.DataFrame:
    """Candidates that compressed and were timed, at sites where production was timed too."""
    ok = rows.filter(pl.col("ok") & pl.col("bytes").is_not_null() & pl.col("decode_ns").is_not_null())
    if exclude:
        ok = ok.filter(~pl.col("variant").is_in(list(exclude)))
    with_prod = ok.filter(pl.col("variant") == PRODUCTION).select(KEY).unique()
    ok = ok.join(with_prod, on=KEY, how="semi")
    # Candidates that produce the same encoding (same size and tree) differ only by timing noise.
    # Give them one decode time, the median of their samples, so noise never picks a winner and
    # ties then go to production.
    same = KEY + ["bytes", pl.col("tree").fill_null("")]
    return ok.with_columns(
        pl.col("decode_ns").median().over(same).alias("decode_ns"),
        pl.len().over(same).alias("same_encoding"),
    )


def picks(rows: pl.DataFrame, bandwidth: float, allowed: list[str] | None = None) -> pl.DataFrame:
    """One row per site: production's candidate and the cheapest candidate (the oracle).

    Ties go to production, then to the variant name, so results are deterministic.
    `allowed` restricts the oracle to a candidate set (production is always in it).
    """
    c = rows.with_columns(cost=cost_expr(bandwidth))
    if allowed is not None:
        c = c.filter(pl.col("variant").is_in(list({PRODUCTION, *allowed})))
    meta = ["dtype", "ptype", "depth", "len", "canonical_bytes"]
    prod = c.filter(pl.col("variant") == PRODUCTION).select(
        KEY
        + meta
        + [
            pl.col("cost").alias("prod_cost"),
            pl.col("bytes").alias("prod_bytes"),
            pl.col("decode_ns").alias("prod_decode_ns"),
            pl.col("compress_ns").alias("prod_compress_ns"),
            pl.col("root").alias("prod_root"),
        ]
    )
    best = (
        c.with_columns(is_prod=(pl.col("variant") == PRODUCTION))
        .sort(KEY + ["cost", "is_prod", "variant"], descending=[False] * len(KEY) + [False, True, False])
        .group_by(KEY, maintain_order=True)
        .first()
        .select(
            KEY
            + [
                pl.col("variant").alias("best_variant"),
                pl.col("cost").alias("best_cost"),
                pl.col("bytes").alias("best_bytes"),
                pl.col("decode_ns").alias("best_decode_ns"),
                pl.col("compress_ns").alias("best_compress_ns"),
                pl.col("root").alias("best_root"),
            ]
        )
    )
    out = prod.join(best, on=KEY, how="inner")
    return out.with_columns(
        regret=pl.col("prod_cost") - pl.col("best_cost"),
        regret_frac=(pl.col("prod_cost") / pl.col("best_cost") - 1).fill_nan(0.0),
        optimal=pl.col("prod_cost") <= pl.col("best_cost") * (1 + 1e-9),
    )


def summarise(p: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Totals of a picks table: cost saved (byte-weighted and per-chunk), ratio, decode, compress."""
    aggs = [
        pl.len().alias("sites"),
        pl.col("canonical_bytes").sum().alias("canonical_bytes"),
        (1 - pl.col("best_cost").sum() / pl.col("prod_cost").sum()).alias("saved"),
        (1 - pl.col("best_cost") / pl.col("prod_cost")).mean().alias("saved_per_chunk"),
        pl.col("optimal").mean().alias("prod_optimal"),
        (pl.col("canonical_bytes").sum() / pl.col("prod_bytes").sum()).alias("prod_ratio"),
        (pl.col("canonical_bytes").sum() / pl.col("best_bytes").sum()).alias("best_ratio"),
        (pl.col("canonical_bytes").sum() / pl.col("prod_decode_ns").sum()).alias("prod_decode_gbps"),
        (pl.col("canonical_bytes").sum() / pl.col("best_decode_ns").sum()).alias("best_decode_gbps"),
        (pl.col("best_compress_ns").sum() / pl.col("prod_compress_ns").sum()).alias("compress_x"),
    ]
    if by:
        return p.group_by(by).agg(aggs).sort(by)
    return p.select(aggs)
