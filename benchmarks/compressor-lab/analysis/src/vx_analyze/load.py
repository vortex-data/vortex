# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Loading a materialized vx-lab dataset into one normalised candidates table.

Two layouts are read:

- today's `vx-lab materialize` output: `rows-<tag>.csv`, `features-<tag>.csv`, `dataset.json`;
- Parquet: `rows/**/*.parquet` and `features/**/*.parquet` (hive partitions allowed), or
  `rows*.parquet` / `features*.parquet` at the top of the directory.

Whatever the layout, `Dataset.rows` has one row per (chunk, site, candidate) with the columns in
`ROW_COLUMNS`. Columns that older datasets lack get defaults: one root site per chunk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

KEY = ["source", "column", "chunk", "site"]
PRODUCTION = "default"

ROW_COLUMNS = {
    "source": pl.Utf8,
    "column": pl.Utf8,
    "chunk": pl.Utf8,
    "site": pl.Utf8,
    "depth": pl.Int32,
    "parent_scheme": pl.Utf8,
    "ptype": pl.Utf8,
    "dtype": pl.Utf8,
    "len": pl.Int64,
    "variant": pl.Utf8,
    "ok": pl.Boolean,
    "canonical_bytes": pl.Int64,
    "bytes": pl.Int64,
    "root": pl.Utf8,
    "tree": pl.Utf8,
    "compress_ns": pl.Float64,
    "decode_ns": pl.Float64,
    "decode_ns_min": pl.Float64,
    "error": pl.Utf8,
}

#: Feature tiers from the plan: 0 context, 1 compressor stats, 2 distinct, 3 one extra pass,
#: 4 derived, 5 sample-based. Unknown features are put in tier 5, the most expensive.
FEATURE_TIERS = {
    "ptype_bits": 0,
    "signed": 0,
    "len": 0,
    "depth": 0,
    "null_frac": 1,
    "bits_bp": 1,
    "bits_for": 1,
    "bits_zz": 1,
    "for_gain": 1,
    "avg_run": 1,
    "run_count": 1,
    "distinct": 2,
    "distinct_frac": 2,
    "top1_frac": 2,
    "top10_cover": 2,
    "entropy": 2,
    "sorted_frac": 3,
    "bits_delta": 3,
    "step_mode_frac": 3,
    "bits_p50": 3,
    "bits_p90": 3,
    "bits_p99": 3,
    "trailing_zeros": 3,
}


def dtype_family(ptype: pl.Expr) -> pl.Expr:
    """Map a physical type name such as `i64` or `f32` to int, float, bool, string or other."""
    p = ptype.str.to_lowercase()
    return (
        pl.when(p.str.contains(r"^[iu]\d+$"))
        .then(pl.lit("int"))
        .when(p.str.contains(r"^f\d+$"))
        .then(pl.lit("float"))
        .when(p.str.contains("bool"))
        .then(pl.lit("bool"))
        .when(p.str.contains("utf8|string|binary"))
        .then(pl.lit("string"))
        .when(p.str.contains("decimal"))
        .then(pl.lit("decimal"))
        .otherwise(pl.lit("other"))
    )


@dataclass
class Dataset:
    """A materialized dataset: candidate rows, per-site features and the manifest."""

    path: Path
    rows: pl.DataFrame
    features: pl.DataFrame
    manifest: dict = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return [c for c in self.features.columns if c not in KEY]

    @property
    def label(self) -> str:
        m = self.manifest
        commit = (m.get("git_commit") or "")[:10]
        machine = m.get("machine_id") or "no timings"
        return f"{self.path.name} · commit {commit or '?'} · machine {machine}"


def _scan(path: Path, stem: str) -> pl.LazyFrame | None:
    part_dir = path / stem
    if part_dir.is_dir() and any(part_dir.rglob("*.parquet")):
        return pl.scan_parquet(part_dir / "**" / "*.parquet", hive_partitioning=True)
    pq = sorted(path.glob(f"{stem}*.parquet"))
    if pq:
        return pl.concat([pl.scan_parquet(p) for p in pq], how="diagonal_relaxed")
    csvs = sorted(path.glob(f"{stem}-*.csv"))
    if csvs:
        return pl.concat([pl.scan_csv(p, infer_schema_length=0) for p in csvs], how="diagonal_relaxed")
    return None


def _normalise_rows(lf: pl.LazyFrame) -> pl.DataFrame:
    cols = lf.collect_schema().names()
    rename = {"decode_ns_median": "decode_ns"}
    lf = lf.rename({k: v for k, v in rename.items() if k in cols})
    cols = [rename.get(c, c) for c in cols]
    defaults = {
        "site": pl.lit("root"),
        "depth": pl.lit(0),
        "parent_scheme": pl.lit(None, pl.Utf8),
        "root": pl.lit(None, pl.Utf8),
        "tree": pl.lit(None, pl.Utf8),
        "compress_ns": pl.lit(None, pl.Float64),
        "decode_ns": pl.lit(None, pl.Float64),
        "decode_ns_min": pl.lit(None, pl.Float64),
        "error": pl.lit(None, pl.Utf8),
        "dtype": pl.lit(None, pl.Utf8),
    }
    lf = lf.with_columns([v.alias(k) for k, v in defaults.items() if k not in cols])
    casts = []
    for name, dt in ROW_COLUMNS.items():
        col = pl.col(name)
        if dt == pl.Boolean:
            casts.append(col.cast(pl.Utf8).is_in(["1", "true", "True"]).alias(name))
        elif dt in (pl.Int64, pl.Int32, pl.Float64):
            casts.append(col.cast(pl.Utf8).replace("", None).cast(pl.Float64).cast(dt).alias(name))
        else:
            casts.append(
                col.cast(pl.Utf8).replace("", None).alias(name) if name != "site" else col.cast(pl.Utf8)
            )
    df = lf.select(casts).collect()
    return df.with_columns(pl.coalesce(pl.col("dtype"), dtype_family(pl.col("ptype"))).alias("dtype"))


def _normalise_features(lf: pl.LazyFrame | None) -> pl.DataFrame:
    if lf is None:
        return pl.DataFrame(schema={k: pl.Utf8 for k in KEY})
    cols = lf.collect_schema().names()
    if "site" not in cols:
        lf = lf.with_columns(pl.lit("root").alias("site"))
    feats = [c for c in cols if c not in KEY]
    return lf.select(
        [pl.col(k).cast(pl.Utf8) for k in KEY]
        + [pl.col(c).cast(pl.Utf8).replace("", None).cast(pl.Float64).alias(c) for c in feats]
    ).collect()


def load(path: str | Path) -> Dataset:
    """Load a dataset directory written by `vx-lab materialize`."""
    path = Path(path)
    rows = _scan(path, "rows")
    if rows is None:
        raise FileNotFoundError(f"no rows-*.csv or rows*.parquet under {path}")
    manifest_path = path / "dataset.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return Dataset(
        path=path,
        rows=_normalise_rows(rows),
        features=_normalise_features(_scan(path, "features")),
        manifest=manifest,
    )
