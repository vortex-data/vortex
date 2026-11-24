# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""`vx-analyze`: report, diff and oracle commands over materialized vx-lab datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from . import sections
from .cost import load_presets, picks, scorable, summarise
from .load import load
from .report import Section, Table, render


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--presets", help="presets TOML (default: the bundled presets.toml)")
    p.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        metavar="VARIANT",
        help="candidate variants to leave out, e.g. forced/pco",
    )


def cmd_report(a) -> int:
    ds = load(a.dataset)
    presets = load_presets(a.presets)
    if a.primary:
        presets = type(presets)(presets.presets, a.primary, presets.sweep)
    scored = scorable(ds.rows, tuple(a.exclude))
    if scored.height == 0:
        print(
            "no site has production and another candidate both compressed and timed; "
            "run `vx-lab run --stages time` before materializing",
            file=sys.stderr,
        )
        return 1
    secs = [
        sections.health(ds, scored),
        sections.headroom(scored, presets),
        sections.regret(scored, presets),
        sections.misses(ds, scored, presets),
    ]
    trials, best_sets = sections.trial_sets(scored, presets)
    secs.append(trials)
    if not a.no_features:
        secs.append(sections.features(ds, scored, presets))
    secs.append(sections.frontier(scored, presets, best_sets))
    meta = ds.label + " · presets " + ", ".join(f"{p.name} {p.bw_label}" for p in presets)
    if a.exclude:
        meta += " · excluded " + ", ".join(a.exclude)
    out = Path(a.out)
    out.write_text(render("vx-lab analysis", meta, secs))
    print(f"wrote {out}")
    return 0


def _aggregate(path: str, presets, exclude) -> pl.DataFrame:
    ds = load(path)
    scored = scorable(ds.rows, tuple(exclude))
    frames = []
    for p in presets:
        pk = picks(scored, p.bandwidth)
        for by in (["source", "dtype"], []):
            s = summarise(pk, by).with_columns(pl.lit(p.name).alias("preset"))
            s = s.with_columns(prod_cost=pl.lit(None, pl.Float64))
            totals = (
                pk.group_by(by).agg(pl.col("prod_cost").sum().alias("prod_cost_s"))
                if by
                else pk.select(pl.col("prod_cost").sum().alias("prod_cost_s"))
            )
            s = s.join(totals, on=by) if by else s.with_columns(totals)
            if not by:
                s = s.with_columns(pl.lit("all").alias("source"), pl.lit("all").alias("dtype"))
            frames.append(s.drop("prod_cost"))
    return pl.concat(frames, how="diagonal_relaxed")


def cmd_diff(a) -> int:
    presets = load_presets(a.presets)
    key = ["preset", "source", "dtype"]
    cols = ["saved", "prod_optimal", "prod_ratio", "prod_decode_gbps", "prod_cost_s"]
    A = _aggregate(a.base, presets, a.exclude).select(key + cols)
    B = _aggregate(a.head, presets, a.exclude).select(key + cols)
    j = A.join(B, on=key, how="full", suffix="_head", coalesce=True)
    j = j.select(
        key
        + [
            pl.col("saved").alias("headroom base"),
            pl.col("saved_head").alias("headroom head"),
            (pl.col("prod_cost_s_head") / pl.col("prod_cost_s") - 1).alias("production cost change"),
            pl.col("prod_ratio").alias("ratio base"),
            pl.col("prod_ratio_head").alias("ratio head"),
            pl.col("prod_decode_gbps").alias("decode base"),
            pl.col("prod_decode_gbps_head").alias("decode head"),
        ]
    ).sort(key)
    note = (
        "Production cost change compares totals, so it is only meaningful when both datasets cover the same "
        "chunks (same spec and data, different commit)."
    )
    if a.out:
        sec = Section(
            "diff",
            "Base vs head",
            note,
            [
                Table(
                    j,
                    f"{a.base} → {a.head}",
                    {
                        "headroom base": "pct",
                        "headroom head": "pct",
                        "production cost change": "spct",
                        "ratio base": "x",
                        "ratio head": "x",
                        "decode base": "gbps",
                        "decode head": "gbps",
                    },
                )
            ],
        )
        Path(a.out).write_text(render("vx-lab diff", f"{a.base} → {a.head}", [sec]))
        print(f"wrote {a.out}")
    else:
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=4):
            print(j)
        print(note)
    return 0


def cmd_oracle(a) -> int:
    ds = load(a.dataset)
    presets = load_presets(a.presets)
    scored = scorable(ds.rows, tuple(a.exclude))
    chosen = [presets.get(n) for n in a.preset] if a.preset else list(presets)
    frames = [
        picks(scored, p.bandwidth).with_columns(
            pl.lit(p.name).alias("preset"), pl.lit(p.bandwidth).alias("bandwidth")
        )
        for p in chosen
    ]
    out = pl.concat(frames)
    out.write_parquet(a.out, compression="zstd", compression_level=3)
    print(f"wrote {out.height:,} rows to {a.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vx-analyze", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report", help="write the HTML report for one dataset")
    r.add_argument("dataset", help="a directory written by `vx-lab materialize`")
    r.add_argument("--out", default="report.html")
    r.add_argument("--primary", help="preset used for regret, misses and features (default from presets)")
    r.add_argument("--no-features", action="store_true", help="skip the model-based features section")
    _common(r)
    r.set_defaults(fn=cmd_report)

    d = sub.add_parser("diff", help="compare headroom and production cost between two datasets")
    d.add_argument("base")
    d.add_argument("head")
    d.add_argument("--out", help="write HTML instead of printing")
    _common(d)
    d.set_defaults(fn=cmd_diff)

    o = sub.add_parser("oracle", help="write per-site oracle picks (training labels) to Parquet")
    o.add_argument("dataset")
    o.add_argument("--out", default="oracle.parquet")
    o.add_argument("--preset", nargs="*", help="presets to label (default: all)")
    _common(o)
    o.set_defaults(fn=cmd_oracle)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
