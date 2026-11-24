# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import numpy as np
import polars as pl
from conftest import write_dataset

from vx_analyze import cli, sections
from vx_analyze.cost import load_presets, picks, scorable, summarise
from vx_analyze.load import KEY, load


def test_load_csv_normalises(dataset):
    ds = load(dataset)
    assert set(ds.rows["site"].unique()) == {"root"}
    assert ds.rows["dtype"].unique().to_list() == ["int"]
    assert ds.rows.filter(~pl.col("ok"))["variant"].unique().to_list() == ["forced/sparse"]
    assert "avg_run" in ds.feature_names


def test_load_parquet_matches_csv(dataset, tmp_path_factory):
    ds = load(dataset)
    pq = tmp_path_factory.mktemp("pq")
    (pq / "rows").mkdir()
    ds.rows.drop("site").rename({"decode_ns": "decode_ns_median"}).write_parquet(
        pq / "rows" / "part-0.parquet"
    )
    ds.features.write_parquet(pq / "features.parquet")
    again = load(pq)
    assert again.rows.sort(KEY + ["variant"]).equals(ds.rows.sort(KEY + ["variant"]))


def test_oracle_is_cheapest_and_ties_go_to_production():
    rows = pl.DataFrame(
        {
            "source": ["s"] * 3,
            "column": ["c"] * 3,
            "chunk": ["1"] * 3,
            "site": ["root"] * 3,
            "dtype": ["int"] * 3,
            "ptype": ["i64"] * 3,
            "depth": [0] * 3,
            "len": [10] * 3,
            "canonical_bytes": [80] * 3,
            "variant": ["default", "forced/a", "forced/b"],
            "ok": [True] * 3,
            "bytes": [40, 40, 20],
            "decode_ns": [1e3, 1e3, 1e9],
            "compress_ns": [1.0, 2.0, 3.0],
            "root": ["x", "a", "b"],
        }
    )
    # At 1 GB/s, b's slow decode loses; a ties production, and production wins the tie.
    p = picks(rows, 1e9)
    assert p["best_variant"][0] == "default" and p["optimal"][0]
    # At 1 B/s bytes dominate and b wins.
    p = picks(rows, 1.0)
    assert p["best_variant"][0] == "forced/b"
    assert np.isclose(p["regret"][0], (40 + 1e-6) - (20 + 1.0))


def test_headroom_grows_with_more_candidates(dataset):
    scored = scorable(load(dataset).rows)
    all_ = summarise(picks(scored, 1e9), [])["saved"][0]
    one = summarise(picks(scored, 1e9, allowed=["forced/for"]), [])["saved"][0]
    assert 0 <= one <= all_ + 1e-12


def test_greedy_trials_never_worse_than_production(dataset):
    scored = scorable(load(dataset).rows)
    _keys, variants, m = sections.cost_matrix(scored, 1e9)
    chosen = sections.greedy_set(m, variants, 3)
    assert chosen[0] == 0
    assert np.min(m[:, chosen], axis=1).sum() <= m[:, 0].sum()


def test_report_end_to_end(dataset, tmp_path):
    out = tmp_path / "report.html"
    assert cli.main(["report", str(dataset), "--out", str(out)]) == 0
    html = out.read_text()
    for key in ["health", "headroom", "regret", "misses", "trials", "features", "frontier"]:
        assert f'id="{key}"' in html
    assert html.count("plotly.js") >= 1


def test_diff_and_oracle(dataset, tmp_path):
    head = tmp_path / "head"
    head.mkdir()
    write_dataset(head, seed=1)
    assert cli.main(["diff", str(dataset), str(head), "--out", str(tmp_path / "diff.html")]) == 0
    out = tmp_path / "oracle.parquet"
    assert cli.main(["oracle", str(dataset), "--out", str(out), "--preset", "size", "memory"]) == 0
    o = pl.read_parquet(out)
    assert set(o["preset"].unique()) == {"size", "memory"}
    assert o.filter(pl.col("best_cost") > pl.col("prod_cost") + 1e-12).height == 0


def test_identical_encodings_share_timings():
    base = {
        "source": ["s"] * 2,
        "column": ["c"] * 2,
        "chunk": ["1"] * 2,
        "site": ["root"] * 2,
        "dtype": ["int"] * 2,
        "ptype": ["i64"] * 2,
        "depth": [0] * 2,
        "len": [10] * 2,
        "canonical_bytes": [80] * 2,
        "ok": [True] * 2,
        "compress_ns": [1.0, 1.0],
        "root": ["x", "x"],
    }
    # Same size and tree: only timing noise differs, so production must stay optimal.
    rows = pl.DataFrame(
        {
            **base,
            "variant": ["default", "sizemodel"],
            "bytes": [40, 40],
            "tree": ["x(a)", "x(a)"],
            "decode_ns": [1100.0, 900.0],
        }
    )
    p = picks(scorable(rows), 1e9)
    assert p["optimal"][0] and p["best_variant"][0] == "default"
    # A different tree with the same size is a different encoding and keeps its own timing.
    rows = rows.with_columns(tree=pl.Series(["x(a)", "x(b)"]))
    assert picks(scorable(rows), 1e9)["best_variant"][0] == "sizemodel"


def test_presets_bundled():
    p = load_presets()
    assert p.main.name == "s3-scan" and p.main.bandwidth == 1e9
