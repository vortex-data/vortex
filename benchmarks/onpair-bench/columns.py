# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Registry of (dataset, column) sources for the OnPair compression benchmark.

Adding a new column is a one-line append to ``COLUMNS``.

Source kinds:

* ``tpch``    — generated locally via the Rust ``gen-tpch`` subcommand (all
                tables, one parquet file each).
* ``parquet`` — an external parquet. Give a download ``url`` (fetched into a
                repo-relative cache on first use) and/or ``local`` paths to
                reuse if already present. Nothing is hard-required: a column
                with no resolvable source is skipped by ``run.py``.

All generated/downloaded data lives **under the repo** (``vortex-bench/data``),
resolved relative to this file — so the benchmark works from any checkout
location with no absolute paths required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Repo-relative roots (resolved from this file's location — no absolute paths).
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "vortex-bench" / "data"
SRC_DIR = DATA_DIR / "onpair-bench-src"

# Download URLs (the same sources Vortex's own loaders use).
CLICKBENCH_URL = "https://datasets.clickhouse.com/hits_compatible/hits.parquet"
FINEWEB_URL = ("https://huggingface.co/datasets/HuggingFaceFW/fineweb/"
               "resolve/v1.4.0/sample/10BT/000_00000.parquet")

# Optional pre-existing local copies to reuse instead of downloading (machine
# specific; ignored if absent).
_LOCAL = {
    "clickbench": [Path("/home/joe/data/hits.parquet")],
    "fineweb": [Path("/home/joe/data/fineweb/sample_10BT_combined.parquet")],
    "book-reviews": [Path("/home/joe/data/book_reviews/book_reviews.parquet")],
}


@dataclass
class Column:
    """One column to benchmark."""

    dataset_id: str
    column: str
    kind: str  # "tpch" | "parquet"
    # tpch
    scale_factor: float = 10.0
    table: str = "lineitem"
    # parquet
    url: str | None = None
    cache: str | None = None  # filename under SRC_DIR/<dataset_id>/
    local: list[Path] = field(default_factory=list)

    def tpch_dir(self) -> Path:
        return SRC_DIR / f"tpch_sf{int(self.scale_factor)}"

    def cache_path(self) -> Path:
        return SRC_DIR / self.dataset_id / (self.cache or f"{self.column}.parquet")

    def parquet_path(self) -> Path:
        """Resolved source: TPC-H generated path, an existing local copy, or the
        repo-relative cache (download target)."""
        if self.kind == "tpch":
            return self.tpch_dir() / "parquet" / f"{self.table}_0.parquet"
        if self.kind == "parquet":
            for p in self.local:
                if Path(p).exists():
                    return Path(p)
            return self.cache_path()
        raise ValueError(f"unknown source kind {self.kind!r}")


def _parquet_cols(dataset_id, columns, *, url, cache):
    return [
        Column(dataset_id=dataset_id, column=c, kind="parquet", url=url, cache=cache,
               local=_LOCAL.get(dataset_id, []))
        for c in columns
    ]


# Every TPC-H string column, by table. Single-char / few-value columns
# (returnflag, linestatus, orderstatus, mktsegment, brand, ...) are intentionally
# included: they show OnPair *expanding* low-cardinality data, where a value
# dictionary wins.
_TPCH_STR_COLS: dict[str, list[str]] = {
    "region": ["r_name", "r_comment"],
    "nation": ["n_name", "n_comment"],
    "supplier": ["s_name", "s_address", "s_phone", "s_comment"],
    "customer": ["c_name", "c_address", "c_phone", "c_mktsegment", "c_comment"],
    "part": ["p_name", "p_mfgr", "p_brand", "p_type", "p_container", "p_comment"],
    "partsupp": ["ps_comment"],
    "orders": ["o_orderstatus", "o_orderpriority", "o_clerk", "o_comment"],
    "lineitem": ["l_returnflag", "l_linestatus", "l_shipinstruct", "l_shipmode", "l_comment"],
}

# The benchmark registry. Append a one-line `Column(...)` to grow the suite.
COLUMNS: list[Column] = [
    *(
        Column(dataset_id="tpch-sf10", column=c, kind="tpch", scale_factor=10.0, table=t)
        for t, cols in _TPCH_STR_COLS.items()
        for c in cols
    ),
    # ClickBench "hits" — long high-cardinality URLs/titles vs short categoricals.
    *_parquet_cols("clickbench", ["URL", "Title", "Referer", "SearchPhrase", "MobilePhoneModel"],
                   url=CLICKBENCH_URL, cache="hits.parquet"),
    # FineWeb 10BT sample — long free text + URLs + low-cardinality categoricals.
    *_parquet_cols("fineweb", ["text", "url", "file_path", "dump", "language"],
                   url=FINEWEB_URL, cache="fineweb_10BT_000.parquet"),
    # OnPair paper's book-reviews corpus (single `text` column). No public URL —
    # reused from a local copy if present, otherwise skipped.
    Column(dataset_id="book-reviews", column="text", kind="parquet",
           cache="book_reviews.parquet", local=_LOCAL["book-reviews"]),
]
