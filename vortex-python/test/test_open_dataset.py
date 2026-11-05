# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Reading many Vortex files as one dataset with :func:`vortex.open_dataset`."""

from pathlib import Path

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from vortex.store import LocalStore

import vortex as vx
import vortex.expr as ve

# Rows of each file, keyed by its path below the dataset directory. Sorted path order is
# a.vortex, b/b.vortex, c.vortex, so the dataset reads rows 0..30 in order.
FILES = {"a.vortex": range(0, 10), "b/b.vortex": range(10, 25), "c.vortex": range(25, 30)}


def table(rows: range) -> pa.Table:
    return pa.table({"x": pa.array(rows, type=pa.int64()), "even": pa.array([row % 2 == 0 for row in rows])})


def expected(*names: str) -> pa.Table:
    return pa.concat_tables([table(FILES[name]) for name in names])


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for name, rows in FILES.items():
        path = tmp_path / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        vx.io.write(table(rows), str(path))
    (tmp_path / "data" / "notes.txt").write_text("not a Vortex file")
    return tmp_path


def test_directory_reads_every_file_below_it_in_order(root: Path):
    ds = vx.open_dataset(root / "data")

    assert ds.file_count == 3
    assert ds.count_rows() == 30
    assert ds.to_table() == expected("a.vortex", "b/b.vortex", "c.vortex")


@pytest.mark.parametrize(
    ("pattern", "names"),
    [
        ("data/*.vortex", ["a.vortex", "c.vortex"]),
        ("data/**/*.vortex", ["a.vortex", "b/b.vortex", "c.vortex"]),
        ("data/b/*.vortex", ["b/b.vortex"]),
    ],
)
def test_glob_reads_the_files_it_matches(root: Path, pattern: str, names: list[str]):
    assert vx.open_dataset(str(root / pattern)).to_table() == expected(*names)


def test_sources_are_read_in_the_order_given(root: Path):
    ds = vx.open_dataset([root / "data" / "c.vortex", str(root / "data" / "a.vortex")])

    assert ds.to_table() == expected("c.vortex", "a.vortex")


@pytest.mark.parametrize(
    ("source", "names"),
    [
        ("data/", ["a.vortex", "b/b.vortex", "c.vortex"]),
        ("data/*.vortex", ["a.vortex", "c.vortex"]),
        ("data/**/*.vortex", ["a.vortex", "b/b.vortex", "c.vortex"]),
        ("data/c.vortex", ["c.vortex"]),
    ],
)
def test_object_store_prefixes_and_globs(root: Path, source: str, names: list[str]):
    ds = vx.open_dataset(source, store=LocalStore(prefix=root))

    assert ds.to_table() == expected(*names)


def test_rows_are_counted_across_files(root: Path):
    ds = vx.open_dataset(root / "data")
    everything = expected("a.vortex", "b/b.vortex", "c.vortex")

    # Row 8 is in the first file and row 27 in the last.
    indices = pa.array([1, 8, 12, 24, 27], type=pa.uint64())
    assert ds.take(indices) == everything.take(indices)
    assert ds.head(12) == everything.slice(0, 12)
    assert ds.head(0) == everything.slice(0, 0)
    assert ds.count_rows(filter=pc.field("x") >= 5) == 25
    assert ds.to_table(filter=ve.column("even")) == everything.filter(pc.field("even"))


def test_fragments_cover_every_file(root: Path):
    ds = vx.open_dataset(root / "data")
    fragments = list(ds.get_fragments())

    assert len(fragments) >= 3
    assert pa.concat_tables([fragment.to_table() for fragment in fragments]) == expected(
        "a.vortex", "b/b.vortex", "c.vortex"
    )


@pytest.mark.parametrize("concurrency", [1, 2, 64])
def test_concurrency_does_not_change_the_rows(root: Path, concurrency: int):
    ds = vx.open_dataset(root / "data", concurrency=concurrency)
    everything = expected("a.vortex", "b/b.vortex", "c.vortex")

    assert ds.concurrency == concurrency
    assert ds.to_table() == everything
    assert ds.to_record_batch_reader(columns=["x"]).read_all() == everything.select(["x"])
    assert ds.to_table(concurrency=1) == everything
    assert ds.filter(pc.field("x") < 3).to_table() == everything.slice(0, 3)


def test_duckdb_reads_every_file(root: Path):
    failures = vx.open_dataset(root / "data", concurrency=4)
    _ = failures  # DuckDB resolves the dataset from the SQL query below.

    rows = duckdb.sql("SELECT even, count(*) FROM failures WHERE x >= 5 GROUP BY even ORDER BY even").fetchall()

    assert rows == [(False, 13), (True, 12)]


def test_polars_reads_every_file(root: Path):
    frame = vx.open_dataset(root / "data", concurrency=4).to_polars()

    result = frame.filter(pl.col("x") >= 5).group_by("even").agg(pl.len().alias("n")).sort("even").collect()

    assert result.to_dicts() == [{"even": False, "n": 13}, {"even": True, "n": 12}]
    assert frame.head(12).collect()["x"].to_list() == list(range(12))


def test_file_reads_take_a_concurrency(root: Path):
    file = vx.open(str(root / "data" / "b" / "b.vortex"))
    rows = table(FILES["b/b.vortex"])

    assert file.to_arrow(concurrency=1).read_all() == rows
    assert file.scan(concurrency=2).read_all().to_arrow_table() == rows
    assert file.to_repeated_scan(concurrency=3).execute().read_all().to_arrow_table() == rows
    assert file.to_dataset(concurrency=4).concurrency == 4
    assert file.to_polars(concurrency=5).collect()["x"].to_list() == list(FILES["b/b.vortex"])


@pytest.mark.parametrize(
    "read",
    [
        lambda root: vx.open_dataset(root / "data", concurrency=0),
        lambda root: vx.open_dataset(root / "data").to_table(concurrency=0),
        lambda root: vx.open(str(root / "data" / "a.vortex")).to_arrow(concurrency=0),
        lambda root: vx.open(str(root / "data" / "a.vortex")).scan(concurrency=0),
    ],
)
def test_concurrency_must_be_positive(root: Path, read):
    with pytest.raises(ValueError, match="concurrency must be at least 1"):
        read(root)


def test_files_must_share_a_dtype(root: Path):
    vx.io.write(pa.table({"y": pa.array(["a"])}), str(root / "data" / "z.vortex"))

    with pytest.raises(RuntimeError, match="same dtype"):
        vx.open_dataset(root / "data")


def test_a_source_must_name_files(root: Path):
    with pytest.raises(RuntimeError, match="no Vortex files found"):
        vx.open_dataset(str(root / "data" / "*.parquet"))
