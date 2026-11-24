# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Reading many Vortex files as one dataset with :func:`vortex.open_dataset`."""

from pathlib import Path

import duckdb
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from vortex.dataset import _open_exact_files
from vortex.store import LocalStore

import vortex as vx
import vortex.expr as ve

# Rows of each file, keyed by its path below the dataset directory. Sorted path order is
# a.vortex, b/b.vortex, c.vortex, so the dataset reads rows 0..30 in order.
FILES = {"a.vortex": range(0, 10), "b/b.vortex": range(10, 25), "c.vortex": range(25, 30)}


def table(rows: range) -> pa.Table:
    return pa.table(
        {
            "x": pa.array(rows, type=pa.int64()),
            "even": pa.array([row % 2 == 0 for row in rows], type=pa.bool_()),
        }
    )


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
    for invalid in ([8, 30], [30], [30, 8]):
        with pytest.raises(IndexError):
            ds.take(pa.array(invalid, type=pa.uint64()))
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


def test_fragment_take_uses_file_relative_indices(root: Path):
    ds = vx.open_dataset(root / "data")
    fragments = list(ds.get_fragments())

    # Second fragment belongs to b/b.vortex (rows 10..25 in the dataset, 0..15 in the file).
    second_file_fragment = fragments[1]
    taken = second_file_fragment.take(pa.array([0, 3, 14], type=pa.uint64()))
    assert taken == table(FILES["b/b.vortex"]).take([0, 3, 14])

    # Third fragment belongs to c.vortex (rows 25..30 in the dataset, 0..5 in the file).
    third_file_fragment = fragments[2]
    taken = third_file_fragment.take(pa.array([1, 4], type=pa.uint64()))
    assert taken == table(FILES["c.vortex"]).take([1, 4])


def test_literal_path_with_glob_characters(tmp_path: Path):
    bracket_file = tmp_path / "part[1].vortex"
    other_file = tmp_path / "part1.vortex"
    vx.io.write(table(range(0, 5)), str(bracket_file))
    vx.io.write(table(range(10, 15)), str(other_file))

    assert vx.open_dataset(bracket_file).to_table() == table(range(0, 5))
    assert vx.open_dataset("part[1].vortex", store=LocalStore(prefix=tmp_path)).to_table() == table(range(0, 5))
    assert _open_exact_files([bracket_file]).to_table() == table(range(0, 5))


def test_url_source_with_question_mark_is_rejected(root: Path):
    with pytest.raises(RuntimeError, match="not supported in URL sources"):
        vx.open_dataset(f"file://{root}/data/?.vortex")


def test_head_stops_without_reading_unneeded_files(tmp_path: Path):
    first = tmp_path / "a.vortex"
    second = tmp_path / "b.vortex"
    vx.io.write(table(range(0, 10)), str(first))
    # Invalid Vortex bytes: if head(0) or head(5) opens this file eagerly or polls past its limit,
    # the scan fails.
    second.write_bytes(b"not a valid vortex file")

    ds = vx.open_dataset([first, second])
    assert ds.head(0) == table(range(0, 0))
    assert ds.head(5, concurrency=1) == table(range(0, 5))


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
    conn = duckdb.connect()
    _ = conn.register("failures", vx.open_dataset(root / "data", concurrency=4))

    rows = conn.sql("SELECT even, count(*) FROM failures WHERE x >= 5 GROUP BY even ORDER BY even").fetchall()

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

    ds = vx.open_dataset(root / "data")
    with pytest.raises(RuntimeError, match="same dtype"):
        ds.to_table()


def test_a_source_must_name_files(root: Path):
    with pytest.raises(RuntimeError, match="no Vortex files found"):
        vx.open_dataset(str(root / "data" / "*.parquet"))


def test_glob_is_case_sensitive(tmp_path: Path):
    vx.io.write(table(range(0, 3)), str(tmp_path / "Train-1.vortex"))
    vx.io.write(table(range(3, 6)), str(tmp_path / "train-2.vortex"))

    assert vx.open_dataset(str(tmp_path / "Train-*.vortex")).to_table() == table(range(0, 3))

    store = LocalStore(prefix=str(tmp_path))
    assert vx.open_dataset("Train-*.vortex", store=store).to_table() == table(range(0, 3))


def test_relative_dataset_ignores_chdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    vx.io.write(table(range(0, 4)), str(dir_a / "part.vortex"))
    vx.io.write(table(range(10, 14)), str(dir_b / "part.vortex"))

    monkeypatch.chdir(dir_a)
    ds = vx.open_dataset("part.vortex")
    monkeypatch.chdir(dir_b)

    assert ds.to_table() == table(range(0, 4))


def test_fragment_reads_only_its_own_file(tmp_path: Path):
    first = tmp_path / "a.vortex"
    second = tmp_path / "b.vortex"
    vx.io.write(table(range(0, 5)), str(first))
    vx.io.write(table(range(5, 11)), str(second))

    ds = vx.open_dataset([first, second])
    fragments = list(ds.get_fragments())
    assert len(fragments) == 2

    # Remove the first file after getting fragments; reading the second fragment must not touch it.
    first.unlink()
    assert fragments[1].count_rows() == 6
    assert fragments[1].to_table() == table(range(5, 11))


def test_relative_glob_from_dir_with_glob_chars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bracket_dir = tmp_path / "batch[1]"
    plain_dir = tmp_path / "batch1"
    bracket_dir.mkdir()
    plain_dir.mkdir()
    vx.io.write(table(range(0, 4)), str(bracket_dir / "part.vortex"))
    vx.io.write(table(range(10, 14)), str(plain_dir / "part.vortex"))

    monkeypatch.chdir(bracket_dir)
    ds = vx.open_dataset("*.vortex")
    assert ds.to_table() == table(range(0, 4))

    monkeypatch.chdir(plain_dir)
    assert ds.to_table() == table(range(0, 4))


def test_recursive_glob_does_not_double_count_symlinked_dir(tmp_path: Path):
    data = tmp_path / "data"
    parts = data / "parts"
    parts.mkdir(parents=True)
    vx.io.write(table(range(0, 5)), str(parts / "part.vortex"))
    (data / "latest").symlink_to(parts, target_is_directory=True)

    assert vx.open_dataset(data).to_table() == table(range(0, 5))
    assert vx.open_dataset(str(data / "**/*.vortex")).to_table() == table(range(0, 5))


def test_trailing_double_star_glob_reads_files(tmp_path: Path):
    data = tmp_path / "data"
    nested = data / "nested"
    nested.mkdir(parents=True)
    vx.io.write(table(range(0, 3)), str(data / "a.vortex"))
    vx.io.write(table(range(3, 6)), str(nested / "b.vortex"))

    assert vx.open_dataset(str(data / "**")).to_table() == table(range(0, 6))
