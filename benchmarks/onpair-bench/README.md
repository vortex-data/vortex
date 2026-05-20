<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->
# OnPair chunked-array compression benchmark

Compresses string columns with [OnPair](../../encodings/onpair) into Vortex
`ChunkedArray`s — **one dictionary per chunk** — sweeping dictionary bit-width,
chunk size, and training threshold, then measures size, compression ratio, and
encode/decode throughput. Each cell is written to real `.vortex` files (one
single-column array, split into ~200 MB files) and read back to verify the
string round-trip.

## Layout

* `onpair-chunk-bench` (Rust, `vortex-bench/src/bin/onpair-chunk-bench.rs`) —
  does the compression + decompression. Subcommands:
  * `gen-tpch --sf <N> --out <parquet>` — generate TPC-H `lineitem` locally.
  * `run --parquet <p> --column <c> ...` — compress one column across the
    matrix, write Vortex files, verify the round-trip, emit JSON.
* `run.py` (this dir) — the single recreate script. Holds the column registry,
  ensures data, drives the binary, aggregates results into a table.
* `columns.py` — the registry. Add a column with a one-line `Column(...)`.

## Run

```bash
python benchmarks/onpair-bench/run.py            # TPC-H sf=10 l_comment, full matrix
python benchmarks/onpair-bench/run.py --dev      # faster build, slower run
python benchmarks/onpair-bench/run.py --sample-bytes 50000000 --chunk-mb 1,10  # quick smoke
```

By default the Python driver runs selected columns with one worker per detected
CPU core. Use `--jobs <N>` to cap column-level parallelism.

The orchestrator is also a `uv` project:

```bash
cd benchmarks/onpair-bench
uv run python run.py --sample-bytes 50000000 --chunk-mb 1,10
uv run python run.py --clean                    # delete generated .vortex output + summaries
```

Defaults: `bits = {12, 16}`, `chunk = {1, 10, 100} MB` (uncompressed budget,
split on equal-ish row boundaries), `threshold = 0.2`, `sample = 1 GB` of raw
string payload, `file-target = 200 MB`.

## Output

* `vortex-bench/data/onpair-bench/<dataset>/<column>/bits<b>_chunk<MB>_thr<t>/`
  — `part_NNNN.vortex` files + `meta.json`.
* `vortex-bench/data/onpair-bench/summary.md` and `summary.json` — all cells.

Two compression ratios are reported, both = raw UTF-8 string payload bytes ÷
compressed bytes:

* `str→codec×` — string bytes ÷ OnPair **in-memory** bytes (codes + per-chunk
  dictionaries). The pure codec ratio.
* `str→files×` — string bytes ÷ total **`.vortex` file** bytes on disk (codec +
  Vortex file framing). Since the files preserve OnPair as-is, this tracks
  `str→codec×` closely.

`uniq` / `uniq%` give the distinct-value count; when `uniq%` is low a plain
value-dictionary may beat OnPair's token dictionary.

Source parquet is cached under `vortex-bench/data/onpair-bench-src/`.
`--clean` removes `vortex-bench/data/onpair-bench/` but leaves source parquet
caches intact.

The FSST paper's DBText corpus is included as dataset id `dbtext`. Its raw text
files are downloaded from `cwida/fsst`, then cached as one-column parquet files.

## Adding datasets / columns

Append to `COLUMNS` in `columns.py`:

```python
Column(dataset_id="tpch-sf10", column="l_shipinstruct", kind="tpch"),       # another TPC-H lineitem column
Column(dataset_id="clickbench", column="URL", kind="parquet",                # an external parquet
       parquet="/path/to/clickbench.parquet"),
```

For new generated datasets, extend `ensure_parquet` in `run.py` (and, if
needed, add a generator alongside `ensure_tpch_lineitem_parquet` in
`vortex-bench/src/onpair_bench.rs`).
