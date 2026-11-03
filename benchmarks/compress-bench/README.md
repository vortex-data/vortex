# Compression benchmark

Measures compression and decompression throughput, plus resulting file sizes, for Vortex,
Parquet, uncompressed Arrow IPC, and optionally Lance.

[Arrow IPC](https://arrow.apache.org/docs/format/Columnar.html#ipc-file-format) is Apache
Arrow's built-in file format, formerly called Feather V2. This suite writes it without
optional buffer compression. Its timings therefore isolate serialization and deserialization
without codec cost. Its file size provides the approximately 1x baseline for compression
ratios. Parquet provides the established reference for a compressed columnar representation.

The suite covers NYC taxi data and several
[Public BI](https://github.com/cwida/public_bi_benchmark) tables (Arade, Bimbo,
CMSprovider, Euro2016, Food, HashTags), TPC-H `l_comment` variants, and synthetic nested
data. This is the workload behind the `Compression` PR comment.

See [`src/main.rs`](./src/main.rs) for the dataset list and CLI flags (`--formats`,
`--datasets`, `--ops compress,decompress`).

## CPU parallelism

Vortex and Parquet get the same CPU budget, and each uses it the way its own default reader and
writer would. The binary runs one multi-threaded Tokio runtime,
sized by `--threads` (default: one worker per available core), and every timed Vortex and
Parquet operation runs as a task on it, so neither format uses threads outside that pool and the
main thread does none of the timed work. `--threads 1` compares single-worker decoders;
the default compares them at full width.

CI runs the timed benchmark on a fixed budget of 16 cores: `scripts/compress-split.py --cpus 16`
(the default) pins the process to 16 CPUs through `scripts/bench-taskset.sh` and passes
`--threads 16`, so the result does not depend on how many cores the benchmark machine has.
Pass `--cpus 0` to use every benchmark CPU instead.

Vortex schedules its own compression and scan work as tasks on the runtime. Parquet runs on
the same runtime the way DataFusion's Parquet sink and scan do with their default options:

- **Compress** follows DataFusion's parallel `ParquetSink`. Input is split into row groups at
  the writer's default `max_row_group_row_count` (1,048,576 rows), the same boundaries
  `ArrowWriter` produces. Every leaf column of the open row group is encoded by its own task,
  fed through a channel of at most 2 arrays (`maximum_buffered_record_batches_per_stream`). When
  a row group is full, a separate task closes its column writers while the next row group
  starts, and at most 1 finished row group waits to be appended to the file
  (`maximum_parallel_row_group_writers`). Encoded bytes leave the writer's buffer every ~1 MB,
  as DataFusion streams them to the object store, and are joined into one buffer at the end.
- **Decompress** follows DataFusion's file scan. The file's bytes are split into one range per
  runtime worker (`target_partitions`), unless the file is under 1 MiB
  (`repartition_file_min_size`), and each row group belongs to the range holding its first page.
  Each range is one task that decodes its row groups in order, in batches of 8,192 rows
  (`batch_size`, also the Vortex writer's row block size), reading strings and binaries as
  `Utf8View` / `BinaryView` (`schema_force_view_types`), the types the Vortex reader produces.
  Spark plans a single file the same way: byte-range splits, each row group read by the split
  holding its midpoint, one sequential reader per split.

The written file has the same row groups, encodings and compression as a sequential
`ArrowWriter` with the same settings. Arrow IPC is a serialization baseline and stays
single-threaded.

## Running locally

```bash
cargo run -p compress-bench --profile release_debug
# single-worker comparison
cargo run -p compress-bench --profile release_debug -- --threads 1
```

## Sparse-wide baseline

`sparse-write-bench` is a focused, synthetic Vortex benchmark that needs no downloads. It measures
complete serialized file sizes and in-memory write times with the default production compressor.
It is a separate, manually invoked runner; the existing compression CI workload is unchanged.

```bash
cargo run -p compress-bench --bin sparse-write-bench --profile release_debug -- \
  --ingest-output /tmp/sparse-write-baseline.jsonl
# Quicker smoke run, including round-trip checks for every output:
cargo run -p compress-bench --bin sparse-write-bench -- \
  --rows 10000 --groups 10 --iterations 1
```

Defaults are 100,000 source rows, 100 mutually exclusive groups, three fields per group, seed 42,
one runtime worker, and three timed iterations after one warmup. Each row has exactly one present
group. Fields hold random integers in `[0, 1_000_000)`, or lists of three such integers. Cases are:

- `flat-runs`: nullable integer fields with clustered group presence.
- `nested-runs`: the integer fields under four nullable struct levels.
- `lists-runs`: the same nesting with list fields.
- `lists-shuffled`: list groups scattered across rows; present-only payloads match `lists-runs`.
- `nested-child-nulls`: nested integers with genuine child nulls in 10% of present values.

Select cases with `--cases nested-runs,lists-shuffled`. `--rows` must be divisible by `--groups`.
`--fields`, `--seed`, `--threads`, and `--iterations` are also configurable. Hold these settings and
the machine constant when comparing commits.

Each case compares four layouts, reporting the sum of all file sizes and sequential write times:

| Layout | Contents |
| --- | --- |
| `wide` | One file containing every group and all source rows. |
| `split-all-rows` | One file per group, retaining absent rows and their nulls. |
| `shredded-present-rows` | One file per group containing only its present rows. |
| `shredded-with-row-ids` | Present-only files plus a `__row_id` column preserving original positions. |

Present-only files omit information about absent rows; the row-ID variant accounts for the cost
of retaining those positions. `wide_over_layout` is the wide file's byte size divided by the
layout's total byte size. Throughput uses source rows, regardless of layout. Timings include the
default writer, compression, and serialization to memory, including per-file overhead. They
exclude data generation, statistic-cache resets, disk I/O, and output decoding. Every warmup file
is read back and checked for matching length, dtype, values, and nulls outside the timer; serialized
sizes must remain stable across iterations.

CSV output reports median write time. `--ingest-output` emits the existing V3 `compression_size`
and `compression_time` JSONL records, with the layout in `dataset_variant`, all iteration timings,
and best-of-N time following the existing ingestion convention. Dataset keys include the case,
shape, seed, and worker count, allowing each subsequent fix to be measured against this baseline.

## GPU decompression

`--gpu-decompress` is opt-in, requires the `cuda` feature, and restricts the suite to the
GPU dataset list in `src/main.rs`. It measures decompression only, for two backends:

- **Vortex** — the file is written with CUDA-compatible BtrBlocks encodings only
  (`only_cuda_compatible`) and a CUDA flat layout, then decoded on the device all the way to
  canonical arrays.
- **Parquet** — the file is rewritten with GPU-friendly writer settings (see below) and read
  back with [cuDF](https://github.com/rapidsai/cudf)'s `read_parquet`, which performs the
  whole read on the device: page header decode, codec decompression, dictionary/RLE/plain
  decoding and column assembly.

Both sides therefore decode all the way to device-resident arrays, which is what makes the
`vortex:parquet-<codec> gpu ratio decompress time` metric a like-for-like comparison.

```bash
cargo run -p compress-bench --profile release_debug \
  --features cuda -- --gpu-decompress

# pick the Parquet page codec the GPU file is written with (default: snappy)
cargo run -p compress-bench --profile release_debug \
  --features cuda -- --gpu-decompress --gpu-parquet-codec zstd
```

### cuDF

cuDF has no Rust binding, so the benchmark drives it out of process: it spawns `python3` running
`scripts/cudf-parquet-read.py`, which imports cuDF from the prebuilt `cudf-cu12` Python package,
reads the file, and prints its timings back as JSON on stdout. Nothing links against libcudf, so
cuDF is a runtime requirement rather than a Rust build dependency:

```bash
uv pip install --extra-index-url https://pypi.nvidia.com cudf-cu12 pandas pyarrow
```

The clock lives inside that script rather than around the subprocess, so process spawn,
interpreter start, `import cudf` and CUDA context creation are all excluded; a warm-up read runs
first for the same reason. The script performs several timed reads per invocation and reports the
fastest, and the harness then takes its own minimum across `--iterations`.

Both backends read a warm file by default. cuDF runs an untimed warm-up read before the timed
one, so its timed read hits the page cache; the Vortex reader therefore does **not** use direct
I/O by default, because `O_DIRECT` would bypass the page cache and compare a Vortex read of the
disk against a cuDF read of RAM. `--gpu-direct-io` turns it back on to measure storage bandwidth
instead — a different question, and the resulting ratio is not a decode comparison.

The remaining asymmetry is the transfer path: the Vortex reader uses pinned buffers, while cuDF
does its own host read and host-to-device copy.

### GPU-friendly Parquet writer settings

Set in `src/gpu/writer.rs`:

| Setting | Value | Why |
| --- | --- | --- |
| writer version | `PARQUET_1_0` | v1 pages compress the whole page body; v2 pages put uncompressed levels ahead of the compressed values in the same body. |
| compression | Snappy (default) or Zstd | Snappy is the Parquet default and has the higher device-side throughput. |
| dictionary | enabled | Keeps the decompressed payload small; the encoding GPU Parquet readers decode fastest. |
| data page size | 1 MiB | Large enough to amortize per-page setup, small enough to keep every SM fed. Matches the page size cuDF targets. |
| data page row limit | 1,000,000 | The 20k-row default caps narrow columns' pages far below 1 MiB. |
| statistics | chunk-level | Page statistics only inflate the headers a reader has to walk. |
| row group size | 1,048,576 rows | Shared with the Vortex side as `GPU_ROW_GROUP_SIZE` — see below. |

### Matching physical partitions

A Parquet row group and a Vortex chunk are the same thing for this comparison: the unit the
reader plans and dispatches over. Both formats are pinned to `GPU_ROW_GROUP_SIZE`
(1,048,576 rows, Parquet's `DEFAULT_MAX_ROW_GROUP_ROW_COUNT`).

Without this the two are not comparable. Parquet reads ~1M-row row groups, while the Vortex
side inherits the Arrow reader's ~8K-row batches — each of which becomes its own chunk, its own
compressed blocks and its own kernel launches, so a single dispatch turns into hundreds.

Setting the Arrow reader's batch size alone is not enough: the reader also breaks at the source
file's row group boundaries, so short batches survive. `parquet_to_vortex_chunks_with_batch_size`
therefore concatenates the source batches and re-slices them on exact boundaries. Those batches
are written straight through as root chunks via `ChunkedLayoutStrategy`, and read back with
`SplitBy::RowCount(GPU_ROW_GROUP_SIZE)` so a scan batch is one whole partition.

### Correctness

`--gpu-verify` cross-checks device output against the CPU decoders on every iteration:

- Parquet: the cuDF-read frame is compared against a CPU Parquet read of the same file.
- Vortex: each GPU-decoded field is copied back and compared against the same field decoded
  on the CPU, through Arrow with a pinned target type.

The check runs before each timed measurement and is not included in it, so a verifying run
still publishes comparable numbers — it just takes considerably longer:

```bash
cargo run -p compress-bench --profile release_debug \
  --features cuda -- --gpu-decompress --gpu-verify --iterations 1
```

Any `--gpu-decompress` run reports on every dataset rather than stopping at the first failure, so
one run shows which datasets decode correctly on the GPU and which do not. The timing tables are
rendered before the failure summary, so a dataset the GPU cannot decode still leaves the rest of
the matrix with numbers — the process exits non-zero either way.

The dataset list in `src/main.rs` therefore holds only datasets a `--gpu-verify` run has confirmed.
It now covers the whole compress suite: the kernel gaps that kept `taxi`, `Arade`, `CMSprovider`,
`Euro2016`, `HashTags` and the `StructListOfInts` wide tables off it — `u16` components in
`date_time_parts`, per-element `RunEnd` validity, and missing `vortex.masked` and `vortex.list`
kernels — have since been closed. Add a new dataset there once verification passes.

`airquality` and `rplace` download from pcodec's public bucket, which the CPU suite skips to avoid
creating egress charges for pcodec. The GPU suite runs every entry on its explicit list, so both
are fetched on each GPU run.
