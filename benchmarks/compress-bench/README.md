# Compression benchmark

Measures compression and decompression throughput, plus resulting file sizes, for Vortex
versus Parquet (and optionally Lance) across a range of datasets: NYC taxi data, several
[Public BI](https://github.com/cwida/public_bi_benchmark) tables (Arade, Bimbo,
CMSprovider, Euro2016, Food, HashTags), TPC-H `l_comment` variants, and synthetic nested
data. This is the workload behind the `Compression` PR comment.

See [`src/main.rs`](./src/main.rs) for the dataset list and CLI flags (`--formats`,
`--datasets`, `--ops compress,decompress`).

## Running locally

```bash
cargo run -p compress-bench --profile release_debug
```

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
The generated files also use the same 1,048,576-row physical partition size: Parquet row groups
and Vortex root chunks. Vortex input batches are concatenated and sliced at those exact boundaries
before writing, so smaller source batches cannot leak into its on-disk layout.

```bash
cargo run -p compress-bench --profile release_debug \
  --features cuda,unstable_encodings -- --gpu-decompress

# pick the Parquet page codec the GPU file is written with (default: snappy)
cargo run -p compress-bench --profile release_debug \
  --features cuda,unstable_encodings -- --gpu-decompress --gpu-parquet-codec zstd

# isolate one backend for diagnostics; the default remains parquet,vortex
cargo run -p compress-bench --profile release_debug \
  --features cuda,unstable_encodings -- --gpu-decompress --formats vortex
```

### Vortex GPU profiling

`--gpu-vortex-profile wall|gpu|nsys` enables opt-in diagnostics for the Vortex backend. `wall`
records host timings, `gpu` also brackets every field dispatch with CUDA events, and `nsys` adds
per-field NVTX ranges. These modes perturb the measured run; use them to explain a result, then
rerun without the flag for the comparison number.

After each timed stream synchronization, the benchmark writes one JSON record to stderr with
`record="vortex_gpu_decompress_profile"`. It includes file/layout sizes and counts, decoded rows,
batches and field dispatches; `stages` contains microsecond wall times; and `encodings` groups calls,
rows and wall time by full encoding tree and field name. In `gpu` mode, each encoding group also has
`gpu_us`; it is `null` in the other modes.

```bash
cargo run -p compress-bench --profile release_debug \
  --features cuda,unstable_encodings -- --gpu-decompress --formats vortex \
  --datasets '^(Arade|Bimbo|CMSprovider)$' --iterations 3 \
  --gpu-vortex-profile gpu 2> /tmp/vortex-gpu-profile.log

# Ignore non-JSON progress/log lines and average the main stages by dataset.
jq -Rs '
  [split("\n")[] | fromjson? |
   select(.record == "vortex_gpu_decompress_profile")]
  | group_by(.dataset)
  | map({
      dataset: .[0].dataset,
      runs: length,
      total_us: (map(.stages.total_us) | add / length),
      read_us: (map(.stages.read_us) | add / length),
      dispatch_us: (map(.stages.field_dispatch_us) | add / length),
      gpu_us: (map([.encodings[].gpu_us // empty] | add) | add / length)
    })
' /tmp/vortex-gpu-profile.log
```

`open_us` covers opening and footer metadata, `scan_plan_us` builds the scan stream, `read_us`
awaits batches, `struct_dispatch_us` materializes each struct batch, `field_dispatch_us` measures
CPU planning/enqueue time for field decodes, and `final_sync_us` is the remaining device tail.
`profile_overhead_us` is the remainder spent collecting diagnostics, primarily encoding-tree
formatting and CUDA-event bookkeeping; it makes the profiler's own perturbation explicit.
Because CUDA work is asynchronous, `read_us` can include I/O, layout execution, backpressure, and
waiting for earlier device work; it is not pure storage time. `gpu_us` is the device-stream time
between field events. Allocation/free, upload, wait, event, and callback counts are not available
from this record; use Nsight Systems for those runtime-wide counts.

### cuDF

cuDF is reached through its prebuilt `cudf-cu12` wheel, so it is a runtime dependency of the
benchmark and never enters the Rust build:

```bash
uv pip install --extra-index-url https://pypi.nvidia.com cudf-cu12 pandas pyarrow
```

`scripts/cudf-parquet-read.py` performs and times the read. Timing is taken inside that
script, so interpreter start, `import cudf` and CUDA context creation are excluded; a warm-up
read runs first for the same reason.

Both backends read a warm file by default. Each runs an untimed full read before a separately
opened timed read, warming the OS page cache, allocator, and CUDA modules. Neither reuses decoded
arrays, and the Vortex CUDA opener disables its data-segment cache. The Vortex reader therefore
does **not** use direct I/O by default, because `O_DIRECT` would bypass the page cache and compare
a Vortex read of the disk against a cuDF read of RAM. `--gpu-direct-io` turns it back on to measure
storage bandwidth instead — a different question, and the resulting ratio is not a decode
comparison.

The remaining asymmetry is the transfer path: the Vortex reader uses pinned buffers, while cuDF
does its own host read and host-to-device copy.

### GPU-friendly Parquet writer settings

Set in `src/gpu_writer.rs`:

| Setting | Value | Why |
| --- | --- | --- |
| writer version | `PARQUET_1_0` | v1 pages compress the whole page body; v2 pages put uncompressed levels ahead of the compressed values in the same body. |
| compression | Snappy (default) or Zstd | Snappy is the Parquet default and has the higher device-side throughput. |
| dictionary | enabled | Keeps the decompressed payload small; the encoding GPU Parquet readers decode fastest. |
| data page size | 1 MiB | Large enough to amortize per-page setup, small enough to keep every SM fed. Matches the page size cuDF targets. |
| data page row limit | 1,000,000 | The 20k-row default caps narrow columns' pages far below 1 MiB. |
| row-group / root-chunk rows | 1,048,576 | Gives both formats the same independently readable physical partitions and amortizes GPU launch overhead. |
| statistics | chunk-level | Page statistics only inflate the headers a reader has to walk. |

### Correctness

`--gpu-verify` cross-checks device output against the CPU decoders on every iteration:

- Parquet: the cuDF-read frame is compared against a CPU Parquet read of the same file.
- Vortex: each GPU-decoded field is copied back and compared against the same field decoded
  on the CPU, through Arrow with a pinned target type.

Verification runs inline, so timings from a verifying run are not comparable to a plain one —
run it as its own pass:

```bash
cargo run -p compress-bench --profile release_debug \
  --features cuda,unstable_encodings -- --gpu-decompress --gpu-verify --iterations 1
```

Any `--gpu-decompress` run reports on every dataset rather than stopping at the first failure, so
one run shows which datasets decode correctly on the GPU and which do not. The timing tables are
rendered before the failure summary, so a dataset the GPU cannot decode still leaves the rest of
the matrix with numbers — the process exits non-zero either way.
