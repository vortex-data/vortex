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

GPU decompression is opt-in and runs only the existing benchmark names allow-listed in
`src/main.rs`:

```bash
cargo run -p compress-bench --profile release_debug \
  --features cuda,unstable_encodings -- --gpu-decompress
```

On Linux, GPU files are read with direct IO (`O_DIRECT`) so repeated iterations measure
storage bandwidth rather than page-cache hits.

Set `VORTEX_GPU_PROFILE=wall` to emit one JSON record per decompression with file/layout counts,
decoded rows, batch and field-dispatch counts, host time for open, scan planning, reads, struct and
field dispatch, and final synchronization, plus wall time grouped by full encoding tree. Set it to
`gpu` to additionally bracket field dispatches with CUDA events and report each encoding group's
device-stream time. `nsys` adds an NVTX range around each field while retaining encoding-tree
metrics; `nsys-ranges` omits tree construction for a lower-overhead Nsight Systems capture.
Profiling perturbs the measurement; rerun without it for comparison numbers.

```bash
VORTEX_GPU_PROFILE=gpu cargo run -p compress-bench --profile release_debug \
  --features cuda,unstable_encodings -- --gpu-decompress --iterations 1 \
  2> /tmp/vortex-gpu-profile.log

jq -Rs '[split("\n")[] | fromjson? |
  select(.record == "vortex_gpu_decompress_profile")]' \
  /tmp/vortex-gpu-profile.log
```
