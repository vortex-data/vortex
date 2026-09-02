# Random Access benchmark

Measures point-lookup latency: fetching individual rows by index from a file, rather than
scanning it. This is the workload behind the `Random Access` PR comment.

[Arrow IPC](https://arrow.apache.org/docs/format/Columnar.html#ipc-file-format) is Apache
Arrow's built-in file format, formerly called Feather V2. This suite writes it without
optional buffer compression, so it provides the established constant-time access reference.
Parquet provides the established reference for a compressed columnar representation. Together,
they let the suite compare Vortex and Lance against both ends of the storage trade-off.

Two access patterns are generated with a fixed seed (see [`src/main.rs`](./src/main.rs)):

- **correlated**: several clusters of consecutive indices scattered across the dataset,
  simulating lookups with spatial locality;
- **uniform**: indices drawn from a Poisson process spread uniformly across the dataset,
  simulating lookups with no locality.

Each pattern runs over four datasets (`taxi`, `feature-vectors`, `nested-lists`,
`nested-structs`) in Arrow IPC, Parquet, Lance, and Vortex. Cached mode performs a one-second
untimed warm-up, then reuses the open file handle. Reopen mode includes file open and metadata
work in each timed iteration. CI drives the full matrix via
[`scripts/random-access-split.py`](../../scripts/random-access-split.py).

## Running locally

```bash
cargo run -p random-access-bench --profile release_debug --features lance
```

Compare random access for a numeric scheme bundle with one of these values:

- `--vortex-numeric-bundle prior-default`
- `--vortex-numeric-bundle block-residual`
- `--vortex-numeric-bundle current-default`

Each bundle uses a separate Vortex file. Existing files remain available for repeated runs.

Add a local Parquet file with `--parquet-path`. A local path replaces the default dataset list.

```bash
cargo run -p random-access-bench --profile release_debug -- \
  --formats vortex --parquet-path /tmp/input.parquet
```

If `--datasets` also selects built-in datasets, the benchmark runs both groups.
