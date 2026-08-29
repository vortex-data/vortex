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

The morsel executor intentionally uses all available parallelism, bounded only by the number of
runnable morsels. Its process-wide worker pool is sized from detected hardware parallelism, not a
fixed constant, and larger explicit requests remain supported with a dedicated pool. Do not add a
fixed worker cap: V1 schedules layout splits across the full runtime, so limiting only the morsel
path can hide substantial sparse-scan parallelism and makes the comparison depend on an unrelated
constant. Keeping workers persistent is equally important because reopen benchmarks must not
create a new set of OS threads for every lookup.

Equivalent reopened Vortex accessors also reuse their morsel plan and executor state. The cache is
keyed by canonical path, format, file length, and modification time, and is bounded by file count.
It retains the fixed-size per-worker execution arenas and natural split metadata, but not ordinary
decoded segments, decoded dictionary values, or scan-local IO state. Dictionary values are shared
between morsels only for the lifetime of one lookup. Rebuilding an arena per worker inside every
lookup is especially expensive for wide plans and does not make reopen semantics more
representative.

The consolidated technical history, retained optimizations, observability guide, and final SSD
comparison against V1 are in the
[`vortex-morsel` SSD optimization record](../../vortex-morsel/README.md#ssd-random-access-optimization-record).

## Running locally

```bash
cargo run -p random-access-bench --profile release_debug --features lance
```
