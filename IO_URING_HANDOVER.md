# Vortex morsel I/O: `io_uring` experiment handover

## Workspace and branch safety

This work belongs to the isolated worktree:

- worktree: `/mnt/sdd/vortex-io-prefetch-bench`
- branch: `ji/io-prefetch-real-benchmarks`
- starting integration commit: `ec00f09c66c7a149a33fcce67edd6faa9383a620`

Do not modify `/mnt/sdd/vortex` or reuse another chat's branch. Create a separate worktree and a
new `ji/` branch for the `io_uring` experiment.

At the time this document was written, the implementation in
`/mnt/sdd/vortex-io-prefetch-bench` was intentionally uncommitted and included the oracle,
morsel-push integration, exact-expiry caches, tracing, and benchmark options. A new worktree made
from the branch ref alone will not contain those changes. Before starting independently, either:

1. ask the current chat to commit/snapshot the current implementation, then branch from that
   commit; or
2. export and apply the complete dirty diff to the new worktree, preserving all existing changes.

Do not silently start from `ec00f09c66` and assume the file-scoped cache or oracle is present.

## Current result

DataFusion normally divides the TPCH SF1 `lineitem.vortex` file into 48 partition scans. Each
partition previously opened its own segment source, coalescing driver, and completed-segment
cache. TPCH Q6 therefore generated 258 physical admissions for only 161 distinct segments.

The current implementation adds a file-scoped `RetainedSegmentSource`:

- `vortex-layout/src/segments/shared.rs`
- `vortex-datafusion/src/persistent/source.rs`
- `vortex-datafusion/src/persistent/opener.rs`

`VortexSource` owns a path-keyed registry of weak segment-source references. All DataFusion
partition openers for one physical file upgrade the same source. Completed segment futures remain
available while any partition still holds the source. The final partition drop releases the source
and every retained buffer; the registry itself cannot extend the lifetime.

Q6 lifecycle trace:

- 48 DataFusion scan partitions retained
- 258 logical demands
- 161 unique segments retained
- 97 cross-partition reuses
- 97-98 physical admissions
- one final `clear` event releasing all 161 segments

The relevant debug target is `vortex_layout::segment_lifecycle` and emits `retain`, `reuse`, and
`clear` events.

## Benchmark evidence

Target throughout:

- TPCH scale factor 1
- DataFusion
- `vortex-file-compressed`
- DataFusion file repartitioning enabled
- ordinary morsel prefetch mode
- local NVMe

### All-query hot A/B

All 22 queries, 15 iterations per query, two process runs per variant, excluding each process's
first iteration:

- sum of per-query medians: 550.45 ms before, 534.44 ms after (`-2.9%`)
- per-query geometric mean: `-2.8%`

Focused 50-iteration results:

| Query | Before | File-scoped cache | Change |
|---|---:|---:|---:|
| Q1 | 37.80 ms | 36.53 ms | -3.3% |
| Q6 | 6.28 ms | 6.13 ms | -2.4% |
| Q11 | 13.35 ms | 13.90 ms | +4.2% |
| Q14 | 13.46 ms | 13.57 ms | +0.8% |
| Q19 | 12.77 ms | 11.22 ms | -12.2% |

### Alternating forced-cold A/B

Six samples per variant. Before every sample, all SF1 compressed Vortex files were passed through
GNU `dd` with `iflag=nocache`. Variant order alternated to reduce order bias.

| Query | Before | File-scoped cache | Change |
|---|---:|---:|---:|
| Q6 | 124.68 ms | 120.45 ms | -3.4% |
| Q11 | 25.78 ms | 25.11 ms | -2.6% |
| Q19 | 165.11 ms | 119.06 ms | -27.9% |

### Physical I/O trace

| Query | Reads before | Reads after | Bytes before | Bytes after |
|---|---:|---:|---:|---:|
| Q6 | 257 | 124 | 57.69 MiB | 41.57 MiB |
| Q11 | 67 | 26 | 45.76 MiB | 16.72 MiB |
| Q19 | 405 | 209 | 64.46 MiB | 74.75 MiB |

Q19 is important: it became substantially faster despite reading more bytes. Cutting request,
blocking-task, and completion count mattered more than the additional contiguous over-read.

Post-cache physical read-size distribution:

| Query | Reads | Average | p50 | p90 | Maximum |
|---|---:|---:|---:|---:|---:|
| Q6 | 124 | 343 KiB | 384 KiB | 769 KiB | 2.3 MiB |
| Q11 | 26 | 659 KiB | 224 KiB | 2.0 MiB | 4.3 MiB |
| Q19 | 209 | 366 KiB | 280 KiB | 899 KiB | 3.0 MiB |

Benchmark artifacts from the A/B run were written under `/tmp` with these prefixes:

- `tpch-file-cache-hot-`
- `tpch-file-cache-focus-`
- `cache-cold-`
- `cache-io-`

They are ephemeral and may not exist in a later environment.

## Confirmed local read path

DataFusion does not currently construct the direct local `FileReadAt`. The default path is:

1. `vortex-datafusion/src/persistent/reader.rs` constructs `ObjectStoreReadAt` for every
   DataFusion object store, including `LocalFileSystem`.
2. `ObjectStoreReadAt` defaults to object-storage tuning:
   - concurrency: 192
   - coalescing distance: 1 MiB
   - maximum coalesced range: 16 MiB
3. `vortex-file/src/read/driver.rs` spatially merges visible requests into contiguous ranges.
4. `vortex-file/src/segments/source.rs` sends batches of those ranges through
   `VortexReadAt::read_ranges`.
5. For a local `object_store` response, `vortex-io/src/object_store/read_at.rs` gets a file payload,
   allocates a buffer, and calls `spawn_blocking` for each physical range.
6. The blocking closure calls `read_exact_at`, which is positional file I/O (`pread` semantics on
   Unix).

`read_ranges` uses one async coordinator for a batch, but it does not turn disjoint ranges into one
kernel operation. Every coalesced contiguous range still becomes its own blocking-pool job,
allocation, syscall, and completion.

The direct `vortex-io/src/std_file/read_at.rs` implementation has
`preadv2(RWF_NOWAIT)`, but the DataFusion `ObjectStoreReadAt` path does not implement
`read_at_nowait`. Consequently, the morsel early-read path gets `Unsupported` here and background
reads ultimately use blocking workers.

## `io_uring` experiment objective

Measure the execution mechanism independently from the scheduler. The primary comparison must
submit the same ordered physical ranges with the same coalescing policy and concurrency:

1. current `ObjectStoreReadAt` local-file path using blocking `pread` jobs;
2. a local `io_uring` reader using submission/completion queues;
3. optionally direct `FileReadAt` as an intermediate baseline.

Do not combine the first `io_uring` patch with a new scheduler, oracle order, cache lifetime, or
coalescing policy. Otherwise a win cannot be attributed to `io_uring`.

Useful measurements:

- end-to-end hot and forced-cold runtime;
- requested ranges and their order, count, and total bytes;
- queue depth over time;
- submission-to-completion latency distribution;
- buffer allocation count and bytes;
- number of blocking-pool jobs (must become zero for the `io_uring` data path);
- time the morsel executor has runnable CPU work versus waiting for I/O;
- cancellations and late completions after a morsel/scan exits.

Start with Q6, Q11, and Q19 because they cover three useful shapes:

- Q6: selective scan, moderate number of reads;
- Q11: short hot query where additional userspace bookkeeping is visible;
- Q19: many requests and the strongest evidence that per-request overhead dominates bytes.

## Coalescing and queue-depth follow-up

After establishing a same-range backend A/B, sweep the scheduler parameters independently:

- coalescing distance: 0, 64 KiB, 256 KiB, 1 MiB, 4 MiB;
- maximum contiguous read: 1, 4, 16, 32 MiB;
- queue depth/concurrency: 8, 16, 32, 64, 192.

The current 1 MiB / 16 MiB / 192 defaults are object-storage defaults and are suspicious for local
NVMe. Do not assume that more coalescing wins: it trades fewer completions for irrelevant bytes and
later availability of individual subranges. `io_uring` may make smaller selective requests cheap
enough that the best coalescing distance decreases.

## Correctness and lifetime requirements

- Preserve DataFusion's 48 partition outputs; do not fall back to one full-file output partition.
- A segment requested by several partitions must have only one underlying read while the shared
  file source is alive.
- The final partition/file-source drop must release all retained buffers and pending I/O state.
- A late `io_uring` completion must not write into freed or reused memory.
- Cancellation must either cancel the kernel request safely or retain its buffer/state until the
  completion is consumed.
- Partial reads, EOF, alignment, and errors must retain existing `VortexReadAt` semantics.
- Object-store/cloud behavior must remain on its existing backend unless explicitly under test.

## Existing checks

The current file-scoped implementation has passed:

- all 22 TPCH queries end to end;
- `vortex-layout` shared-source tests;
- DataFusion repartitioned versus non-repartitioned scan correctness;
- strict clippy for `vortex-layout` and `vortex-datafusion`;
- `cargo +nightly fmt --all -- --check`;
- `cargo check -p vortex-datafusion`;
- `git diff --check`.

Run narrow `vortex-io` tests for the new backend first, then the existing file/layout/DataFusion
tests, then the Q6/Q11/Q19 A/B. Do not use broad workspace checks until the backend behavior and
benchmark attribution are stable.

