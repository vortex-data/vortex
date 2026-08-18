# PR #9416 random-access handover

## Scope

- PR: https://github.com/vortex-data/vortex/pull/9416
- Branch: `ji/partial-flat-random-access-batched-ring`
- Worktree: `/mnt/vortex-ssd/worktrees/random-access-pr9416-ring`
- Workload: feature-vectors / uniform, pinned to cores 0-7.
- The decoded ALPRD patch cache was removed. Do not reintroduce a data cache for the I/O comparison.
- The `action/bench-random-access` label was applied after removing the cache.

## Retained change

The ALPRD partial-read path no longer calls `clear_stats` on the newly constructed partial array.
Those `BitPacked`, `Patches`, `ALPRD`, and `FixedSizeList` arrays begin with empty statistics. The
fixed-width path still clears statistics inherited from its serialized array tree.

Three five-second cached runs on 8 cores:

| Variant | Vortex runs | Median | Lance median |
|---|---|---:|---:|
| No-cache baseline | 1.612 / 1.621 / 1.680 ms | 1.621 ms | 1.059 ms |
| Skip redundant ALPRD stats clear | 1.594 / 1.602 / 1.607 ms | 1.601 ms | 1.058 ms |

This is a 1.2% Vortex improvement and changes neither I/O nor serialization. A final cold rerun is
still desirable, although the change occurs entirely after I/O.

## Segment and patch measurements

The complete 105-row measurement is in `feature-vectors-uniform-segments.csv`.

- Uniform selects 105 physical Flat segments.
- Every segment has 256 vectors x 1,024 values = 262,144 values.
- Left buffer: 98,304 bytes per segment; right buffer: 753,664 bytes per segment.
- Patches per segment: min 953, mean 1,060.70, max 1,793.
- Patch density: 0.404627%.
- Patch bytes per segment: min 5,718, mean 6,364.23, max 10,758.
- Totals: 27,525,120 resident values, 111,374 patches, 668,244 patch bytes.
- The query selects 107,520 values and should intersect only about 435 patches, but the reader
  reconstructs all 111,374 patches before slicing.
- Main data is already row-sliced: 384 left bytes + 2,944 right bytes per selected vector, or
  349,440 bytes total. Patch buffers remain unsliced and add 668,244 bytes.

Conclusion: 256-vector Flat segments are not the main problem because left/right buffers support
partial row reads. Patch read/reconstruction granularity is the leak.

## Rejected experiments

1. `OnceLock<Patches>` cache: hot 1.491 ms, but skipped patch reads and decode on repeated takes.
   Removed as an invalid apples-to-apples I/O comparison.
2. Search bitpacked patch indices without bulk decode: median 1.640 ms versus 1.621 ms baseline.
   The scalar probes were slower; removed.
3. Merge patch-index/value reads: median 1.741 ms. `FileSegmentSource` already coalesces nearby
   reads, so this added slicing work; removed, including the temporary buffer API.
4. Latency-based blocking/io_uring routing: about 1.28 ms hot but about 26.7 ms cold because a small
   cold metadata read falsely classified the file as hot; removed.

## Profile and next work

A 10-second no-cache Samply profile kept all eight Tokio workers busy. Dominant resolved self
frames were AArch64 atomics and mimalloc allocation/free, pointing to task/array/future construction
overhead rather than worker serialization. The local profile is
`/tmp/no-cache-feature-uniform.profile.json.gz` and is not portable with this branch.

Next, count allocations and short-lived objects inside `resolve_alprd_pages` and final
canonicalization. Avoid two-stage patch I/O unless separately proven hot and cold: it can reduce
patch-value bytes but adds an I/O round. If a future format change is allowed, serialize ALPRD patch
chunk offsets. The 1,024-value patch chunk exactly matches one feature vector, enabling direct
lookup of the roughly four relevant patches.

## Benchmark environment

```text
taskset -c 0-7
TOKIO_WORKER_THREADS=8
RAYON_NUM_THREADS=8
LANCE_IO_THREADS=8
VORTEX_EXPERIMENTAL_PATCHED_ARRAY=1
FLAT_LAYOUT_INLINE_ARRAY_NODE=1
VORTEX_IO_URING=1
VORTEX_IO_URING_RINGS=4
VORTEX_IO_URING_QUEUE_DEPTH=128
VORTEX_IO_URING_MIN_READ_SIZE=0
VORTEX_IO_URING_MAX_IN_FLIGHT=512
```

Validation completed: nightly formatting, `git diff --check`, `cargo check -p vortex-layout`,
release benchmark builds, feature-vectors/uniform smoke runs, and repeated hot comparisons.
