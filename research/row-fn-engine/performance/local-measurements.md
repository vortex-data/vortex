<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# ARM overhead measurements

These September 21 measurements predate the [RowFn updates](../current-system/recent-changes.md).
They do not measure the current output storage, allocator, or mask paths.

The measured framework has batch overhead, and its output collector also affects cost.
A matched experiment separates those effects for one canonical `i64` operation on Apple M4 Max.
These results describe that operation and configuration. They do not establish a universal RowFn
percentage or predict an Arrow, DataFusion, or DuckDB adapter.

## The two comparisons

Every timed call computes `value.wrapping_add(1)` and returns a newly allocated canonical `i64` array.
Inputs are canonical, non-nullable, non-constant arrays with values `0..n`. Each call includes input
handle cloning, canonical decode, output construction, and output destruction.

| Mode | Implementation | What its comparison means |
| --- | --- | --- |
| Direct iterator, mode `0`. | Decode once, collect a standard slice iterator into `Vec<i64>`, and construct the output array. | The full array-to-array cost of RowFn relative to a direct specialized kernel. |
| Direct collector, mode `4`. | Decode once, then use `i64::build_from` on the native slice. | Holds the RowFn output collector fixed to examine the extra batch wrapper and argument source. |
| RowFn, mode `1`. | Call `execute_rows` with reused `VecExecutionArgs`. | Includes RowFn planning, policy, validation, decoding, the collector, and finalization. |
| Fresh arguments, mode `2`. | Construct `VecExecutionArgs` inside each RowFn call. | Adds the caller's argument allocation, input clone, and releases. |
| Direct control, mode `3`. | Call exactly the same direct function as mode `0`. | Detects movement from order, cache state, and host noise. |

The direct collector intentionally retains a framework component. Its comparison cannot establish
that all RowFn overhead is about 100 ns. The direct iterator is an equally important baseline.
Both baselines have narrower preconditions than the general framework. The measured input domain
satisfies those preconditions, and each mode produces identical values, dtype, and length.

## Results

Times are medians of 160 observations per size and mode, pooled across five processes.
Each observation times many complete calls and reports nanoseconds per call. The final column
reports a median paired difference with its interquartile range, not a difference of medians.

| Rows | Direct iterator, ns | Direct collector, ns | RowFn, ns | Fresh arguments, ns | RowFn minus collector, ns [IQR] |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 59.7 | 59.6 | 157.5 | 162.5 | 98.0 [96.3, 99.3] |
| 1 | 63.0 | 62.8 | 166.3 | 171.3 | 103.6 [102.0, 105.0] |
| 8 | 65.9 | 66.7 | 170.2 | 175.5 | 103.6 [102.1, 105.3] |
| 64 | 75.9 | 75.2 | 179.2 | 184.4 | 103.9 [102.1, 105.2] |
| 1,024 | 164.2 | 141.2 | 242.2 | 247.9 | 100.9 [99.1, 103.6] |
| 16,384 | 1,439.3 | 2,434.1 | 2,532.8 | 2,551.2 | 99.4 [62.2, 132.5] |
| 262,144 | 34,165.5 | 38,402.3 | 38,305.0 | 38,555.3 | -74.4 [-468.3, 244.0] |

For 1 through 1,024 rows, the additional batch-wrapper cost under the shared collector is about
101 to 104 ns. This is an empirical difference. It does not assign nanoseconds to dispatch,
reference counts, or validation individually.

At 16,384 rows, RowFn takes about 2.53 us versus 1.44 us for the direct iterator. The direct collector
takes about 2.43 us. Most of the iterator comparison therefore remains when the RowFn batch wrapper
is removed but its collector is retained. The collector is a relevant part of the total result.

At 262,144 rows, the wrapper comparison does not resolve an added cost. Its paired interquartile
range crosses zero, and the process medians have both signs. This does not prove zero overhead.
The direct iterator remains faster in this experiment.

The full paired comparisons are:

| Rows | RowFn minus iterator, ns [IQR] | Collector minus iterator, ns [IQR] | Fresh minus reused arguments, ns [IQR] | Direct control / direct, median ratio |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 97.7 [96.5, 98.8] | -0.3 [-1.1, 0.7] | 5.2 [3.6, 6.4] | 1.004 |
| 1 | 103.2 [102.0, 104.6] | -0.2 [-0.7, 0.5] | 5.4 [4.0, 6.8] | 1.001 |
| 8 | 104.2 [102.3, 106.1] | 0.8 [-0.0, 1.8] | 5.3 [3.7, 6.8] | 1.001 |
| 64 | 103.0 [101.5, 104.4] | -0.8 [-1.5, -0.0] | 5.5 [3.7, 7.9] | 1.000 |
| 1,024 | 78.9 [63.7, 90.8] | -20.7 [-39.4, -9.4] | 5.5 [2.9, 8.3] | 1.003 |
| 16,384 | 1,092.0 [1,037.5, 1,130.7] | 992.5 [943.5, 1,023.5] | 12.5 [-15.2, 35.3] | 1.003 |
| 262,144 | 4,105.1 [3,300.1, 5,168.3] | 4,160.5 [3,394.0, 5,368.2] | 191.9 [-175.1, 580.4] | 0.998 |

The unchanged control ratios stay near 1.00 when pooled. Individual process medians and all raw
observations are in [the data leaf](raw-observations.md). No significance test or confidence interval
is claimed. Interquartile ranges describe observed dispersion.

## Allocation requests

A separate allocator-instrumented binary ran the same paths after warm-up. Counters were enabled
only during one complete call, including destruction. The timed binary uses ordinary mimalloc,
without these counters.

| Input length | Direct iterator | Direct collector | Reused-argument RowFn | Fresh-argument RowFn |
| --- | --- | --- | --- | --- |
| Empty. | 3 requests, 624 bytes. | 3 requests, 624 bytes. | 3 requests, 624 bytes. | 4 requests, 640 bytes. |
| Nonempty, `n` rows. | 4 requests, `8*n + 624` bytes. | 4 requests, `8*n + 624` bytes. | 4 requests, `8*n + 624` bytes. | 5 requests, `8*n + 640` bytes. |

The reused RowFn path does not issue extra heap requests in this fixture. The fresh argument wrapper
adds one 16-byte request. Its time difference also includes ownership operations, so that difference
is not the time of one allocation in isolation.

These are requested byte counts. They are not allocator block sizes, retained memory, or peak memory.
The experiment does not measure atomic reference-count time directly.

## What the compiler shows

Both direct baselines and RowFn use vector arithmetic on this target. The ordinary iterator loop
processes eight `i64` values per vector-loop iteration. The shared collector processes full 64-lane
chunks, with a separate remainder path. Optimized IR and assembly show this distinction.
See [the compiler evidence](compiler-evidence.md).

This supports the need to distinguish the collector from batch setup. It does not prove that the
chunk length alone causes the timing difference. No chunk-size change or production optimization
was tested. The collector also changes how the compiler sees the traversal.

## Environment and protocol

| Item | Value |
| --- | --- |
| Vortex commit. | `96bd521eb0565555def2af7b8e97e96891728da6`. |
| Measurement date. | 2026-09-21, approximately 17:34 to 17:35 UTC. |
| Host. | Apple M4 Max, 12 performance cores, 4 efficiency cores, 128 GB memory. |
| OS and target. | Darwin 24.6.0, `aarch64-apple-darwin`. |
| Compiler. | `rustc 1.98.0 (88d9e12ae 2026-08-18)`, LLVM 22.1.8. |
| Cargo. | `cargo 1.98.0 (797e8a9bc 2026-08-05)`. |
| Profile. | Release, optimization level 3, one codegen unit, `lto = "off"`, default release debug and panic settings. |
| Flags. | `-C force-frame-pointers=yes`, inherited from repository configuration. No `target-cpu` override. |
| Allocator. | `mimalloc 0.1.52`, `libmimalloc-sys 0.1.49`, default features. |
| Compiler wrapper. | `kache 0.14.2`, inherited from `~/.cargo/config.toml`. |
| FlatBuffers compiler. | `flatc 25.12.19`. |
| Dependencies. | The temporary package copied the repository lockfile. Every retained dependency name, version, and source matches it. |
| Vortex features. | `vortex-array/unstable_row_fns`, without the benchmark dev-dependency feature set. |
| Process control. | One timed process at a time. No explicit CPU affinity, frequency lock, or thermal control. |
| Input and context. | One reused input, argument wrapper, session, and execution context per size. |
| Warm-up. | 1,000 calls for each of five modes before sampling each size. |
| Calibration. | Double the direct-iterator repetition count until the measured block reaches 2 ms. Reuse that count for all modes at that size. |
| Samples. | 32 groups per size per process, five processes, seven sizes, five modes, 5,600 observations. |
| Order. | Even groups: `0,4,1,2,3`. Odd groups: `3,2,1,4,0`. |
| Pair definition. | Same process, size, and sample-group index. |
| Timer. | `std::time::Instant`, amortized over calibrated repeated complete calls. |
| Output consumption. | `black_box`, then drop the output inside timing. |

The dependency version comparison found only the new temporary package outside the repository lock.
The original worktree was clean. Research Markdown was the only worktree change during measurement.
The initial dependency build took 2m42s with the configured cache available. This is an environment
observation, not a compile-time benchmark or a RowFn extraction cost estimate.

[Exact source and commands](reproduction.md) reproduce the experiment. The ordinary timing binary's
SHA-256 is `1bb7b2a891f9b3b6905d974bc8b48bb9edacc33a67aa57b1825f2f5fccca4089`.

## Limits

The row function has a fixed `i64` signature and trivial options. It does not exercise a large
runtime type-dispatch match, nested dtype equality, parameterized output types, or expensive binding.

These are warm canonical-input measurements. They exclude cold session creation, fresh contexts,
nullable input, constants, other arities, deferred errors, filtered execution, strings, compressed
input, host conversion, and query execution. The result is specific to Apple ARM and this compiler
configuration. The repository benchmark profile is a different configuration.

The experiment validates the operation's values, dtype, and length before timing. It does not run
the repository test suite, Clippy, formatting, or unrelated benchmarks. No production Rust source
was changed. Compiler artifacts and the temporary package remain outside the repository.
