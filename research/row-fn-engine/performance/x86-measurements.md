<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# x86 overhead measurements

These historical timings predate the [RowFn updates](../current-system/recent-changes.md). They do
not quantify the remaining UTF-8 validation cost or the updated Boolean, allocation, and mask paths.

[Performance overview](README.md). This record uses Vortex `f5b3b26` on 2026-09-21.

The recorded run used the existing `row_fn_output` Divan benchmark and a temporary row-count sweep.
The sweep compared several RowFn policies with direct kernels. Small dense invocations cost about
0.5 us, or about 1.3 us with array-backed validity in this fixture. Matched dense in-cache slopes
show little additional per-row cost. Nullable execution, retry, and UTF-8 validation add other costs.

The tables retain the reported medians and derived estimates. The temporary sweep source, raw
output, and fitting script were not retained. These results were not rerun for the synthesis.
They cannot serve as an exact reproduction of the deleted harness.

## Machine and build context

| Item | Value |
|------|-------|
| CPU | Intel Xeon Processor @ 2.80 GHz (cloud VM), 4 cores, 1 thread per core |
| SIMD available | SSE4.2, AVX, AVX2, AVX-512F/BW/VL, POPCNT, BMI2 |
| Caches | L1d 32 KiB per core, L2 1 MiB per core, L3 33 MiB shared |
| Toolchain | rustc 1.98.0, cargo 1.98.0, flatc 25.12.19 |
| Profile | `[profile.bench]`: `codegen-units = 16`, `lto = false`, no `target-cpu` flag, so stock x86-64 codegen (SSE2 baseline) plus runtime multiversioning where the code opts in |
| Allocator | mimalloc (both benches set it as the global allocator) |
| Timer | divan reports 20 ns precision |
| Load | idle otherwise (load average 0.1 to 0.5 during runs) |
| Commit | `f5b3b26` working tree |

Three `i64` columns at 65536 rows are 1.5 MiB, which exceeds the 1 MiB per-core L2. The 4096 to
65536 slope therefore includes memory effects and varied by about 15 percent between runs. The 512
to 4096 slope is in cache and is the number to use for per-row comparisons.

## Existing bench: `row_fn_output` (`ROWS = 16384`, 100 samples of 1 iteration)

Command: `cargo bench -p vortex-array --features unstable_row_fns --bench row_fn_output`.

| Case | Median | ns per row |
|------|--------|-----------|
| `infallible_bool` i32, per-row/per-row | 3.09 µs | 0.19 |
| `infallible_bool` i64, per-row/per-row | 16.62 µs | 1.01 |
| `deferred_bool` i32 | 3.47 µs | 0.21 |
| `deferred_bool` i64 | 11.85 µs | 0.72 |
| `infallible_bool_constant` ConstantPerRow / PerRowConstant | 15.76 µs / 15.59 µs | 0.96 / 0.95 |
| `deferred_bool_constant` ConstantPerRow / PerRowConstant | 15.61 µs / 11.14 µs | 0.95 / 0.68 |
| `deferred_i64` (checked add) | 10.75 µs | 0.66 |
| `filtered_owned_i64` AllValid / OneNullInEight / NineNullsInTen | 4.95 µs / 25.03 µs / 12.65 µs | 0.30 / 1.53 / 0.77 |
| `filtered_sink_i64` AllValid / OneNullInEight / NineNullsInTen | 5.73 µs / 27.66 µs / 12.45 µs | 0.35 / 1.69 / 0.76 |

The measured `i64` comparisons cost more than the `i32` cases. Visitor choice and constant position
also affect the measurements. This run used the stock x86-64 target despite broader CPU support.
No IR or assembly inspection ran, so the tables do not establish the vectorization state or its
cause. The single-size benchmark also lacks a matched direct baseline.

## Sweep: fixed cost and per-row cost (temporary bench, 100 samples of 50 iterations)

Each row-function case calls `execute_rows` on prebuilt `VecExecutionArgs`; an `ExecutionCtx` is
created outside the timed region. Baselines start from already decoded `Buffer<i64>` slices and
build the same `PrimitiveArray` output.

### Medians by row count

| Case | N=0 | N=1 | N=8 | N=64 | N=512 | N=4096 | N=65536 |
|------|-----|-----|-----|------|-------|--------|---------|
| base: hand loop, `Vec` only | 2 ns | 10 ns | 12 ns | 24 ns | 177 ns | 1.75 µs | 71.7 µs |
| base: hand loop + `PrimitiveArray` | 89 ns | 93 ns | 95 ns | 111 ns | 257 ns | 1.73 µs | 69.5 µs |
| base: `LaneZip::map_checked_into` + array (the framework's own kernel, called directly) | 91 ns | 94 ns | 98 ns | 133 ns | 495 ns | 3.04 µs | 81.0 µs |
| base: hand nullable add, eager bit AND + array | 222 ns | 288 ns | 291 ns | 277 ns | 432 ns | 2.02 µs | 80.1 µs |
| base: hand `x + 512` + array | 88 ns | 92 ns | 93 ns | 108 ns | 235 ns | 1.55 µs | 48.1 µs |
| base: `collect_bool` over raw string views | 146 ns | 146 ns | 152 ns | 173 ns | 394 ns | 2.40 µs | 42.8 µs |
| S1a infallible add, non-nullable | 521 ns | 572 ns | 572 ns | 591 ns | 740 ns | 2.29 µs | 66.2 µs |
| S1b deferred add, non-nullable | 504 ns | 531 ns | 533 ns | 566 ns | 892 ns | 3.37 µs | 78.2 µs |
| S2 infallible add, all-true array validity | 1.25 µs | 1.29 µs | 1.27 µs | 1.30 µs | 1.50 µs | 3.08 µs | 70.1 µs |
| S3 infallible add, one null in eight, `Dense` | 1.27 µs | 1.31 µs | 1.29 µs | 1.31 µs | 2.17 µs | 3.15 µs | 83.4 µs |
| S3b deferred add, one null in eight, `DenseWithRetry`, no failure | 1.27 µs | 1.27 µs | 1.29 µs | 1.32 µs | 1.68 µs | 4.15 µs | 84.3 µs |
| S4a integer division sink, one null in eight, `ValidOnly` direct | 2.41 µs | 1.68 µs | 2.90 µs | 3.05 µs | 4.66 µs | 17.5 µs | 312 µs |
| S4b `FilteredI64 + 1`, one null in eight, `ValidOnly` filtered | 1.12 µs | 391 ns | 3.19 µs | 4.98 µs | 5.71 µs | 12.3 µs | 187 µs |
| S5 deferred add with a retry | 1.20 µs | 8.16 µs | 9.60 µs | 9.76 µs | 10.8 µs | 17.6 µs | 212 µs |
| S6 infallible add, constant rhs | 1.05 µs | 516 ns | 511 ns | 540 ns | 691 ns | 2.15 µs | 53.4 µs |
| S6 deferred add, constant rhs | 1.03 µs | 509 ns | 516 ns | 540 ns | 856 ns | 3.06 µs | 63.7 µs |
| S7 all constant | 1.49 µs | 663 ns | 660 ns | 663 ns | 660 ns | 660 ns | 657 ns |
| S8 all-null input | 272 ns | 272 ns | 274 ns | 273 ns | 274 ns | 275 ns | 274 ns |
| S9 nullary | 180 ns | 192 ns | 194 ns | 204 ns | 347 ns | 1.52 µs | 47.1 µs |
| U1 `visit_bool` over `Utf8Column`, `len > 4` | 748 ns | 774 ns | 980 ns | 1.98 µs | 10.4 µs | 74.8 µs | 1.20 ms |
| K1 `Utf8Sink` copying each string | 725 ns | 768 ns | 1.09 µs | 2.92 µs | 17.4 µs | 141 µs | 2.40 ms |
| Vortex `Binary` `Add` expression path | 896 ns | 1.52 µs | 1.52 µs | 1.55 µs | 1.89 µs | 4.01 µs | 85.1 µs |
| P0 `row_fn_return_dtype` (planning only) | 37 ns | | | | | | |

### Fitted model

`fixed` is the N=1 median. Slopes are finite differences of medians.

| Case | fixed | per_row in cache (512 to 4096) | per_row large (4096 to 65536) |
|------|-------|-------------------------------|------------------------------|
| base: hand loop + array | 93 ns | 0.41 ns | 1.10 ns |
| base: `LaneZip::map_checked_into` + array | 94 ns | 0.71 ns | 1.27 ns |
| base: hand nullable add + array | 288 ns | 0.44 ns | 1.27 ns |
| base: hand `x + 512` + array | 92 ns | 0.37 ns | 0.76 ns |
| base: `collect_bool` over raw views | 146 ns | 0.56 ns | 0.66 ns |
| S1a infallible add | 572 ns | 0.43 ns | 1.04 ns |
| S1b deferred add | 531 ns | 0.69 ns | 1.22 ns |
| S2 all-true array validity | 1.29 µs | 0.44 ns | 1.09 ns |
| S3 partial, `Dense` | 1.31 µs | 0.27 ns (0.51 ns on rerun) | 1.31 ns |
| S3b partial, `DenseWithRetry`, no failure | 1.27 µs | 0.69 ns | 1.30 ns |
| S4a division sink, `ValidOnly` direct | 1.68 µs (2.41 µs at N=0) | 3.59 ns | 4.79 ns |
| S4b filtered fallback | 3.19 µs (N=8; N=1 hits the all-null shortcut) | 1.83 ns | 2.85 ns |
| S5 retry | 8.16 µs | 1.89 ns | 3.17 ns |
| S6 infallible, constant rhs | 516 ns | 0.41 ns | 0.83 ns |
| S6 deferred, constant rhs | 509 ns | 0.61 ns | 0.99 ns |
| S7 all constant | 663 ns | 0 | 0 |
| S8 all null | 272 ns | 0 | 0 |
| S9 nullary | 192 ns | 0.33 ns | 0.74 ns |
| U1 UTF-8 predicate | 774 ns | 17.97 ns | 18.36 ns |
| K1 UTF-8 sink | 768 ns | 34.6 ns | 36.7 ns |
| Vortex `Binary` `Add` | 1.52 µs | 0.59 ns | 1.32 ns |

A rerun of S1a, S2, S3 and two baselines with 200 samples of 100 iterations gave: base hand loop +
array 94 ns and 0.44 ns per row; S1a 588 ns and 0.45 ns; S2 1.29 µs and 0.48 ns; S3 1.29 µs and 0.51
ns. The small-N and in-cache numbers are stable to a few percent between runs; the 65536-row numbers
moved by up to 15 percent.

### Overhead derived from the fits

| Comparison | Fixed overhead | Per-row overhead (in cache) | Reading |
|------------|----------------|-----------------------------|---------|
| S1a vs hand loop + array | +480 ns | +0.02 ns (noise) | No added per-row cost resolved by these finite differences. |
| S1b vs the same kernel called directly | +437 ns | -0.02 ns (noise) | The deferred kernel itself costs 0.28 ns per row more than a plain add (0.71 vs 0.43); that is the kernel, not the framework. |
| S2 vs S1a | +720 ns | 0 | Array-backed validity adds two lazy nodes, two optimizer runs and one extra array node per batch, and never touches the rows. |
| S3 vs hand nullable add | +1.0 µs | 0 (framework defers the bit AND; the baseline does it eagerly) | Same as S2. |
| S3b vs S1b | +740 ns | 0 | Same as S2, deferred form. |
| S4b vs hand `x + c` | +3.1 µs | +1.5 ns per row (about 4x) | Filter copy of the input, O(N) placeholder fill, sparse set-bit traversal, materialized mask. |
| S4a | 1.7 to 2.4 µs | kernel-dominated (`idiv`) | Not separable without a division baseline; the framework adds mask materialization and an O(N) fill. |
| S5 vs S3b | +6.9 µs | +1.2 ns per row | The rejected error is a candidate for much of the fixed part (see below); the per-row part is the second pass plus the fill. |
| S6 vs hand `x + c` | +424 ns | +0.04 ns (noise) | Consistent with effective constant handling. Unswitching was not inspected. |
| S7 | 663 ns flat | 0 | O(1) in N as designed. |
| S8 | 272 ns flat | 0 | One planning dispatch plus argument collection; this is the cost of the prelude alone. |
| U1 vs `collect_bool` over raw views | +630 ns | **+17.4 ns per row (32x)** | `decode_utf8` rebuilds and re-validates the `VarBinViewArray` on every call (`vortex-array/src/scalar_fn/unstable/row/types/element/utf8.rs:155`). This O(N) per-batch validation dominates any cheap string predicate. |
| Vortex `Binary` path vs S1b | +990 ns | -0.1 ns (noise) | Expression-array construction, optimization and execution wrapper around the same `RowFn`. |

Attribution of the 572 ns fixed cost of S1a, using the other points: planning dispatch 37 ns (P0);
prelude without decode or execution 272 ns (S8, which includes P0, argument collection, validity
conjoin and a `ConstantArray`); `PrimitiveArray` construction about 83 ns (hand loop with array
minus `Vec` only at N=1). The remaining roughly 200 ns is the execution dispatch with its
revalidation, decode by downcast, and the two output validations. These are estimates from
differences, not direct measurements.

The N=0 anomalies for S6 and S7 (1.0 to 1.5 µs instead of 0.5 to 0.66 µs) come from the
`!array.is_empty()` guard in `ArgColumn::decode` (`element_tuple.rs:41`) and the `row_count > 0`
guard (`batch/execute/mod.rs:71`): an empty constant is sent through `execute::<PrimitiveArray>`,
which canonicalizes it. These cases need separate empty-batch comparisons.

The S5 N=1 case contains a null row whose payload overflows. Rejected evidence constructs an error,
then the resolved all-null mask suppresses it. Subtracting the S4b N=1 shortcut gives about 7.7 us.
That subtraction crosses scenarios and is an attribution estimate, not an isolated error benchmark.

`RUST_BACKTRACE=1` was set in the recorded environment. `VortexError` captures a backtrace during
construction, which makes it a plausible contributor. No disabled-backtrace control establishes
its exact share. The source confirms that the partial-validity path can discard the constructed
error before selected replay. The [candidate analysis](optimization-candidates.md) describes the
contract that a cheaper retry signal must preserve.

## Codegen inspection

No optimized IR or assembly inspection ran for this x86 experiment. Timing slopes alone cannot
establish vectorization, loop unswitching, or the cause of a visitor difference. The
[ARM compiler evidence](compiler-evidence.md) describes a different target and configuration.

## Historical experiment setup

- Installed `/usr/local/bin/flatc` 25.12.19 (outside the repository) and deleted `target/release/build/vortex-{array,layout,file,ipc}-*` once so the build script would rerun.
- Created `vortex-array/examples/row_fn_sweep.rs` (about 560 lines, divan, 23 cases times 7 row counts plus 2 single cases), built it with `cargo build --profile bench --example row_fn_sweep -p vortex-array --features unstable_row_fns`, ran `--test` then `--bench --sample-count 100 --sample-size 50`, then deleted the file and the `examples/` directory. The report records no remaining source changes after cleanup.
- Raw Divan output and the fitting script (`fit.py`) were not retained in this repository.

## Recorded commands and reproduction limits

The existing benchmark can be rerun from the pinned source. The sweep commands below require the
missing temporary harness. [The protocol](x86-measurement-plan.md) describes a new sweep, but does
not contain that harness. The setup commands record the original environment only.

```bash
# toolchain prerequisite
curl -fsSL -o /tmp/flatc.zip https://github.com/google/flatbuffers/releases/download/v25.12.19/Linux.flatc.binary.clang++-18.zip
unzip -o /tmp/flatc.zip -d /tmp/flatc && install /tmp/flatc/flatc /usr/local/bin/flatc
rm -rf target/release/build/vortex-{array,layout,file,ipc}-*   # only if an older flatc already ran

# existing bench
cargo bench -p vortex-array --features unstable_row_fns --bench row_fn_output

# Historical sweep command. Requires the deleted row_fn_sweep.rs harness.
cargo build --profile bench --example row_fn_sweep -p vortex-array --features unstable_row_fns
./target/release/examples/row_fn_sweep --bench --sample-count 100 --sample-size 50 | tee sweep.log
```
