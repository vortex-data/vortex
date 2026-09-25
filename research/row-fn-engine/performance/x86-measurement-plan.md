<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# x86 sweep protocol

This protocol records the original sweep. A new run must use the
[updated execution paths](../current-system/recent-changes.md), including direct packed Boolean
retry and allocator-aware output, and record its own source and artifacts.

[Performance overview](README.md). This record uses Vortex `f5b3b26` on 2026-09-21.

This plan defines how to measure the `RowFn` framework's overhead as a linear model, `time(N) =
fixed_per_batch + N * per_row`, and how to compare `per_row` against a hand-written loop and against
Vortex's other paths for the same operation. It lists the scenarios, the row counts to sweep, the
fitting method, what the existing divan bench already covers, how to inspect the generated code, and
how to split the fixed cost into phases. Every command is given in full. The companion [results](x86-measurements.md)
record the original experiment.

The temporary sweep implementation was deleted after the recorded run. This page retains scenario
and methodology notes, not its exact source. A new implementation needs fresh baseline review.
The [results](x86-measurements.md) are historical tables and were not rerun for this synthesis.

## The model

For a `RowFn` `F` executed on a batch of `N` rows:

```text
time_F(N)          = fixed_F + N * per_row_F
time_baseline(N)   = fixed_B + N * per_row_B
fixed_overhead     = fixed_F - fixed_B
per_row_overhead   = per_row_F - per_row_B
```

`per_row` is measured against two references:

| Reference | What it is | Why |
|-----------|------------|-----|
| (a) hand-written loop | Two `&[i64]` slices already decoded, `Vec::with_capacity(N)`, one loop applying the same closure, then `PrimitiveArray::new(vec, Validity::NonNullable).into_array()` so both sides produce an `ArrayRef`. A second form without the array construction isolates the array node cost. | This is the floor a user could write by hand against the same decoded buffers. |
| (b) Vortex's own path for the same operation | `lhs.binary(rhs, Operator::Add)?.execute::<PrimitiveArray>(ctx)`, which is what `vortex-array/benches/binary_ops.rs:355` measures. Note that there is no independent vectorized primitive kernel any more: `execute_numeric_primitive` at `vortex-array/src/scalar_fn/fns/binary/numeric/row.rs:36` builds a `VecExecutionArgs` and calls `execute_rows` on `NumericBinary` (`row.rs:44`). Only decimal arithmetic keeps a separate columnar kernel (`numeric/decimal.rs`). | (b) therefore measures expression-array construction plus the `RowFn`, not an alternative kernel. The closest independent kernel is the framework's own inner loop called directly: `LaneZip::new(lhs, rhs).map_checked_into(out, f)` from `vortex-compute/src/lane_kernels/map_into.rs:244`. Measuring it directly isolates the loop from the framework wrapper. |

The dense comparisons ask whether an added per-row cost remains after compilation.
The [pipeline trace](pipeline-trace.md) identifies candidate batch and row costs. The result depends
on the operation, collector, target, and compiler configuration.

## Scenarios

| Id | Scenario | Inputs | Expected policy / path | Reference |
|----|----------|--------|------------------------|-----------|
| S1a | non-nullable, infallible `visit` | two `PrimitiveArray<i64>` | dense fast path, `execute_owned_infallible`, `map_into` | (a) hand loop |
| S1b | non-nullable, deferred `visit_deferred` | same | dense fast path, `execute_owned`, `map_checked_into` over `ArgColumnSource` | (a) and the direct `LaneZip::map_checked_into` |
| S2 | nullable, array-backed all-true validity | two `PrimitiveArray<i64?>` with `Validity::Array(all true)` | policy path, `Dense` or `DenseWithRetry`, lazy mask | S1 numbers plus the lazy node cost |
| S3 | partially valid, `Dense` | one null in eight, infallible | `execute_dense`, output masked lazily | (a) plus an eager `BitBuffer & BitBuffer` |
| S3b | partially valid, `DenseWithRetry`, no failure | one null in eight, deferred, null payloads do not overflow | `execute_owned_dense_attempt`, `LaneZip` | same |
| S4a | partially valid, `ValidOnly`, direct | nullable inputs, `visit_into` with `VortexResult<InitializedElement>` (integer division) | `execute_mask`, `execute_sink_valid_rows` | hand-written masked loop over set bits |
| S4b | partially valid, `ValidOnly`, filtered fallback | an element with `DENSE_SAFE = false` and the default `can_decode_null_tolerant` (the bench's `FilteredI64`) | `execute_filtered`, three dispatches | same |
| S5 | `DenseWithRetry` with a retry | lhs nullable with `i64::MAX` stored under nulls, rhs all ones | dense pass, rejected evidence, `execute_mask`, valid-rows pass | S3b plus one extra pass |
| S6 | one constant operand | per-row lhs, `ConstantArray` rhs | dense fast path with `ArgColumnSource::Constant` | hand loop `x + c` |
| S7 | all constant | two `ConstantArray` | `execute_all_constant`, one closure call | O(1) |
| S8 | all-null input | lhs with `Validity::AllInvalid` | `all_null()`, one dispatch | O(1) |
| S9 | nullary | no inputs, `visit::<(), i64>` | `execute_nullary_rows` | `vec![7; N]` |
| U1 | UTF-8 input | `VarBinViewArray` of mixed inlined and referenced strings, `visit_bool::<(Utf8Column,), false>(len > 4)` | dense, `execute_owned_infallible_bool`, plus `decode_utf8` re-validation | `BitBuffer::collect_bool(N, |i| views[i].len() > 4)` over the raw views |
| K1 | sink output | `visit_into::<(Utf8Column,), Utf8Sink, ()>` copying each string | dense `execute_sink` | not needed for the overhead question; report absolute cost |
| P0 | planning only | `row_fn_return_dtype(&F, &opts, &dtypes)` | one dispatch, no execution | isolates the planning phase |

## Row counts and fitting

Sweep `N` in `{0, 1, 8, 64, 512, 4096, 65536}`.

- `N = 0` and `N = 1` give the fixed cost almost directly. `N = 0` skips even the single closure call, and avoids the all-constant shortcut (`batch/execute/mod.rs:71` requires `row_count > 0`), so compare it with `N = 1` before trusting it.
- `N = 8` and `N = 64` sit inside one `map_into` chunk (`CHUNK_LEN = 64`, `vortex-compute/src/lane_kernels/mod.rs:49`) and one bit word.
- `N = 512` and `N = 4096` are the vectorized steady state in L1 (three `i64` columns at 4096 rows are 96 KiB).
- `N = 65536` is 1.5 MiB for three columns, which still fits this machine's 1 MiB-per-core L2 only partly. Watch for a bend in the curve here; if the slope between 4096 and 65536 differs from the slope between 512 and 4096 by more than about 20 percent, report both and attribute the difference to memory, not to the framework.

Fit with ordinary least squares over all points, and also report two robust estimators that do not
depend on the fit: `fixed = t(N=1)` and `per_row = (t(65536) - t(4096)) / (65536 - 4096)`. The
framework overhead is the difference between the row-function estimators and the baseline estimators
for the matching scenario. Report medians, not means; divan prints `fastest`, `slowest`, `median`,
`mean`.

Use `--sample-size` of at least 50 for the small `N` points so each sample spans several
microseconds; the timer precision divan reports on this machine is 20 ns.

## What the existing bench covers

`vortex-array/benches/row_fn_output.rs` measures `execute_rows` end to end (`row_fn_output.rs:381`
to `row_fn_output.rs:393`) at a single `ROWS = 1 << 14` (`row_fn_output.rs:51`). It constructs **no
hand-written baseline**: every case is a `RowFn`, and comparisons are only possible between cases
(infallible vs deferred, `i32` vs `i64`, constant orientation, validity patterns). Because `ROWS` is
fixed, it cannot separate `fixed_per_batch` from `per_row`.

| Existing case | Scenario covered | Not covered |
|---------------|------------------|-------------|
| `infallible_bool` (`i32`, `i64`), `PerRowPerRow` | S1a with `Out = bool` through `visit` (not `visit_bool`), so it exercises `bool::build_from` (`types/element/bool.rs:102`) | `visit_bool` and the `MULTIVERSIONED = true` path |
| `deferred_bool` (`i32`, `i64`) | S1b with packed bools through `visit_deferred_bool::<.., false>` | `MULTIVERSIONED = true` |
| `infallible_bool_constant`, `deferred_bool_constant` | S6, both orientations, bool output only | S6 with a primitive output |
| `deferred_i64` `PerRowPerRow` | S1b (the checked-add kernel that `NumericBinary` uses) | S1a with a primitive output |
| `filtered_owned_i64`, `filtered_sink_i64` with `AllValid`, `OneNullInEight`, `NineNullsInTen` | S4b (filtered fallback) for owned and sink outputs. The `AllValid` case uses `Validity::from_iter` (`row_fn_output.rs:375`), which collapses to `Validity::AllValid`, so it measures the dense path of a non-dense-safe element, not an array-backed all-true validity. | S4a direct path |
| nothing | S2, S3, S3b, S5, S7, S8, S9, U1, K1, P0, and all baselines | |

## Codegen inspection

Build the bench once and keep the binary:

```bash
cargo bench -p vortex-array --features unstable_row_fns --bench row_fn_output --no-run
```

Option 1, `cargo-show-asm` (not installed during the recorded run):

```bash
cargo install cargo-show-asm
cargo asm -p vortex-array --features unstable_row_fns --bench row_fn_output --profile bench --rust \
  "execute_owned_dense_attempt" 0
```

Run it once with no symbol name to list the monomorphized functions and pick the exact one; the
interesting symbols are `execute_owned_infallible`, `execute_owned`, `execute_owned_dense_attempt`,
`execute_owned_bool`, `execute_sink`, and, because they are `#[inline]`, the inlined lane loops will
appear inside those symbols rather than as their own.

Option 2, plain `rustc` output:

```bash
RUSTFLAGS="--emit asm -C debuginfo=0" cargo bench -p vortex-array --features unstable_row_fns \
  --bench row_fn_output --no-run
ls target/release/deps/row_fn_output-*.s
```

What to look for in the dense loop of `i64 + i64`:

- A loop body with two vector loads, one vector add, one vector store, and no `call`. On a stock x86-64 build this is `movdqu` / `paddq` / `movdqu` on `xmm` registers, unrolled 2x or 4x. With `-C target-cpu=native` on this machine it becomes `vpaddq` on `ymm` or `zmm` registers.
- For the deferred kernel, an extra `por` or `vpor` into a reduction register per iteration and a final horizontal reduce after the loop. The `bool` failure lane appears as a compare (`pcmpgtq` needs SSE4.2; with only SSE2 the compiler emulates it, which is why `i64` comparisons are markedly slower than `i32` on the stock build).
- No `panic_bounds_check` or `slice_index_len_fail` call target inside the loop body. If one is present the loop still contains a bounds check.
- No `cmp` on a loop-invariant enum discriminant inside the loop. If the `ArgColumnSource` `match` was not unswitched, the loop body will branch on it per lane.

Confirm with hardware counters on the sweep binary (`perf` was not present in the recorded experiment; install
`linux-tools-generic` or use `samply`):

```bash
perf stat -e instructions,cycles ./target/release/examples/row_fn_sweep s1_infallible_add_nonnull --sample-count 200 --sample-size 100
```

Divide instructions by rows: a vectorized `i64` add should stay well under 2 instructions per row.

## Attributing the fixed cost to phases

| Phase | How to isolate | Command or code |
|-------|----------------|-----------------|
| Planning dispatch | `row_fn_return_dtype` runs `ensure_arity`, one `dispatch` with `BatchPlanner`, `BatchPlan::new`, `result_dtype` (`vtable.rs:93`) | bench `phase_planning_only` in the sweep |
| Whole fixed cost | `N = 0` and `N = 1` points of each scenario | sweep |
| Validity conjoin | difference between S1a and S2 at `N = 0` (S2 adds two `Validity::Array` clones, one lazy `Binary And` node, one `Mask` node, one optimizer run, one extra `PrimitiveArray`) | sweep |
| Decode | difference between S1a and the hand loop at `N = 0` is planning + decode + finalize; subtract `phase_planning_only` to get decode + finalize | sweep |
| Allocation and array construction | difference between `base_add_nonnull_array` and `base_add_nonnull_vec_only` | sweep |
| Finalize (`finalize_kernel_output` + `finalize_output`) | not separately reachable from outside the crate; estimate as the remainder, or add a `#[cfg(test)]` microbench inside `batch/execute/output.rs` | remainder |
| Mask materialization | S4a minus S3 at `N = 65536` less the sparse loop; or bench `Validity::execute_mask` directly on the input validity | `bencher.bench_local(|| validity.execute_mask(n, &mut ctx))` |
| Retry | S5 minus S3b at each `N` | sweep |
| Sampling profile of the fixed cost | run one scenario at `N = 64` for a few seconds under `samply` and read the inverted call tree | `samply record ./target/release/examples/row_fn_sweep s1_infallible_add_nonnull --sample-count 5000 --sample-size 1000` |

## Commands recorded for the original experiment

```bash
# flatc must match the flatbuffers crate (25.12.19). apt's 2.0.8 generates incompatible code.
curl -fsSL -o /tmp/flatc.zip https://github.com/google/flatbuffers/releases/download/v25.12.19/Linux.flatc.binary.clang++-18.zip
unzip -o /tmp/flatc.zip -d /tmp/flatc && install /tmp/flatc/flatc /usr/local/bin/flatc
# If an older flatc already ran, drop its stale build-script output first:
rm -rf target/release/build/vortex-{array,layout,file,ipc}-*

# Existing bench
cargo bench -p vortex-array --features unstable_row_fns --bench row_fn_output

# Temporary sweep (file lived at vortex-array/examples/row_fn_sweep.rs and was deleted afterwards)
cargo build --profile bench --example row_fn_sweep -p vortex-array --features unstable_row_fns
./target/release/examples/row_fn_sweep --test
./target/release/examples/row_fn_sweep --sample-count 100 --sample-size 50 > sweep.log
python3 fit.py sweep.log   # parses divan medians and prints the fixed / per-row table
```
