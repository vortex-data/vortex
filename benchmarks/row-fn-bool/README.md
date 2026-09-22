# RowFn bool benchmark

Measures a deferred `i64` predicate through public RowFn execution, including output destruction.
The matrix covers two columns and constants in either position, accepted dense execution, partial
validity, null-only rejection, and observable errors. Partial validity excludes one row in eight.
Each shape uses both values of the multiversioned collector flag.

## Reproduce

Run these commands from this PR's checkout. The baseline uses the same benchmark source and lockfile.
The two worktrees differ only by this PR's production change and tests.

```bash
git worktree add --detach /tmp/row-fn-bool-baseline 133aacdb7e
cp -R benchmarks/row-fn-bool /tmp/row-fn-bool-baseline/benchmarks/
(cd /tmp/row-fn-bool-baseline && \
  CARGO_TARGET_DIR=/tmp/row-fn-bool-before cargo rustc \
  --manifest-path benchmarks/row-fn-bool/Cargo.toml --locked --release \
  -- --emit=llvm-ir,asm,link)
CARGO_TARGET_DIR=/tmp/row-fn-bool-after cargo rustc \
  --manifest-path benchmarks/row-fn-bool/Cargo.toml --locked --release \
  -- --emit=llvm-ir,asm,link
python3 benchmarks/row-fn-bool/run.py \
  /tmp/row-fn-bool-before/release/row-fn-bool-probe \
  /tmp/row-fn-bool-after/release/row-fn-bool-probe \
  /tmp/row-fn-bool-results/run
python3 benchmarks/row-fn-bool/summarize.py \
  'before=/tmp/row-fn-bool-results/run-before-*.csv' \
  'after=/tmp/row-fn-bool-results/run-after-*.csv'
```

Use unused temporary paths. Do not run builds or other benchmarks during the timed comparison.
The build emits optimized LLVM IR and assembly beside the executable.

## Protocol

The allocator is mimalloc 0.1.52. Release builds use optimization level 3, 16 codegen units, and no LTO.
Repository configuration enables frame pointers. The measurements use the native target without a
`target-cpu` override. The lockfile pins dependency versions.

Each case warms 200 calls, then doubles its iteration count until a block takes at least 2 ms.
Each process records 12 blocks. The runner alternates before/after process order across three pairs,
which gives 36 observations per case and variant. Each binary calibrates independently.
The summary reports pooled medians and inclusive quartiles, which are not confidence intervals.
Fixture construction, session creation, and argument allocation are outside the timed calls.
`RUST_BACKTRACE=0` applies to both variants.

[Raw results and environment](results/) record the independent PR comparison.
ARM measurements do not establish x86 performance. The host has no CPU affinity, frequency lock,
or thermal control. No end-to-end query improvement is established.
