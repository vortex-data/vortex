# RowFn utf8 benchmark

Measures complete `Utf8Column::decode` calls for inline and external strings, including output
destruction. Both variants validate the same strings and sanitize the same validity domain.
The inputs are already resident in host memory, so this does not measure cold buffer access.

## Reproduce

Run these commands from this PR's checkout. The baseline uses the same benchmark source and lockfile.
The two worktrees differ only by this PR's production change and tests.

```bash
git worktree add --detach /tmp/row-fn-utf8-baseline 133aacdb7e
cp -R benchmarks/row-fn-utf8 /tmp/row-fn-utf8-baseline/benchmarks/
(cd /tmp/row-fn-utf8-baseline && \
  CARGO_TARGET_DIR=/tmp/row-fn-utf8-before cargo rustc \
  --manifest-path benchmarks/row-fn-utf8/Cargo.toml --locked --release \
  -- --emit=llvm-ir,asm,link)
CARGO_TARGET_DIR=/tmp/row-fn-utf8-after cargo rustc \
  --manifest-path benchmarks/row-fn-utf8/Cargo.toml --locked --release \
  -- --emit=llvm-ir,asm,link
python3 benchmarks/row-fn-utf8/run.py \
  /tmp/row-fn-utf8-before/release/row-fn-utf8-probe \
  /tmp/row-fn-utf8-after/release/row-fn-utf8-probe \
  /tmp/row-fn-utf8-results/run
python3 benchmarks/row-fn-utf8/summarize.py \
  'before=/tmp/row-fn-utf8-results/run-before-*.csv' \
  'after=/tmp/row-fn-utf8-results/run-after-*.csv'
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
