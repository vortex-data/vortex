# Native CI comparison for the RowFn performance stacks

The Boolean measurements below are historical. The [capture investigation](ci-bool-capture.md)
identifies the x86 compiler cause and compares a correction with a revised benchmark protocol.

The UTF-8 measurements below are historical. The [revised UTF-8 comparison](ci-utf8-guide.md)
uses cases above the wall-time floor and excludes output destruction, as the benchmark guide
requires. Do not use the old tiny-case results as CI proof for the revised PR.

The benchmark-only parents establish a baseline with the same benchmark source as each fix.
All four CodSpeed workflows completed successfully on September 22, 2026, including all twelve
native timing jobs. Successful execution does not mean the performance changes are acceptable.

UTF-8 single-row decoding improves on every target. Larger batches have mixed results, including
an AVX-512 regression. The Boolean fix improves several constant-operand cases but introduces
large x86 multiversioned regressions. The Boolean fix is not ready to merge.

## Compared revisions

| Stack layer | Source revision | CI run |
| --- | --- | --- |
| UTF-8 benchmark, #9987 | `6c33a302726e85e78528719efd26627a352600ec` | [Before](https://github.com/vortex-data/vortex/actions/runs/35767825603) |
| UTF-8 fix, #9985 | `97f8c2638090009006a8c1b6e3c14ffcdd4c53ad` | [After](https://github.com/vortex-data/vortex/actions/runs/35768383383) |
| Boolean benchmark, #9988 | `a01de348c7d0093b150fc86bd9b24550ec83c876` | [Before](https://github.com/vortex-data/vortex/actions/runs/35767877888) |
| Boolean fix, #9986 | `31c9d94ee356a9a7b59fe7e2dbe890dab254ea71` | [After](https://github.com/vortex-data/vortex/actions/runs/35768386889) |

The retained CodSpeed check reports confirm that each fix compares against its benchmark parent.
The benchmark source is identical across each pair. Production patches are unchanged from the
previously tested versions; rebasing only adds the benchmark parent.

## Method and raw evidence

AVX2 and AVX-512 ran natively on Intel Xeon Platinum 8488C, `c7i.metal-24xl`. NEON ran natively
on Graviton3, `c7g.metal`. Each comparison uses matching target features and CPU families. The
targets are separate series; timings across ARM and x86 are not interchangeable.

The existing workflow uses Rust 1.98.0, the bench profile, mimalloc, `RUST_BACKTRACE=1`, and
`DIVAN_SAMPLE_COUNT=1000`. Fixture and execution-context setup are outside timing. Decoding,
execution, allocation, and output or error destruction are inside timing.

[Retained results](results/ci-stacks/) include complete job logs, benchmark stdout excerpts,
workflow metadata with job URLs, and CodSpeed check reports. [summary.csv](results/ci-stacks/summary.csv)
and [summary.json](results/ci-stacks/summary.json) transcribe all 360 reported case results into
nanoseconds. These are the rounded fastest, slowest, median, and mean values printed by Divan,
not individual sample distributions. Tables below use the printed Divan median. CodSpeed's
comparison statistic is different, so its percentages should not be mixed with these tables.

There is one before/after CI run per target, on separate runner instances. Some control cases
also move. CodSpeed flags the custom runners as an unknown wall-time environment, and its
repository-wide report includes unrelated cases compared across different environments. The
job logs identify the matching hardware for the RowFn cases. Repeated paired runs are still
needed to establish the smaller differences.

## UTF-8 decode

The source change removes an intermediate `VarBinViewArray`. It still calls the same validation
and sanitation routine. These timings compare equivalent checked decoding, including nullable
inputs; they do not compare against prevalidated raw views.

Single-row, all-valid medians, in nanoseconds per decode:

| Target | Inline before | Inline after | External before | External after |
| --- | ---: | ---: | ---: | ---: |
| AVX2 | 374.2 | 107.4 | 572.5 | 208.5 |
| AVX-512 | 394.7 | 155.0 | 589.5 | 286.5 |
| NEON | 412.8 | 119.8 | 535.6 | 254.6 |

At 16,384 rows, all-valid medians, in microseconds per batch:

| Target | Inline before | Inline after | External before | External after |
| --- | ---: | ---: | ---: | ---: |
| AVX2 | 212.1 | 205.0 | 352.1 | 364.5 |
| AVX-512 | 198.5 | 225.5 | 364.7 | 352.4 |
| NEON | 107.6 | 107.3 | 221.6 | 221.8 |

The small-batch improvement is confirmed in CI. There is no established large-batch improvement.
The AVX-512 inline movement, including nullable inputs at 190.4 to 207.8 microseconds, remains
unresolved. This report does not attribute it to a compiler transformation.

## Boolean dense retry

The source change overrides the default Boolean visitor in `ExecuteDenseWithRetry` and packs
accepted dense attempts directly. It preserves the requested collector flag. The benchmark
separates plain and multiversioned collectors, both constant positions, and four scenarios:
all-valid, partial validity with an accepted dense attempt, failure only in null payloads, and
an observable failure.

At 16,384 rows with partial validity and an accepted dense attempt, plain collector medians,
in microseconds per batch:

| Target | Columns before | Columns after | Constant LHS before | Constant LHS after | Constant RHS before | Constant RHS after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AVX2 | 14.20 | 13.30 | 29.56 | 13.33 | 26.14 | 13.30 |
| AVX-512 | 5.621 | 5.808 | 29.44 | 5.061 | 26.10 | 5.071 |
| NEON | 10.13 | 10.99 | 21.87 | 9.195 | 22.29 | 8.617 |

The same cases with the multiversioned collector:

| Target | Columns before | Columns after | Constant LHS before | Constant LHS after | Constant RHS before | Constant RHS after |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AVX2 | 14.22 | 51.78 | 29.53 | 52.45 | 26.17 | 52.16 |
| AVX-512 | 5.611 | 50.74 | 29.46 | 50.17 | 26.10 | 52.13 |
| NEON | 9.801 | 12.55 | 21.83 | 8.951 | 22.29 | 8.798 |

The x86 multiversioned regressions also affect all-valid and failure cases. AVX-512 two-column
all-valid execution moves from 4.913 to 50.48 microseconds, and null-only failure moves from
34.87 to 80.75 microseconds. Some AVX2 all-valid controls are already slow in the baseline;
the complete results retain those cases rather than excluding them.

The CI benchmark confirms gains in specific plain constant-operand cases, but it rejects a
general improvement claim for this patch. The x86 compiler cause has not been inspected, and
the earlier local ARM compiler evidence does not explain this result. Keep #9986 as a draft
until the multiversioned regressions are understood and resolved.

## Correctness and reproduction

Both benchmark parents pass targeted benchmark Clippy with `-D warnings`. Parent and fix
revisions pass the NEON-tagged Divan `--test` smoke run. The Boolean fixture checks expected
success or failure before timing. The unchanged production patches previously passed 78
UTF-8 and VarBinView tests, 175 RowFn tests, and all-target, all-feature `vortex-array` Clippy.
Correctness coverage includes empty and sliced inputs, constants in both positions, all-valid,
all-null, partial validity, null-payload sanitation, and terminal decode failures.

Check out either member of a pair and use the same benchmark target, flags, and hardware for
both revisions. The pinned workflow defines the full AVX-512 feature list and runner setup:

- [UTF-8 workflow](https://github.com/vortex-data/vortex/blob/97f8c2638090009006a8c1b6e3c14ffcdd4c53ad/.github/workflows/codspeed.yml)
- [Boolean workflow](https://github.com/vortex-data/vortex/blob/31c9d94ee356a9a7b59fe7e2dbe890dab254ea71/.github/workflows/codspeed.yml)

For example, on matching AVX2 hardware, build the UTF-8 benchmark with the workflow's settings:

```bash
RUSTFLAGS='-C target-feature=+avx2 -C force-frame-pointers=yes' \
VORTEX_BENCH_VARIANT=avx2 VORTEX_BENCH_PREFIX='avx2::' VORTEX_BENCH_SUFFIX=_avx2 \
cargo codspeed build --locked -m walltime --profile bench -p vortex-array --bench row_fn_utf8

RUST_BACKTRACE=1 DIVAN_SAMPLE_COUNT=1000 \
bash scripts/bench-taskset.sh cargo codspeed run -- '.*::avx2::'
```

Use `row_fn_bool_retry` for the Boolean pair. Reproducing the CI environment also requires the
workflow's dedicated-host setup and CodSpeed action. The raw logs record their exact commands.
Local smoke tests only verify benchmark execution; they are not the reported CI measurements.
