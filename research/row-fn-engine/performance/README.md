<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Performance findings

[Overview](../README.md)

The [September 22 follow-up](follow-up.md) records local fixes, reverted experiments, and new native
ARM measurements. The measurements below are the original research results.

RowFn adds batch work, and some execution paths also change the per-row algorithm. The evidence
supports several specific improvements. It does not establish one framework overhead percentage.

## What the measurements show

| Experiment | Result | What the comparison establishes |
| --- | --- | --- |
| Apple M4 Max, canonical unary `i64`, shared output collector. | About 101 to 104 ns of extra batch work at 1 to 1,024 rows. | The wrapper cost while retaining a framework collector in both paths. |
| Same ARM experiment, 16,384 rows. | RowFn 2.53 us, direct collector 2.43 us, direct iterator 1.44 us. | Output collection matters beyond batch dispatch. Both loops use vector arithmetic. |
| Same ARM fixture, allocation requests. | Reused arguments match both direct kernels. Fresh arguments add one 16-byte request. | No extra allocation request in the reused fixture. This does not measure allocator block sizes. |
| Xeon sweep, binary primitive operations, stock x86-64 target. | About 0.5 us per small dense invocation, about 1.3 us with array-backed validity. | Total invocation estimates in that fixture. Matched in-cache slopes resolve little additional dense per-row cost. |
| Xeon sweep, cheap UTF-8 predicate. | About 18 ns/row versus 0.56 ns/row over prevalidated raw views. | Decoding and validation dominate this comparison. The baseline has stronger input preconditions. |
| Xeon sweep, rejected evidence with `RUST_BACKTRACE=1`. | About 7.7 us inferred for a discarded rich error. | A candidate retry cost, estimated by subtraction. No isolated backtrace experiment established the split. |

The ARM run uses one codegen unit and a unary wrapping operation. The x86 run uses 16 codegen units,
different operations, and different baselines. Their numbers cannot be averaged, ranked as competing
implementations, or used to predict a portable adapter.

The ARM report retains the harness, raw observations, and compiler excerpts. The x86 report retains
result tables and the protocol. Its temporary sweep source and raw output were not retained.
The [evidence record](../evidence.md) states these different reproduction limits.

## What deserves further work

Source inspection supports repeated type validation, UTF-8 sanitation, rich retry errors, and
missing specialized Boolean visits on selected paths. Lazy validity operations and filtered input
execution also add costs outside the closure. These are optimization candidates, not verified fixes.

The [candidate analysis](optimization-candidates.md) gives each cause, required invariant, and
proposed comparison. The [pipeline trace](pipeline-trace.md) maps costs to execution scenarios.

No experiment here measures an extracted library, a real host adapter, or an end-to-end query.
Dense primitive results do not cover nullable strings, geometry decoding, or filtered execution.

## Detailed evidence

- [ARM measurements](local-measurements.md), [exact harness](reproduction.md), and
  [raw observations](raw-observations.md).
- [ARM compiler evidence](compiler-evidence.md), including the collector distinction and its limits.
- [x86 measurements](x86-measurements.md) and [the x86 measurement protocol](x86-measurement-plan.md).
- [Cost model](cost-model.md), with binding, decoding, rows, validity, output, and host costs separated.
- [Remaining measurement plan](measurement-plan.md), including selections and real host integration.
