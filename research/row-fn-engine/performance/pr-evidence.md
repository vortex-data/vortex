# Evidence for the RowFn performance PRs

This branch retains the research, benchmark source, raw measurements, compiler excerpts, and
verification logs. It is an evidence archive and is not intended to merge into `develop`.
The two fix PRs contain only production changes and correctness tests. Each is stacked on a
benchmark-only PR so CI can compare the same cases before and after the change.

The [native CI comparison](ci-comparison.md) records the stacked PR results, including the
x86 Boolean multiversioned regressions that block merging that fix.

| PR | Evidence | Original measured snapshot |
| --- | --- | --- |
| [#9985](https://github.com/vortex-data/vortex/pull/9985), UTF-8 decode | [Benchmark and results](../../../benchmarks/row-fn-utf8/) | `6b507e8ac9012f58487431dea24ea33e142c5e00` |
| [#9986](https://github.com/vortex-data/vortex/pull/9986), Boolean retry | [Benchmark and results](../../../benchmarks/row-fn-bool/) | `5cdf87a89651d4ca0405afad3cff0e066f9db58d` |

The original snapshots remain ancestors of this branch. Each snapshot contains its independent
production change and its own benchmark package. The evidence branch tip contains both changes.
For an independent reproduction, create a worktree at the relevant snapshot and follow its
benchmark README. Do not use the combined branch tip as an independent PR measurement.

```bash
git worktree add --detach /tmp/row-fn-utf8-evidence 6b507e8ac9012f58487431dea24ea33e142c5e00
git worktree add --detach /tmp/row-fn-bool-evidence 5cdf87a89651d4ca0405afad3cff0e066f9db58d
```

The [full investigation](follow-up.md) retains the other candidates and rejected experiments.
The original local measurements are native ARM results. The newer CI comparison reports native
ARM and x86 separately. Neither establishes end-to-end query performance.
