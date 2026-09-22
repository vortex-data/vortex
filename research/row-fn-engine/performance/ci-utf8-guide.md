# UTF-8 CI comparison after applying the benchmark guide

The benchmark in #9987 now uses four 8,192-row cases: inline and external strings, each with
all-valid or one-in-eight-null validity. The 1-row and 64-row cases were below the guide's
wall-time floor. The 16,384-row case was reduced to leave more room in the 1 MiB cache budget
for views, external string bytes, validity, and sanitized output views.

The timed closure now returns the decoded column, so Divan drops it after timing. Fixture and
execution-context setup remain outside timing. No minimum-duration override was added. The
guide requires tens of microseconds of work per wall-time iteration, with a hard ceiling below
1 ms per iteration. Increasing the total sampling duration would not fix an undersized case.

The production patch in #9985 is unchanged. `git range-diff` confirms that rebasing only updates
its benchmark parent. The benchmark source is identical on both stack layers.

| Layer | Source revision | CI run |
| --- | --- | --- |
| Benchmark parent, #9987 | `50995f872e11d5fcb60f4cc60c9ed7fcb3f4c524` | [Before](https://github.com/vortex-data/vortex/actions/runs/35772035049) |
| UTF-8 fix, #9985 | `82348005c4f8767650f9c5d61f29546fad5f6e16` | [After](https://github.com/vortex-data/vortex/actions/runs/35772036461) |

These runs use the existing native AVX2, AVX-512, and NEON jobs. AVX2 and AVX-512 use
`c7i.metal-24xl`, while NEON uses `c7g.metal`. Each target is compared separately with matching
hardware and flags. Each case uses 1,000 samples. Reproduce with the command in the
[original report](ci-comparison.md#correctness-and-reproduction), using the revisions above.

The raw job logs, benchmark stdout, workflow metadata, and transcribed timing summaries are
retained in [results/ci-utf8-guide](results/ci-utf8-guide/). Summary values come from Divan's
rounded output, not individual sample distributions. There is one before/after run per target,
so small movements do not establish a speedup or a compiler cause.

Targeted benchmark Clippy passed with `-D warnings`. Both revisions ran all four cases locally
with NEON tagging. Those local runs verify execution but are not the CI comparison. Their logs
are retained alongside the CI evidence. No production tests were rerun because the production
patch is unchanged from the previously tested revision.

## Native results

All six native timing jobs completed successfully. Across both revisions, the printed medians
range from 53.76 to 182.8 microseconds per iteration. Every printed slowest sample is below
1 ms, with a maximum of 204.6 microseconds. All cases clear the wall-time floor.

Divan medians in microseconds per 8,192-row decode:

| Target | Layout | Validity | Before | After |
| --- | --- | --- | ---: | ---: |
| AVX2 | External | All valid | 176.5 | 182.8 |
| AVX2 | External | One null in eight | 154 | 160.3 |
| AVX2 | Inline | All valid | 106.3 | 112.8 |
| AVX2 | Inline | One null in eight | 98.18 | 95.24 |
| AVX-512 | External | All valid | 179.7 | 177.1 |
| AVX-512 | External | One null in eight | 156.8 | 165.1 |
| AVX-512 | Inline | All valid | 116.6 | 109.4 |
| AVX-512 | Inline | One null in eight | 107.5 | 101.1 |
| NEON | External | All valid | 113.4 | 112.8 |
| NEON | External | One null in eight | 116.9 | 115.2 |
| NEON | Inline | All valid | 54.1 | 53.76 |
| NEON | Inline | One null in eight | 61.76 | 60.8 |

The results do not establish a consistent throughput improvement for #9985 at this batch size.
NEON is close across the pair. AVX2 and AVX-512 show mixed movements. The previous tiny-case
speedup claim is not evidence for these revised cases.

The [CodSpeed check report](results/ci-utf8-guide/codspeed-check.md) confirms comparison against
the new benchmark parent. Its own statistic flags AVX-512 inline, all-valid decoding as faster
at 93.8 to 83.0 microseconds. That statistic differs from the Divan medians above and does not
establish an improvement across all cases. The overall performance check fails on other
benchmark cases, including filtered output and `take_fsl`. Their movement has not been
attributed to this patch. No regressions were acknowledged or waived.

CodSpeed also warns about the custom wall-time environment and different environments among
its repository-wide comparisons. The retained native job logs identify matching hardware for
these UTF-8 pairs. More independent runs are needed before attributing the smaller movements.
