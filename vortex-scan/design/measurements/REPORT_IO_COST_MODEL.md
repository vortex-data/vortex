<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Cold SSD coalescing cost experiment

The offline cost prototype was built, tested and measured, then removed from the working patch.
It did not establish a worthwhile improvement over the existing IO-only configuration. The
independent Q23 validation showed a 2.38% reduction in median latency against the previous binary,
but only 7/12 paired wins and a 1.08% increase in paired geometric-mean latency. A descriptive
paired bootstrap spans both improvement and regression. The earlier IO-only changes remain;
this experiment contributes no additional claimed speedup.

[The follow-up investigation](REPORT_IO_COST_FOLLOWUP.md) tests identical controls, concurrent
read service, first-needed payload timing, and exact speculative-member reuse. It confirms
that small merges can help the backend while the scan policy still lacks a reliable win.

## Hypothesis and implementation

The proposal compared the marginal cost of enlarging a read with the expected cost of a later
read avoided. Fixed read cost was expressed in bytes: `H * bandwidth`. For cumulative additional
physical extent `extra`, newly covered announced payload `useful`, and an explicit probability
prior `p`, the prototype required:

```text
extra < p * (fixed_cost_bytes + useful)
extra <= speculative_byte_budget
```

One fixed-cost credit was available per coalesced read, rather than per announced segment.
Reachable demanded requests were chosen before speculative extensions, preserving their access
to the existing maximum coalesced size. Gaps and alignment prefixes counted toward the budget;
overlapping payload was not credited twice. Announcements already covered by the physical read
could share its result. Rejected announcements remained registered for later demand or pruning.
Counters measured accepted extensions, added extent, announced payload and rejected candidate
checks. A rejected candidate could be checked again during expansion.

This was an offline service-cost approximation. It did not learn demand probabilities, estimate
future physical read groups, model available parallel slots or predict driver critical paths.
A byte cap bounded added transfer, but was not a measured deadline guarantee. No remote or S3
latency model was fitted.

## Cold read probe

The SSD probe completed 276 cold batches: 60 single reads and 216 merged/serial/parallel pairs.
File-page residency was verified zero before each batch. It used persistent descriptors and
Python `pread`, including its allocation overhead, with warm metadata/device caches. This
calibrates an approximate service curve; it is not a Vortex query benchmark.

The fitted overhead was 101.98 microseconds, bandwidth 2854.59 MB/s and fixed-cost equivalent
291099.83 bytes. Residual error reached 19.55% across read-size medians. The serial cost rule
predicted the faster option in 9/12 pair cases, falling to 6/12 when the separate reads ran
concurrently. Always splitting would match the concurrent winner in 9/12 of these particular
cases, so the formula did not establish a useful latency predictor.

For two 64 KiB reads separated by 64 KiB, the merged batch took 159.55 microseconds and the
parallel batch 86.01 microseconds. Their process block-read accounting was 332 KiB and 128 KiB,
respectively. Request-size dependent kernel traffic and parallel completion both matter.

## Scan comparisons

All timings used DataFusion/Vortex V2, direct IO, descriptor reuse, native concurrency 32,
unchanged compute kernels/morsels/scheduling and one iteration in a fresh process. Dataset,
executables, temporary files and results used instance SSD `nvme0n1`. File pages were evicted
and verified nonresident. Metadata/device caches remained warm. Configuration order was balanced;
observed competing builds/benchmarks caused entire rounds to be retained, rejected and repeated.
No accepted outliers were removed. Native diagnostics ran separately after timings.

The initial screen used eight configurations, eight rounds and four queries: 256 accepted
timings and 32 native diagnostics. It compared full optional extent growth, demand-only growth,
16/64/256 KiB budgets, probability priors and demand-only with announcements disabled.
The central 64 KiB/50% setting improved the four-query geometric mean by 0.93%.

The probability sweep required correction: with a 291100-byte fixed cost, the 25%, 50% and 75%
priors all permit every extension within a 64 KiB budget. These are equivalent policies, not
independent probability decisions. Their differing timings illustrate scan/scheduling variance.
The second window added 5% and 10% priors so the cost predicate actually constrained expansion.
It completed 128 accepted timings and 16 native diagnostics. The best four-query geometric-mean
improvement was 1.07%, with two slightly slower query medians:

| Query | Demand-only ms | 64 KiB / 10% ms | Latency reduction | Paired wins |
|---|---:|---:|---:|---:|
| ClickBench Q23 | 440.585 | 422.203 | 4.17% | 7/8 |
| TPC-H Q19 | 69.367 | 68.690 | 0.98% | 7/8 |
| ClickBench Q24 | 35.554 | 35.688 | -0.38% | 3/8 |
| ClickBench Q19 | 34.179 | 34.369 | -0.55% | 4/8 |

The selected 10% candidate then received independent Q23 validation: four configurations in
twelve balanced rounds, 48 accepted timings and four native diagnostics. The previous executable
control uses the earlier IO-only binary with reuse/native32/demand-only enabled. It is already
optimized IO, and is not the untouched PR or V1.

| Configuration | Median ms | Median latency reduction vs previous binary | Paired wins vs previous |
|---|---:|---:|---:|
| Previous IO-only executable, demand-only | 431.137 | — | — |
| New executable, cost policy disabled | 430.471 | 0.15% | 6/12 |
| Cost policy, 64 KiB / 10% | 420.857 | 2.38% | 7/12 |
| 32 KiB cap, nonbinding 50% cost prior | 430.309 | 0.19% | 9/12 |

Pairing tells a different story from ratios of configuration medians. The cost candidate is
1.08% slower by paired geometric mean against the previous executable, 0.26% faster against the
new executable with the policy disabled, and 4.44% slower against the cap control. It wins only
4/12 pairs against that cap. Its descriptive 95% paired-bootstrap interval against the previous
executable spans -8.83% to +6.36% latency reduction. This is insufficient evidence to retain the
cost policy or claim that it outperforms a simple cap.

Announcements still showed a role in the initial fixed-configuration screen: disabling them
made Q23 4.37% slower, losing all eight pairs, while Q24 was 1.54% faster with 6/8 wins. Their
usefulness depends on the workload; their remaining sharing benefit was measured independently
of optional extent expansion.

## Artifacts and checks

Raw results, commands, source snapshots, calibration, analysis and the prototype restoration patch:
`/mnt/vortex-ssd/votex-4/results/coalesce-cost-20261008T124118Z/`.

The audit verifies 432 accepted cold timings, 52 separate native diagnostics and matching result
row counts. Five rejected rounds contain 20 retained samples. Row counts are not value checksums.
The prototype was not screened across all 65 queries, or on warm/remote data, because the focused
evidence did not justify promoting it.

Prototype executable:
`/mnt/vortex-ssd/votex-4/bin/datafusion-coalesce-cost`, SHA-256
`cee27145aea66c8cdb7d8ec2214de59611ac50b914d0378709a526777f6fa542`.
Previous executable:
`/mnt/vortex-ssd/votex-4/bin/datafusion-io-only`, SHA-256
`f1d851df83356b7d38ef86d5f30e489b89adb03cd841dd574d57d4657091bc54`.

The release build and affected-crate clippy passed. All 227 file tests passed with default
settings; 223 passed with the cost policy enabled, including actual array checks. Four tests
explicitly exercising the old optional-growth boolean were excluded in that second run and
passed in the default run. All 47 Python tool tests passed. Runtime tests used SSD executables,
working directories and temporary files. Ruff, ty, pinned Rust formatting and patch whitespace
checks passed. The archived restoration patch passes `git apply --check`.

The experimental Rust was restored byte-for-byte to the previous IO-only source. The benchmark
retains explicit previous-binary/demand-only/no-announcement controls and clears archived cost
flags from inherited environments. The final release executable was rebuilt after restoration;
its hash and final checks are recorded in `COMPLETED.json`.
