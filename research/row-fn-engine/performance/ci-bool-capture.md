# Boolean retry: x86 regression and callback capture

The large regression in #9986 comes from a scalar row loop in the multiversioned collector. The
shared executor changed inlining across code-generation units. Once the collector stayed out of
line, separate closure captures lost useful alias information for the failure accumulator. The
compiler then repeated input classification and failure stores for every row.

The first correction keeps the source, prepared value, callback, and failure accumulator in one
local state. The closure borrows that state mutably. With the original benchmark, optimized Linux IR
retains exclusive access to the state and vectorizes both x86 collector clones. Native CI with the
revised benchmark removes the AVX-512 regression, but an AVX2 regression remains. Its runtime
AVX-512 clone is still scalar. This first correction changes only the RowFn executor.

The second correction adds `#[inline(always)]` to `collect_bool_word_scalar`. The tail is already a
small generic loop, and keeping it inside the collector prevents callback state from escaping
through that call. With the revised benchmark, Linux IR again has `captures(none)` on the state
argument and vector comparisons in the runtime AVX-512 clone. This is an inlining change, not a
rewrite of the 64-row collection or packing algorithm.

## Controlled comparison

Each native case measures eight executions over the same 16,384-row input. Fixture and context setup
are outside timing. The closure returns all eight results, so output and error destruction also stay
outside timing. Both collector flags, both constant positions, and all four execution scenarios
remain covered. The old 64-row cases are removed.

The input working set is about 256 KiB for two columns, plus validity and small output buffers.
Repeating the batch gives fast paths enough work without increasing the input working set. The
benchmark guide requires tens of microseconds per wall-time iteration and a strict limit below 1 ms.
A longer total sampling period does not satisfy that requirement by itself.

| Revision | Source | Native CI run |
| --- | --- | --- |
| Benchmark parent | `2449ffba98047e03b6a4a9477fe6949ef7c89640` | [Baseline](https://github.com/vortex-data/vortex/actions/runs/35776817261) |
| Original production patch | `5aef26728ba69a87dfb7be1e0680fe577370a66f` | [Regressed](https://github.com/vortex-data/vortex/actions/runs/35776817970) |
| Callback capture correction, incomplete | `2ed0398caf41241ea4ca65fe41780195f823a033` | [Capture](https://github.com/vortex-data/vortex/actions/runs/35777490085) |
| Capture plus inline tail | `363bac72b5cc62b751347352e61494b27ef7970f` | [Corrected](https://github.com/vortex-data/vortex/actions/runs/35778932351) |

All four revisions have identical benchmark source. The regressed revision only rebases the original
production patch onto the revised benchmark parent. Each correction changes one file in its own
commit. The comparison isolates each correction from the benchmark protocol change.

The workflow uses Rust 1.98.0, LLVM 22.1.8, 16 code-generation units, no LTO, mimalloc,
`RUST_BACKTRACE=1`, and 1,000 samples per case. AVX2 and AVX-512 run on Intel Xeon Platinum 8488C
(`c7i.metal-24xl`). NEON runs on Graviton3 (`c7g.metal`). Each target is a separate comparison. The
AVX2 host also supports AVX-512, so runtime multiversioning can select the AVX-512 clone.

Raw logs, benchmark stdout, workflow metadata, and timing summaries are in
[results/ci-bool-capture](results/ci-bool-capture/). The summaries transcribe Divan's rounded
output. They do not contain individual sample distributions. One run per revision and target can
establish the large regression, but small movements still need repeated paired runs.

## Native results

All twelve native timing jobs completed successfully. Across the four revisions, every printed
slowest sample is below 1 ms. The maximum is 994.3 microseconds in the regressed revision. Final
medians range from 28.05 to 394.4 microseconds, with a maximum printed sample of 555.8 microseconds.
All retained cases clear the wall-time floor.

The tables report Divan medians in microseconds per **eight 16,384-row batches**. They are not
single-batch timings. The full 288 case results remain in `summary.csv` and `summary.json`.

Multiversioned collector:

| Target | Input | Scenario | Baseline | Regressed | Capture only | Final |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| avx2 | Columns | AllValid | 241.9 | 388.3 | 225.9 | 37.36 |
| avx2 | Columns | PartialAccepted | 111 | 398.7 | 232.4 | 44.53 |
| avx2 | ConstantLhs | PartialAccepted | 217.1 | 399.3 | 232.3 | 34.58 |
| avx2 | ConstantRhs | PartialAccepted | 206.8 | 416 | 232.3 | 34.46 |
| avx2 | Columns | NullOnlyFailure | 348.8 | 639.3 | 476.1 | 297.5 |
| avx2 | Columns | ObservableFailure | 291.6 | 448.5 | 286.1 | 96.04 |
| avx512 | Columns | AllValid | 38.81 | 397.5 | 37.95 | 38.58 |
| avx512 | Columns | PartialAccepted | 42.81 | 401.4 | 44.88 | 45.07 |
| avx512 | ConstantLhs | PartialAccepted | 206.9 | 402.4 | 37.42 | 37.25 |
| avx512 | ConstantRhs | PartialAccepted | 210.1 | 412.2 | 37.38 | 37.29 |
| avx512 | Columns | NullOnlyFailure | 280.5 | 644.2 | 285.9 | 286.7 |
| avx512 | Columns | ObservableFailure | 95.29 | 454.7 | 95.16 | 95.41 |
| neon | Columns | AllValid | 76.29 | 76.74 | 63.18 | 63.54 |
| neon | Columns | PartialAccepted | 68.32 | 84.2 | 69.6 | 71.01 |
| neon | ConstantLhs | PartialAccepted | 173.3 | 57.81 | 55.71 | 57.62 |
| neon | ConstantRhs | PartialAccepted | 177.9 | 56.79 | 51.99 | 53.76 |
| neon | Columns | NullOnlyFailure | 384.1 | 410.1 | 392.8 | 394.4 |
| neon | Columns | ObservableFailure | 182.5 | 181 | 164.1 | 164.3 |

Plain collector, accepted partial-validity attempts:

| Target | Input | Baseline | Regressed | Capture only | Final |
| --- | --- | ---: | ---: | ---: | ---: |
| avx2 | Columns | 110.9 | 104.2 | 104.2 | 104 |
| avx2 | ConstantLhs | 217 | 104.6 | 104.8 | 104.5 |
| avx2 | ConstantRhs | 206.7 | 104.5 | 104.8 | 104.4 |
| avx512 | Columns | 42.95 | 45.21 | 44.85 | 45.26 |
| avx512 | ConstantLhs | 207 | 39.25 | 39.16 | 39.08 |
| avx512 | ConstantRhs | 216.3 | 39.18 | 39.23 | 39.09 |
| neon | Columns | 70.75 | 72.06 | 69.81 | 71.15 |
| neon | ConstantLhs | 174.3 | 58.47 | 55.8 | 57.74 |
| neon | ConstantRhs | 178.2 | 53.88 | 52.22 | 54.61 |

The large x86 regressions are resolved. The final patch also improves constant-operand partial
execution on every target. This candidate is **confirmed and improved** for those paths. It does not
establish an improvement for every input shape. AVX-512 two-column partial execution moves from
42.81 to 45.07 microseconds, and NEON moves from 68.32 to 71.01 microseconds. Those smaller
differences need repeated paired runs before attributing them to this patch.

No final case has a median more than 10% above its baseline in this comparison. That is a
description of this run, not a general no-regression guarantee. The benchmark covers this Boolean
predicate and does not establish end-to-end query performance.

The final [CodSpeed check](results/ci-bool-capture/final-codspeed-check.md) compares the correct
stack heads and flags no `row_fn_bool_retry` regressions. Its overall check still fails on
`filtered_sink_i64_avx512[OneNullInEight]` and two `take_fsl` simulation cases. Those alerts remain
unattributed and are not waived. The check also warns about the custom wall-time hosts and different
environments in its repository-wide comparisons. The retained job logs identify matching hardware
and feature flags for the Boolean comparisons above.

## Compiler evidence

The original AVX-512 CI binaries provide a direct comparison before and after the production patch.
Their disassembly is retained under [codegen](results/ci-bool-capture/codegen/). The baseline
`execute_owned_bool` contains vector comparisons. The regressed binary calls an out-of-line
`collect_bool_words_avx512` with a scalar row loop.

The regressed loop at `0x34cf10..0x34cf75` repeatedly tests the source tags, loads individual
values, and updates the failure byte through a pointer. Its `orb %sil,(%rbx)` stays inside the row
loop. The loop writes byte results before vector packing. The cost is in that generated loop, not
merely the extra function call per batch.

Matching Linux-target builds reproduce the scalar loop. Their IR shows a `readonly` closure
aggregate containing separate pointers, including a pointer to mutable failure evidence. After the
correction, the closure argument points directly to the owned state with `noalias`. With the
original benchmark, the failure accumulator becomes an SSA reduction and both x86 clones contain
vector comparisons. With the revised benchmark, the runtime AVX-512 clone in the AVX2 build still
retains failure stores inside the row loop. The exact CI executable and a local build of that
benchmark agree. Its scalar-tail helper stays out of line, so the callback state escapes through the
tail call. The function no longer has `captures(none)`. The compiler still moves source
classification outside the loop, which explains only part of the improvement. Forcing the scalar
tail to inline restores `captures(none)` and removes the failure stores from the row loop. The
revised Linux AVX2 build then has vector comparisons in its runtime AVX-512 clone. The
multiversioned collector still stays out of line at the CPU feature boundary.

The final native AVX2 CI executable agrees with the cross-build. Its runtime AVX-512 collector at
`0x338b90` contains vector comparisons and is 4,194 bytes. The capture-only CI executable has a
739-byte scalar collector at `0x338cf0`. This confirms the transformation in the binaries that CI
measured, rather than inferring it from local timing.

The collector symbols become larger because they contain vectorized paths for the different constant
positions. In the original-benchmark cross-build, the AVX-512 clone grows from 494 to 4,343 bytes.
Those historical size measurements use the old harness. The native timings below use the revised
harness and establish runtime effects separately.

These observations identify the compiler consequence and a source correction. They do not identify
the exact inlining heuristic threshold that the shared helper crossed. They also do not justify a
general claim that the collector's temporary Boolean array always survives optimization.

## Rejected experiments and limits

Moving the source and callback into the closure while keeping a separate mutable failure capture did
not restore vectorization. Adding `#[inline]` to the AVX-512 collector helped a statically enabled
AVX-512 build, but did not solve the runtime feature boundary in an AVX2 build. Neither change is
retained.

A tuple containing the entire state restored vectorization. The retained version uses named fields
so the invariant is clearer. A per-row method on the state did not fix the remaining AVX2 loop. No
public API, callback order, failure aggregation, or selection behavior changes. Terminal decode
errors remain terminal.

An initial cross-build for `x86_64-apple-darwin` gave mixed results across feature variants. That
build is not evidence for Linux CI. The retained Linux IR uses the CI target and feature flags.
Those cross-built executables were not run on ARM. All reported runtime comparisons come from native
CI hosts.

## Correctness and reproduction

The corrected executor passed all 175 RowFn tests and all-target, all-feature `vortex-array` Clippy
with `-D warnings`. Existing tests cover empty and sliced inputs, word boundaries, constants in both
positions, all-valid, all-null, partial validity, observable failures, null-only failures, and
terminal decode errors. The revised benchmark passed targeted Clippy and all 24 cases in its native
ARM smoke run. The inline-tail change also passed all 14 native ARM Boolean packing tests, another
run of all 175 RowFn tests, and all-target, all-feature `vortex-buffer` Clippy.

The baseline workflow encountered an EC2 Spot interruption in the unrelated FastLanes and decimals
simulation shard. The capture-only workflow encountered the same infrastructure failure in its
strings, tensor, and spatial simulation shard. These interruptions are separate from the native
RowFn timing jobs.

For native timing, check out one of the revisions above and follow the existing workflow. Use the
same hardware and flags for every revision. For example, the AVX2 build is:

```bash
RUSTFLAGS='-C target-feature=+avx2 -C force-frame-pointers=yes' \
VORTEX_BENCH_VARIANT=avx2 VORTEX_BENCH_PREFIX='avx2::' VORTEX_BENCH_SUFFIX=_avx2 \
cargo codspeed build --locked -m walltime --profile bench -p vortex-array --bench row_fn_bool_retry

RUST_BACKTRACE=1 DIVAN_SAMPLE_COUNT=1000 \
bash scripts/bench-taskset.sh cargo codspeed run -- '.*::avx2::'
```

For optimized IR and assembly, use the same environment and replace the build command with:

```bash
cargo rustc --locked --target x86_64-unknown-linux-gnu -p vortex-array \
  --bench row_fn_bool_retry --profile bench -- --emit=llvm-ir,asm,link \
  -C llvm-args=-pass-remarks-missed=loop-vectorize
```

The compiler excerpts compare the original patch at `31c9d94ee356a9a7b59fe7e2dbe890dab254ea71` with
the isolated correction at `45223404924723ec1e886b02b1c440be0420fac5`. They use the original
identical benchmark on both sides. The native comparison above uses the revised identical benchmark
on all four revisions. The `linux-revised-state-*` and `linux-inline-tail-*` excerpts also use that
revised benchmark. On the ARM development host, Zig 0.15.1 supplies the Linux linker and C compiler.
This changes neither the Rust target nor its feature flags, but it is not the CI linker environment.
The complete build logs retain the target configuration and compiler remarks.

For the AVX-512 series, replace the variant, prefix, and suffix with `avx512` and use:

```bash
RUSTFLAGS='-C target-feature=+avx512f,+avx512bw,+avx512cd,+avx512dq,+avx512vl,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512vnni,+avx512bitalg,+avx512vpopcntdq,+avx512bf16,+avx512fp16 -C force-frame-pointers=yes'
```

Download the original CI executable with `gh run download RUN -n codspeed-cpu-benchmarks-avx512`.
Extract `codspeed/walltime/vortex-array/row_fn_bool_retry` from the artifact tarball. The retained
binary manifest records the run IDs and SHA-256 hashes. `llvm-nm --print-size` and `llvm-objdump
--disassemble --demangle` produce the symbol and assembly excerpts.
