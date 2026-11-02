# Benchmarking

Vortex has two categories of benchmarks: microbenchmarks for individual operations, and SQL
benchmarks for end-to-end query performance.

## Microbenchmarks

Microbenchmarks use the Divan framework and live in `benches/` directories within individual crates.
Every microbenchmark binary **must** use `mimalloc` as its global allocator (see
[Use `mimalloc` as the global allocator](#use-mimalloc-as-the-global-allocator)).

Run microbenchmarks for a specific crate with:

```bash
cargo bench -p <crate-name>
```

## Best Practices

### Separate setup from profiled code

Always use `bencher.with_inputs(|| ...)` so fixture construction is excluded from timing:

```rust
bencher
    .with_inputs(|| bench_fixture()))
    .bench_refs(|(array, indices)| {
        array.take(indices.to_array()).unwrap()
    });
```

### Exclude `Drop` from measurements

Divan measures only the closure body, **not** the `Drop` of its return value.
Structure your benchmark so that expensive drops happen via the return value or
via bench_refs inputs.

- **Return the value** from the closure — Divan will drop it after timing stops:

  ```rust
  bencher
      .with_inputs(|| make_big_vec())
      .bench_values(|v| transform(v))  // drop of the result is NOT timed
  ```

- **Use `bench_refs`** — the input is dropped after the entire sample loop, not per-iteration:

  ```rust
  bencher
      .with_inputs(|| make_big_vec())
      .bench_refs(|v| v.sort())  // v is dropped outside the timed region
  ```

Structure your benchmark so that expensive drops happen via the return value or via `bench_refs` inputs.

### Black-box inputs to prevent compiler optimization

The compiler can constant-fold or eliminate work if it can prove that inputs are known at
compile time.

Values provided through `with_inputs` are automatically black-boxed by Divan — no action
needed:

```rust
// ✓ `array` and `indices` are automatically black-boxed by Divan
bencher
    .with_inputs(|| (&prebuilt_array, &prebuilt_indices))
    .bench_refs(|(array, indices)| array.take(indices.to_array()).unwrap());
```

### Captured variables

Variables captured from the surrounding scope are _not_ black-boxed. Wrap them with
`divan::black_box()` or pass them through `with_inputs` instead:

```rust
let array = make_array();

// ✗ `array` is captured — the compiler may optimize based on its known contents
bencher.bench(|| process(&array));

// ✓ Option A: pass through with_inputs
bencher
    .with_inputs(|| &array)
    .bench_refs(|array| process(array));

// ✓ Option B: explicit black_box on the capture
bencher.bench(|| process(divan::black_box(&array)));
```

### Return values and manual loops

Return values are automatically black-boxed. You only need explicit
`black_box` for side-effect-free results inside manual loops:

```rust
bencher
    .with_inputs(|| (&array, SESSION.create_execution_ctx()))
    .bench_refs(|(array, ctx)| {
        for idx in 0..len {
            divan::black_box(array.execute_scalar(idx, ctx).unwrap());
        }
    });
```

### Use deterministic, seeded RNG

Always use `StdRng::seed_from_u64(N)` for reproducible data generation:

```rust
let mut rng = StdRng::seed_from_u64(0);
```

### Parameterize with `args`, `consts`, and `types`

Use Divan's parameterization features and define parameter arrays as named constants:

```rust
const NUM_INDICES: &[usize] = &[1_000, 10_000, 100_000];
const VECTOR_SIZE: &[usize] = &[16, 256, 2048, 8192];

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE)]
fn my_bench<const N: usize>(bencher: Bencher, num_indices: usize) { ... }
```

### Keep per-iteration execution time under 1 ms

**1 ms is the maximum, not a soft target.** Each individual iteration of the benchmarked
closure must complete in less than 1 ms. This is to keep benchmarks snappy, locally and on
CI.

A benchmark that needs longer than that is measuring too much work at once. Shrink the
input size until a single iteration fits, split it into smaller parameterized cases, or
gate it with `#[cfg(not(codspeed))]` if it genuinely cannot be made to fit.

The number to check against the budget is the per-iteration time, not the time the whole
benchmark binary takes. CodSpeed reports exactly that: its performance report on a pull
request lists the per-iteration time under `HEAD` for every benchmark the pull request adds
or changes, so check any new benchmark there before merging.

### Keep per-iteration work above the harness floor

The budget has a floor as well as a ceiling. CodSpeed runs each benchmark once and adds a fixed
cost of roughly half a microsecond of reported time around the closure. A closure that does tens
of nanoseconds of real work, such as one small allocation or a fast path that finds nothing to
do, reports mostly that floor, and the floor moves by more than 10% between runs of identical
code. Such benchmarks flag regressions on pull requests that do not touch Rust at all.

Aim for at least a few microseconds of real work per iteration:

- When the operation itself is tiny, repeat it a fixed number of times inside the closure and
  black-box each result, as `vortex-buffer/benches/allocation.rs` does.
- Drop degenerate inputs, such as a zero-byte allocation or a compaction with nothing to move.
- Size inputs by bytes rather than element count so narrow and wide types land in the same
  range, as `vortex-array/benches/filter_fixed_width.rs` does.

Benchmarks tagged `#[cpu_features]` run on the walltime legs instead, where the floor is timer
resolution and per-iteration jitter. Give each sample of those at least tens of microseconds,
with more work per iteration or with a fixed `sample_size` (see
[How CodSpeed measures a benchmark](#how-codspeed-measures-a-benchmark)). Keep the working set
inside the L2 cache of the leg machines (1 MiB on the Graviton leg) when the benchmark is about
kernel code rather than memory bandwidth.

### Gate CodSpeed-incompatible benchmarks

Use `#[cfg(not(codspeed))]` for benchmarks that are incompatible with CodSpeed.

### Keep third-party and frozen baselines out of CodSpeed

A benchmark of code that Vortex does not own, such as an arrow-rs kernel over the same data, or
of a frozen copy of an old Vortex implementation, cannot regress because of a pull request, so a
change in its number is never actionable. On the walltime legs these baselines were among the
noisiest series in the suite. Keep them for local `cargo bench` comparisons, but gate them with
`#[cfg(not(codspeed))]` and leave them untagged, as `vortex-compute/benches/lane_kernels.rs` and
`vortex-buffer/benches/collect_bool.rs` do.

### How CodSpeed measures a benchmark

Divan runs the benchmark closure in _iterations_ and groups them into _samples_. The _sample size_
is the number of iterations in one sample. Divan times each sample as a whole and reports the time
per iteration.

CodSpeed measures a benchmark with one of two instruments. Benchmarks without `#[cpu_features]`
run in _simulation_. Benchmarks with `#[cpu_features]` run on the _walltime legs_.

**Simulation.** [CodSpeed's CPU simulation](https://codspeed.io/docs/instruments/cpu) runs each
benchmark **exactly once** and estimates CPU cycles from the instruction trace, including cache
and memory access costs. As a result:

- `sample_count` and `sample_size` have no effect.
- System load does not change the result, because the estimate does not come from a clock. Some
  benchmarks still switch between two values on code that did not change (see
  [Known unstable benchmarks](#known-unstable-benchmarks)).
- System calls are not measured. A benchmark that depends on I/O or kernel calls **must** use the
  [walltime instrument](https://codspeed.io/docs/instruments/walltime) or be gated with
  `#[cfg(not(codspeed))]`. The simulation job keeps mimalloc and glibc from returning freed
  memory to the OS, so that allocator system calls do not land in a measured iteration.

**Walltime legs.** A walltime leg is one CI job that measures wall-clock time on a dedicated
machine for one CPU feature set (AVX2, AVX-512 or NEON). `.github/workflows/codspeed.yml` defines
the legs. On each leg:

- Divan takes 1000 samples (`DIVAN_SAMPLE_COUNT` in the workflow).
- Divan picks the sample size from the first sample. If the first sample takes more than 100 times
  the timer precision (about 3 µs), divan keeps one iteration per sample. The first sample is cold
  and slow, so most walltime benchmarks run with one iteration per sample.
- CodSpeed compares the **fastest** sample of the pull request run with the fastest sample of the
  base run, not the median.

With one iteration per sample, the fastest sample is the single fastest iteration out of 1000, so
one unusually fast iteration decides the result. A benchmark close to the 3 µs limit can also get a
different sample size in each run, and each sample size reports a different time.

For a short walltime benchmark, set `sample_size` in `#[divan::bench]` so that one sample takes
tens of microseconds, as `vortex-buffer/benches/collect_bool.rs` and
`encodings/fastlanes/benches/bitpack_blocked.rs` do. CodSpeed then compares the average of several
iterations, and the sample size is the same in every run. Write the value as a literal: the
simulation build ignores the setting, so a named constant is unused under `--cfg codspeed`.

Do not set `DIVAN_SAMPLE_SIZE` in the workflow instead. The variable overrides the setting of every
benchmark, and divan keeps the inputs and outputs of a whole sample alive until the sample ends. A
large global sample size changes the memory behavior of the benchmarks that allocate large
outputs.

### Use `mimalloc` as the global allocator

Every microbenchmark binary **must** use `mimalloc` as its global allocator. This includes the
CUDA and DuckDB benchmarks. Put these lines in the main file of each benchmark target, and keep
them when you rewrite the file:

```rust
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
```

The cost of the system allocator depends on its state, which differs between runner images and
between runs. Allocation inside a timed region made several simulation benchmarks flip between two
values on pull requests that did not change the measured code. `mimalloc` still depends on allocator
state. Using it consistently removes differences caused by choosing different allocators, but does
not guarantee repeatable measurements. Each crate with benchmarks already has the `mimalloc`
dev-dependency.

## What Not to Do

The mistakes below have made CodSpeed report regressions on pull requests that did not change the
measured code. Such reports teach reviewers to ignore CodSpeed, and then real regressions pass
unnoticed. When a section above explains a rule, the rule links to that section.

- **Do not write a benchmark that does tens of nanoseconds of work, or that uses degenerate input
  such as a zero-byte allocation.** See
  [Keep per-iteration work above the harness floor](#keep-per-iteration-work-above-the-harness-floor).
- **Do not leave a walltime benchmark of a few microseconds to divan's sample-size tuning.** See
  [How CodSpeed measures a benchmark](#how-codspeed-measures-a-benchmark).
- **Do not judge a walltime benchmark by its median alone.** CodSpeed compares the fastest sample,
  so a benchmark with a stable median can still report a 12% change. See
  [How CodSpeed measures a benchmark](#how-codspeed-measures-a-benchmark).
- **Do not measure code that a pull request cannot change.** See
  [Keep third-party and frozen baselines out of CodSpeed](#keep-third-party-and-frozen-baselines-out-of-codspeed).
- **Do not leave one-time process setup to the first measured iteration.** In simulation, each
  benchmark runs one cold iteration. Lazy statics, such as the session id interner or the state
  that the first regex compilation creates, then initialize inside whichever benchmark runs
  first, often with system calls. Run the operation once in `main()` before `divan::main()`, as
  `vortex-array/benches/like.rs` and `vortex-array/benches/run_end_filter.rs` do.
- **Do not drop the `mimalloc` global allocator when you rewrite a benchmark file.** See
  [Use `mimalloc` as the global allocator](#use-mimalloc-as-the-global-allocator).
- **Do not read a one-time change in later benchmarks as a regression after you add or remove
  benchmarks.** All benchmarks of one binary run in one process and share its allocator state, so
  a change to the earlier benchmarks can move the later ones once.
- **Do not acknowledge a flagged benchmark in CodSpeed without a cause.** If the pull request
  cannot reach the flagged code, open the log of the CodSpeed job and compare the divan table of
  the run with the table of the base run. If the median did not move, the report comes from the
  fastest-sample comparison. Fix the benchmark, or exclude it from automatic runs with evidence
  in the table below.

### Benchmarks excluded from automatic runs

These cases remain available locally. They show unexplained variation on unchanged code,
so automatic reports cannot reliably distinguish regressions from noise. The cause of the Rust
cases is unknown. Larger iterations did not fix the NEON filter and take cases.

| Benchmark | Excluded instrument | Evidence |
| --- | --- | --- |
| `bitpacking_take`: `threshold[64, 80]` | NEON walltime | 1.4 to 2 times slower in 15% to 30% of runs. |
| `bitpacking_filter`: `filter[64, 80]` | NEON walltime | 1.4 to 2 times slower in 15% to 30% of runs. |
| `binary_ops`: `mul_u64_nonnull` | NEON walltime | About 20 or 40 µs. |
| `row_fn_utf8_input`: `starts_with[8 and 64, VarBin]` | AVX2 and AVX-512 walltime | 12% to 17% shifts between runs. |
| `row_fn_utf8_input`: `starts_with[64, VarBin]` | NEON walltime | 6.38% spread over 30 processes on three hosts with one binary. |
| `chunk_array_builder`: `chunked_opt_bool_canonical_into[(1000, 10)]` | Simulation | About 55 or 65.5 µs. |
| Python `test_scan_scalar_at[int]` and `test_repeated_scan_scalar_at[int]` | Simulation | Variable reactor polling inside measured iterations. |

The walltime exclusions match Divan tree paths in `codspeed.yml`. Simulation uses a smaller Rust
argument list and Python collection skips. Stable CPU variants and other arguments still run.

The historical Rust measurements come from [PR #10808](https://github.com/vortex-data/vortex/pull/10808).
Recent repeats include the string alert on the workflow-only
[PR #11095](https://github.com/vortex-data/vortex/pull/11095#issuecomment-6577817281).
For the Python repeated scalar scan on
[PR #11102](https://github.com/vortex-data/vortex/pull/11102), the head trace contains five reactor
polls and three clock calls absent from the base trace. Both also call `getpid` twice.
Memory mapping and `VORTEX_MAX_THREADS=0` do not remove those calling-thread runtime operations.
Simulation excludes syscall time and cannot measure their full cost. Keep the scan operation
intact for local measurements instead of replacing it with an in-memory lookup under the same name.

### Check repeatability before restoring a case

Use `scripts/codspeed_repeatability.py` to run independent benchmark processes. It records the
binary hash, command, compiler, CPU, environment, and per-process values. It reads instrument
results, not process elapsed time. For a built walltime binary on matching Linux hardware:

```bash
python3 scripts/codspeed_repeatability.py collect \
  --instrument walltime --binary /path/to/benchmark --output /tmp/host-a \
  --repeats 10 -- /path/to/benchmark --bench 'benchmark_filter'
python3 scripts/codspeed_repeatability.py summarize \
  /tmp/host-{a,b,c}/report.json
```

Each process writes into a fresh `CODSPEED_CARGO_WORKSPACE_ROOT`. Walltime uses the minimum sample,
matching CodSpeed's reported metric. Raw results and process logs remain beside each report.
For simulation, provide a command that runs the simulator and exports one JSON record per URI
under that directory. Select its metric with `--result-glob`, `--value-field`, and `--scale`.
Do not label native walltime as simulation.

Qualification requires at least 30 processes across three hosts, with a maximum spread of 5%.
First compare one binary, then repeat with independent builds using `--allow-rebuilds`. Compare a
temporary variant with 20% additional measured work to confirm that regressions remain visible.
A failing case stays excluded until a demonstrated fix passes these checks.

Fixed sampling also matters when one binary is repeatable on most runs. With 1000 samples per
process, `filter_avx2[80]` varied by 9.36% across 30 processes on three `c7i.metal-24xl` hosts.
Setting `sample_size=64` reduced that spread to 1.45% with the same executable. Across independent
base and head AVX2 builds, its worst spread was 4.66% over 60 processes on three hosts. The AVX-512
filter cases stayed within 1.78% over 30 processes. The take cases use `sample_size=16`, which
reduced AVX-512 `threshold[80]` from 8.26% to 1.69%.

The head measurements used artifacts from workflow run `41449104113`, source commit
`44bdefff6e69`. The independent base build came from run `41442182591`, commit `7c9bb58e39fe`.
The NEON measurements used the head run's artifact on three `c7g.metal` hosts. These are comparisons
of identical executables, not proof that unrelated source changes cannot affect code placement.
A separate calibration loop with 20% additional measured CPU work reported a 19.99% increase,
with less than 0.05% spread in both versions over 30 processes on three x86 hosts.

Record the exact base and head commits when investigating a report. An unexpected comparison base
is inconclusive. Separate instruction changes from cache, allocator, and syscall effects before
attributing a change to a kernel. Do not raise the repository's regression threshold to hide noise.

## SQL Benchmarks

SQL benchmarks measure end-to-end query performance across different engines and file formats.
The `vortex-bench` crate provides a common `Benchmark` trait that each benchmark suite
implements, defining its queries, data generation, and expected results.

Available suites include TPC-H, TPC-DS, ClickBench, FineWeb, and others. Each suite can be
run against multiple engines (DataFusion, DuckDB) and formats (Parquet, Vortex, Vortex Compact,
Lance, DuckDB native).

### Data Generation

Before running SQL benchmarks, test data must be generated:

```bash
uv run --project bench-orchestrator vx-bench prepare-data <benchmark> --format parquet,vortex
```

The data generator creates base Parquet data and converts it to each requested format. Scale
factors are configurable per suite (e.g. `--opt scale-factor=10.0` for TPC-H SF=10).

### Running SQL Benchmarks

SQL benchmarks can be run directly via their per-engine binaries:

```bash
cargo run --release --bin datafusion-bench -- <benchmark>
cargo run --release --bin duckdb-bench -- <benchmark>
```

## Orchestrator

The `bench-orchestrator` is a Python CLI tool (`vx-bench`) that coordinates running SQL
benchmarks across multiple engines, stores results, and provides comparison tooling.

See [`bench-orchestrator/README.md`](https://github.com/vortex-data/vortex/blob/develop/bench-orchestrator/README.md) for installation,
commands, and example workflows.

For CI, the reusable SQL workflow now drives `vx-bench` directly:

```bash
uv run --project bench-orchestrator vx-bench prepare-data tpch \
  --formats-json '["parquet","vortex","vortex-compact"]' \
  --opt scale-factor=1.0

uv run --project bench-orchestrator vx-bench run tpch \
  --targets-json '[{"engine":"datafusion","format":"parquet"},{"engine":"duckdb","format":"vortex"}]' \
  --output results.json \
  --no-build
```

## CI Benchmarks

Benchmarks run automatically on all commits to `develop` and can be run on-demand for PRs:

- **Post-commit** -- compression, string encoding, random access, and SQL benchmarks run on every
  commit to `develop`, with results uploaded for historical tracking.
- **Random access** -- `action/bench-random-access` runs only the random-access benchmark.
- **Random access (S3)** -- `action/bench-random-access-s3` runs the random-access benchmark
  against data uploaded to S3 instead of local disk.
- **Compression** -- `action/bench-compress` runs only the compression benchmark.
- **String encoding** -- `action/bench-string` runs only the string encoding benchmark.
- **GPU compression** -- `action/bench-gpu-compress` runs the allow-listed Vortex decompression
  cases on a GPU runner.
- **SQL** -- `action/bench-sql` runs the `pr` preset, a quick core subset of the SQL benchmarks
  that excludes `vortex-compact`, Clickbench Sorted, Appian, JOB, statpopgen, FineWeb on S3,
  TPC-H SF=10 on S3, and Vortex queries.
- **SQL Extended** -- `action/bench-sql-extended` runs the `pr-full` preset: every regular SQL
  benchmark, including `vortex-compact` and DuckDB-format targets.
- **SQL Compact** -- `action/bench-sql-compact` runs the `pr-compact` preset, which benchmarks
  `vortex-compact` plus Parquet control rows used to distinguish code changes from runner drift.
- **All CPU benchmarks** -- `action/bench-all` runs random access, compression, string encoding,
  and the `pr-all` SQL preset, the union of the `pr`, `pr-compact`, and `pr-full` coverage, without
  repeating shared jobs. Do not combine it with other benchmark labels;
  GPU compression is the only exception.

All CI benchmarks run on dedicated instances with the `release_debug` profile and
`-C target-cpu=native` to produce representative numbers.

Results can be viewed at [bench.vortex.dev](https://bench.vortex.dev).
