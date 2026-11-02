# Benchmarking

Vortex uses microbenchmarks for individual operations and SQL benchmarks for end-to-end query
performance.

## Microbenchmarks

Rust microbenchmarks use Divan and live in each crate's `benches/` directory. Run a crate's
benchmarks with:

```bash
cargo bench -p <crate-name>
```

### Separate setup from measured work

Use `with_inputs` to construct fixtures outside the timed closure. Return the operation's result so
Divan drops it after timing stops:

```rust
bencher
    .with_inputs(|| bench_fixture())
    .bench_refs(|(array, indices)| array.take(indices.to_array()).unwrap());
```

`bench_refs` also keeps input destruction outside the timed region. Any other value dropped inside
the closure contributes to the measurement.

Initialize one-time process state before `divan::main()`. Lazy session setup or regex initialization
inside the first measured iteration can dominate the operation, especially under simulation.

### Prevent the compiler from removing work

Divan black-boxes inputs from `with_inputs` and the closure's return value. Captured values do not
receive that protection. Pass them through `with_inputs` or use `divan::black_box` explicitly:

```rust
let array = make_array();
bencher
    .with_inputs(|| &array)
    .bench_refs(|array| process(array));
```

Inside a manual loop, black-box each result so the compiler cannot discard earlier iterations:

```rust
bencher.bench(|| {
    for value in divan::black_box(&values) {
        divan::black_box(process(value));
    }
});
```

### Choose reproducible inputs

Use a seeded generator such as `StdRng::seed_from_u64(0)` for random data. Parameterize distinct
workloads with Divan's `args`, `consts`, and `types`, and keep shared parameter lists in named
constants.

Choose inputs that exercise useful work. A zero-byte allocation or a compaction with nothing to move
can measure mostly the benchmark harness. For tiny operations, repeat a fixed amount of work and
black-box each result. Size inputs by bytes when comparing element types of different widths.

Keep each iteration below 1 ms. Reduce the input size or split a larger workload into parameterized
cases. For CPU kernels, keep the working set within the runner's cache unless memory bandwidth is
the intended measurement.

### Use `mimalloc` as the global allocator

Every microbenchmark binary must use `mimalloc`, including CUDA and DuckDB benchmarks:

```rust
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
```

Each crate with benchmarks already has the dev-dependency. A consistent allocator removes one source
of differences between binaries, but allocation can still depend on process state and earlier
benchmarks. Keep the allocator declaration when rewriting a benchmark file.

## CodSpeed

[`.github/workflows/codspeed.yml`](https://github.com/vortex-data/vortex/blob/develop/.github/workflows/codspeed.yml)
defines the measurement jobs. Rust CPU benchmarks use two instruments:

- [CPU simulation](https://codspeed.io/docs/instruments/cpu) runs benchmarks without
  `#[vortex_bench_support::cpu_features]`. It estimates cycles from one execution's instruction
  trace, including cache and memory access costs. Divan's sampling settings have no effect.
- [Walltime](https://codspeed.io/docs/instruments/walltime) runs CPU feature variants on dedicated
  AVX2, AVX-512, and NEON machines. It measures elapsed time, so sampling and machine state matter.

### Give short benchmarks enough measured work

Simulation has a fixed harness cost. Aim for at least a few microseconds of useful work per
iteration so that this cost does not dominate the result.

For walltime, Divan groups iterations into samples and reports time per iteration. CodSpeed compares
the fastest sample, so a stable median alone does not establish a repeatable benchmark. CI collects
1000 samples, but one unusually fast sample can still decide the result.

For a short walltime benchmark, set a fixed `sample_size` so each sample takes at least tens of
microseconds. This averages several iterations and avoids changes caused by Divan choosing a
different sample size between runs:

```rust
#[vortex_bench_support::cpu_features]
#[divan::bench(args = [64, 80], sample_size = 64)]
fn filter(bencher: Bencher, selected_per_chunk: usize) {
    // ...
}
```

Choose the sample size for the operation. Do not set a large global `DIVAN_SAMPLE_SIZE`: Divan keeps
sample inputs and outputs alive until the sample ends, so batching changes memory use for operations
that allocate. Use a literal in the attribute because simulation ignores the setting and can leave
a named constant unused.

### Keep incompatible benchmarks out of automatic runs

Simulation excludes system-call time. Use walltime for operations that depend on I/O or kernel calls.
Memory mapping and disabling background workers do not remove runtime polling on the calling thread.
Moving an operation entirely into memory changes what the benchmark measures.

Use automatic reports to track maintained Vortex entry points. Keep third-party kernels and frozen
implementations available for local comparisons.

Use `#[cfg(not(codspeed))]` for a Rust benchmark that cannot produce useful CodSpeed measurements.
When only one argument or CPU variant is unreliable, exclude that case and keep the others. Record
the reason beside the exclusion and keep the supporting measurements in the PR.

### Investigate reports on unchanged code

Check the exact base and head commits and the measurement instrument before attributing a change.
An old comparison base can include unrelated changes. For walltime, compare the Divan tables in the
job logs, including the minimum and the median. For simulation, inspect the callgraph for changes in
measured work, allocation, or runtime polling.

Repeat the same executable in fresh processes on matching hardware to check measurement noise, then
compare independent builds to check sensitivity to code placement. Keep commands, sampling settings,
and environment fixed. One repeatable executable does not establish that unrelated source changes
cannot affect its measurements.

If a case remains noisy after fixing setup and sampling, exclude it from automatic runs with the
evidence in the PR. Restore it after demonstrating repeatability and sensitivity to a real
regression. Do not raise the regression threshold to hide noise.

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
