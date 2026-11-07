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

Use `with_inputs` to construct fixtures outside the timed closure:

```rust
bencher
    .with_inputs(|| bench_fixture())
    .bench_refs(|(array, indices)| array.take(indices.to_array()).unwrap());
```

Initialize one-time process state before `divan::main()`. Lazy session setup or regex initialization
inside the first measured iteration can dominate the operation, especially under simulation, which
measures one cold iteration. See `vortex-array/benches/like.rs` for an example of initializing this
state before measurement.

### Exclude destruction when it is not part of the operation

Divan drops the closure's return value after timing stops. Return the result instead of discarding
it inside the closure:

```rust
bencher
    .with_inputs(|| make_big_vec())
    .bench_values(|v| transform(v));
```

Use `bench_refs` when the operation mutates an input in place. Divan keeps the input alive until the
sample ends, so its destruction is outside the timed region:

```rust
bencher
    .with_inputs(|| make_big_vec())
    .bench_refs(|v| v.sort());
```

Other values dropped inside the closure contribute to the measurement. Keep those drops only when
they are part of the cost being measured, such as an allocate-and-free benchmark.

### Prevent the compiler from removing work

The compiler can fold or remove work when it knows the inputs. Divan black-boxes inputs from
`with_inputs` and the closure's return value to prevent that optimization. Captured values do not
receive that protection:

```rust
let array = make_array();
bencher.bench(|| process(&array));
```

Pass the input through `with_inputs` instead:

```rust
let array = make_array();
bencher
    .with_inputs(|| &array)
    .bench_refs(|array| process(array));
```

An explicit `black_box` also protects a captured input:

```rust
bencher.bench(|| process(divan::black_box(&array)));
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

Use a seeded generator such as `StdRng::seed_from_u64(0)` for random data. Unseeded random data can
change the measured work between runs even when the code stays the same.

Choose inputs that exercise useful work. A zero-byte allocation or a compaction with nothing to move
can measure mostly the benchmark harness. For tiny operations, repeat a fixed amount of work and
black-box each result. Size inputs by bytes when comparing element types of different widths.

Keep each iteration below 1 ms so the suite remains practical to run. This is the cost of one
closure invocation, not the total time of the benchmark binary. Check the per-iteration `HEAD`
value in the CodSpeed report before merging a new benchmark. Reduce the input size, split a larger
workload into parameterized cases, or keep it local if it cannot fit this budget.

For CPU kernels, keep the working set within the runner's L2 cache unless memory bandwidth is the
intended measurement. `vortex-array/benches/filter_fixed_width.rs` shows how sizing by bytes keeps
different element widths above the harness cost without making the widest types too slow.

### Parameterize workloads

Use Divan's `args`, `consts`, and `types` for distinct workloads. Keep shared parameter lists in named
constants:

```rust
const NUM_INDICES: &[usize] = &[1_000, 10_000, 100_000];
const VECTOR_SIZE: &[usize] = &[16, 256, 2048, 8192];

#[divan::bench(args = NUM_INDICES, consts = VECTOR_SIZE)]
fn my_bench<const N: usize>(bencher: Bencher, num_indices: usize) { /* ... */ }
```

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

The simulation workflow disables mimalloc purging and limits glibc heap trimming to keep allocator
system calls out of measured iterations. These settings reduce noise but do not make allocation
independent of process history. A target's simulation benchmarks share one process, so adding,
removing, or reordering earlier benchmarks can change the allocator state seen by later ones.

## CodSpeed

[`.github/workflows/codspeed.yml`](https://github.com/vortex-data/vortex/blob/develop/.github/workflows/codspeed.yml)
defines the measurement jobs. Rust CPU benchmarks use two instruments:

- [CPU simulation](https://codspeed.io/docs/instruments/cpu) runs benchmarks without
  `#[vortex_bench_support::cpu_features]`. It estimates cycles from one execution's instruction
  trace, including cache and memory access costs. Divan's sampling settings have no effect.
- [Walltime](https://codspeed.io/docs/instruments/walltime) runs CPU feature variants on dedicated
  AVX2, AVX-512, and NEON machines. It measures elapsed time, so sampling and machine state matter.

### Give short benchmarks enough measured work

Simulation has a fixed harness cost. A single small allocation can take only tens of nanoseconds,
so most of its reported time can come from that cost. Aim for at least a few microseconds of useful
work per iteration. `vortex-buffer/benches/allocation.rs` repeats allocate-and-drop pairs a fixed
number of times and black-boxes each result to keep the operation visible above this cost.

For walltime, Divan groups iterations into samples and reports time per iteration. CodSpeed compares
the fastest sample, so a stable median alone does not establish a repeatable benchmark. CI collects
1000 samples, but one unusually fast sample can still decide the result.

Divan chooses the sample size from early timing measurements. A cold first iteration can be slow
enough that Divan keeps one iteration per sample even when later iterations are much faster. With
1000 single-iteration samples, the CodSpeed result is then the single fastest iteration. A case
near Divan's sampling cutoff can also get a different sample size in each run.

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

For example, the AVX2 bitpacked filter varied by 9.36% across 30 processes on three matching hosts
with one iteration per sample. Setting `sample_size = 64` reduced that spread to 1.45% with the same
executable. [PR #11110](https://github.com/vortex-data/vortex/pull/11110) records the configuration
and measurements.

Choose the sample size for the operation. Do not set a large global `DIVAN_SAMPLE_SIZE`: Divan keeps
sample inputs and outputs alive until the sample ends, so batching changes memory use for operations
that allocate. Use a literal in the attribute because simulation ignores the setting and can leave
a named constant unused.

### Keep incompatible benchmarks out of automatic runs

Simulation excludes system-call time. Use walltime for operations that depend on I/O or kernel calls.
Memory mapping and disabling background workers do not remove runtime polling on the calling thread.
Moving an operation entirely into memory changes what the benchmark measures.

Use automatic reports to track maintained Vortex entry points. Keep third-party kernels and frozen
implementations available for local comparisons, gated with `#[cfg(not(codspeed))]` and without
`#[vortex_bench_support::cpu_features]`. These baselines help compare implementations locally, but
unexplained changes in their automatic results do not identify a change in maintained Vortex code.

Use `#[cfg(not(codspeed))]` for a Rust benchmark that cannot produce useful CodSpeed measurements.
For a tagged benchmark, `#[vortex_bench_support::cpu_features(exclude = ["neon"])]` skips the named
walltime leg while keeping the other legs and local runs. When only one input case is unreliable,
filter the benchmark's argument list instead. Record the reason beside the exclusion and keep the
supporting measurements in the PR. Keep CI commands independent of individual benchmark names.

### Investigate reports on unchanged code

Check the exact base and head commits and the measurement instrument before attributing a change.
An old comparison base can include unrelated changes. For walltime, compare the Divan tables in the
job logs, including the minimum and the median. For simulation, inspect the callgraph for changes in
measured work, allocation, or runtime polling.

If the median stays stable while the minimum moves, investigate sampling before treating the alert
as an operation becoming slower. If several later benchmarks shift after an earlier benchmark is
added or removed, investigate shared allocator state. Neither pattern proves the cause on its own.
Simulation avoids clock noise, but changes in allocation or runtime polling can still change the
instruction trace on unchanged code.

Repeat the same executable in fresh processes on matching hardware to check measurement noise, then
compare independent builds to check sensitivity to code placement. Keep commands, sampling settings,
and environment fixed. One repeatable executable does not establish that unrelated source changes
cannot affect its measurements.

Record the executable identity, base and head commits, hardware, commands, and environment with the
results. Use the instrument's metric, such as minimum sample time for walltime, rather than process
elapsed time. Check spread across processes with `max / min - 1`. Include enough repeats and hosts
to expose occasional slow states, rather than relying on several samples from one process.

If a case remains noisy after fixing setup and sampling, exclude it from automatic runs with the
evidence in the PR. Before restoring it, demonstrate repeatability and check that a deliberate
increase in measured work remains visible. Stable results are not useful if the measurement no
longer responds to a real regression. Do not raise the regression threshold to hide noise.

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
  that includes TPC-DS SF=10 on NVMe and excludes `vortex-compact`, Clickbench Sorted, Appian,
  JOB, nested TPC-H, statpopgen, FineWeb on S3, TPC-H SF=10 on S3, and Vortex queries.
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
