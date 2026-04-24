# Benchmarking

Vortex has two categories of benchmarks: microbenchmarks for individual operations, and SQL
benchmarks for end-to-end query performance.

## Microbenchmarks

Microbenchmarks use the Divan framework and live in `benches/` directories within individual crates.

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
bencher.with_inputs(|| &array).bench_refs(|array| {
    for idx in 0..len {
        divan::black_box(array.scalar_at(idx).unwrap());
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

### Keep per-iteration execution time under ~1 ms

Each individual iteration of the benchmarked closure should complete in
**less than 1ms**. This is to keep benchmarks snappy, locally and on CI.

### Gate CodSpeed-incompatible benchmarks

Use `#[cfg(not(codspeed))]` for benchmarks that are incompatible with CodSpeed.

### CodSpeed's single-run model

CI benchmarks run under [CodSpeed's CPU simulation](https://codspeed.io/docs/instruments/cpu),
which executes each benchmark **exactly once** and estimates CPU cycles from the instruction
trace — including cache and memory access costs. This has several implications:

- **`sample_count` and `sample_size` have no effect** — CodSpeed always runs one iteration.
- **Results are deterministic** — the simulated cycle count is derived from the instruction
  trace, not wall-clock time, so there is no noise from system load or scheduling.
- **System calls are excluded** — CodSpeed only measures user-space code. Benchmarks that
  rely on I/O or kernel interactions will not reflect those costs, so they should use the
  [walltime instrument](https://codspeed.io/docs/instruments/walltime) or be gated with
  `#[cfg(not(codspeed))]`.

### Prefer `mimalloc` for throughput benchmarks

Throughput benchmarks should use `mimalloc` as the global allocator to reduce system allocator
noise:

```rust
use mimalloc::MiMalloc;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
```

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

- **Post-commit** -- compression, random access, and SQL benchmarks run on every commit to
  `develop`, with results uploaded for historical tracking.
- **PR benchmarks** -- triggered by the `action/benchmark` label. Results are compared against
  the latest `develop` run and posted as a PR comment.
- **SQL benchmarks** -- triggered by the `action/benchmark-sql` label. Runs a parametric matrix
  of suites, engines, formats, and storage backends (NVMe, S3).

All CI benchmarks run on dedicated instances with the `release_debug` profile and
`-C target-cpu=native` to produce representative numbers.

Results can be viewed at [bench.vortex.dev](https://bench.vortex.dev).
