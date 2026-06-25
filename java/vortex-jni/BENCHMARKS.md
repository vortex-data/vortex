# vortex-jni read-boundary benchmarks

Two benchmarks stress the **vortex-jni read boundary** — reading a Vortex file with column projection
and filter pushdown — from two sides that read the **same canonical file**:

- **`VortexJniReadBenchmark`** (JMH, Java) — reads *through* JNI + the Arrow C Data Interface, the
  path an Iceberg `FormatModel` takes to read Vortex from the JVM. Lives in this module's `jmh`
  source set (`src/jmh/java/dev/vortex/bench/`).
- **`read_boundary`** (Divan, Rust) — the **native floor**: the same `scan → Arrow` work entirely in
  Rust, with no JNI crossing and no Arrow C Data export. Lives in the `vortex-jni` *crate*
  (`vortex-jni/benches/read_boundary.rs`).

Because both read the exact same bytes, run the same lanes, and report **input rows scanned per
second**, the numbers are directly comparable: the gap between them is the cost of the JNI + Arrow C
Data boundary on top of the underlying format read.

## The lanes

- `fullScan` / `full_scan` — read all six columns.
- `projection` — native projection of `id,y` (two of six columns).
- `selectiveFilter` / `selective_filter` — native filter `cat = 'alpha'` (~1/16 selectivity).

`projection` vs `fullScan` isolates native **projection** pushdown (same rows, fewer columns
materialized); `selectiveFilter` vs `fullScan` isolates native **filter** pushdown (same scan, fewer
rows produced).

**Single-threaded vs pooled**: each lane runs in two threading modes. The single-threaded mode drives
the scan on the consuming thread only (the JNI default). The **pooled** mode adds a background worker
pool sized to `available_parallelism() - 1`, so the scan's split tasks run in parallel while the
consuming thread drains results. The Rust bench runs the Vortex→Arrow conversion inside the scan's
`map` (on those handle-spawned split tasks), so the pool parallelizes both the decode and the Arrow
conversion — column-heavy `full_scan` speeds up several-fold; lanes whose Arrow output is tiny
(`projection`, `selective_filter`) are unaffected. The Rust pooled lanes are `*_pooled`
(`full_scan_pooled`, …) backed by a `CurrentThreadWorkerPool`; the JMH side is parameterized by
`workerThreads` (`0` = single-threaded, `-1` = available parallelism) via `NativeRuntime`.

Each lane scans the full table and **materializes every result batch to Arrow, then sums the batch
row counts** (`getRowCount()` / `RecordBatch::len()`) — no per-value work. So the numbers reflect
scan + materialization + (for JMH) the boundary, not consume-side arithmetic. `@OperationsPerInvocation(ROWS)`
(JMH) and `ItemsCount::new(ROWS)` (Divan) both make this **rows scanned/s**. Vortex coalesces to
~64K-row read batches regardless of the writer's chunk size (the `VortexJniBatchDiagnostic` tool
prints this), so boundary cost is amortized over large batches by construction.

The JMH `@Setup` validates the file before any measurement (exact row count, `cat='alpha'` returns
exactly `ROWS/|CATS|`, projection schema is exactly `[id, y]`) and fails the trial otherwise — a
corrupt file or broken filter must not silently produce impressive throughput.

**View types**: both benches downgrade Utf8View → Utf8 (and BinaryView → Binary) when materializing
to Arrow — the JMH path because the production `scanArrow` does so for Spark and other view-less
consumers, and the Rust bench via a matching stripped target field — so the two materialize identical
Arrow types. (With row-count consumption neither actually touches the string columns.)

## The shared canonical file

Both benches read one canonical `.vortex` file at `target/vortex-jni-bench/data.vortex`, written by
the **Rust generator** (`vortex-jni/benches/canonical/mod.rs`, exposed as the `gen_bench_data`
example). It is a deterministic, six-column, 2M-row table (2× int64, 2× float64, 2× Utf8View) built
from a fixed formula — no RNG — so it is byte-reproducible and the two benches measure identical data.
Generation is idempotent: an existing file is reused, so whichever side runs first writes it and the
other reads it. Delete the file to regenerate.

## Running

**Java (JMH), through the boundary** — the `jmh` task wires everything up:

```bash
cd java
./gradlew :vortex-jni:jmh
```

It (a) builds a release-optimized native lib via `buildJmhNativeLib`
(`cargo build --profile release_debug -p vortex-jni`) and stages it on the runtime classpath — the
dev `makeTestFiles` debug build, which would make numbers meaningless, is skipped while the benchmark
is in the task graph; (b) generates the canonical file via `generateBenchFile`; and (c) passes the
Arrow `--add-opens` flags and `-Dvortex.jni.bench.file=<path>` into the forked JVM. Results land in
`vortex-jni/build/results/jmh/results.txt`.

**Rust (Divan), the native floor**:

```bash
# from repo root
cargo bench -p vortex-jni --bench read_boundary
```

The bench generates the canonical file on first run if absent. (`release_debug` is the `release`
profile plus full debug info, good for profiling; it lives in `target/release_debug/`, separate from
`target/debug` and `target/release`.)

**Batch-granularity diagnostic** — `VortexJniBatchDiagnostic`, a standalone tool (not a JMH
benchmark) that prints read-batch row counts per writer chunk size:

```bash
cd java
./gradlew :vortex-jni:batchDiagnostic
```

## Comparing the two

Line up the JMH `ops/s` against the Divan `rows/s` lane-for-lane. The Rust floor is faster on every
lane; the ratio is the JNI + Arrow C Data boundary overhead for that access pattern (it is largest
where the most data crosses the boundary, e.g. `fullScan`, and smallest where pushdown means little
crosses it, e.g. `projection`). Both are **synthetic, single-machine, warm-cache — directional, not a
leaderboard**: `id` is sequential, `cat` is a periodic 16-value low-cardinality column (non-null, so
filter selectivity is exactly 1/16), `tag` is high-cardinality with a 10% null rate. This shape is
friendly to compression and pushdown.

## Future work

- A less-compressible / higher-null workload (the current shape favors compression and pushdown).
- A wider/narrower-row and a multi-file variant.
