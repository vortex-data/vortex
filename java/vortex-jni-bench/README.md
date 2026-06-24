# vortex-jni-bench

JMH microbenchmarks that stress the **vortex-jni read boundary** — JNI plus the Arrow C Data
Interface — which is the path an Iceberg `FormatModel` takes to read Vortex from the JVM.

`VortexJniReadBenchmark` writes a synthetic six-column table (2M rows: 2× int64, 2× float64,
2× Utf8View) and reads it back, consuming columns at the buffer level (numeric sums, view lengths,
null counts) so the numbers reflect format + boundary cost, not per-row Java allocation.

Each invocation scans the full 2M-row table, so `@OperationsPerInvocation(ROWS)` makes JMH report
**input rows scanned per second** directly (not scans/s that you have to convert by hand).

Benchmarks:

- `fullScan` — read all six columns.
- `projection` — native projection of `id,y` (two of six columns).
- `projectionControl` — full scan, but consume only `id,y` in Java (NO native projection).
- `selectiveFilter` — native filter `cat = 'alpha'` (~1/16 selectivity).
- `filterControl` — full scan, evaluate `cat = 'alpha'` in Java (NO native filter).

The **controls are the point**: `projection` vs `projectionControl` and `selectiveFilter` vs
`filterControl` do the same Java-side work with and without native pushdown, so the remaining
speedup isolates native projection / filter pushdown from "fewer vectors / fewer rows touched in
Java." Comparing a pushdown lane against `fullScan` alone conflates the two and overstates pushdown.

`@Setup` validates the generated file before any measurement (exact row count, `cat='alpha'` returns
exactly `ROWS/|CATS|`, projection schema is exactly `[id, y]`) and fails the trial otherwise — a
corrupt write or broken filter must not silently produce impressive throughput.

`ScanOptions` has no read-batch knob, and Vortex coalesces to ~64K-row read batches regardless of
the writer's chunk size, so the boundary is amortized over large batches by construction. Run the
`main` method to see the batch-granularity diagnostic across chunk sizes.

**Workload caveats** (it is synthetic, single-machine, warm-cache — directional, not a leaderboard):
`id` is sequential, `cat` is a periodic 16-value low-cardinality column (kept non-null so filter
selectivity is exactly 1/16), `tag` is high-cardinality with a 10% null rate, numerics use a fixed
seed. This shape is friendly to compression and pushdown; a less-compressible / higher-null workload
is future work. These lanes measure throughput and pushdown *through* the boundary, **not** the
boundary's overhead versus a native floor (that comparison is the v2 TODO below).

## Running

The benchmark **must** run against a `--release` native lib (the dev `makeTestFiles` task builds a
debug lib, which would make the numbers meaningless). The `jmh` task is **guarded** to fail unless
`VORTEX_SKIP_MAKE_TEST_FILES=true`, so a plain `./gradlew :vortex-jni-bench:jmh` cannot silently
rebuild and measure the debug lib. Build the release lib once, drop it into vortex-jni's resources,
then run:

```bash
# from repo root: build the release cdylib
cargo build --release -p vortex-jni
# place it for the host arch (example: macOS arm64)
cp target/release/libvortex_jni.dylib \
   java/vortex-jni/src/main/resources/native/darwin-aarch64/

cd java
VORTEX_SKIP_MAKE_TEST_FILES=true ./gradlew :vortex-jni-bench:jmh
```

Results land in `vortex-jni-bench/build/results/jmh/results.txt`. The JMH fork adds Arrow's
`--add-opens` flags via `@Fork(jvmArgsAppend=...)`.

Batch-granularity diagnostic:

```bash
VORTEX_SKIP_MAKE_TEST_FILES=true ./gradlew :vortex-jni-bench:jmhJar
java --add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED \
  -cp vortex-jni-bench/build/libs/*-jmh.jar dev.vortex.bench.VortexJniReadBenchmark
```

## TODO (v2)

These benchmarks measure absolute throughput and pushdown effectiveness *through* the boundary, not
the boundary's *overhead*. To quote a "<X% over native" figure, add a Rust criterion read of the
same file (scan → Arrow) as the native floor and compare.
