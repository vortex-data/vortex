# vortex-jni-bench

JMH microbenchmarks that stress the **vortex-jni read boundary** — JNI plus the Arrow C Data
Interface — which is the path an Iceberg `FormatModel` takes to read Vortex from the JVM.

`VortexJniReadBenchmark` writes a synthetic six-column table (2M rows: 2× int64, 2× float64,
2× Utf8View) and reads it back three ways, consuming columns at the buffer level (numeric sums /
null counts) so the numbers reflect format + boundary cost, not per-row Java allocation:

- `fullScan` — read all six columns.
- `projection` — read two of six (projection pushdown).
- `selectiveFilter` — `cat = 'alpha'` (~1/16 selectivity; filter pushdown).

`ScanOptions` has no read-batch knob, and Vortex coalesces to ~64K-row read batches regardless of
the writer's chunk size, so the boundary is amortized over large batches by construction. Run the
`main` method to see that batch-granularity diagnostic.

## Running

The benchmark **must** run against a `--release` native lib (the dev `makeTestFiles` task builds a
debug lib, which would make the numbers meaningless). Build it once and drop it into vortex-jni's
resources, then run with `VORTEX_SKIP_MAKE_TEST_FILES=true` so the debug rebuild doesn't clobber it:

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
