# PCO numeric benchmarks

Run the compression and decompression cases with:

```sh
cargo bench -p vortex-pco --bench columns
```

Each case generates 65,536 values at startup:

- `NullablePeriodic`: `i % 101`, with every thirteenth position null. This is the
  nullable integer pattern used in the Vail diagnostic that motivated the PCO audit.
- `SmallIntegers`: deterministic pseudorandom integers from 0 through 100.
- `WideIntegers`: deterministic pseudorandom values spanning the i32 range.
- `RandomDoubles`: deterministic pseudorandom f64 values in `[0, 1)`.

The random cases use a fixed seed and a wrapping linear congruential generator, so
inputs are reproducible without data downloads or additional dependencies. These
are synthetic arrays, not extracted Hail columns or models of their distributions.

The benchmark explicitly calls PCO, using its default compression level and 8,192
values per page. It does not measure the default compressor's encoding selection.

Generation, fixture compression for decompression, and full round-trip checks of
values, validity, and dtype happen outside timing. Each timed operation gets a
fresh execution context. The existing CodSpeed encoding shard discovers the
benchmark target without workflow changes.
