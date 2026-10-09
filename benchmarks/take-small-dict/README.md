<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Small dictionary take on NEON

Run on AArch64 with NEON:

```sh
cargo run --release --manifest-path benchmarks/take-small-dict/Cargo.toml > results.jsonl
```

This standalone experiment compares checked scalar lookup, native NEON, and generic
`fearless_simd` 1.0.0 kernels. It does not add a dependency to the Vortex workspace.
For widths 1, 2, 4, and 8, it includes the production NEON vector loop directly. Widths
16 and 32 use an experimental native loop to measure whether extending dispatch would help.
It deliberately measures combinations outside production dispatch limits as well.

All variants use identical preallocated outputs and deterministic non-identity data. Timed
calls include table preparation, dispatch, bounds checks, lookup, and the scalar tail;
they exclude allocation and Vortex array execution. Each case rotates variant order over
31 samples. Each sample performs `clamp(2_000_000 / output_bytes, 1, 256)` iterations.
The JSONL output records the median, minimum, and maximum nanoseconds per call.

Cases cover 1-, 2-, 4-, 8-, 16-, and 32-byte records, dictionaries through 256 total bytes,
non-power-of-two cardinalities, uniform and 90%-zero-code distributions, and 64, 1,024,
65,536, and 1,000,000 rows. Startup checks cover unaligned inputs, vector/tail boundaries, and invalid codes;
each measured case also compares every variant with a scalar reference.

The generic kernel uses `Simd` functions dispatched through `Level::new()`. Precise byte
swizzles return zero outside each table bank, allowing multiple banks to be combined with
OR. For 2-, 4-, and 8-byte values, both native and generic kernels transpose the dictionary
into one table per byte position and interleave lookup results back into records. The generic
version uses wide swizzles and splits its vectors for the library's 128-bit interleaved stores.
The native version uses NEON TBL and ST2/ST4. The scalar baseline corresponds to the fallback algorithm, not the existing NEON
fast path for small byte dictionaries.

For end-to-end measurements, including allocation and dictionary execution, build and save
the Vortex benchmark executable separately for the baseline and candidate revisions:

```sh
cargo bench -p vortex-array --bench take_small_dict --no-run
<saved-benchmark-binary> --bench --sample-count 31 --sample-size 4
```

That benchmark checks decoded values before timing and covers all primitive integer and
floating-point types. Keep the expanded benchmark source identical between revisions.
Run saved binaries serially in alternating order. Report results separately from this
kernel-only experiment; their timings have different boundaries and data generators.

Production NEON dispatch accepts u8 codes and arbitrary cardinalities up to 256, 128, 64,
and 32 for 1-, 2-, 4-, and 8-byte values respectively. The 8-byte path requires at least
1,024 output rows to amortize setup; the other widths require 64. These physical widths
cover primitive integers, floats, and narrow decimal storage. Wider records and wider
code types retain their existing fallback. The combined PR also has AVX2 paths for up to
64 one-byte or 32 two-byte values, and AVX-512 VBMI paths for up to 256 one-byte or
128 two-byte values. This standalone benchmark measures NEON; use `dict_take_small`
for the existing x86 and generic SIMD comparisons.
