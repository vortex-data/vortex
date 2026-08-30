# Encoding benchmark harness

This directory preserves the benchmark used for the results in
[`../docs/encoding-performance.md`](../docs/encoding-performance.md).

- `../src/bin/encode_bench.rs` measures Rust training, native parsing, verified
  packed parsing, and end-to-end compression. `full_fair_ms` is the comparison
  value.
- `original-rust` pins and measures the paper's original Rust implementation at
  its only supported 16-bit dictionary width and merge threshold 5.
- `onpair_cpp_bench.cpp` performs the same in-memory phases using the Boost-based
  C++ implementation and verifies decompression outside the timed regions.
- `build_cpp.sh` builds GCC and Clang binaries with the recorded optimization
  flags.
- `run_matrix.sh` runs both widths for optimized Rust and C++, plus the original
  Rust 16-bit mode, over every 32 MiB and 128 MiB corpus, pinned to one CPU. It
  defaults to median-of-five for 32 MiB and median-of-three for 128 MiB.

Inputs use the `ONPAIR01` framing described by the harness loaders. Files are
read and flattened before timing; only in-memory compression is measured.
