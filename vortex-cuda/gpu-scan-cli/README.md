# gpu-scan-cli

A CLI tool for benchmarking CUDA-accelerated scans of Vortex files.

## What it does

1. Reads a Vortex file from disk
2. Recompresses it using only GPU-compatible encodings
3. Executes a full scan on the GPU via CUDA
4. Outputs tracing information about kernel execution times

## Usage

```bash
FLAT_LAYOUT_INLINE_ARRAY_NODE=true RUST_LOG=vortex_cuda=trace,info \
    cargo run --release --bin gpu-scan-cli -- ./path/to/file.vortex
```

Use `--json` for JSON-formatted trace output.
