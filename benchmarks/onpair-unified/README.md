# Unified OnPair compression benchmark

This standalone crate compares the three final OnPair implementations on identical
`ONPAIR01` corpora. Each timed sample compresses the complete corpus as independent,
whole-row blocks and includes dictionary training, matcher construction, parsing, and
packing. Every retained output is decoded and checked row by row.

The compared implementations are the optimized Rust snapshot used by Vortex, the
original Rust paper implementation, and the original C++/Boost implementation. Earlier
experimental Rust ports were removed after they failed to beat the snapshot.

The JSON result reports compression latency and a canonical payload ratio:

```text
(dictionary bytes + 4-byte dictionary offsets + bit-packed codes + 4-byte row offsets)
/ input payload bytes
```

`run_matrix.py` executes one fresh benchmark process at a time. Use `--cpu` to pin
every cell to a single logical CPU. AVX-512 labels fail instead of silently falling
back when their dense vector path is unavailable. `paper-rust16` is the original
Rust implementation and only supports 16-bit codes.

Example:

```bash
RUSTFLAGS='-C target-cpu=native' cargo build --release \
  --manifest-path benchmarks/onpair-unified/Cargo.toml
python3 benchmarks/onpair-unified/run_matrix.py \
  --binary benchmarks/onpair-unified/target/release/onpair-unified-bench \
  --corpus-dir /path/to/corpora \
  --output /path/to/results.jsonl \
  --datasets fineweb-32mib \
  --blocks 2 4 8 16 --bits 12 16 --cpu 2
```
