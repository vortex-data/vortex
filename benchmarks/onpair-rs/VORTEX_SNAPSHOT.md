# Vortex OnPair performance snapshot

This directory preserves the standalone `spiraldb/onpair` source from commit
`a957f3329118739d9ca1e0439c3de30ae09f9a9d` on the local branch
`ji/onpair-encode-performance`.

It contains the single retained implementation from the encoding-performance
investigation. Slower experimental direct tables, unconditional SIMD layouts,
alternate long-bucket packing, and timed output-packing variants are excluded.

The implementation changes are:

- a reserved, prehashed raw pair-frequency table for training;
- reserved training-time longest-prefix maps;
- a read-only two-byte short-prefix directory for eligible dictionaries;
- contiguous structure-of-arrays candidates with scalar and AVX2 matching;
- a compact per-prefix length bitmap that jumps over impossible candidates at
  short row tails while preserving mixed-length SIMD density;
- workload and dictionary-size gating for both static indexes;
- an 8 KiB membership filter plus prehashed table for small, completed
  long-prefix maps, with the mutable hash map retained for larger dictionaries.

The complete benchmark protocol, every per-corpus winner, aggregate results,
strict 16-bit comparison, and output-format caveat are in
[`docs/encoding-performance.md`](docs/encoding-performance.md).

This is deliberately a standalone snapshot rather than a Vortex workspace
member. It does not replace the `onpair` crates.io dependency used by
`vortex-onpair`.

Run its checks from the Vortex repository root with:

```shell
cargo test --manifest-path benchmarks/onpair-rs/Cargo.toml --lib
cargo clippy --manifest-path benchmarks/onpair-rs/Cargo.toml --all-targets --all-features -- -D warnings
```
