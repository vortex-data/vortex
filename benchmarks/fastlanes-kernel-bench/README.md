# fastlanes-kernel-bench

Microbenchmarks for the FastLanes 1024-element bit-packing kernels, with no
Vortex array, validity, patch, or allocation overhead included in the timed
region.

The crate **vendors** the relevant kernel files from the upstream
[`fastlanes` 0.5.0](https://crates.io/crates/fastlanes/0.5.0) crate verbatim
(`src/bitpacking.rs`, `src/ffor.rs`, `src/macros.rs`) plus the trimmed
`FastLanes` trait and helpers from `src/lib.rs`. Vendoring is intentional: we
want to be able to modify the kernel locally for experiments without affecting
the Vortex production path that still depends on the published crate.

## What is measured

For every `(unsigned type, bit width W)` in:

- `u8`  &times; `W` &in; `1..=8`
- `u16` &times; `W` &in; `1..=16`
- `u32` &times; `W` &in; `1..=32`
- `u64` &times; `W` &in; `1..=64`

three variants of decoding one 1024-element block are timed:

| name           | what it does                                                         |
|----------------|----------------------------------------------------------------------|
| `bare_unpack`  | `BitPacking::unpack::<W, B>(&packed, &mut out)` &mdash; decompress only |
| `unfused_for`  | the same, then a separate `for i in 0..1024 { out[i] = out[i].wrapping_add(reference) }` pass |
| `fused_for`    | `FoR::unfor_pack::<W, B>(&packed, reference, &mut out)` &mdash; the FoR reference application is fused into the same kernel via the upstream `unpack!` macro |

`bare_unpack` is the baseline. The `fused_for` vs `unfused_for` pair is the
real comparison: does fusing the wrapping-add into the unpack kernel beat
running it as a separate vectorisable pass over the output buffer?

### Why this is "runtime only"

- Every benchmark allocates `input`, `packed`, and `output` on the stack
  *outside* the `bencher.bench_local(|| ...)` closure.
- The closure body only calls the kernel (and, for `unfused_for`, the manual
  add loop). There is no `Vec` growth, no Vortex `Buffer` construction, no
  validity tracking, no patch handling.
- `divan` automatically computes the per-iteration time over a calibrated
  number of inner repetitions.

The kernels themselves are data-independent (no value-dependent branches), so
the choice of input pattern does not bias timings.

## Signed vs unsigned: one kernel + transmute is enough

Upstream `BitPacking` and `FoR` are only implemented for `u8`/`u16`/`u32`/`u64`.
The signed variants (`i8`/`i16`/`i32`/`i64`) deliberately reuse the same code:

1. Bit-packing is purely shift-and-mask; the bit pattern produced is
   identical regardless of whether the operands are interpreted as signed or
   unsigned.
2. `wrapping_add` / `wrapping_sub` on two's-complement integers produce the
   same bit pattern whether the inputs are `i32` or `u32`. So FoR with a
   negative reference works correctly under reinterpretation.

That is why the existing Vortex integration (see
`encodings/fastlanes/src/bitpacking/array/bitpack_compress.rs` &mdash;
`reinterpret_cast(parray.ptype().to_unsigned())`) just bit-casts the slice and
runs the unsigned kernel. The upstream `FastLanesComparable` trait in
`fastlanes/src/lib.rs` does the same with `core::mem::transmute`.

**Conclusion: do not duplicate kernels for signed types.** The unsigned
benchmark numbers below apply directly to the corresponding signed widths.
The signed types are therefore intentionally not benchmarked in this crate.

## Running

Run every case (360 benchmarks total &mdash; takes a while):

```bash
cargo bench -p fastlanes-kernel-bench
```

Filter by type or bit width:

```bash
# All u32 cases
cargo bench -p fastlanes-kernel-bench -- u32

# Just W=10 across all types
cargo bench -p fastlanes-kernel-bench -- '__w10$'

# Just the three variants of u64 W=33
cargo bench -p fastlanes-kernel-bench -- 'u64__w33$'
```

Speed up iteration with a smaller sample count:

```bash
cargo bench -p fastlanes-kernel-bench -- u32__w10 --sample-count 100
```
