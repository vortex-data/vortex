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

> **Use the helper script** (`scripts/bench.sh`) to compile the kernels with
> `target-cpu=native` and a single codegen unit. Plain `cargo bench` builds at
> the `x86-64-v1` baseline (SSE2 only) and leaves a large speedup on the table
> &mdash; see "Why the helper script matters" below.

Run every case (360 benchmarks total &mdash; takes a while):

```bash
./benchmarks/fastlanes-kernel-bench/scripts/bench.sh
```

Filter by type or bit width (filters are regexes against the function name):

```bash
# All u32 cases
./benchmarks/fastlanes-kernel-bench/scripts/bench.sh u32

# Just W=10 across all types
./benchmarks/fastlanes-kernel-bench/scripts/bench.sh '__w10$'

# Just the three variants of u64 W=33
./benchmarks/fastlanes-kernel-bench/scripts/bench.sh 'u64__w33$'

# Smaller sample count
./benchmarks/fastlanes-kernel-bench/scripts/bench.sh u32__w10 --sample-count 100
```

For a reproducible portable baseline (skylake-class instead of host CPU):

```bash
RUSTFLAGS_NATIVE='-C target-cpu=x86-64-v3' \
    ./benchmarks/fastlanes-kernel-bench/scripts/bench.sh
```

### Why the helper script matters

Profiling the binary built by plain `cargo bench` shows scalar SSE2 code, e.g.
the `<u32 as BitPacking>::unpack` body is a stream of:

```
movdqu  xmm1, [rdi+rax*1+0x80]   # 128-bit load
pand    xmm2, xmm0               # mask
psrld   xmm2, 0x8                # shift right
movdqu  [rsi+rax*1+0x280], xmm2  # 128-bit store
```

i.e. 4-wide u32 vectors with no AVX VEX-encoded ops at all. With the script's
`-C target-cpu=native` we get AVX2:

```
vmovdqu ymm3, [rdi+rax*1+0x80]   # 256-bit load
vpand   ymm4, ymm3, ymm0
vpshufb ymm4, ymm3, ymm1         # SSSE3/AVX2 byte permute
vmovdqu [rsi+rax*1+0x280], ymm4
```

i.e. 8-wide u32 vectors plus byte-shuffle. The fused FoR variant additionally
gets `vpaddd ymm, ymm, broadcast(reference)` interleaved with the unpack chain,
which the compiler can only do when the kernel body is in one codegen unit.

Measured medians on a Sapphire-Rapids-class Xeon @ 2.1&nbsp;GHz, 1024-element
block, in nanoseconds (lower is better):

| case             | SSE2 baseline | AVX2 + cgu=1 | speedup |
|------------------|--------------:|-------------:|--------:|
| u8  W=3 bare     |  21           |  17          | 1.24x   |
| u8  W=3 fused    |  47           |  17          | 2.76x   |
| u16 W=3 fused    |  46           |  45          | 1.02x   |
| u32 W=3 fused    |  92           |  85          | 1.08x   |
| u32 W=10 fused   | 140           |  77          | 1.82x   |
| u32 W=17 fused   | 135           |  77          | 1.75x   |
| u64 W=3 fused    | 198           | 148          | 1.34x   |
| u64 W=17 fused   | 202           | 172          | 1.17x   |
| u64 W=33 fused   | 226           | 185          | 1.22x   |

Headline: **with proper SIMD compile flags the fused FoR kernel is uniformly
the fastest option**, often by ~2x vs the unfused two-pass alternative. With
SSE2-only compilation the win is much smaller or even negative for narrow
types, which is misleading -- without the helper script you would conclude
that fusing FoR into the unpack barely matters.

Note: LLVM defaults to 256-bit (`ymm`) on this host even though `avx512f`
is available, because the default `prefer-vector-width` for a generic
`native` CPU model favours 256-bit to avoid the AVX-512 frequency
licence-domain dip on older Xeons. To force 512-bit, use
`RUSTFLAGS_NATIVE='-C target-cpu=sapphirerapids'` (or your CPU-specific
codename that sets `prefer-vector-width=512`).
