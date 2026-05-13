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

### Cost of the FoR `wrapping_add` itself (`bare_unpack` vs `fused_for`)

Direct answer to "is the FoR `wrapping_add` more expensive than just
unpacking": **for narrow types it is essentially free; for wide types it
adds 10-50&nbsp;ns over a bare unpack**. The reason is µop-level
parallelism -- the broadcast-add fits in the unpack's existing
load/shift/mask pipeline for narrow lanes, but as the per-vector lane count
shrinks (32 for u8 &rarr; 16 for u16 &rarr; 8 for u32 &rarr; 4 for u64 at
256-bit ymm), the kernel runs out of slack and the `vpaddd` becomes visible.

Median best-of-3 runs (`--min-time 0.5`):

| case      | `bare_unpack` (no add) | `fused_for` (unpack + add) | add overhead |
|-----------|----------------------:|---------------------------:|-------------:|
| u8  W=1   |  17.6 ns              |  17.5 ns                   |   -0.2 ns    |
| u8  W=5   |  17.8 ns              |  20.0 ns                   |   +2.2 ns    |
| u8  W=8   |  17.8 ns              |  17.7 ns                   |   -0.1 ns    |
| u16 W=7   |  35.1 ns              |  43.2 ns                   |   +8.1 ns    |
| u16 W=16  |  34.7 ns              |  34.7 ns                   |    0.0 ns    |
| u32 W=8   |  76.8 ns              |  76.8 ns                   |    0.0 ns    |
| u32 W=17  |  77.1 ns              |  78.3 ns                   |   +1.3 ns    |
| u32 W=24  |  80.4 ns              |  89.5 ns                   |   +9.0 ns    |
| u32 W=32  |  98.6 ns              | 108.6 ns                   |  +10.0 ns    |
| u64 W=11  | 146.4 ns              | 159.6 ns                   |  +13.2 ns    |
| u64 W=33  | 154.3 ns              | 173.2 ns                   |  +18.9 ns    |
| u64 W=55  | 163.7 ns              | 210.7 ns                   |  +47.0 ns    |
| u64 W=64  | 153.5 ns              | 171.4 ns                   |  +17.9 ns    |

Compare with the cost of running the same `wrapping_add` as a *separate*
loop after a bare unpack (the `unfused_for` column from the next table):

* u32 W=17: separate add adds **+72&nbsp;ns**; fused add adds **+1.3&nbsp;ns**.
  Fusing recovers ~55x of the cost.
* u64 W=33: separate add adds **+166&nbsp;ns**; fused add adds **+19&nbsp;ns**.
  Fusing recovers ~9x of the cost.

So yes, the `wrapping_add` *is* extra work, but fusing it into the unpack
kernel lets it overlap with the existing memory + shift + mask µop chain,
turning a 30-170&nbsp;ns sequential dependency into a 0-50&nbsp;ns
co-scheduled instruction. The wider the type and the higher the bit width,
the more visible the residual cost.

### Fused vs unfused FoR (the headline comparison)

Measured medians on a Sapphire-Rapids-class Xeon @ 2.1&nbsp;GHz, AVX2 build via
`scripts/bench.sh` (1024-element block, sample-count=500):

| case      | without fused FoR (`unfused_for`) | with fused FoR (`fused_for`) | speedup |
|-----------|----------------------------------:|-----------------------------:|--------:|
| u8  W=3   |  33.6 ns                          |  17.0 ns                     | **1.97x** |
| u8  W=5   |  49.3 ns                          |  25.0 ns                     | **1.97x** |
| u16 W=11  |  66.9 ns                          |  39.3 ns                     | **1.70x** |
| u32 W=8   | 144.6 ns                          |  79.7 ns                     | **1.81x** |
| u32 W=17  | 149.0 ns                          |  75.9 ns                     | **1.96x** |
| u32 W=25  | 181.6 ns                          | 114.6 ns                     | **1.58x** |
| u64 W=11  | 340.6 ns                          | 178.6 ns                     | **1.91x** |
| u64 W=33  | 319.6 ns                          | 198.6 ns                     | **1.61x** |
| u64 W=55  | 346.6 ns                          | 227.6 ns                     | **1.52x** |

Fusing the wrapping-add into the unpack kernel is **1.5x–2x faster** than the
unfused two-pass version across every type and width tested. The win comes
from:

* one pass over the output buffer instead of two (better L1 reuse);
* the wrapping-add merges into the unpack's load-shift-mask µop chain rather
  than emitting an independent `vpaddd` loop after the kernel returns;
* the unpack kernel is `#[inline(never)]`, so without fusion the add loop has
  to start cold after a function-call boundary that drains register state.

These conclusions only hold with proper SIMD flags. At the SSE2 default the
fused-vs-unfused difference shrinks to near zero or even inverts for narrow
types -- see "Why the helper script matters" below.

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

## Signed vs unsigned FoR: one unsigned kernel covers both directions

Short answer: **no, you do not need a separate signed kernel** to support FoR
in either "direction" (values above the reference or values below). One
unsigned kernel handles signed types and bidirectional deltas via bit-level
transmute. This is proven by the round-trip tests in
`tests/signed_for_via_transmute.rs` (run with
`cargo test -p fastlanes-kernel-bench --release`).

Why this works:

1. **Bit-packing is shift-and-mask.** The bit pattern produced is invariant
   under signed-vs-unsigned reinterpretation; nothing in `pack` / `unpack`
   ever asks "is this number negative".
2. **`wrapping_add` and `wrapping_sub` on a `T`-bit integer produce identical
   bit patterns regardless of whether the operands are `iT` or `uT`.** Both
   are just modular arithmetic on the underlying bits, and two's-complement
   makes the modular ring the same in both signed and unsigned views.

So FoR encode (`packed = value - reference`) and FoR decode
(`value = packed + reference`) round-trip the bit pattern losslessly through
the unsigned kernel, even for `iT` inputs reinterpreted as `uT`.

### The "both directions" caveat

There are two distinct "directions" to be careful about. The kernel handles
the first cleanly; the second requires picking the reference correctly.

* **Direction 1 - encode / decode.** Encode runs `wrapping_sub`, decode runs
  `wrapping_add`. Same kernel covers both. Nothing to do.
* **Direction 2 - deltas above and below the reference.** `BitPacking::unpack`
  *zero-extends* the W-bit packed value before adding the reference. If you
  pick `reference = min(values)`, every delta is non-negative as a signed
  number, every delta fits in `W = ceil(log2(max - min + 1))` bits as an
  unsigned number, and the round-trip works directly. This is the
  conventional FoR rule and what Vortex's bitpacking path enforces.
  If you instead pick a non-min reference so deltas straddle zero, the W-bit
  zero-extended unpack will not reconstruct negative deltas correctly unless
  `W == T` (full width, i.e. zero compression). The fix is *not* a new
  kernel; it is to set `reference = min(values)`. The round-trip test
  `i32_round_trip_through_unsigned_kernel` demonstrates the canonical case;
  `i32_with_arbitrary_reference_round_trips_when_w_is_full_width`
  demonstrates the corner case where `W == T` lets a non-min reference work.

This matches what Vortex already does -- see
`encodings/fastlanes/src/bitpacking/array/bitpack_compress.rs`:
`reinterpret_cast(parray.ptype().to_unsigned())` and the upstream
`FastLanesComparable` trait in `fastlanes/src/lib.rs`, both of which run the
unsigned kernel after a `transmute`.

**Conclusion: do not duplicate kernels for signed types.** The unsigned
benchmark numbers in this crate apply directly to the corresponding signed
widths. Signed types are therefore intentionally not benchmarked here; they
would produce identical timings.

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
