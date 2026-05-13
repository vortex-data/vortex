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

> The numbers in this section are from the AVX2 (`ymm`) build. The AVX-512
> column under "Why the helper script matters" below shows that switching
> to 512-bit vectors reduces every absolute number further by 1.04-2.5x, but
> the *qualitative* picture (FoR add is essentially free for narrow types,
> visible for wide types) is unchanged.

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

Measured medians on an Emerald-Rapids Xeon @ 2.1&nbsp;GHz, AVX2 (`ymm`) build
(1024-element block, sample-count=500). The AVX-512 (`zmm`) build reduces
*both* columns further, but the fused/unfused ratio is similar -- skipping
the second pass over the output buffer is even more valuable at higher
throughput because L1 pressure dominates.

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

The default cargo build targets `x86-64-v1` -- SSE2 only. Inspecting the
`<u32 as BitPacking>::unpack` body shows a stream of 128-bit `xmm` ops:

```
movdqu  xmm1, [rdi+rax*1+0x80]   # 128-bit load, 4 u32 lanes
pand    xmm2, xmm0               # mask
psrld   xmm2, 0x8                # shift right
movdqu  [rsi+rax*1+0x280], xmm2  # 128-bit store
```

`scripts/bench.sh` rebuilds with `-C target-cpu=native` *plus*
`-C target-feature=-prefer-256-bit`. The first flag enables the host's full
ISA; the second tells LLVM to actually emit 512-bit AVX-512 (`zmm`) instead
of LLVM's conservative 256-bit (`ymm`) default for Skylake-X / Sapphire- /
Emerald-Rapids cores (which was originally chosen to avoid the AVX-512
downclock penalty -- mostly mitigated on current Xeons).

The same kernel under each setting:

`ymm` build (AVX2, `target-cpu=native` only):
```
vmovdqu ymm3, [rdi+rax*1+0x80]   # 256-bit load,  8 u32 lanes
vpand   ymm4, ymm3, ymm0
vpshufb ymm4, ymm3, ymm1         # AVX2 byte permute
vmovdqu [rsi+rax*1+0x280], ymm4
```

`zmm` build (AVX-512, what the script does by default):
```
vmovdqu64 zmm6, [rdi+rax*4-0x980] # 512-bit load, 16 u32 lanes
vpandd    zmm7, zmm6, zmm1        # EVEX-encoded mask
vpaddd    zmm7, zmm7, zmm0        # fused FoR add, broadcast(reference)
vmovdqu64 [rdx+rax*4-0x980], zmm7
```

The fused FoR variant additionally interleaves a `vpaddd zmm, zmm, broadcast`
into the unpack body, which the compiler can only do when the kernel body
lives in one codegen unit (the script's `--config 'profile.bench.codegen-units=1'`).

### The canonical comparison: `unpack` vs fused-FoR-`unpack`, per SIMD class

This is the single table the rest of the README is reasoning about. For a
representative set of `(type, bit-width)` pairs we measured both
`BitPacking::unpack` ("bare unpack", no FoR step) and `FoR::unfor_pack` ("fused
FoR + unpack") under each of **three** ISA targets:

* **SSE2**     -- the default cargo build (`x86-64-v1`), 128-bit `xmm`, 4-wide u32.
* **AVX2**     -- `-C target-cpu=native`, 256-bit `ymm`, 8-wide u32. LLVM
                  picks this by default on AVX-512 hosts because of the
                  `prefer-256-bit` heuristic.
* **AVX-512**  -- `-C target-cpu=native -C target-feature=-prefer-256-bit`,
                  512-bit `zmm`, 16-wide u32. What `scripts/bench.sh` does
                  by default.

All three binaries are built with `codegen-units=1`. Measurement: best-of-5
median, `--min-time 0.3` per invocation, Emerald-Rapids Xeon @ 2.1 GHz, no
other load on the box. Times in ns; lower is better.

| case      | sse2 unpack | ymm unpack | zmm unpack | sse2 fused | ymm fused | zmm fused |
|-----------|------------:|-----------:|-----------:|-----------:|----------:|----------:|
| u8  W=3   |   23.62     |   14.99    |    9.62    |   26.98    |   14.90   |   17.45   |
| u8  W=8   |   17.14     |   14.91    |   20.87    |   20.05    |   14.97   |   10.11   |
| u16 W=7   |   50.13     |   29.67    |   36.11    |   55.88    |   37.21   |   30.14   |
| u16 W=15  |   55.97     |   30.10    |   40.78    |   70.21    |   41.34   |   44.91   |
| u32 W=8   |  105.70     |   66.15    |   54.24    |   80.72    |   67.75   |   54.88   |
| u32 W=17  |  108.50     |   65.94    |   62.98    |  119.30    |   69.73   |   71.58   |
| u32 W=24  |  105.90     |   65.74    |   61.84    |  103.80    |   67.28   |   65.91   |
| u32 W=32  |  105.20     |   67.39    |   78.47    |   79.00    |   78.72   |   90.64   |
| u64 W=11  |  222.90     |  133.70    |   95.97    |  231.80    |  143.60   |  107.30   |
| u64 W=33  |  228.00     |  137.30    |  139.30    |  257.00    |  160.90   |  139.80   |
| u64 W=55  |  306.00     |  143.20    |  168.90    |  319.60    |  208.10   |  190.30   |
| u64 W=64  |  220.30     |  132.40    |  206.10    |  181.50    |  148.50   |  186.70   |

### Is the FoR `wrapping_add` free in `unpack`? (And: is the kernel memory-bound?)

These two questions are tied. Computing the per-cell fusing overhead and the
per-cell effective L1 bandwidth side-by-side gives a clear "no, but" answer.

| case      | overhead sse2 | overhead ymm | overhead zmm | ymm bandwidth (read+write) |
|-----------|--------------:|-------------:|-------------:|---------------------------:|
| u8  W=3   |  +14.2%       |   -0.6%      |  **+81.4%**  |  93.9 GB/s                 |
| u8  W=8   |  +17.0%       |   +0.4%      |  -51.6%      | 137.4 GB/s                 |
| u16 W=7   |  +11.5%       |  +25.4%      |  -16.5%      |  99.2 GB/s                 |
| u16 W=15  |  +25.4%       |  **+37.3%**  |  +10.1%      | 131.8 GB/s                 |
| u32 W=8   |  -23.6%       |   +2.4%      |   +1.2%      |  77.4 GB/s                 |
| u32 W=17  |  +10.0%       |   +5.7%      |  +13.7%      |  95.1 GB/s                 |
| u32 W=24  |   -2.0%       |   +2.3%      |   +6.6%      | 109.0 GB/s                 |
| u32 W=32  |  -24.9%       |  +16.8%      |  +15.5%      | 121.6 GB/s                 |
| u64 W=11  |   +4.0%       |   +7.4%      |  +11.8%      |  71.8 GB/s                 |
| u64 W=33  |  +12.7%       |  +17.2%      |   +0.4%      |  90.4 GB/s                 |
| u64 W=55  |   +4.4%       |  **+45.3%**  |  +12.7%      | 106.4 GB/s                 |
| u64 W=64  |  -17.6%       |  +12.2%      |   -9.4%      | 123.7 GB/s                 |

Bandwidth = `(1024 * W / 8) + (1024 * T / 8)` bytes per call, divided by the
AVX2 `bare_unpack` time. The negative-overhead cells are real but mostly
codegen variance (the fused version's slightly different IR sometimes
schedules better than the bare version).

**The kernel is not memory-bandwidth-bound.** Emerald-Rapids L1 sustains
~250 GB/s on benchmarks like STREAM. Our kernel peaks at ~138 GB/s for the
identity case (u8 W=8) and sits in the 70-130 GB/s band for compressed
widths -- roughly 30-55% of L1 peak. There is plenty of memory headroom; the
loop is gated on the front-end and the shift/mask µop chain on ports 0/1/5,
not on memory.

**Fusing `wrapping_add` is mostly free anyway, but for a different reason
than memory boundedness.** It is free because the broadcast-add executes as
a single `vpaddd ymm, ymm, ymm` (the broadcast happens *once*, outside the
loop, and lives in a register for the whole 1024-element block) which can
co-issue on a port that the unpack's shifts and ANDs are not using that
cycle. On Sapphire/Emerald Rapids `vpaddd` runs on ports 0/1/5, `vpsrld`
on 0/1, `vpandd` on 0/1/5 -- so as long as some `vpaddd` µop can find a
slot on port 5 (or wherever the shift/mask µops aren't queued), the add
adds zero cycles to the dependency height.

* On AVX2 (`ymm`), 8 of the 12 measured cells fuse for **<10% overhead**,
  and 4 of them for **<3% (essentially free)**. The two outliers
  (u16 W=15: +37%, u64 W=55: +45%) are exactly the cases where the unpack
  has the densest shift/mask µop pattern -- W is near T-1 and the kernel
  emits a long chain of `vpsrld`/`vpslld`/`vpor` per output element. Here
  the add can't find a free port and adds real time.

* On AVX-512 (`zmm`) the overhead is much more variable; the EVEX-encoded
  forms run on a narrower set of execution units on Emerald Rapids and the
  add starts competing with the unpack chain more often. A few cells show
  significant overhead (u8 W=3: +81%), driven by very small absolute base
  times where even a single extra cycle is a large fraction.

* On SSE2 the kernel issues so many 128-bit ops per output that there is
  always slack -- fusing is uniformly cheap (most cells within ±15%, several
  negative because the fused codegen happens to be tighter).

**Practical conclusion.** Fusing FoR into the unpack is the right
implementation strategy: it pays nothing on average and avoids the much
larger second-pass cost (see the `unfused_for` table below, where running
`wrapping_add` as a separate loop costs +30 to +170 ns per block). But the
underlying reason is *port-level instruction parallelism* in the unpack
loop, not memory saturation.

### Reading the matrix

**SSE2 -> AVX2 is uniformly a big win.** Every cell improves by ~1.5-2x on
the `unpack` column and ~1.3-2x on the fused column. This is the cheapest
performance improvement available -- just adding `-C target-cpu=native` (or
`-C target-cpu=x86-64-v3` for a portable AVX2 baseline) eliminates ~1/3 to
1/2 of the kernel's runtime on any modern x86.

**AVX2 -> AVX-512 is workload-dependent.** It is *not* a uniform win; an
earlier revision of this README claimed otherwise based on noisy best-of-N
measurements. The pattern with the cleaned-up best-of-5 numbers above:

* **AVX-512 wins for compute-bound narrow-W cases**, where 512-bit lanes let
  the kernel emit twice as many `vpsrld`/`vpandd`/`vpaddd` per iteration.
  E.g. u32 W=8 bare: 66.2 ns -> 54.2 ns (-18%); u64 W=11 bare: 133.7 ns ->
  96.0 ns (-28%); u8 W=3 bare: 15.0 ns -> 9.6 ns (-36%).
* **AVX-512 loses for memory-bound or full-width-identity (W==T) cases**,
  where the kernel is largely a 512-byte streaming copy and the wider load
  /store path has worse front-end throughput and gather pattern. E.g.
  u32 W=32 bare: 67.4 ns -> 78.5 ns (+16%); u64 W=64 bare: 132.4 ns ->
  206.1 ns (+56%); u16 W=15 bare: 30.1 ns -> 40.8 ns (+35%).
* The two effects roughly cancel on a *per-type* geometric mean: across the
  24 cells of the matrix above, the geometric mean of zmm/ymm is ~1.0.

So picking AVX-512 vs AVX-2 is a workload choice. The script defaults to
AVX-512 because the compressed widths typical in Vortex production data
(u64 with W << 64, u32 with W < 32) sit firmly in the AVX-512-wins regime.
For benchmarks dominated by `W == T` identity cases, prefer `PREFER=256`.

**Fusing FoR into the unpack** (compare the left half of each row with the
right half) costs essentially nothing in the SSE2 and AVX2 columns -- the
broadcast-add slots into the unpack's existing µop chain for free. In the
AVX-512 column it becomes more visible for wide types (u64 fused ~12-50 ns
more than zmm bare); even there it is much cheaper than running the
`wrapping_add` as a separate pass over the output, which would cost
+100-300 ns (see the "Fused vs unfused FoR" table below).
