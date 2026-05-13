<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Fused Delta + FFoR + BitUnpack decoding — design note

Status: design only. No production kernel in this branch.

## Motivation

With signed-integer support added to `DeltaArray`, the natural compressed shape for
signed columns is:

```
DeltaArray {
    bases:  PrimitiveArray<T>                   // T ∈ {i8..i64, u8..u64}
    deltas: BitPackedArray { encoded: FoRArray { encoded: …, ref: min_d } }
}
```

That is: delta-encode → frame-of-reference (subtract `min_d`) → bit-pack. This is the
"DELTA + FFoR + BP" stack recommended by the FastLanes paper and the ADMS '24
follow-up, and it is the only stack that keeps the bit-packing width small when the
deltas can be negative (a single negative delta in two's complement otherwise
sets every high bit, forcing `W = T`).

Today, decoding such an array makes **three separate passes** over the packed
buffer (and intermediate buffers):

1. `BitPackedArray::execute` — unpack `W` bits per element into a full-width
   primitive array.
2. `FoRArray::execute` — element-wise `wrapping_add(reference)`.
3. `DeltaArray::execute` → `delta_decompress` → `Delta::undelta` — element-wise
   cumulative-sum (`wrapping_add(prev)`) within each lane.

Each pass reads and writes 1024 × `size_of::<T>()` bytes per chunk. For the common
case where `T = i32` and `W = 8`, that is 3 × 4 KiB of bandwidth per chunk to do
work whose minimum information-theoretic cost is one read of 1 KiB (the packed
buffer) and one write of 4 KiB (the output).

## Upstream building blocks

`fastlanes` 0.5.0 already ships partial fusions:

| Kernel | Fuses | Type bound |
|---|---|---|
| `BitPacking::unpack<W, B>` | unpack only | `Self: FastLanes` (unsigned) |
| `FoR::unfor_pack<W, B>` | unpack + `wrapping_add(ref)` | `Self: BitPacking` |
| `Delta::undelta_pack<LANES, W, B>` | unpack + lane-cumsum undelta | `Self: BitPacking` |
| `Delta::undelta<LANES>` | undelta on already-unpacked values | `Self: BitPacking` |

What is **missing upstream** is a triple-fused kernel: unpack + `wrapping_add(ref)`
+ undelta in a single pass. The two existing fused kernels each pair *one* of the
two reductions with `unpack`; neither pairs both.

## Proposed kernel

```rust
/// Triple-fused decode: unpack W-bit values, add a FoR reference, and undo a
/// per-lane delta in one pass.
///
/// `input`  — packed buffer of `B = 1024 * W / T` elements of width `T`
/// `base`   — `LANES`-element per-lane bases (already in the natural type)
/// `reference` — FoR reference added to every unpacked element before undelta
/// `output` — 1024 reconstructed values, in lane-transposed order
fn undelta_for_pack<const LANES: usize, const W: usize, const B: usize>(
    input: &[Self; B],
    base: &[Self; LANES],
    reference: Self,
    output: &mut [Self; 1024],
);
```

Sketch (compare with upstream `Delta::undelta_pack` and `FoR::unfor_pack`):

```rust
for lane in 0..Self::LANES {
    let mut prev = base[lane];
    unpack!(T, W, input, lane, |idx, packed_elem| {
        // (1) restore FoR offset, (2) cumulative wrapping-add along the lane
        let d = packed_elem.wrapping_add(reference);
        let next = d.wrapping_add(prev);
        output[idx] = next;
        prev = next;
    });
}
```

Memory traffic per 1024-chunk: one read of `1024 * W / 8` bytes (packed), one
read of `LANES * size_of::<T>()` bytes (bases), one scalar `reference`, and one
write of `1024 * size_of::<T>()` bytes. For `T = i32`, `W = 8`: 1 KiB read + 128 B
read + 4 KiB write = ~5 KiB total, versus ~13 KiB for the 3-pass path.

### Type bounds

The kernel naturally inherits `Self: BitPacking`, which upstream restricts to
unsigned types (`u8`/`u16`/`u32`/`u64`). Signed inputs reuse the kernel via
`FastLanesComparable::Bitpacked` — the same transmute trick used by this branch's
non-fused signed-support change — so a single set of macro instantiations
(`u8`/`u16`/`u32`/`u64`) covers all eight integer types.

## Where the kernel lives

Two options, in increasing order of effort:

1. **Vortex-local kernel** in `encodings/fastlanes/src/delta/undelta_for_pack.rs`,
   built with the same `seq_t!` / `pack!` / `unpack!` macros that upstream exports.
   Pros: lands in one PR, no upstream churn. Cons: duplicates the lane-iteration
   skeleton; future upstream fixes (e.g. patches to the bit-shuffling order) have
   to be mirrored.

2. **Upstream `fastlanes` PR** adding `Delta::undelta_for_pack` next to
   `Delta::undelta_pack`. Pros: shares the macro skeleton with the existing
   fused kernels. Cons: depends on a release and a workspace pin bump.

Option 1 is the right starting point. If benchmarks show the win we expect, the
kernel can be lifted upstream with a thin wrapper kept locally.

## Integration into the decode path

`delta_decompress` currently calls `array.deltas().clone().execute(ctx)?` and
then `Delta::undelta` lane-by-lane. To use the fused kernel:

1. Inspect the `deltas` child. The fast path applies only when it is exactly
   `BitPacked` *or* `FoR(BitPacked)`.
2. For `FoR(BitPacked)`: read the FoR `reference` scalar; read the packed
   buffer, bit-width, and patches from the `BitPacked` child; dispatch to
   `undelta_for_pack::<LANES, W, B>` for each 1024-chunk.
3. For `BitPacked` (no FoR layer): dispatch to upstream `Delta::undelta_pack`
   (already exists, no new kernel needed).
4. For anything else (e.g. a generic primitive deltas slot): fall through to
   the current non-fused path.
5. Handle patches (the BitPacked layer's exception store) after the fused decode,
   the same way `for/array/for_decompress.rs::fused_decompress` does it today.

The signed-vs-unsigned dispatch is the same `reinterpret_cast` trick used in
this branch: rewrap as the unsigned counterpart, call the fused kernel, rewrap
the output. The bases and the FoR reference participate in the same transmute.

## Benchmark plan (before committing to the kernel)

A microbench in `vortex-bench/` over four sorted signed columns of 10M elements:

| Column shape | Expected `W` | Hypothesis |
|---|---|---|
| `i32` monotone increasing from 0 | small | fused wins, no FoR step does much |
| `i32` monotone increasing from −1e9 | small | fused wins; FoR ref nontrivial |
| `i32` near-monotone with 5 % decreases | small | fused wins by larger margin |
| `i32` random in `[−100, +100]` | medium | fused ≈ 3-pass; bandwidth less dominant |

Decode throughput on a single core; compare 3-pass vs proposed fused kernel.
Worth landing if the fused path is ≥ 1.5× on the first three rows.

## Out of scope

- The encoding side: `delta_compress` already runs in one pass; FoR + bit-pack
  on the produced deltas is a separate sequential composition that is already
  fused well enough by the existing FoR and BitPacked encoders.
- A symmetric `delta_for_pack` (fused encode) — only worth doing once the
  decode-side wins are confirmed.
