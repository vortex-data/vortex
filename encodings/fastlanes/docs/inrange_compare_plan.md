# Plan: in-range constant compare for bit-packed arrays

## Status today

`encodings/fastlanes/src/bitpacking/compute/compare.rs` accelerates
`BitPackedArray op ConstantArray` only when `c` is **outside** the packable range
`[0, 2^bit_width - 1]`. That case reduces every packed lane to the same boolean
under `op`, so the result is a `ConstantArray<bool>` (no work on the buffer) or a
`BitBuffer` filled with that constant plus a per-position overlay at any patched
indices.

**In-range** constants (those that could equal a packed lane) fall through to the
canonical "decompress to `PrimitiveArray`, then run Arrow's vectorized compare"
path. For `Eq`/`NotEq`/`Lt`/`Lte`/`Gt`/`Gte` this is correct but does two SIMD
passes' worth of work (unpack + compare) and writes the unpacked primitive to
memory along the way.

## Why the obvious approach (FastLanes `unpack_cmp`) doesn't win

`fastlanes::BitPackingCompare::unchecked_unpack_cmp` fuses unpack + compare and
emits `[bool; 1024]` without materializing the primitive array. It is
`#[inline(never)]` and applies the comparator closure to every element
individually. We tried wiring it in: at 65 K u32 elements (bit_width 4) the
fused path measured ~170 µs against ~91 µs for the canonical "unpack then
Arrow compare" path. Both Arrow's primitive compare and FastLanes' `unpack` are
heavily SIMD-vectorized; the per-element closure call defeats vectorization in
`unpack_cmp`. Reverted in commit `cc586c6`.

## Proposal: bit-parallel compare on the packed buffer

Pack the constant into a 1024-element template once (we already have a
constant-only pack kernel in `bitpack_compress::bitpack_constant`, which
synthesizes the FastLanes bit pattern analytically — no `BitPacking::pack`
call). Then for each 1024-chunk of the input, do the comparison directly on the
packed bytes via SIMD/SWAR. No materialization, less memory traffic
(`~3W·128` bytes per chunk vs `12·1024` bytes for unpack + compare on `u32`),
and the loop is fully vectorizable.

### Equality (`Eq` / `NotEq`)

The clean case.

```
diff = packed_chunk  ^  c_packed_chunk        // SIMD XOR per word
eq_per_element = "every W-bit slot of diff is zero"
```

Per the FastLanes layout, lane `l`'s `W` output words contain bits
`[k·T, (k+1)·T)` of the per-lane stream `c, c, c, …` for `k ∈ 0..W`. After
XOR with the same-layout `c_packed`, element `r`'s `W` bits land at known
positions inside the lane's `W` output words.

* **`W` divides `T` (W ∈ {1, 2, 4, 8, 16, 32} for u32):** each element's `W`
  bits are contained in a single output word. The classic SWAR "any byte is
  zero" trick works for `W = 8`:
  ```
  let v = diff_word;
  let zero_byte = !v & (v.wrapping_sub(0x01010101)) & 0x80808080;
  // bit 7 of each byte set iff that byte was 0
  ```
  Analogous masks `0x55555555` (`W=2`), `0x11111111` (`W=4`),
  `0x00010001` (`W=16`) cover the other power-of-2 widths.
* **`W` does not divide `T` (e.g. 3, 5, 7, 9, 11, 13, 15):** elements straddle
  word boundaries. The "OR-reduce W shifted copies" idea still applies but the
  mask depends on the rotation; easiest implementation is per-width SWAR
  unrolled at compile time via `match_each_bit_width!`.

Pack the resulting per-element bits into the output `BitBuffer`. We already do
this for the out-of-range short-circuit's patches overlay; the same code
applies.

### Ordering (`Lt` / `Lte` / `Gt` / `Gte`)

The harder case. Two routes; pick one per width.

#### Route A — SWAR less-than (preferred for `W ∈ {8, 16, 32}`)

For `W = 8` and `u32` storage, each output word holds 4 packed elements as
bytes. The classic SWAR unsigned less-than is:

```
let A = packed_word;
let B = c_packed_word;                        // a constant per chunk
let mask = 0x80808080;
let lt = ((A | mask) - (B & !mask)) ^ ((A ^ B) | mask);
let lt_bits = lt & mask;                      // high bit per byte = 1 iff A_byte < B_byte
```

Extract bit 7 of each byte (e.g. `_pext_u32(lt_bits, 0x80808080)` on BMI2, or a
shift-and-mask sequence) and pack into the result `BitBuffer`. `W = 16` uses
`0x80008000`; `W = 32` is the trivial single-element-per-word case.

Derive the other three operators from `Lt`:
* `Gt(a, c) = Lt(c, a)` → swap operands.
* `Lte(a, c) = !Gt(a, c)` → SWAR less-than with swapped operands, then invert.
* `Gte(a, c) = !Lt(a, c)` → invert.

For `W = 4` the same SWAR pattern works on nibbles with mask `0x88888888`.

#### Route B — bit-sliced compare (covers all `W`)

Generic alternative: for each output word, treat the contained `W`-bit slots
as a vertical stack of `T_bits / W` slot values, and run the standard
bit-sliced comparator on the lane's `W` output words at once. This is
layout-aware (uses the FastLanes lane order) but doesn't need per-width
SWAR masks. Slower than Route A on widths Route A supports, but simpler to
write and works uniformly.

### Patches and validity

Same overlay pattern as the out-of-range path: compute the per-position
ordering bit from the packed buffer, then for each `(idx, value)` patch set the
bit at `idx - patches.offset()` to `op(value, c)`. Apply the validity mask
at the end via `BoolArray::new(bits, validity)`.

### Sliced arrays

`lhs.offset() != 0` means the first chunk's packed bytes do not align with
element 0; defer to the canonical path until we have proper offset handling
inside the SWAR loop (drop the first `offset` bits before writing).

## Order of work

1. **`Eq` in-range via XOR + SWAR zero-detect.** Add the per-width SWAR masks
   for `W ∈ {1, 2, 4, 8, 16, 32}` first; widths in between can fall through
   to canonical until step 3. NotEq is the same kernel inverted.
2. **`Gt` / `Gte` in-range via SWAR less-than.** Land `W ∈ {8, 16, 32}`,
   derive the four ordering operators from a single `Lt(a, b)` primitive.
3. **Non-power-of-2 widths.** Pick Route B (bit-sliced compare) or
   per-width SWAR; benchmark.
4. **Sliced offsets and patches.** Handle `offset != 0` inside the SWAR loop
   so we don't fall back on sliced inputs.

Each step is independently shippable; the kernel already returns `Ok(None)`
for any case it doesn't accelerate, so the canonical path remains the
correctness fallback throughout.

## Benchmarks to land alongside

Add cases to `benches/bitpack_compare.rs` for an **in-range** constant
(currently only the out-of-range fast path is benched there). Compare:

* the SWAR fast path
* the canonical "execute to `PrimitiveArray`, then Arrow compare" baseline

across `bit_width ∈ {4, 8, 16}` and `len ∈ {1 024, 65 536}` for both `Eq`
and `Gt`. We need to beat the baseline at 64 K to be worth landing —
otherwise the canonical path's SIMD throughput is already the right answer
and we should drop this idea.
