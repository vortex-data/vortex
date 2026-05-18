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
  word boundaries. Use Route C below — Knuth's broadword zero-test against a
  non-uniform guard mask, with rotation tables generated per width.

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

FastLanes packs each lane's 32 elements as `W` output words where word `k`
holds bit `k` of every element in element-index order. That layout is the
natural input for a bit-sliced unsigned comparator: process the `W` bit
planes top-down, maintaining two state words per lane —

* `lt` — bit `i` set once element `i` has been proven `< c` at some prior
  (more significant) bit.
* `eq` — bit `i` set while element `i` has matched `c` at every prior bit.

```
let mut lt = 0u32;
let mut eq = !0u32;
for k in (0..W).rev() {
    let a_k = lane[k];              // bit k of every element in this lane
    let b_k = if (c >> k) & 1 == 1 { !0u32 } else { 0u32 };
    lt |= eq & !a_k & b_k;          // A had 0, B had 1 at bit k, still equal above
    eq &= !(a_k ^ b_k);             // remain equal iff this bit matches
}
// lt[i] set iff lane element i < c.
```

Because `c` is a single scalar, every `b_k` is a compile-time-known `0u32` or
`!0u32`. Specialise the inner body per bit of `c`:

```
if (c >> k) & 1 == 0 {
    eq &= !a_k;                     // lt unchanged: b_k & anything = 0
} else {
    lt |= eq & !a_k;
    eq &= a_k;
}
```

That's 1–2 word-ops per bit plane and ~`W` to `2W` ops total per lane (32
elements). For `W = 7` it's ~10–14 ops vs. Route C's ~22; for `W ∈ {8, 16,
32}` Route A's single SWAR + extract still wins.

**Equality.** Drop `lt`, keep only `eq`. The per-bit specialisation collapses
to one op per bit plane (either `eq &= a_k` or `eq &= !a_k`), so Eq is `W`
word-ops per lane — strictly cheaper than the SWAR XOR + zero-detect for
non-power-of-2 widths.

**Deriving the other operators.** Same identities as Route A:
`Gt(a, c) = Lt(c, a)` swaps operand roles, which in this loop is one toggle
of every `b_k` (i.e. flip the `(c >> k) & 1` test); `Lte` and `Gte` invert
the final `lt` word and mask off any trailing bits past the lane's end.

**Output ordering.** `lt` already carries its bits in element-index order
within the lane — bit `i` is element `i`. Assembling into the chunk-wide
`BitBuffer` is one `OR ((lt as u32) << (lane_idx * 32))` per lane (assuming
lanes pack consecutively); otherwise apply the FastLanes lane permutation
on the way out.

**Patches.** Same overlay pattern as Routes A and C — see *Patches and
validity* below.

**Layout precondition.** Route B's whole pitch depends on the lane's `W`
output words actually being bit planes (one bit-position per word). If
FastLanes instead stores horizontally-packed `W`-bit slots — the layout
assumed by Routes A and C — Route B needs a one-time per-chunk bit-transpose
to extract the planes, which costs roughly as much as a full unpack and
nullifies the win. Confirm against the FastLanes source before committing;
if the layout is horizontal, drop Route B in favour of Routes A (`W` a
power of 2) + C (otherwise).

#### Route C — Knuth broadword with rotation tables (covers all `W`)

Knuth's broadword less-than (TAOCP 4A §7.1.3, Hacker's Delight §5-3)
generalises Route A to non-power-of-2 widths via a *non-uniform* guard mask:

```
diff    = (A | H) - B
lt_bits = !diff & H
```

`H` has 1 at the MSB (guard bit) of each field fully contained in the word.
Both `A` and `B` must be 0 at every bit of `H`. The guard positions in `H`
do **not** need to be uniformly spaced — Knuth's identity holds for arbitrary
field layouts, which is what makes this work for widths whose slots straddle
word boundaries.

**Rotation tables.** Width `W` over `T_bits = 32` storage repeats with period
`P = W / gcd(W, T_bits)` words, holding exactly `T_bits` elements per period.
Compile-time generate per width via `match_each_bit_width!`:

* `H[0..P]` — guard mask per word position in the period.
* `(shift, output_idx)[0..P-1]` — stitch recipe for each straddler that
  crosses a word boundary in the period.

Examples: `W = 3` → `P = 3`, ~10 in-word fields and 2 straddlers per period;
`W = 5` → `P = 5`, ~6 + 4; `W = 7` → `P = 7`, ~4 + 6.

**Inner loop.** For each period of `P` words: one Knuth SWAR per word over
its `H`-covered fields, plus one scalar stitch + compare per straddler —

```
let straddled = ((chunk[k] >> shift) | (chunk[k + 1] << (T_bits - shift)))
                & ((1u32 << W) - 1);
set_bit(output, straddler_output_idx, straddled < c);
```

Result bits land in the output `BitBuffer` at the per-position indices baked
into the tables.

**Guard-bit precondition.** The 3-op form needs every `H`-position MSB to be
0. Free when both:

* `c < 2^(W-1)` — the constant-packed `B` template has 0 at every guard by
  construction.
* `max(lhs) < 2^(W-1)` — cheap stat lookup on the bit-packed array.

If both hold, use the 3-op form. If only the constant is small, the 4-op
general SWAR Lt with the same per-position `H` still works, just at higher
cost per word. If neither, defer to Route B or canonical.

**Cost vs. Route B.** At `W = 7`: ~7 SWAR + 6 stitch ≈ 22 word-ops per 32
elements. Bit-sliced is `W` word-ops per 32 elements *if* FastLanes' internal
layout exposes bit-planes cheaply — verify in source. Route C wins when the
layout is horizontally packed (no cheap bit-planes); Route B wins when it
isn't. They aren't strict alternatives — Route C can be the fallback inside
Route B's outer dispatch.

**Equality reuse.** The same `H` infrastructure covers Eq: XOR the chunk
against the constant template, then run Knuth's broadword zero-test
(Hacker's Delight §6-1) keyed on `H`. One set of tables, two operators.

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
3. **Non-power-of-2 widths.** Bench Route B (bit-sliced compare) against
   Route C (Knuth broadword with rotation tables) at `W ∈ {3, 5, 7, 11, 13}`
   and `len ∈ {1024, 65536}`. Route C's 3-op form is gated on guard-bit
   stats; include both the stat-friendly and stat-unfriendly cases.
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
