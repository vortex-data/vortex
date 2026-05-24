# Validity-aware iteration

Many compute and aggregate kernels walk a value buffer alongside a validity bitmap (a
`vortex_mask::Mask` / `vortex_buffer::BitBuffer`, one bit per element). How the bitmap is
traversed dominates the throughput of these kernels, and the naive idiom is several times slower
than it needs to be. This page describes the fast pattern, the evidence behind it, and a plan to
roll it out across the codebase.

## The pattern

**Slow (avoid):** zipping a per-bit `bool` iterator with the values and branching per lane.

```rust
// ❌ scalar, branchy — does not vectorize, degrades further as nulls appear
for (v, valid) in values.iter().zip(mask.bit_buffer().iter()) {
    if valid {
        out.push(checked(v)?);   // per-element fallible op + early return
    }
}
```

This loses on three independent counts, each of which alone defeats the autovectorizer:

1. a per-element fallible op with an early `return`/`?` inside the loop;
2. a per-bit `bool` iterator (`bit_buffer().iter()`, `iter_bits`, `BitIterator`, `threshold_iter`);
3. a data-dependent `if valid { … }` branch on the bit.

**Fast (prefer):** consume the bitmap in 64-bit words via [`BitBuffer::chunks`], gate each lane
branch-free, and decide once at the end.

```rust
// ✅ branch-free; vectorizes; cost is independent of null density
for word in mask.bit_buffer().chunks().iter() {     // offset-normalized u64 words
    for (j, &v) in block.iter().enumerate() {
        let valid = (word >> j) & 1 != 0;
        // gate with a SELECT, not a branch: fold invalid lanes against a neutral value
        let folded = if valid { v } else { NEUTRAL };
        acc = combine(acc, folded);                 // e.g. min/max/sum
    }
}
// one check after the loop, never inside it
```

The rules, in order of impact:

- **Never early-return / bail per element.** Accumulate a flag (or running min/max) and check once
  after the loop. Early exits serialize the reduction.
- **Walk the bitmap in word chunks, read the chunk once.** Use [`BitBuffer::chunks`] — it yields
  offset-normalized `u64` words, so the same code is correct for sliced (bit-offset) buffers. Never
  index `bits[i / 64] >> (i % 64)` inside the value loop.
- **Gate with a select, not a branch.** Fold invalid lanes against a neutral that can't affect the
  result (e.g. `+∞`/`T::MAX` for a running min, `T::MIN` for a max, `0` for a sum), or pre-zero the
  invalid inputs. A branch on the validity word reintroduces misprediction on null-bearing data.
- **Leave invalid output lanes as garbage when you can.** The result validity already hides them, so
  masking/zeroing them is wasted work. (Zeroing the output is near-free only when the output type is
  at least as wide as the validity granularity; for narrow outputs prefer zeroing the *input*.)

See the [`vortex_buffer::BitBuffer`] docs for the same guidance from the bitmap side.

## Two shapes of kernel

The right strategy depends on whether the kernel is a reduction or an elementwise map.

- **Reductions** (`sum`, `min`/`max`, `nan_count`, the cast's range check) *must* consult validity
  to skip null lanes. Use the word-chunked branch-free fold above: fold invalid lanes against a
  neutral, accumulate, decide once.
- **Elementwise maps** (`a + b`, `cast`, `take`) should **not** gate the value computation by
  validity at all. Compute the result for *every* lane at full SIMD width — null-position values
  are arbitrary but masked — and produce the result validity as a cheap, separate **bitwise op on
  the input bitmaps** (e.g. `result_nulls = a_nulls & b_nulls`, a vectorized word-wise AND). This is
  the approach arrow-rs uses for arithmetic (`arrow-arith`'s `binary`/`try_binary`), and it keeps
  the hot value loop completely branch- and validity-free.

  The only wrinkle is a **fallible** elementwise op (checked add, checked cast): an overflow at a
  *null* lane must not error. Handle it exactly as the cast does — compute on all lanes, detect the
  fault branch-free per lane, **AND the fault mask with validity**, and bail once at the end if any
  *valid* lane faulted. Never branch on validity in the per-lane body.

## Evidence

This pattern was validated while tuning the primitive cast kernel
(`vortex-array/src/arrays/primitive/compute/cast.rs`). Rewriting the fallible path from the slow
idiom to the branch-free word-chunk form:

- **int → int (e.g. `u32 → u8`): scalar → SIMD.** The old kernel emitted a scalar byte loop with an
  early-return branch (no vector instructions). The new kernel emits packed `vpmovdb` +
  `vpminud`/`vpmaxud` + a mask-register select — pure safe Rust, no `std::arch`.
- **nullable `f64 → i32`: ~493 µs → ~200 µs (≈2.4×)** at N=100k, 50% nulls. Notably this win came
  from making the loop *branch-free* (the scalar convert was unchanged), so it is
  density-independent where the old branchy version got worse as nulls increased.

The same rewrite on the nullable aggregate kernels (`aggregate_fn/fns/...`), benchmarked at N=100k,
50% nulls, before → after:

| kernel | before | after | speedup |
| --- | --- | --- | --- |
| `nan_count` (f64) | 140.7 µs | 79.3 µs | 1.77× |
| `sum` (f64) | 720.9 µs | 430.5 µs | 1.67× |

The branch-free `sum` is *bit-identical* to the old scalar version (adding `0.0` for invalid/NaN
lanes is exact and preserves order); `nan_count` is an order-independent count. The measured
per-kernel branch-misprediction penalty at 50% nulls was 3–8× (`nan_count` 8×, `sum` 3.9×,
`min_max` 3×) — that penalty is what the branch-free form removes.

### Fallible elementwise: checked add with validity (prototype, N=65,536)

A standalone prototype of `i32 + i32` with per-operand validity compared three strategies:

| strategy | 100% valid | 50% valid |
| --- | --- | --- |
| naive per-lane zip + branch + early return | 1.61 ns/elem | 7.58 ns/elem |
| arrow-style scalar valid-index gather (`try_for_each_valid_idx`) | 1.17 | 0.52 |
| **dense branch-free (compute all, AND fault with validity, bail once)** | **0.27** | **0.28** |

The dense form vectorizes to `vpaddd` + `vpternlogd` (the branch-free signed-overflow test
`((x^s) & (y^s)) < 0` in one op) + `vpsrld` + `vporq`, and is **4.3× faster than arrow's approach**
and density-independent. Note arrow-rs *cannot* do this for checked arithmetic: its `op` is a
generic fallible closure that would trap on the garbage at null slots, so it gathers valid indices
instead. Rust's non-trapping `i32::overflowing_add` (or the explicit sign test) lets us stay dense
and branch-free — overflow at a *null* lane is computed but its fault bit is masked out by the
`& validity` before the final bail. Result validity is the cheap word-wise `a_nulls & b_nulls`.

### `take` / gather with validity

Per arrow-rs (`arrow-select`) and confirmed by inspection, `take` is **gather-bound, not bitmap- or
branch-bound**: gather the values (`out[i] = values[indices[i]]`) and build the result validity in
a separate pass (gather the source validity bits at the indices, AND with the indices' own
validity; when the source has no nulls, just propagate the indices' null buffer). No mainstream
crate uses SIMD here — the random gather dominates — so the validity-iteration pattern above buys
little; the win, if any, is a hardware gather (Vortex already has an AVX2 `take`). The "checked"
part is an index bounds-check, which vectorizes trivially (`indices < len`, OR-reduced).

### What does the validity guard actually cost?

Decomposing checked `i32 + i32` (N=65,536; ns/elem; ratios matter, absolute numbers are machine-relative):

| variant | density 1.0 | density 0.5 |
| --- | --- | --- |
| bare add (no validity, no overflow check) | 0.20 | 0.21 |
| + overflow check, no validity | 0.20 (+1%) | 0.21 |
| **+ validity, arrow-style (flat dense values, separate word-level `a & b`)** | **0.21 (+4%)** | **0.21** |
| + validity *fused per-lane into the value loop* (anti-pattern) | 0.97 (~5×) | 0.97 |

Three conclusions:

- **The overflow/fault check is essentially free** (~1% over a bare add) — it rides alongside the
  arithmetic (`vpaddd` + `vpternlogd` + OR-reduce). Never avoid checking to "save time".
- **For an elementwise op, validity is also nearly free (~4%)** *if done right*: keep the value
  computation a **flat dense loop** (it vectorizes exactly like the non-nullable case) and produce
  the result validity as a **separate word-level bitwise AND** of the input bitmaps. This is arrow's
  strategy and there is no meaningful penalty for nullability.
- **The ~5× cost is an anti-pattern, not inherent.** Fusing the per-lane validity bit into the
  value loop (`for word { for j in 0..64 { … bit j … } }`) breaks vectorization of the arithmetic.
  An earlier prototype did this and measured ~3–5× — the cost was the broken value loop, not the
  validity. Do **not** thread the bitmap through the elementwise value loop.

**Reductions are different.** `sum`/`min`/`max`/`nan_count` *cannot* compute densely and AND
bitmaps — they must skip null lanes inside the fold — so they pay a real (but density-independent)
cost to gate, which is exactly why the branch-free fold above matters for them. For elementwise
maps, prefer the dense-values + separate-bitmap split; for reductions, use the branch-free gated
fold.

For a *fallible* elementwise op (checked add/cast) the overflow-anywhere flag is computed ungated in
the dense pass; only if it trips do you correlate fault-with-validity (rare), so the common path
stays at the ~4% figure.

### Caveats learned the hard way

- **Benchmark on the real build, not just a microbenchmark.** A "byte per 8 lanes" traversal that
  read a flat `&[u8]` was faster in isolation, but re-deriving the byte from a `u64` chunk word
  (`(word >> b*8) as u8`) *regressed* the real cast kernel (201 µs → 269 µs). It was reverted.
- **Saturating `as` does not vectorize for float → int.** `v as i32` lowers to scalar
  `vcvttsd2si` + clamp. Getting SIMD there needs `to_int_unchecked` after a branchless clamp (pure
  Rust, vectorizes to `vcvttpd2dq`) or an `std::arch` backend — a separate, reviewed change.
- **End-to-end overhead can mask a kernel win.** `cast_u32_to_u8_nullable` barely moved end-to-end
  even though the kernel went scalar → SIMD, because that path is dominated by mask materialization
  and array construction, not the cast loop. Profile to confirm the kernel is the bottleneck.
- **Beware the statistics cache when benchmarking aggregates.** Calling `sum`/`min_max`/`nan_count`
  on a cloned array measures a cached-stat lookup (~100 ns for 100k elements), not the kernel.
  Construct a fresh `PrimitiveArray` (cheap buffer/validity `clone`s) per timed iteration so the
  stats cache is empty and the kernel actually runs.

## Iterator interop & API ergonomics

A natural question is whether the ergonomic `iterator.map(…).collect()` style can be used without
losing the vectorization. Measured findings (N=65,536, i32 transform / f64 reduction):

- **`Iterator::collect()` over a packed bitmap does not vectorize** (~2–4 ns/elem). The blocker is
  twofold: per-element `next()` extracts the bit with a scalar shift, and `FromIterator`'s
  `Vec`-growth defeats a tight store loop.
- **`TrustedLen` / `ExactSizeIterator` helps allocation, not vectorization.** Telling `collect`
  the exact size lets it preallocate + use the tight `extend_trusted` loop (~1.2× — 1.01 → 0.84
  ns/elem), but the per-element bitmap `next()` stays scalar. Sizing is necessary, not sufficient.
- **Slice iterators *do* vectorize through `collect`.** `values.iter().zip(bools.iter()).map(sel)
  .collect()` hits ~0.16 ns/elem — but only because the mask is a `&[bool]` slice (cheap
  pointer-bump `next()`); the packed bit is what blocks it. Expanding a bitmap to `&[bool]` costs a
  pass + 8× memory, so it only pays if validity is already byte-per-lane or reused.
- **The fast ergonomic form is a `for_each`/fill into a *sized* target**, not `collect`:
  `out.chunks_exact_mut(64).zip(values.chunks_exact(64)).zip(bits).for_each(per-64)` is pure
  `std::iter` and vectorizes (~0.19). Collecting into a sized value is fine too — allocate
  **uninitialized** (`with_capacity` + `set_len`) and fill once (~0.18); never `vec![0; n]` (wasted
  zeroing) or growable `FromIterator`.
- **Chunk width matters: use ≥32 (64 is ideal).** Letting the backend vectorize the inner per-lane
  loop only kicks in at width ≥32; chunk-8 and chunk-16 fall back to scalar (~4 ns/elem). The
  bitmap's native 64-bit word is both the natural and the fast granularity.

So the shipped combinator is closure-shaped and word-chunked on purpose: it keeps the
`.map_if(…).collect()` / `.fold(…)` ergonomics while running the word loop into a sized buffer,
which is the only form that is fast for *packed* bitmaps. A reductions consumer should capture a
`&mut` accumulator (like [`BitBuffer::zip_lanes`]) rather than thread it through `Iterator::fold`,
which went scalar in testing.

### Guarding against silent scalar regressions

Vectorization is an LLVM outcome, not a type-level guarantee — there is no "this `next()` is
SIMD-able" marker. Verify it stays vectorized with either: a **divan threshold** on the kernel
bench (primary — robust to codegen details), or an **asm check** that the monomorphized body
contains packed ops (`zmm`/`vpaddd`/`vcvttpd2dq`) and no scalar fallback (`vcvttsd2si`); the latter
is a good diagnostic when the threshold trips. The byte-per-8 cast attempt that silently went
scalar is exactly what such a guard catches.

## Rollout plan

A shared, branch-free `(values, validity)` reducer now exists — [`BitBuffer::zip_lanes`] — and the
primitive `sum`, `min_max`, and `nan_count` kernels route through it (see Evidence above). Remaining
work, hottest first, **each behind a benchmark**:

   | tier | sites | status |
   | --- | --- | --- |
   | 1 — aggregates | `sum/primitive`, `min_max/primitive`, `nan_count/primitive` | **done** (1.8–3.7×) |
   | 1 — aggregates | `sum/decimal`, `min_max/decimal` | candidate (same reduction shape) |
   | 2 — metadata / strings / bridges | `is_sorted/{primitive,bool,decimal}`, `arrays/varbinview/compact` | candidate (warm) |
   | 3 — decompression / bridges | `encodings/runend/compress`, `vortex-duckdb/convert/vector` | candidate |
   | — bottlenecked elsewhere | `fastlanes fill_null_forward` (sequential carry), `runend decompress` (per-run), `compressor/stats/integer` (HashMap), `take`/`patches` (gather) | **skip** — validity walk isn't the cost |

   The decimal aggregates are the strongest remaining candidates (identical structure to the
   primitives that won 1.8–3.7×). Several "hot"-looking sites are bottlenecked by a sequential
   carry, per-run granularity, a `HashMap`, or a random gather, so the combinator would not help —
   verify what dominates each loop before rewriting.
3. **Don't trust unbenchmarked estimates.** Treat per-site projections as "worth measuring," not
   promises — the cast work produced at least one change that looked good on paper and regressed in
   practice.

[`BitBuffer::chunks`]: https://docs.rs/vortex-buffer/latest/vortex_buffer/struct.BitBuffer.html
[`vortex_buffer::BitBuffer`]: https://docs.rs/vortex-buffer/latest/vortex_buffer/struct.BitBuffer.html
