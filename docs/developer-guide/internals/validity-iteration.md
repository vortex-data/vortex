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

## Rollout plan

The goal is one shared, branch-free `(values, validity)` reducer/mapper rather than rewriting each
kernel by hand — the same "avoid N bespoke kernels" principle the cast work demonstrated.

1. **Extract a shared helper** (in `vortex-array`, near the aggregate plumbing, building on
   [`BitBuffer::chunks`]) that drives a word-chunked, branch-free fold over `(values, validity)`
   with a caller-supplied combine + neutral. Aggregates and simple masked maps route through it.
2. **Retrofit the hottest sites first, each behind a benchmark.** For every site: add/keep a divan
   bench, confirm scalar → SIMD in the emitted asm, and measure before/after on the real build.

   Priority order (hottest first; these run on every scan batch):

   | tier | sites |
   | --- | --- |
   | 1 — aggregates | `aggregate_fn/fns/sum/{primitive,decimal}.rs`, `aggregate_fn/fns/min_max/{primitive,decimal}.rs`, `aggregate_fn/fns/nan_count/primitive.rs` |
   | 2 — metadata / strings / bridges | `aggregate_fn/fns/is_sorted/{primitive,bool,decimal}.rs`, `arrays/varbinview/compact.rs`, `vortex-duckdb/src/convert/vector.rs` |
   | 3 — decompression | `encodings/runend/src/compress.rs`, `encodings/fastlanes/src/lib.rs` (`fill_null_forward`) |

   `sum` and `min_max` (primitive + decimal) are the highest leverage; `min_max` is also the exact
   reduction the cast kernel already does well, so it is the natural first consumer of the helper.
3. **Don't trust unbenchmarked estimates.** Treat per-site projections as "worth measuring," not
   promises — the cast work produced at least one change that looked good on paper and regressed in
   practice.

[`BitBuffer::chunks`]: https://docs.rs/vortex-buffer/latest/vortex_buffer/struct.BitBuffer.html
[`vortex_buffer::BitBuffer`]: https://docs.rs/vortex-buffer/latest/vortex_buffer/struct.BitBuffer.html
