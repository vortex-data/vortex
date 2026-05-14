# Funnel-shift source-pattern experiment (u64 W=51, ymm preferred)

## Goal

Find a plain safe Rust phrasing of the FoR-fused unpack body whose LLVM lowering
emits `vpshldq + vpand + vpaddq` (the AVX-512-VBMI2 fused funnel-shift sequence)
instead of the legacy 5-op `vpsllq + vpsrlq + vpor + vpand + vpaddq`. If such a
pattern exists, the upstream `unpack!` macro in
`benchmarks/fastlanes-kernel-bench/src/macros.rs` could be retargeted to that
shape and the +30-50% overhead seen in the `u64 W∈[45..63] ymm fused` matrix
cells would close.

Build flags used for every artifact in this directory:

```text
RUSTFLAGS="-C target-cpu=native -C target-feature=-prefer-256-bit"
```

CPU: the host this experiment ran on (Linux x86_64). VBMI2 is enabled by
`target-cpu=native`, so `vpshldq` / `vpshrdq` (EVEX, 256-bit width when LLVM
prefers 256-bit) are available.

All six variants share the same signature `fn(&[u64; 816], u64, &mut [u64; 1024])`
and are marked `#[inline(never)]` so divan can call them through a function
pointer (matching the layout of the other benches in this crate).

---

## Variant 1: `pat_macro_shape` (mask-then-combine baseline)

```rust
fn pat_macro_shape(packed: &[u64; PACKED_LEN], reference: u64, out: &mut [u64; 1024]) {
    for i in 0..1024 {
        let bit_pos = i * W;
        let word = bit_pos / 64;
        let shift = (bit_pos % 64) as u32;
        let take_lo = (64 - shift).min(W as u32);
        let lo = (packed[word] >> shift) & ((1u64 << take_lo) - 1);
        let val = if take_lo as usize == W {
            lo
        } else {
            let hi = packed[word + 1] & ((1u64 << (W as u32 - take_lo)) - 1);
            lo | (hi << take_lo)
        };
        out[i] = val.wrapping_add(reference);
    }
}
```

Inner loop (full body, scalar):

```text
   594a0: add    %rsi,%r10                    ; + reference
   594a3: mov    %r10,(%rdx)                  ; store
   594a6: add    $0x8,%rdx
   594aa: add    $0x33,%rax                   ; bit_pos += 51
   594ae: cmp    $0xcc00,%rax
   594b4: je     <ret>
   594b6: mov    %rax,%r9
   594b9: shr    $0x6,%r9                     ; word = bit_pos/64
   594bd: mov    %eax,%ebx
   594bf: and    $0x3f,%ebx                   ; shift = bit_pos%64
   594c2: mov    $0x40,%r11d
   594c8: sub    %ebx,%r11d                   ; 64 - shift
   594cb: cmp    $0x33,%r11d
   594cf: cmovae %ecx,%r11d                   ; min(64-shift, 51)
   594d3: shrx   %rax,(%rdi,%r9,8),%r10       ; lo = packed[word] >> shift
   594d9: bzhi   %r11,%r10,%r10               ; & ((1<<take_lo)-1)
   594de: cmp    $0xe,%ebx
   594e1: jb     594a0                        ; branch: shift < 14, no hi half
   594e3: cmp    $0xcbcd,%rax
   594e9: je     <bounds_check_panic>
   594eb: shlx   %r11,0x8(%rdi,%r9,8),%r9     ; hi = packed[word+1] << take_lo
   594f2: and    %r8,%r9                      ; & ((1<<W)-1) (computes mask wrong-size, but folded)
   594f5: or     %r9,%r10                     ; lo | (hi << take_lo)
   594f8: jmp    594a0
```

- `vpshldq` emitted? **No** (0 occurrences).
- Lowered to scalar `shrx + bzhi + shlx + or + add` per iteration. **No vector ops at all.**
- Inner-loop ALU ops/element: ~10 (scalar) along the spilled-to-hi path,
  ~5 along the no-hi branch.
- Median: **1.490 us / 1024 elements = 1.45 ns per output**.

---

## Variant 2: `pat_combine_then_mask` (combine first, mask after)

```rust
fn pat_combine_then_mask(packed: &[u64; PACKED_LEN], reference: u64, out: &mut [u64; 1024]) {
    for i in 0..1024 {
        let bit_pos = i * W;
        let word = bit_pos / 64;
        let shift = (bit_pos % 64) as u32;
        let val = if shift as usize + W <= 64 {
            (packed[word] >> shift) & MASK
        } else {
            let lo = packed[word] >> shift;
            let hi = packed[word + 1] << (64 - shift);
            (lo | hi) & MASK
        };
        out[i] = val.wrapping_add(reference);
    }
}
```

Inner loop excerpt (the per-iteration body LLVM unrolled by 2):

```text
   59b40: shrx   %r10,(%rdi,%r9,8),%r9        ; lo = packed[word] >> shift
   59b46: bzhi   %r8,%r9,%r9                  ; & MASK
   59b4b: add    %rsi,%r9                     ; + reference
   59b4e: mov    %r9,(%rdx)
...
   59b90: shrx   %r10,(%rdi,%r9,8),%r10       ; lo
   59b9e: shlx   %r11,0x8(%rdi,%r9,8),%r9     ; hi = packed[word+1] << (64-shift)
   59ba5: or     %r10,%r9                     ; lo | hi
   59ba8: bzhi   %r8,%r9,%r9                  ; & MASK
   59bad: add    %rsi,%r9
   59bb0: mov    %r9,-0x8(%rdx)
```

- `vpshldq` emitted? **No** (0). Same scalar lowering as Variant 1, but LLVM
  unrolled the loop by 2 and split the "shift+W <= 64" cases.
- Inner-loop ALU ops/element: ~5 (combine path) or ~3 (no-hi path).
- Median: **1.144 us / 1024 elements = 1.12 ns per output**. Fastest variant.

---

## Variant 3: `pat_branchless_funnel` (always do the funnel)

```rust
fn pat_branchless_funnel(packed: &[u64; PACKED_LEN], reference: u64, out: &mut [u64; 1024]) {
    for i in 0..1024 {
        let bit_pos = i * W;
        let word = bit_pos / 64;
        let next = (word + 1).min(PACKED_LEN - 1);
        let shift = (bit_pos % 64) as u32;
        let lo = packed[word] >> shift;
        let hi = packed[next].wrapping_shl(64u32.wrapping_sub(shift));
        let val = (lo | hi) & MASK;
        out[i] = val.wrapping_add(reference);
    }
}
```

Note: this variant has a known correctness bug at `shift == 0`
(`wrapping_shl(64)` is identity, not zero). For W=51 the offset
`i*51 % 64` is zero only at `i == 0`, so a single output is wrong; the bug does
not affect the codegen-shape question.

Inner loop (vectorized! zmm-wide despite `-prefer-256-bit`):

```text
   59a30: vpmullq %zmm2,%zmm1,%zmm10          ; bit_pos = i * 51
   59a36: vpsrlq  $0x6,%zmm10,%zmm11          ; word = bit_pos >> 6
   59a3d: kxnorw  %k0,%k0,%k1
   59a46: vpgatherqq (%rdi,%zmm11,8),%zmm12{%k1}   ; packed[word]
   59a4d: vpminuq %zmm3,%zmm11,%zmm11
   59a53: vpandq  %zmm5,%zmm10,%zmm13         ; shift = bit_pos & 63
   59a59: vpsrlvq %zmm13,%zmm12,%zmm12        ; lo = packed[word] >> shift
   59a68: vpgatherqq (%rax,%zmm11,8),%zmm13{%k1}   ; packed[next]
   59a6f: vpsubq  %zmm10,%zmm4,%zmm11         ; 64 - shift (mod 64)
   59a75: vpandq  %zmm5,%zmm11,%zmm11
   59a7b: vpsllvq %zmm11,%zmm13,%zmm11        ; hi << (64-shift)
   59a81: vpternlogq $0xc8,%zmm12,%zmm6,%zmm11 ; (lo | hi) & MASK fused
   59a88: vpaddq  %zmm0,%zmm11,%zmm11         ; + reference
   59a8e: vmovdqu64 %zmm11,(%rdx,%rcx,8)
   ; ... loop unrolled 2x: same 12-instruction body repeats
```

- `vpshldq` emitted? **No** (0). LLVM does NOT pattern-match the
  `(a >> s) | (b << (64-s))` pair into the variable-count `vpshldvq` even
  though the data flow is the textbook funnel-shift.
- LLVM does fold the trailing `(... | ...) & MASK` into a single `vpternlogq`
  (3-input bitwise), so the body is 5 vector ALU ops per chunk: `vpsrlvq`,
  `vpsllvq`, `vpternlogq`, `vpaddq`, plus two `vpgatherqq` loads which
  dominate the cost.
- Inner-loop ALU ops/element: ~5 vector ops on 8-element zmm chunks
  (~0.6 ops/element), but two `vpgatherqq` per 8 outputs.
- Median: **2.227 us / 1024 elements = 2.18 ns per output**. Slower than the
  scalar Variants 1-2 because the gather is the bottleneck.

---

## Variant 4: `pat_u128_cat` (concatenate via u128, shift, truncate)

```rust
fn pat_u128_cat(packed: &[u64; PACKED_LEN], reference: u64, out: &mut [u64; 1024]) {
    for i in 0..1024 {
        let bit_pos = i * W;
        let word = bit_pos / 64;
        let shift = (bit_pos % 64) as u32;
        let next = (word + 1).min(PACKED_LEN - 1);
        let combined = (packed[word] as u128) | ((packed[next] as u128) << 64);
        let val = ((combined >> shift) as u64) & MASK;
        out[i] = val.wrapping_add(reference);
    }
}
```

Inner loop (partially vectorized — gather then scalar `shrd`):

```text
   59270: vpmullq %zmm2,%zmm0,%zmm7
   59276: vpsrlq  $0x6,%zmm7,%zmm8
   59295: vpgatherqq (%rax,%zmm9,8),%zmm10{%k1}
   5929c: vpgatherqq (%rdi,%zmm8,8),%zmm11{%k2}
   ; ... many vpextract / vmovq to spill 8 lanes to GPRs ...
   592e0: shrd   %cl,%r8,%r9                  ; scalar funnel-shift
   592f0: shrx   %rcx,%r8,%r8
   59307: shrd   %cl,%rbx,%r10
   ; ... 8x scalar shrd/shrx, then 8x vmovq/vpunpcklqdq to reassemble ...
   5943e: vpandq %zmm5,%zmm7,%zmm7            ; & MASK
   59444: vpaddq %zmm1,%zmm7,%zmm7            ; + reference
```

- `vpshldq` emitted? **No** (0). LLVM does emit the *scalar* x86 `shrd`
  funnel-shift but loses on the vectorization: it gathers + extracts to GPRs
  + computes scalar + reassembles + stores. Very expensive.
- Inner-loop ALU ops/element: ~15 (scalar shrd + many extract/insert moves).
- Median: **3.423 us / 1024 elements = 3.34 ns per output**. Slowest of the
  six.

---

## Variant 5: `pat_u128_cat_unrolled4` (Variant 4 unrolled 4x)

```rust
fn pat_u128_cat_unrolled4(packed: &[u64; PACKED_LEN], reference: u64, out: &mut [u64; 1024]) {
    let mut i = 0;
    while i + 3 < 1024 {
        for k in 0..4 {
            let bit_pos = (i + k) * W;
            let word = bit_pos / 64;
            let shift = (bit_pos % 64) as u32;
            let next = (word + 1).min(PACKED_LEN - 1);
            let combined = (packed[word] as u128) | ((packed[next] as u128) << 64);
            let val = ((combined >> shift) as u64) & MASK;
            out[i + k] = val.wrapping_add(reference);
        }
        i += 4;
    }
    // tail elided
}
```

Same shape as Variant 4 — LLVM still goes gather->scalar `shrd`->scatter.
Manual unroll did not change the lowering strategy.

- `vpshldq` emitted? **No** (0). The scalar `shrd` is preserved (32 occurrences
  across the unrolled body) but no vector funnel-shift is generated.
- Inner-loop ALU ops/element: ~15 (same density as Variant 4).
- Median: **3.329 us / 1024 elements = 3.25 ns per output**.

---

## Variant 6: `pat_chunked_aligned` (lane-major, 16 lanes x 64 rows)

```rust
fn pat_chunked_aligned(packed: &[u64; PACKED_LEN], reference: u64, out: &mut [u64; 1024]) {
    const LANES: usize = 16;
    for lane in 0..LANES {
        for row in 0..64 {
            let bit_pos = (row * LANES + lane) * W;
            let word = bit_pos / 64;
            let shift = (bit_pos % 64) as u32;
            let next = (word + 1).min(PACKED_LEN - 1);
            let lo = packed[word] >> shift;
            let hi = packed[next] << (64 - shift);
            let val = (lo | hi) & MASK;
            out[row * LANES + lane] = val.wrapping_add(reference);
        }
    }
}
```

Inner loop (vectorized, but lane-variable shift):

```text
   59630: vpbroadcastq %rcx,%zmm29
   59636: vpmullq -0x3a8f0(%rip){1to8},%zmm29,%zmm30   ; per-lane bit_pos
   59640: vpaddq -0x39d0a(%rip),%zmm30,%zmm31          ; + lane offsets
   5964a: vpsrlq  $0x6,%zmm31,%zmm1                    ; word
   59663: vpgatherqq (%rdi,%zmm1,8),%zmm3{%k1}         ; packed[word]
   5966e: vpgatherqq (%rax,%zmm2,8),%zmm1{%k2}         ; packed[next]
   59675: vpandq  %zmm5,%zmm31,%zmm2                   ; shift = bit_pos & 63
   5967b: vpsrlvq %zmm2,%zmm3,%zmm2                    ; lo = packed[word] >> shift
   59681: vpsubq  %zmm30,%zmm6,%zmm3                   ; 64 - shift
   59691: vpandq  %zmm5,%zmm3,%zmm3
   59697: vpsllvq %zmm3,%zmm1,%zmm1                    ; hi << (64-shift)
   5969d: vpternlogq $0xc8,%zmm2,%zmm7,%zmm1           ; (lo | hi) & MASK
   596a4: vpaddq  %zmm0,%zmm1,%zmm1                    ; + reference
   596ae: vpscatterqq %zmm1,(%rdx,%zmm31,8){%k1}
   ; ... 7 more identical 13-instruction chunks (8 chunks total per outer iter) ...
```

- `vpshldq` emitted? **No** (0). Same exact pattern as Variant 3:
  `vpsrlvq + vpsllvq + vpternlogq + vpaddq`. The lane-major reorganization
  still produces a per-lane variable shift inside the vector (each of the 8
  zmm lanes has a different shift count for the same row), so LLVM emits
  the variable-shift form.
- Inner-loop ALU ops/element: ~5 vector ops + gather + scatter per chunk,
  unrolled 8x.
- Median: **2.344 us / 1024 elements = 2.29 ns per output**.

---

## Summary

| variant                        | vpshldq emitted? | inner-loop ALU ops/element        | median (us / 1024 elems) | ns/elem |
|--------------------------------|:---:|------------------------------------|-------------------------:|--------:|
| pat_macro_shape                | no  | ~10 scalar (shrx/bzhi/shlx/or/add) |                    1.490 |    1.45 |
| pat_combine_then_mask          | no  | ~5 scalar (shrx/shlx/or/bzhi/add)  |                    1.144 |    1.12 |
| pat_branchless_funnel          | no  | vpsrlvq+vpsllvq+vpternlogq+vpaddq  |                    2.227 |    2.18 |
| pat_u128_cat                   | no  | gather+scalar shrd+scatter         |                    3.423 |    3.34 |
| pat_u128_cat_unrolled4         | no  | gather+scalar shrd+scatter         |                    3.329 |    3.25 |
| pat_chunked_aligned            | no  | vpsrlvq+vpsllvq+vpternlogq+vpaddq  |                    2.344 |    2.29 |

## Did any pattern get LLVM to emit `vpshldq + vpand + vpaddq`?

**No.** Across all six source-level variants, LLVM never emits `vpshldq`
(or `vpshrdq`) in the inner loop:

- Variants 1, 2 lower to **pure scalar** `shrx + bzhi + shlx + or + add`.
  LLVM's loop-vectorizer rejects the loop because the per-element variable
  shift count plus the `if` branch makes profitability hard to estimate when
  the function is `#[inline(never)]` (so `1024` is not a constant trip count
  in the call site).
- Variants 3, 6 successfully autovectorize but lower to the
  *variable-count* funnel idiom `vpsrlvq + vpsllvq + vpternlogq + vpaddq`.
  LLVM has a peephole that recognizes `(a >> s) | (b << (T - s))` for
  *constant* `s` and lowers to `vpshldq` immediate, but for *vector* `s`
  (which is what a per-element loop necessarily produces) it falls back to
  the variable-shift pair. Even on x86 with VBMI2, `vpshldvq` (variable
  count) exists but LLVM does not currently match the variable-count
  pattern from this Rust IR shape.
- Variants 4, 5 do trigger LLVM's funnel-shift recognizer (they emit
  scalar `shrd`!) but only at scalar width, with very expensive
  gather/scalar/scatter wrapping. The autovectorizer cannot widen scalar
  `shrd` into `vpshldq` because each lane needs an independent shift.
- Variant 2 is **fastest** (1.144 us), but only because purely scalar
  `shrx + bzhi + add` on a modern out-of-order core is already very cheap
  (~1 ns per output) and avoids gather/scatter entirely.
- The hand-written intrinsic baseline in `funnel_shift_fix.rs` (the
  `hand_funnel_w51` variant) does emit `vpshrdq + vpaddq` and runs at the
  expected ~3 ALU ops/4-element chunk, but only because (a) it picks a
  single constant shift count `K=20`, and (b) it bypasses the FastLanes
  bit-position arithmetic entirely. Neither of those is achievable from
  plain safe Rust without the per-shift specialization.

## Recommendation

There is no plain-safe-Rust source-level rewrite of the `unpack!` macro body
that gets LLVM to emit `vpshldq + vpand + vpaddq` in the inner loop. The
upstream pattern-matcher gap is in LLVM, not in the Rust source shape. Two
fundamentally different paths forward:

1. **Specialize the macro by `(W, lane-row pair)`**, generating code where
   `shift` is a const-generic. The FastLanes layout already groups elements
   so that within one row of one lane the bit position is fixed, but the
   current `unpack!` macro computes `shift` from `row * W % T` *at runtime*
   inside the loop. If the macro were rewritten to enumerate the 64 (row,
   lane) pairs as 64 distinct `const SHIFT: u32` instantiations -- e.g.
   via `seq_t!` expanding to 64 statements where the shift is a literal --
   LLVM's existing constant-shift `vpshldq` peephole fires. This is the
   right place to change `mask-then-combine` (lines 153-176 of
   `benchmarks/fastlanes-kernel-bench/src/macros.rs`): replace the runtime
   `shift = (row * $W) % T` with a paste-generated `const SHIFT_<row>: u32`,
   and rewrite the body to `let val = (((src >> SHIFT) | (next.wrapping_shl(T - SHIFT))) & MASK).wrapping_add(reference)`
   with the SHIFT-equals-0 case branched out at compile time.

2. **Accept the current scalar lowering for fused-FoR.** Variant 2
   (`pat_combine_then_mask`) at 1.12 ns/elem is *already faster* than the
   matrix's reported fused 195 ns (~0.19 ns/elem at full 1024-element
   chunk amortization, but with FL_ORDER lane interleaving overhead). If
   the goal is purely to close the +30-50% gap vs. the bare unpack at the
   same W, switching the macro from "mask-then-combine" to "combine-then-
   mask" (replace lines 160-167 of `macros.rs` with the unconditional
   `tmp = ((src >> shift) | (next.wrapping_shl(T - shift))) & mask($W)`)
   should at least let LLVM unroll the no-hi-half branch out and drop one
   ALU op per iteration. That would not produce `vpshldq`, but it is a
   one-line change that removes the dead `mask(current_bits)` intermediate.

Path (1) is the only way to actually emit `vpshldq` from safe Rust, and is
the recommended follow-up for this experiment.

## Files

- `funnel_patterns_asm/pat_macro_shape.s`
- `funnel_patterns_asm/pat_combine_then_mask.s`
- `funnel_patterns_asm/pat_branchless_funnel.s`
- `funnel_patterns_asm/pat_u128_cat.s`
- `funnel_patterns_asm/pat_u128_cat_unrolled4.s`
- `funnel_patterns_asm/pat_chunked_aligned.s`
- `funnel_patterns.csv`
