# `collect_bool` / bitmask-pack SIMD audit

Audit of every place in Vortex that packs one bit per element into an LSB-first
bitmask ("collect-bool"), classified by whether it can benefit from an AVX-512
mask-compare lowering. No source kernels are changed by this commit — this is an
analysis artifact only.

## The insight

Packing a per-element predicate into a dense bitmask is, on AVX-512, ideally a
single mask-producing compare into an opmask (`k`) register plus a `kmov` store:

| operation | ideal kernel | elements / iter |
| --- | --- | --- |
| `&[bool]` → bits (`b != 0`) | `vmovdqu64` → `vptestmb` → `kmovq` | 64 |
| `&[i32]` → bits (`v > 0`) | `vpcmpd`/`vpcmpltd (mem),zmm,k` → `kmovw` | 16 |

But the natural scalar idiom

```rust
let mut packed = 0u64;
for i in 0..len { packed |= (pred(i) as u64) << i; }   // collect_bool_word
```

does **not** lower to that. With the constant 64-trip loop fully unrolled, LLVM's
**SLP vectorizer** rewrites the straight-line `shl`/`or` chain into a per-lane
variable shift (`vpsllvq`) plus an `llvm.vector.reduce.or` shuffle tree — dozens
of µops to fold 64 lanes into one word. (For the `&[i32]` predicate case the
auto-vectorizer did even worse, emitting `vpgatherqd`.)

### Compiler-pass provenance (rustc 1.91 / LLVM 21)

Traced with `-Cllvm-args=-print-after-all`:

- rustc hands LLVM **fully scalar** IR (`%_21 = shl i64 %_22, %23`, `or i64`, in a
  `phi` loop) — verified with `-Cno-prepopulate-passes`. **The Rust frontend is
  faithful; this is not a rustc bug.**
- `LoopVectorizePass` runs and **declines** (0 vector ops after it).
- `SLPVectorizerPass` is the culprit — first vector op to appear is
  `call i8 @llvm.vector.reduce.or.v4i8(...)` in *IR Dump After SLPVectorizerPass*.
- The **good** `vptestmb`/`vpcmpd` form is produced by the **X86 SelectionDAG
  instruction selector**, not by any IR pass: it recognizes `icmp <N x iM>` +
  `store <N x i1>` (or `bitcast <N x i1> to iN`) and materializes a `k` register.
  The intrinsic version reaches ISel already in that shape, so the backend "just
  does the right thing."

**Conclusion: it's an LLVM middle-end optimization gap (movemask/bitmask-pack
idiom not recognized), not a Rust problem.** Fix options, easiest first:

1. **Library fix (in our control, recommended):** express the hot kernels as a
   vector compare → bitmask via `core::arch` intrinsics behind
   `target_feature` + `is_x86_feature_detected!` (stable), or `std::simd`
   `Mask::to_bitmask` (portable, nightly today). Deterministically yields
   `vptestmb`/`vpcmpd` + `kmov`.
2. **LLVM patch (upstream, real work):** teach `VectorCombine`/SLP to fold
   `or-reduce(zext(icmp_i) << i)` into `bitcast(<N x i1> cmp) to iN`. Hazards:
   arbitrary predicates, lane/bit order, not regressing genuine shift-OR. Only
   helps once a newer LLVM ships in rustc.
3. **rustc:** not the right layer (would just be a stdlib specialization = #1).

### Measured impact (best-of-12, pinned to one core, inputs/outputs `black_box`ed)

`collect_bool` (`&[bool]`, 1 byte in / 1 bit out):

| working set | SSE2 autovec | AVX-512 autovec | **`vptestmb` (opt)** | opt vs SSE2 | opt vs AVX-512 autovec |
| --- | --- | --- | --- | --- | --- |
| 32 KiB (L1) | 7.4 Gelem/s | 12.0 | **~145 (140 GiB/s)** | **~20×** | **~12×** |
| 1 MiB (L2/L3) | 7.7 | 12.7 | ~89 | ~11.6× | ~7× |
| 16 MiB (DRAM) | 3.9 | 5.2 | ~22 | ~5.6× | ~4.3× |

`collect_gt0` (`&[i32]`, 4 bytes in / 1 bit out):

| working set | autovec | **`vpcmpd` (opt)** | speedup |
| --- | --- | --- | --- |
| 128 KiB | ~7 Gelem/s | ~22 (87 GiB/s) | ~3.2× |
| 4 MiB | ~4 | ~7 | ~1.7× |
| 64 MiB | ~4.3 | ~6.5 | ~1.4× |

The win is largest when compute-bound (cache-resident); as data spills to DRAM
both converge on the bandwidth ceiling. `collect_bool` wins more than
`collect_gt0` because it reads 4× fewer bytes per element, so it stays
compute-bound far longer.

## Codebase classification

Categories:

- **A — Primary candidate.** Scalar shift-OR pack whose predicate is a simple
  comparison over a **contiguous** slice. Directly maps to `vpcmpd`/`vptestmb` + `kmov`.
- **B — Gather-bound.** Predicate reads through index indirection or carries
  loop state / heavy work (DFA). SIMD needs gather+compress or restructuring; smaller win.
- **C — Already SIMD.** Uses BMI2 / AVX-512 / NEON intrinsics with runtime dispatch. **Do not touch.**
- **D — Delegates / not-a-pack.** Delegates to Arrow, the primitive below, or batch fill.
- **E — Cold / low-volume.** Correctness / small-N path; not worth it.

### Category A — primary candidates

| site | what | why A |
| --- | --- | --- |
| `vortex-buffer/src/bit/mod.rs:34` `collect_bool_word` | `packed \|= (f(i) as u64) << i` | **the primitive**; benefits every contiguous caller below |
| `vortex-buffer/src/bit/buf_mut.rs:188` `BitBufferMut::collect_bool` | calls `collect_bool_words` into a `u64` slice | **the chokepoint** — all bulk packing funnels here |
| `vortex-buffer/src/bit/buf_mut.rs:564` `From<&[bool]>` | `collect_bool(len, \|i\| value[i])` | stride-1 contiguous load → `vptestmb` |
| `vortex-buffer/src/bit/buf_mut.rs:571` `From<&[u8]>` | `collect_bool(len, \|i\| value[i] > 0)` | contiguous `> 0` → `vpcmpub`+`kmov` |
| `vortex-array/src/arrays/primitive/compute/between.rs:108` `between_impl_` | `lower_fn(lower,s[i]) & upper_fn(s[i],upper)` over `&[T]` | **hottest end-user path** (between on any primitive column) |
| `vortex-array/src/arrays/decimal/compute/between.rs:135` `between_impl` | two-sided compare over contiguous decimal buffer | same shape as primitive between (cooler) |
| `vortex-array/src/arrays/varbin/compute/compare.rs:149` `compare_offsets_to_empty` | `offsets[i] == offsets[i+1]` over contiguous offsets | empty-string filter → `vpcmpeqd`+`kmov` |
| `encodings/fastlanes/src/bitpacking/compute/stream_predicate.rs:62` `stream_predicate` | `pack_bools_into_words(.., \|i\| predicate(block[i]))` over decoded 1024-elem block | FastLanes hot scan path; contiguous scratch block |

### Category B — gather-bound / heavy predicate

| site | why B |
| --- | --- |
| `vortex-array/src/arrays/bool/compute/filter.rs:50` `filter_sparse` (indices) | `get_bit(buf, offset + indices[i])` — gather |
| `vortex-array/src/arrays/bool/compute/filter.rs:76` `filter_set_bits` | reads non-contiguous set-bit positions |
| `vortex-array/src/arrays/bool/compute/take.rs:70` `take_byte_bool` | `bools[indices[i]]` — arbitrary permutation |
| `vortex-array/src/arrays/bool/compute/take.rs:78` `take_bool_impl` | `get_bit(buf, indices[i])` — gather |
| `vortex-array/src/patches.rs:716` patch mask | `!masked.value(patch_indices[i]-offset)` — sparse gather, cold |
| `vortex-buffer/src/bit/buf.rs:144` `BitBuffer::from_indices` | sparse scatter-set |
| `encodings/runend/src/compute/filter.rs:89` `filter_run_end_primitive` | loop-carried `start`/`count` cursor; variable-length run walk |
| `encodings/fsst/src/dfa/mod.rs:297` `dfa_scan_to_bitbuf` | predicate is a full DFA traversal over variable-length bytes |

### Category C — already SIMD (do not touch)

| site | technique |
| --- | --- |
| `vortex-buffer/src/bit/count_ones.rs` | AVX2 VPSHUFB / AVX-512 `_mm512_popcnt_epi64` popcount, dispatched |
| `vortex-array/src/arrays/bool/compute/filter.rs:110` `filter_pext_bmi2` (+fallback) | BMI2 `_pext_u64` with software byte-LUT fallback |
| `encodings/fastlanes/src/bitpacking/compute/compare_fused.rs:103` `stream_compare_fused` | FastLanes fused `unchecked_unpack_cmp` + SIMD `untranspose_bits` |
| `encodings/fastlanes/src/bit_transpose/x86.rs:49` `transpose_bits_bmi2`/`_vbmi` | BMI2 PEXT/PDEP and AVX-512 VBMI `_mm512_permutexvar_epi8` |
| `encodings/fastlanes/src/bit_transpose/mod.rs:50` `transpose_bits`/`untranspose_bits` | runtime dispatch wrappers over the above |
| `vortex-mask/src/intersect_by_rank.rs:131` `pdep_bmi2` | BMI2 `_pdep_u64` with portable fallback |

### Category D — delegates / not-a-pack

`vortex-buffer/src/bit/mod.rs:52` `collect_bool_words`,
`vortex-buffer/src/bit/mod.rs:110` `pack_bools_into_words`,
`vortex-buffer/src/bit/buf.rs:164` `BitBuffer::collect_bool`,
`vortex-array/src/scalar_fn/fns/binary/compare.rs:114` `execute_compare` (→ Arrow),
`vortex-array/src/validity.rs:482` `FromIterator<bool> for Validity`,
`vortex-array/src/arrays/bool/array.rs:333` `FromIterator<bool> for BoolArray`,
`vortex-array/src/builders/bool.rs` `BoolBuilder` (batch `append_n`),
`encodings/fastlanes/src/bitpacking/compute/between.rs:133` `between_constant_typed` (→ `stream_predicate`),
`vortex-mask/src/lib.rs:189` `Mask::from_buffer`/`from_indices`/`from_excluded_indices`.

(These either funnel into the Category-A primitive — and so improve for free if
it is fixed — or hand the bitmap off to Arrow.)

### Category E — cold / low-volume

`vortex-buffer/src/bit/mod.rs:89` `splice_word_at_bit` (constant-work helper),
`vortex-buffer/src/bit/buf_mut.rs:583` `FromIterator<bool> for BitBufferMut`
(scatter prefix then 1-bit `append`; no contiguous slice).

## Recommendation

One change unlocks most of the value: give `collect_bool_word` /
`collect_bool_words` a **concrete-slice fast path** (e.g. `&[bool]`, `&[u8]`, and
the typed comparison kernels) implemented with a vector compare → bitmask
(`core::arch` + runtime dispatch, or `std::simd::Mask::to_bitmask`), instead of
relying on the opaque `FnMut(usize) -> bool` closure that LLVM's SLP vectorizer
mishandles. Because `BitBufferMut::collect_bool` is the chokepoint that every
bulk path funnels through, and `primitive between` + FastLanes `stream_predicate`
are the hottest contiguous callers, fixing the primitive (plus exposing a
slice/compare entry point those two can call) captures the Category-A wins
without touching the already-optimal Category-C kernels.

### Reproduction

The four kernels and the benchmark/`-print-after-all` methodology used above are
standalone (rustc `--emit=llvm-ir,asm`, `target-cpu=x86-64` vs `x86-64-v4`,
`#[inline(never)]`, `std::hint::black_box` on inputs each iteration and outputs
after each call). The optimal kernels:

```rust
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn collect_bool_avx512(input: &[bool], out: &mut [u64]) {
    let inp = input.as_ptr() as *const __m512i;
    for i in 0..input.len() / 64 {
        let v = _mm512_loadu_si512(inp.add(i));
        *out.get_unchecked_mut(i) = _mm512_test_epi8_mask(v, v);   // vptestmb
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn collect_gt0_avx512(input: &[i32], out: &mut [u16]) {
    let inp = input.as_ptr() as *const __m512i;
    let zero = _mm512_setzero_si512();
    for i in 0..input.len() / 16 {
        let v = _mm512_loadu_si512(inp.add(i));
        *out.get_unchecked_mut(i) = _mm512_cmpgt_epi32_mask(v, zero); // vpcmpd
    }
}
```
