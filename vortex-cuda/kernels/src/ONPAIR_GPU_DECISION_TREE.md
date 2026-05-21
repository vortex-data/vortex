# OnPair GPU decode — optimization decision tree

Every avenue explored for the OnPair string-decompression kernels on GH200
(Hopper, sm_90), with implementation, measured outcome, and the reason. Verdict
tags: ✅ adopt · ❌ reject · ⟷ neutral/tie · 🔬 untested (reasoned or prior-art).
Numbers are GPU decode throughput unless noted; see `ONPAIR_GPU_FINDINGS.md` for
raw NCU. Profiling premise: the baseline is **L1/TEX cache-request bound (86–93%),
DRAM idle (17%), register-capped at 64 regs/thread → 50% occupancy**. So the only
real lever is *reduce L1/TEX request work*; everything else just moves it.

```
OnPair GPU decode optimization
│
├─ AXIS 1 — DICTIONARY LAYOUT (where the dict lives + how a token is read)   ◀ the dominant axis
│   │  Per token: read code → look up (bytes,len) → copy bytes to output.
│   │  Access is a RANDOM GATHER (32 lanes → 32 arbitrary dict entries). This
│   │  pattern is the crux of every decision below.
│   │
│   ├─ padded-16, global  (dict_padded, uint4 16B/entry)  ── BASELINE `onpair_shmem_4tpt`   ✅ reference
│   │     impl: t = *(uint4*)(dict_padded + code*16); len = lens[code].
│   │     found: L2-resident (94% L2 hit) but L1 hit only 31%, 31% bytes/sector
│   │            (uncoalesced) → L1/TEX 86–93% = the wall. DRAM idle.
│   │     why:  simplest correct; aligned 16B reads; over-reads short tokens but
│   │           one 32B sector/token regardless. The bottleneck to beat.
│   │
│   ├─ split-read 8B  (dict_s8 32KB for lo, dict_padded for hi)  `onpair_shmem_4tpt_split8read`   ✅ bits12 / ❌ bits16
│   │     impl: lo = uint2 (8B) from dict_s8; if len>8 also hi = uint2 from
│   │           dict_padded+8. Same scan/drain as 4tpt.
│   │     found: bits12 +4–11% (fineweb 1.11×, wiki 1.09×, book-rev 1.04×,
│   │            ps_comment tie); bits16 −5–26% (ps_comment 0.74×). NCU: L1/TEX
│   │            86%→73%.
│   │     why GOOD (bits12): dict_s8 = 4096·8 = 32KB fits L1; reading 8B not 16B
│   │           pushes ~half the dict bytes through the saturated L1/TEX pipe.
│   │           First idea that *reduces* pipe work instead of relocating it.
│   │     why BAD (bits16): dict_s8 = 65536·8 = 512KB, no L1 benefit; and long
│   │           tokens (len>8) pay a 2nd gather into the 1MB dict_padded.
│   │     → SELECT only when dict ≤ ~4096 entries.
│   │
│   ├─ split-read at 4 B  `onpair_shmem_4tpt_split4read`   ❌ regresses (−13% bits12)
│   │     impl: 4 B (uint) from the 16 KB `dict_s4` common case; `len>4` high 12 B
│   │           from `dict_padded`. Motivated by the histogram (fineweb/b12: 77%
│   │           of tokens <= 4 B).
│   │     found: fineweb/b12 1.83 (vs split8read 1.60), bookrev/b12 0.875 (vs
│   │           0.787). Validated.
│   │     why BAD: L1/TEX works at **32 B sector granularity** — a 4 B and an 8 B
│   │           gather both cost one sector, so reading 4 B saves no sectors. And
│   │           the 23% `len>4` fallback hits the **64 KB** `dict_padded`,
│   │           enlarging the working set vs split8read's mostly-32 KB. The
│   │           byte-reduction lever bottoms out at 8 B; split8read is the sweet
│   │           spot.
│   │
│   ├─ variable-stride length-bucket  `onpair_shmem_4tpt_lenbucket`   ❌ bits12 −26% / ⟷ bits16 tie
│   │     impl: entries packed at stride {4,8,12,16} by width into 4 regions;
│   │           code value selects bucket via 3 thresholds; aligned per-bucket
│   │           reads, no over-read, single read (vs split8read's double read).
│   │           Decode-side repack via `ONPAIR_DICT_REORDER=lenbucket`.
│   │     found: bits12 2.28 ms (vs 4tpt 1.81, split8read 1.60); bits16 2.03 ms
│   │           (vs 4tpt 2.01, split8read 2.19). Validated byte-exact.
│   │     why BAD: the per-token bucket branch diverges the warp (4 paths, mixed
│   │           read widths across lanes) — that cost exceeds the smaller-dict L1
│   │           benefit, which is zero on bits12 (dict already fits L1) and only
│   │           a wash on bits16. split8read (2 buckets, uniform 8B common read)
│   │           is the better point on this curve.
│   │
│   ├─ padded-16, shared (persistent grid)  `onpair_shmem_4tpt_pdict`   ❌ −27%
│   │     impl: cooperatively load 64KB padded dict to shared once/block,
│   │           grid-stride over chunks; uint4 reads from shared.
│   │     found: global gather eliminated (549M→21M sectors, 31%→92% coalesced)
│   │            BUT 82M shared bank conflicts + occupancy halved (64KB → 1
│   │            block/SM → 24%).
│   │     why BAD: random uint4 shared reads hit 4 banks each → 32 lanes collide;
│   │           and 64KB shared starves occupancy. Doubly wrong.
│   │
│   ├─ var-len packed, shared (persistent)  `onpair_shmem_4tpt_vdict`   ❌ −53%
│   │     impl: ~17KB packed dict_bytes in shared (off,len from global
│   │           dict_table); byte-granular shared→shared copy.
│   │     found: 78M shared bank conflicts; occ better than pdict (37.5%, smaller
│   │            footprint) but serial byte-copy + conflicts dominate.
│   │     why BAD: a random gather conflicts regardless of element width; byte
│   │           loop is serial. Footprint intuition was right, conflicts win.
│   │     ── CONCLUSION for "dict in shared": dead end. The dict already lives in
│   │        L2 at 94% hit; L2's gather machinery beats shared-memory banks for
│   │        random access. Trades tolerable L1-request pressure for intolerable
│   │        bank-conflict pressure. (Matches prior `_tma` history failures.)
│   │
│   ├─ var-len packed, global (read len bytes from dict_bytes)   🔬 rejected by reasoning
│   │     why: byte-granular unaligned global gathers; worse coalescing than the
│   │          16B aligned read. Not implemented.
│   │
│   ├─ stride-8 only  `onpair_shmem_s8*`   ⟷ only if max_len ≤ 8 (prior art)
│   │     found: inapplicable to our text columns (max_len > 8). dict_s8 alone
│   │            can't represent len 9–16. (split-read 8B generalises it.)
│   │
│   ├─ stride-4 / s4l1  `onpair_shmem_s4l1*`   ⟷ only if max_len ≤ 4 (prior art)
│   │     note: an s4 *shared* variant was removed earlier — regressed on A100.
│   │
│   └─ const1 / const2  (1 or 2 bytes/entry)   ⟷ only if all entries len 1/2
│         use: low-cardinality categoricals (flags, codes). Auto-selected.
│
├─ AXIS 2 — TOKEN EMISSION / OUTPUT STAGING (how decoded bytes reach global mem)
│   │
│   ├─ shared staging + byte-ladder  (BASELINE)   ✅
│   │     impl: scatter token bytes into a per-warp shared buffer at prefix-sum
│   │           offsets, then drain as aligned 16B uint4 global stores.
│   │     why:  turns variable-length unaligned token writes into coalesced
│   │           aligned global stores. Essential.
│   │
│   ├─ split-8 write  `onpair_shmem_4tpt_split8`   ⟷ ~tie/slightly slower
│   │     impl: split the byte-ladder at 8B to cut predicated-off stores
│   │           (~21% of stores are masked off, mean len ~6).
│   │     why BAD: the write side is NOT the bottleneck (shared-store conflicts
│   │           only 3.9M); predication-off stores don't consume LSU bandwidth.
│   │
│   ├─ split-4 write   ❌ slower (prior art)
│   │
│   └─ direct global scatter (no staging)   🔬 rejected by reasoning
│         why: unaligned, uncoalesced global writes — the reason staging exists.
│
├─ AXIS 3 — THREAD↔WORK MAPPING (tokens per thread)
│   ├─ 1 tpt `onpair_shmem` / 2 tpt `_2tpt`   ❌ slower — under-amortise the warp scan.
│   ├─ 4 tpt  (BASELINE)   ✅ best — 128 tokens/warp amortises scan + head/tail epilogue.
│   └─ 8 tpt (stride16)   ❌ driver rejected PTX (CUDA_ERROR_INVALID_PTX), prior art.
│
├─ AXIS 4 — OCCUPANCY / REGISTERS / LAUNCH BOUNDS
│   │  Baseline: 64 regs/thread, 512 threads, __launch_bounds__(512,2) → 2
│   │  blocks/SM = 50% (register-limited).
│   ├─ 8 warps/block  `_wpb8`   ❌ −12% — fewer warps, no occupancy gain (still 50%).
│   ├─ wpb8 + occ tuning  `_wpb8_occ` (__launch_bounds__(256,4))   ❌ 10-iter NOISE
│   │     The prior handoff's "winner". At 300 iters it ties-or-loses vs 4tpt.
│   │     Root cause of the false win: unlocked GPU clocks + 1.6ms warmup. See
│   │     `ONPAIR_GPU_FINDINGS.md` "Measurement methodology".
│   └─ register reduction → 3 blocks/SM   ❌ rejected by analysis
│         need ≤42 regs (have 64) for 75% occ — infeasible (4tpt holds 4 tokens'
│         state across the scan). And at 93% L1/TEX, more occupancy can't help.
│
├─ AXIS 5 — DICTIONARY PREFETCH INTO SHARED (enabling techniques for AXIS-1 shared variants)
│   ├─ cooperative load + __syncthreads   ❌ −22–33% (prior art) — barrier + no amortisation.
│   ├─ per-thread cp.async.cg   ❌ −22–33% + ILLEGAL_ADDRESS on max_len=16 (prior art).
│   ├─ TMA cp.async.bulk  `onpair_shmem_tma`   🔬 gated behind ONPAIR_ENABLE_TMA; padded dict; fragile.
│   └─ persistent grid-stride (my addition)   ⟷ correctly amortises the load…
│         …but the shared-dict variants it enables still lose to bank conflicts
│         (AXIS 1). Good technique, wrong target.
│
├─ AXIS 5b — DICT CODE ORDERING (decode-side relabel; permutation = byte-exact, no compressor/disk change)
│   │  Tested via env `ONPAIR_DICT_REORDER` in the GpuOnPairChunk builder.
│   │  Key fact: this relabel PROVES OnPair does not currently order codes by
│   │  frequency (it changes results, so codes aren't already sorted).
│   │
│   ├─ frequency-sort (hot codes → low indices)   ✅ bits16 (+8–13%) / ⟷ bits12
│   │     impl: count code freq over the stream, sort dict entries descending.
│   │     found (NCU 4tpt): bits16 L1 sector hit 35%→45%, 1.87→1.71 ms (8.6%);
│   │           split8read 2.28→1.99 (12.7%). bits12 ~neutral (1.82→1.80) —
│   │           its 64KB dict already fits L1 (88% hit), nothing to cluster.
│   │     why GOOD: clustering hot entries makes the *effective* working set
│   │           small enough to stay L1-resident → fewer L2 trips. Same mechanism
│   │           as split8read, via ordering; COMPOSES with it.
│   │     caveat: to be free it must live in the ENCODER (decode-side relabel is
│   │           an O(tokens) host pass). Out of current scope (no compressor
│   │           change) — but a validated, high-value encoder recommendation,
│   │           esp. for high-cardinality bits16 columns.
│   │
│   └─ length-sort (cluster by token width)   ❌ HURTS (bits16 4tpt 1.87→2.02)
│         why BAD: length does not correlate with frequency, so length-clustering
│         SCATTERS the hot entries → worse L1 locality than the original order.
│         Ordering by length is only useful as the basis for a variable-stride
│         compact dict (AXIS 1 → lenbucket), not as a relabel by itself.
│
├─ AXIS 6 — CHUNK SIZE (compression vs decode; one OnPair dict per chunk)
│   ├─ bits12 (dict saturates ~4096 entries / ~17KB within ~10MB text)
│   │     compression ratio FLAT across 10/100/1000MB (fineweb 2.24–2.26).
│   │     → chunk size is free; pick large for fewer/faster GPU launches.
│   └─ bits16 / high-cardinality (dict does NOT saturate)
│         small chunks replicate a big dict: wiki/bits16 dict 456KB→26MB and
│         ratio 2.815→2.345 going 1000→10MB. → prefer LARGE chunks for compression.
│   note: bigger chunks also decode faster (fineweb b12: 1.64 / 1.72 / 2.43 ms
│         for 1000 / 100 / 10 MB) — fewer, larger kernel launches.
│
└─ AXIS 7 — KERNEL SELECTION POLICY (the actionable output) — ✅ ROLLED IN
      pick_auto_kernel (decode-side, no layout/result change):
        if all entries len==1 → const1 ; len==2 → const2
        elif max_len ≤ 4      → s4l1_16tpt ; ≤ 8 → s8_4tpt
        elif (dict ≤ 4096 entries) AND (token-weighted frac(len≤8) ≥ 0.90)
                              → ✅ split8read
        else                  → ✅ 4tpt        (was 2tpt — 4tpt is faster)
      `frac_le8` is computed per chunk (one pass over codes). Verified routing:
      fineweb/book-reviews text → split8read; l_comment/ps_comment/URL → 4tpt.
      Captures every measured win, avoids every regression.

      Per-column data (bits12, freq, 80 iters; ms; best in bold conceptually):
        column          mean  ≤8%    4tpt    split8read   pick
        fineweb         4.27  98.5   1.796   1.602(-11%)  split8read
        wikipedia       4.17  98.2   1.312   1.183(-10%)  split8read
        book-reviews    4.84  95.7   0.821   0.792(-4%)   split8read
        i_item_desc     4.93  94.3   0.0212  0.0208       split8read
        clickbench/URL  5.22  80.4   1.354   1.343        4tpt (tie)
        l_comment       8.16  57.6   0.946   0.987(+4%)   4tpt
        p_name         11.10  62.2   0.064   0.066        4tpt
        ps_comment      9.57  32.6   0.811   0.814        4tpt
      Crossover ≈ frac(len≤8) 0.90; threshold chosen there.
```

## AXIS 8 — gaps vs GPU-(de)compression research (not yet explored)

Framing from the literature: OnPair decode is a **read-only random gather into a
small, hot, reused table, with idle DRAM** — the textbook prescription is "keep
the table in the fastest cache and cut per-access cost." We've cut per-access
bytes (split8read, lenbucket) and improved residency via ordering (freq-sort).
What standard techniques remain:

1. **L2 persistence / access-policy window (Hopper)** — ❌ TESTED, NO-OP.
   Implemented (`apply_l2_persist`, env `ONPAIR_L2_PERSIST`): set
   `CU_LIMIT_PERSISTING_L2_CACHE_SIZE` + an `accessPolicyWindow` over the dict.
   bits16 4tpt: 2.000→1.999 ms; freq+L2 = freq exactly. NCU: L2 (LTS) hit rate
   **93.7% with AND without** persistence, DRAM read identical. The dict already
   lives in L2 (1 MB ≪ 50 MB L2; DRAM only 17%), so there is no L2 eviction to
   prevent. The bottleneck is **L1** residency (35% hit) + L1/TEX request
   throughput — which L2 persistence cannot touch. freq-sort (an L1 fix) is what
   helps. Clean negative with NCU proof.

2. **Software-pipelined gather prefetch (cp.async, Ampere+)** — ❌ reasoned out.
   The kernel is L1/TEX-request-*throughput* bound, not latency bound; cp.async
   overlaps the gather but does not reduce the *number* of L1/TEX requests, so it
   cannot move a throughput wall. It also stages through shared → reintroduces the
   bank-conflict problem that sank `pdict`/`vdict`. Not built.

3. **Register/`__shfl` hot-code cache** (`onpair_shmem_4tpt_regcache`) — ❌ TESTED,
   wash/loss. Each lane caches one of the 32 hottest entries (freq-sorted); a
   token with code < 32 is served by `__shfl` instead of a gather.
   Found: bits12 freq 1.96 ms (−9% vs 4tpt 1.80); bits16 freq 1.68 ms (tie with
   4tpt 1.69); bits16 without freq 2.08 (regress). Validated byte-exact.
   why BAD: top-32 of 4096–65536 codes covers too little of the long Zipfian text
   tail, so most tokens still gather AND every token pays wasted shuffles; the +5
   warp-wide registers (already at the 64 cap) and hot/cold divergence cancel the
   saved gathers. A larger K needs multi-level shuffle (more overhead). Dead end.

4. **Token sorting for gather coalescing** — ❌ reasoned out. Sorting the 32/128
   tokens by code would coalesce/broadcast dict reads (classic gather→sort trick),
   but OnPair output must stay in token order, so it needs an inverse-scatter that
   costs more than it saves and breaks the streaming drain.

5. **Constant / texture-cache dict** — ⟷ reasoned marginal. `__constant__`
   broadcasts same-address but *serialises divergent* reads (same failure as bank
   conflicts) for a random gather; the read-only/`__ldg` path is already in use via
   `__restrict`. No clear win.

6. **Decouple length-scan from byte-copy across the block (FSST-GPU style)** — ⟷.
   Output offsets are already precomputed per chunk (`chunk_offsets`); the
   remaining per-warp scan is cheap (4× `shfl_up`). Limited headroom.

4. **`__ldcs` streaming-load on `codes`** (`onpair_shmem_4tpt_ldcs`) — ⟷ TESTED,
   neutral (±1.5%, noise). Hypothesis: the read-once `codes` stream pollutes L1
   and evicts the reused dict, so mark codes streaming to protect dict residency.
   Found (explicit per-kernel labels): ps_comment/b16 1.052 vs 4tpt 1.067 (−1.4%);
   fineweb/b16 2.031 vs 2.002 (+1.5%); fineweb/b12 1.846 vs 1.822 (+1.3%).
   Validated byte-exact.
   why NEUTRAL: the hypothesis is essentially wrong — `codes` are read *coalesced*
   (32 lanes → 64 contiguous bytes ≈ 2 cache lines), so they are tiny and
   L1-friendly, NOT the dict's evictor (the 1 MB bits16 dict thrashes L1 on its
   own). De-caching codes therefore neither protects the dict nor costs much →
   a wash. Lesson confirmed: protect the dict by shrinking *it* (split8read) or
   clustering it (freq-sort), not by touching the already-cheap codes.

Outcome: all four plausible research-informed levers (L2 persistence, register
hot-code cache, cp.async prefetch, __ldcs codes) were tested or reasoned out and
**none helped**.
This *confirms* the diagnosis: OnPair decode is bound by L1/TEX **request
throughput** on an irreducible random gather, with the dict already L2-resident
(94%) and DRAM idle. The only effective levers are the two we found — **fetch
fewer bytes per token (split8read, bits12)** and **improve L1 residency by
clustering hot codes (freq-sort, bits16)**. Both are now validated; everything
else moves the work without shrinking it.

## Format / encoder change suggestions (NOT implemented — proposals only)

These would help decode but require touching the OnPair encoder / on-disk layout,
so they are out of the "no layout change" scope. Ranked by value.

1. **Frequency-ordered code assignment** (highest value). Assign code 0..N to
   dict entries by descending in-stream frequency. We *validated* the decode win
   (bits16 +8–13%, L1 sector hit 35%→45%) via a decode-side relabel; it is free
   at encode (a sort during dict construction), keeps on-disk size and
   compression ratio identical, and the decode output is byte-identical. Helps
   CPU decode too (cache locality). Also unlocks future hot-prefix tricks.

2. **Tunable max symbol length (FSST-style ≤8 B)**. If the encoder split symbols
   longer than 8 B, `dict_s8` (8 B stride) would cover *every* token → split8read
   needs no `dict_padded` fallback, and the bits16 dict halves (512 KB vs 1 MB)
   → better L1. Trade-off: more codes ⇒ slightly worse ratio + more tokens to
   decode. Best for long-token columns (ps_comment: 84% of tokens > 8 B today,
   which is exactly where split8read regresses). Expose as a per-column knob and
   pick on a ratio-vs-decode-speed curve.

3. **Per-chunk decode hint (1–2 bytes)**. Store the token-weighted length class
   (e.g. ≤4 / ≤8 / ≤12 / >12 dominant bucket) per chunk so the decoder selects
   the right kernel (split8read vs 4tpt) without scanning. Cheap metadata; makes
   adaptive selection exact instead of heuristic.

4. **Prefer bits12 when ratio permits**. bits12 decodes much faster than bits16
   (64 KB dict fits L1; split8read applies). For columns whose dict saturates
   (e.g. fineweb text: ratio 2.264 b12 vs ~same b16), the encoder should pick
   bits12 for decode speed. Weight decode speed in the bit-width heuristic, not
   just ratio.

5. **Store the dict pre-split** (`dict_s8` + overflow) on disk to skip the
   load-time derivation. Low value (derivation is cheap), minor.

## One-line verdict table

| axis | option | impl | result | adopt? |
|------|--------|------|--------|--------|
| dict | padded-16 global | `4tpt` | L1/TEX-bound baseline | ✅ ref |
| dict | **split-read 8B** | `4tpt_split8read` | bits12 +4–11%, bits16 −5–26% | ✅ bits12 |
| dict | padded-16 shared | `4tpt_pdict` | −27% (82M bank conflicts, occ 24%) | ❌ |
| dict | var-len packed shared | `4tpt_vdict` | −53% (78M conflicts, byte loop) | ❌ |
| dict | var-len packed global | — | reasoned worse (unaligned gather) | 🔬 |
| dict | variable-stride lenbucket | `4tpt_lenbucket` | bits12 −26%, bits16 tie (branch divergence) | ❌ |
| order | frequency-sort codes | reorder (decode-side) | bits16 +8–13%, bits12 ~neutral; L1 hit 35→45% | ✅ encoder |
| order | length-sort codes | reorder (decode-side) | hurts (scatters hot entries) | ❌ |
| cache | L2 persistence (access-policy window) | `apply_l2_persist` | no-op (dict already 94% L2-resident; L1 is the wall) | ❌ |
| latency | cp.async gather prefetch | — | reasoned out (throughput-bound; shared conflicts) | ❌ |
| cache | register/shfl hot-code cache | `4tpt_regcache` | bits12 −9%, bits16 tie (top-32 too small) | ❌ |
| cache | __ldcs streaming-load on codes | `4tpt_ldcs` | neutral ±1.5% (codes are coalesced/L1-cheap, not the evictor) | ⟷ |
| dict | stride-8 / stride-4 / const | `s8*`/`s4l1*`/`const*` | only when max_len/cardinality allows | ⟷ |
| emit | shared staging + ladder | `4tpt` | coalesces global stores | ✅ ref |
| emit | split-8 / split-4 write | `_split8`/`_split4` | tie/slower (write not bottleneck) | ⟷/❌ |
| emit | direct global scatter | — | reasoned worse (uncoalesced) | 🔬 |
| map | 1/2 tpt | `_2tpt` etc | under-amortised | ❌ |
| map | 4 tpt | `4tpt` | best amortisation | ✅ |
| map | 8 tpt | — | invalid PTX | ❌ |
| occ | wpb8 | `_wpb8` | −12% | ❌ |
| occ | wpb8+occ | `_wpb8_occ` | 10-iter noise; ties/loses at 300 iters | ❌ |
| occ | register cut → 3 blocks/SM | — | need ≤42 regs (have 64); pipe-saturated | ❌ |
| prefetch | coop-load / cp.async / TMA | `_tma` | −22–33% / fragile | ❌/🔬 |
| prefetch | persistent grid-stride | (in `_pdict`/`_vdict`) | amortises load, wrong target | ⟷ |
| chunk | large chunks | — | bits16 better ratio; faster decode | ✅ |
| chunk | small chunks | — | bits12 ratio-neutral; more launch overhead | ⟷/❌ |
