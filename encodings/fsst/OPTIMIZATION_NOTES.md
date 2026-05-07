# FSST `LIKE` optimization journey

A focused effort to optimize `SELECT COUNT(*) FROM hits WHERE "URL" LIKE '%google%'`
on FSST-compressed data. Workload: 1M synthetic ClickBench-style URLs, headline
bench `clickbench_url_google::like_google_full`. All numbers are on a Skylake-X
class Xeon (CPUID: avx2 + avx512f + avx512bw + bmi2; **no AVX-512 VBMI**).

## Final state on this branch

| Stage | Fastest | Median | vs Original |
|---|---|---|---|
| Original `FlatContainsDfa` | 44.3 ms | 44.9 ms | — |
| + Folded 2N+1 DFA | 40.9 ms | 41.4 ms | -8% |
| + Monomorphic `scan_to_bitbuf` | 38.8 ms | 39.5 ms | -12% |
| + AVX2 anchor scan + tzcnt advance | **26.7 ms** | **27.6 ms** | **-40%** |
| Theoretical floor (memmem on decompressed corpus) | 12.8 ms | 12.8 ms | — |

Cross-pattern validation on the `fsst_like` suite confirmed the win
generalises whenever the state-0 progressing-code set is large enough that
`SkipStrategy::Bitmap` is selected:

| Pattern | Before | After | Δ |
|---|---|---|---|
| `%yandex%` (cb) | 3.84 ms | 2.34 ms | -39% |
| `%target/release%` (path) | 2.02 ms | 1.20 ms | -41% |
| `%google%` (urls) | 1.93 ms | 1.25 ms | -36% |
| `%Googlebot%` (log) | 3.05 ms | 2.98 ms | -2% |
| `%gmail%` (email) | 1.84 ms | 1.87 ms | 0% |
| `%enterprise%` (json) | 4.02 ms | 4.01 ms | 0% |
| `%xyzzy%` (rare) | 1.60 ms | 1.59 ms | 0% |
| All `fsst_prefix` | unchanged across the board | | 0% |

3 substantial wins, 0 regressions. Patterns where the 0% rows occur use
`Memchr1/2/3` skip already — that path was already optimal.

## Ablation matrix

Systematic 2-runs-per-config measurement of each subset of {Folded DFA (F),
Monomorphic dispatch (M), Anchor scan (A)} on this host. Numbers are
`like_google_full` median across 2 interleaved bench runs.

| Config | Median | vs HEAD |
|---|---|---|
| 000 (baseline `FlatContainsDfa`) | 45.43 ms | +97% |
| 100 (F only) | 42.86 ms | +86% |
| 010 (M only) | 40.86 ms | +77% |
| 110 (F + M) | 39.14 ms | +70% |
| 111 (F + M + A, HEAD) | **22.99 ms** | — |

**Total reduction 000 → 111: −49.4%**.

### Marginal contributions

| Step | Δ Median | Headline win |
|---|---|---|
| F alone (000 → 100) | −5.6% | Modest. Sentinel-branch removal helps when the inner DFA loop is hot — usually it isn't, because state-0 skip dominates. |
| M alone (000 → 010) | −10.0% | Monomorphising `FsstMatcher` dispatch into the bit-packing closure. |
| F + M (000 → 110) | −13.8% | F and M roughly additive, slight sub-additive overlap. |
| A on top of F+M (110 → 111) | **−41.2%** | The dominant optimization. AVX2 PSHUFB-Mula bitmap scan + tzcnt-driven state-0 advance. |

### Interactions

- **A's win does not depend on M** for `%google%`. A's `scan_with_anchor_bitset`
  → `matches_with_bitset` path bypasses the `scan_to_bitbuf_with` closure that
  M monomorphises. So when A fires, M contributes ~0 to the hot path.
- **M's win is only measurable when A is absent** — i.e., on patterns that
  use `Memchr1/2/3` skip (email, json, rare) where A doesn't fire and the
  generic scan_to_bitbuf path is taken. M is "polish that matters when A
  isn't doing the heavy lifting."
- **A depends architecturally on F**. A's bitset routines are methods on
  `FoldedContainsDfa`. Configurations 001, 011, 101 are not cleanly
  buildable: there's no host DFA for A's logic without F. The algorithm
  could in principle apply to `FlatContainsDfa`, but the implementation as
  committed does not.

### Is the stack justified?

Each layer pulls weight, but unevenly:

| Layer | Standalone marginal win | Why it's there |
|---|---|---|
| F (folded DFA) | ~5–6% | Modest as perf, but it is the structural foundation A is built on. |
| M (monomorphic) | ~9% standalone; ~0% when A fires | Independent polish. Helps non-A patterns and the general scan_to_bitbuf path (Flat / Prefix / Suffix / Multi). |
| A (anchor scan) | ~41% on Bitmap-skip patterns | Dominant. Justifies the entire stack on its own. |

No layer is redundant. The most defensible framing: **F is plumbing for A;
A is the headline; M is independent polish that helps when A is absent.**

`dfa_inner_only` (per-string `FsstMatcher::matches`, no scan_to_bitbuf):
all five configs land at 41 ± 1 ms. None of F / M / A acts on the per-string
`matches` path itself — they all act on the surrounding scan/bitbuf
machinery. This confirms the inner DFA step is NOT the bottleneck for
`%google%` on this corpus; the time is in everything else.

## What shipped, and why each piece earns its place

### 1. Folded 2N+1 escape-aware DFA (`encodings/fsst/src/dfa/folded_contains.rs`)

The original `FlatContainsDfa` had a sentinel state in its transition table.
Each code byte stepped the DFA by:

```rust
let next = transitions[state * 256 + code];
if next == sentinel {
    state = escape_transitions[...];   // separate table for escape literals
} else {
    state = next;
}
```

Per-iteration sentinel branch.

The folded variant encodes "in escape from state s" directly into the state
space (states `0..N` for normal, `N+1..2N` for in-escape, plus an accept
state). Total `2N+1` states for an N-byte needle, fits in `u8` for N ≤ 127.
Inner loop becomes a single packed lookup — no sentinel branch, no
secondary table.

**Marginal contribution: ~8% faster.**

### 2. Monomorphic `scan_to_bitbuf` (`encodings/fsst/src/dfa/folded_contains.rs`, etc.)

Originally the FSST `LikeKernel` called a generic
`dfa_scan_to_bitbuf(n, offsets, all_bytes, negated, |codes| matcher.matches(codes))`,
and inside `matcher.matches` there was an enum match on `MatcherInner`
(MatchAll / Prefix / Suffix / Contains / FoldedContains / MultiContains).

That enum match runs **once per string** (1M times in the headline bench).
Even with branch prediction making it cheap, the dispatch through a
closure + non-fully-inlined method blocks LLVM from keeping the matcher's
internals in registers across the loop.

The monomorphic refactor adds per-DFA `scan_to_bitbuf<T: IntegerPType>`
methods; `FsstMatcher::scan_to_bitbuf` dispatches once per LIKE call (not
per string), routing to the right specialized method. Inside the specialized
method, the inner loop is 100% the same DFA's `matches` body inlined.

**Marginal contribution: ~4% on top of folded DFA.**

### 3. AVX2 global anchor scan + tzcnt-driven state-0 advance (`encodings/fsst/src/dfa/anchor_scan.rs`)

The big one. The DFA's state-0 fast path uses `SkipStrategy::Bitmap` for
patterns with 4+ progressing codes (which includes `%google%` —
6 progressing codes including `ESCAPE_CODE`). The original Bitmap path
was a scalar byte-by-byte bit-test loop:

```rust
for (i, &code) in slice.iter().enumerate() {
    if bm[code >> 6] & (1 << (code & 63)) != 0 {
        return Some(i);
    }
}
```

Called once per string (~1M times) on slices of ~30-40 bytes.

The optimization is twofold:

**Phase 1 — Global anchor scan**. One AVX2 PSHUFB-Mula pass over the
ENTIRE compressed `all_bytes` buffer (~36 MB) marks every byte that is a
progressing code. Output: a 4.5 MB `BitBuffer`. This pass runs at
SIMD-memory-bandwidth speed (~30 GB/s on this CPU) — about 7 ms total.

**Phase 2 — Bitset-driven state-0 advance**. The DFA's state-0 fast path
now reads the bitset via `u64`-chunked `tzcnt` instead of byte-by-byte
bitmap probes. For each per-string call, instead of scanning the
compressed bytes, we scan the bitset's u64 words for the first set bit
within the string's range — a single `tzcnt` per word.

Crucially the win comes from BOTH phases. An earlier attempt that used the
bitset only to filter out non-candidate strings (skip the DFA entirely on
strings with no progressing codes) was nearly a no-op on this corpus
because ~46% of strings contain at least one progressing code, leaving
most of the DFA work intact. Driving the DFA's state-0 jumps off the
precomputed bitset is what unlocks the ~30% speedup.

**Marginal contribution: ~30% on top of folded DFA + monomorphic.**
Confirmed to fire on patterns where Bitmap skip is selected; no-op on
Memchr1/2/3 patterns; no-op on prefix patterns.

## What we tried that didn't work (and why)

### Decompress + memmem path (multiple attempts)

The intuition: memmem is heavily SIMD-optimised. If we decompress the FSST
codes and run memmem on the decompressed bytes, we get the SIMD
matching floor "for free."

Three independent implementations, all losing to HEAD:
- A naive `Box<dyn Fn>` per-string memmem (~31% claimed but on a noisy
  host; reality on this host: ~50 ms — slower than baseline).
- Aggressive `unsafe` per-string memmem with branchless decompression
  (~47-48 ms).
- Bulk-decompress + memmem with thread-local scratch + chunked decode
  (~38 ms).

**Wall**: fsst-rs's scalar decoder runs at ~1.5-1.8 GB/s on this CPU
(measured ~33-55 ms standalone for the 50 MB output). Even with that
"free", the path's floor is `decode + memmem ≈ 33 + 13 = 46 ms`, already
worse than HEAD's 26.7 ms.

### Vectorized FSST decoder (Phases 1-3)

Tried three implementations to break the decompression bottleneck:

| Phase | Approach | Decode standalone (50 MB) | Bulk total |
|---|---|---|---|
| Reference | fsst-rs scalar | 33-55 ms | 46-71 ms |
| 1 | Scalar 8-codes vendored (autovec hope) | 82 ms | 75 ms |
| 2 | AVX2 explicit gather+scatter | **171 ms** | 184 ms |
| 3 | AVX-512BW 16-codes/iter | 110 ms | 127 ms |

**Walls discovered**:

1. **`vpgatherqq`/`vpgatherdq` on Skylake-X is slow** — ~5-cycle
   reciprocal throughput per 4-lane gather. Three gathers per 8 codes
   ≈ 25 cycles, vs scalar's ~4 cycles per code. The FSST paper's claimed
   12 GB/s decoder relies on Ice Lake+ gather hardware OR AVX-512 VBMI's
   `vpermi2b` (we have neither).

2. **No VBMI = no wide-store compaction.** Without `vpermt2b` you can't
   compact 16 variable-length symbols into one contiguous 128-byte ZMM
   pair. You're stuck with 16 narrow `vmovq` stores serialised through
   port 5 → store-port bound.

3. **Phase 1's "compute prefix sums up front then store at offsets"
   creates a serial dependency chain** between consecutive stores
   (`p_i = p_{i-1} + l_{i-1}`) that LLVM can't break. fsst-rs's
   `out_ptr += lengths[code]` per-code shape has the same arithmetic but
   a single running accumulator dependency, which is shorter and lets
   the compiler pipeline.

Net: **vectorized FSST decode is dead on Skylake-X.** It would likely win
on Ice Lake+ / Sapphire Rapids (faster gather) or any CPU with AVX-512
VBMI — code is preserved on the agent worktrees for that future.

### Inner-loop micro-optimisations on the DFA

Multiple agents chased ways to make the inner DFA step faster.

| Attempt | Result | Why |
|---|---|---|
| Half-row nibble-packed DFA (2 KiB → 1 KiB transitions) | regression | DFA already fits L1; state-0 skip dominates anyway |
| 4-byte hierarchical composition DFA (≤7 byte needles) | +3.6% | Inner loop is only ~4 ms of total; ceiling too low |
| N-way ILP across strings (4/8 lanes) | neutral | Per-batch overhead nullifies the throughput gain on ~30-byte slices |
| AVX-512 64-way SIMD DFA with overrun | -70% | `vpgatherdq zmm` is 22-cycle latency on Skylake-X; sequential state chain is latency-bound |
| AVX-512BW PSHUFB Mula bitmap skip (64 bytes/iter) | -3% | Slices too short (~30 codes) for 64-byte body to fire; AVX-512 frequency-license downclock dominates |
| AVX2 PSHUFB Mula bitmap skip (32 bytes/iter) | ~0% (noise) | `#[target_feature]` blocks inlining; per-call dispatch eats the win on short slices |
| AVX-512F broadcast-vpcmpb bitmap skip | -10% | Same as above, plus AVX-512 frequency cost |
| SWAR-8 bitmap skip (no intrinsics, fully inlinable) | -60% | Loop typically exits in first 1-4 bytes; per-chunk SIMD cost wasted |
| Specialise `scan_to_bitbuf` per skip-strategy | -5% | LLVM already constant-folds the dispatch via inlining; duplicated inner loops thrash icache |

**Common wall across all SIMD attempts on the inner skip loop**: the
state-0 skip's average exit position is just 1-4 bytes. Any per-chunk
fixed cost exceeds the actual scan work, so wider SIMD is strictly worse.
The per-call `#[target_feature]` ABI boundary is also a tax that erases
small wins.

The successful anchor scan dodged this by doing one big SIMD pass
GLOBALLY (over 36 MB) rather than per-string.

## Diagnostic bench variants

`encodings/fsst/benches/clickbench_url_google.rs` includes four variants
to triangulate where time goes:

| Variant | Fastest | Tells us |
|---|---|---|
| `like_google_full` | 26.7 ms | Full LIKE expression evaluation |
| `dfa_inner_only` | ~40 ms | Per-string `FsstMatcher::matches`, no bitbuf, no expression dispatch |
| `memmem_per_string` | ~25 ms | Decompressed bytes + per-string memmem call |
| `memmem_concat_corpus` | ~13 ms | One global memmem on whole decompressed corpus — the matching-work-only floor |

Useful for triangulating cost: `memmem_concat_corpus` is the lower bound
on any path that has to find needle occurrences. The gap from 13 ms to
HEAD's 26.7 ms is split between per-candidate DFA cost (the anchor scan
already eliminates most of the per-byte scanning) and the unavoidable
per-string call overhead (~13-15 ns/string × 1M).

## Headroom remaining

Roughly 14 ms separate HEAD from the absolute matching-work floor. Two
plausible levers:

### 1. Per-symbol pre-classification (untouched)

For a given (FSST symbol table × needle), classify each of the 256 codes:
- **ACCEPT_ALONE**: this code's expansion contains the full needle. The
  code's mere presence in a string means the string matches.
- **INERT**: this code's expansion contains zero needle bytes and can't
  extend any partial match.
- **AMBIGUOUS**: needs the DFA.

Then per-string scan: AVX2 `vpcmpeqb` against the (typically tiny)
ACCEPT_ALONE set first; instant match on hit, fall through to DFA on
miss. For `%google%` over ClickBench URLs the FSST trainer probably
learned a "google"-containing symbol — ACCEPT_ALONE then catches a large
fraction of true positives without DFA cost.

Plausible 26.7 → 18-22 ms.

### 2. Tighten the anchor scan itself

The AVX2 PSHUFB-Mula anchor scan is ~7 ms over 36 MB. Two angles:
- AVX-512BW for the anchor scan loop. Unlike the per-string SIMD attempts
  that failed on frequency licensing, this is a single 36 MB bulk pass —
  the up-clock cost amortises cleanly and the SIMD throughput dominates.
- Software prefetch of upcoming offsets/codes so per-candidate DFA
  pulls into L1 ahead of time.

Plausible 7 → 3-4 ms, total → ~22-23 ms.

### 3. Future hardware

On Ice Lake+ / Sapphire Rapids / Zen 4, the vectorised FSST decoder
becomes viable. The Phase 2 / Phase 3 implementations were left committed
on agent worktrees ready to revisit. Add a runtime CPU-feature gate
(`is_x86_feature_detected!("avx512vbmi")`) to ship both paths.

## Repository state on this branch

Substantive optimisation commits (in order):

1. `e69284207` — folded 2N+1 contains DFA
2. `3d63b6668` — monomorphise matcher dispatch in scan_to_bitbuf
3. `9cf410fb1` — AVX2 global anchor scan for FoldedContainsDfa Bitmap skip

Bench / infrastructure commits:

- `cc2562c0e` — focused `clickbench_url_google::like_google_full` bench
- `66c16d8dc` — diagnostic bench variants (`dfa_inner_only`,
  `memmem_per_string`, `memmem_concat_corpus`)
- `55987b412` — expose `FsstMatcher` under `_test-harness` so the
  `dfa_inner_only` bench can call it without a boxed dispatch wrapper
- `0240e3f11`, `6e1c47ca9` — `.gitignore` updates for `.agents/worktrees`
  and `.claude/worktrees`
- `633c75f1f` + `41c6f5d6f` — earlier 64-block bitbuf packing attempt
  + revert (replaced by the cleaner monomorphic dispatch)
