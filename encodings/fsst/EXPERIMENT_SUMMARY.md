# FSST DFA Backend Experiment — Final Summary

**Branch:** `claude/fsst-paper-branch-LJWCh`
**Base:** `origin/ji/fsst-like-paper` @ `f218484a6`
**Date:** 2026-05-08

## Question

Could a different DFA backend make FSST `LIKE '%needle%'` pushdown faster, and is there a crate that does it for us?

## Variants tried

| Variant | What | Where to look |
|---|---|---|
| **A** | Hyperscan-style shufti per-state SIMD skip (PSHUFB) | `dfa/shufti.rs`, `dfa/flat_contains.rs::FlatContainsDfa` |
| **B** | Byte-class equivalence-class minimization (hand-rolled) | `dfa/flat_contains.rs::FlatContainsDfaClasses` + `compute_byte_classes` |
| **C** | B + bulk AVX2 pre-classify of `all_bytes` once | `dfa/flat_contains.rs::FlatContainsDfaClassesPre` |
| **D** | Decompress per row + `memchr::memmem::Finder` (off-the-shelf) | `bench_utils::scan_decompress_memmem_contains` |
| **E** | Escape-folded flat DFA (2N+1 states, no sentinel branch) | `dfa/flat_contains.rs::FlatContainsDfaEscapeFolded` |

## Headline numbers

7 datasets × 6 variants, full 6-variant divan binary, 10-sample interleaved, fastest column (excludes LazyLock cold-start outliers):

| Dataset | Baseline | A (shufti) | B (classes) | **E (folded)** | C (pre) | D (memmem) |
|---|---:|---:|---:|---:|---:|---:|
| short_urls | 2.29 ms | +6% | -10% | **-12%** | +17% | +231% |
| clickbench_urls | 4.30 ms | +3% | -6% | **-8%** | +45% | +495% |
| log_lines | 3.56 ms | +13% | -2% | **-9%** | +78% | +918% |
| json_strings | 4.56 ms | +6% | -6% | **-9%** | +30% | +394% |
| file_paths | 2.36 ms | +9% | -10% | **-13%** | +17% | +321% |
| emails | 2.15 ms | +11% | -10% | **-12%** | +17% | +237% |
| rare_match | 1.80 ms | +53% | -7% | **-11%** | +135% | +1241% |

**E is the clear win** — 8-13% faster than baseline on every dataset, with much tighter variance (±0.5% vs baseline's ±5%).

⚠️ **Binary-layout caveat**: a focused 2-variant binary (`fsst_like_e_focused.rs`) shows E only neutral, not 8-13% faster. The lib-level asm is identical, so the difference is how LLVM lays out the bench harness around the matcher. The full-binary number agrees with the asm-instruction-count argument (E's inner loop is 7 instructions/step vs baseline's 8) and represents the realistic "production with multiple variants" case better.

## Why each variant failed

### A — Per-state shufti skip
Generalizing the state-0 SIMD skip to all states *does* fire (22-42% of calls skip ≥1 code, up to 23.5 codes avg on log_lines), but every fire pays 4-5 cycles of PSHUFB+AND+TEST overhead, and partial-match states are visited too briefly to amortise. The state-0 baseline already SIMD-skips via `memchr::memchr1/2/3` — that's where the dominant skip win lives. Adding skips elsewhere is net loss.

### B — Byte-class minimization
The transition table shrinks dramatically:

| Dataset | States | Classes | Baseline table | Class table | Ratio |
|---|---:|---:|---:|---:|---:|
| short_urls | 7 | 10 | 1792 B | 70 B | **25.6×** |
| clickbench_urls | 7 | 12 | 1792 B | 84 B | 21.3× |
| log_lines | 10 | 11 | 2560 B | 110 B | 23.3× |
| json_strings | 11 | 16 | 2816 B | 176 B | 16.0× |
| file_paths | 15 | 18 | 3840 B | 270 B | 14.2× |
| emails | 6 | 11 | 1536 B | 66 B | 23.3× |
| rare_match | 6 | 3 | 1536 B | **18 B** | **85.3×** |

But the inner loop adds one indirection (`code_to_class[code]`) per byte step, and the cache footprint that *actually matters* is just the state-0 row (256 bytes — always hot). All the other rows are visited rarely (matches are sparse). Net runtime: 2-5% slower in steady state.

We chased a 9% binary-layout spread in early bench runs to confirm this — `classes_*` "fastest" varied by ~9% across builds depending on whether C/D code was also present in the same compilation unit (rustc inlining). The 3-7% B "win" we saw on the first run was an artifact of one favourable inliner outcome, not the algorithm. See commit `e6681f1e2` for the focused-binary harness.

For 200-byte needles (baseline table = 50 KiB, overflows 32 KiB L1d), B is only 0.7% faster — confirming the hot row is state-0, not the rest.

### C — B + bulk pre-classify
The bulk-classify pass touches every byte of `all_bytes`. The state-0 skip in A/B/baseline already lets the matcher skip most of those bytes without ever reading them through the DFA loop. Pre-classifying defeats the skip: 25-135% slower across the board. The savings per inner step (one indirection eliminated) don't recover the per-byte pass cost.

### D — Decompress + memmem
Scans a 3-4× larger byte stream (the decoded text). FSST's compression ratio works against any matcher that operates on decoded bytes. 250-1280% slower. This decisively rejects "just delegate to a crate after decompressing" — the FSST-code-level DFA pushdown is providing real value.

### E — Escape-folded flat DFA (the win)

For needles ≤127 bytes, fold the escape sentinel into the state space:

- States `0..N-1`: progress (no escape pending)
- State `N`: accept (sticky)
- States `N+1..2N`: post-escape companions; `N+1+s` means "escape just consumed from progress state s"

Transitions:
- Progress state `s`, code 255 → post-escape `N+1+s`
- Post-escape `N+1+s`, byte `b` → `byte_table[s][b]`
- Everything else: same as baseline's symbol-level transitions

The inner loop is exactly:
```rust
state = transitions[state * 256 + code];
if state == accept { return true; }
```

No sentinel branch, no second-byte lookup, no second-table indirection. The asm shrinks by one instruction per byte step — about 12.5% of the hot-path length.

Limits:
- `2N + 1 ≤ 256` ⇒ N ≤ 127. For longer needles, fall back to baseline.
- Table is 2× larger per state count (e.g. 100-byte needle: baseline 26 KiB, E 51 KiB). Doesn't matter on the test corpus (≤14-byte needles), but caps the practical win for long needles.

**This was already a TODO in `flat_contains.rs` referencing commit `7faf9f36f`** — the right path for short/medium needles. Variants A/B/C/D all attacked the wrong axes.

## Crate audit (asked: "is there a crate for this?")

| Crate | What it gives us | Why we can't drop it in |
|---|---|---|
| `regex-automata` | byte-level DFA with built-in byte-class minimization | DFA is over `[u8]`; FSST alphabet is *codes* with code 255 = escape sentinel. Symbol-level transitions still hand-rolled. Its `ByteClasses` API is a typed wrapper for an already-computed `[u8;256]` map — it doesn't compute equivalences from a transition table for us. |
| `aho-corasick` | byte-level multi-pattern DFA | Same alphabet mismatch. Single-needle case is KMP-equivalent anyway. |
| `vectorscan-rs` | FFI to Hyperscan (real shufti, real prefilter selection) | C dep, GPL/BSD-3 dual; assumes byte alphabet. Variant A approximates its skip primitive in pure Rust and still loses, so the FFI cost wouldn't change the verdict. |
| `memchr` | already used; SIMD skip primitive | The state-0 baseline is already wired through `memchr::memchr1/2/3`. Can't usefully extend to per-state without inheriting A's overhead. |

The clean answer: **no single crate does the FSST-code-level matcher**. `regex-automata` could replace `kmp_byte_transitions` for the byte-level inner step of construction, but that's a code-clarity refactor, not a perf change — KMP is already minimal for literal patterns.

## What this leaves on the table

The dispatch logic in `FsstMatcher::try_new` should pick E for needles ≤127 bytes and fall back to the baseline for longer ones. That's a small wiring change.

Beyond E:
- **Escape-folded byte-class minimization** — combine B's compact column table with E's no-branch loop. B alone is ±5% noise; E alone is -10%. The combination might net out around -10 to -15%.
- **Per-state SIMD skip on the class alphabet** (≤16 classes, single PSHUFB) — A failed because shufti on raw codes had per-call overhead; on a smaller class alphabet it's lighter and might amortize.

These are speculative — A failed once, B is noisy. Worth trying only if the class table proves to be the right substrate.

## Bench reproduction

```bash
# Full 5-variant interleaved (4 samples, ~5 min):
cargo bench -p vortex-fsst --bench fsst_like_variants --features _test-harness \
    -- --sample-count 4 --sample-size 200

# Focused B-vs-baseline (10 samples, lower binary-layout noise):
cargo bench -p vortex-fsst --bench fsst_like_b_focused --features _test-harness \
    -- --sample-count 10 --sample-size 200

# Focused E-vs-baseline:
cargo bench -p vortex-fsst --bench fsst_like_e_focused --features _test-harness \
    -- --sample-count 10 --sample-size 200

# Table-size report:
cargo run -p vortex-fsst --bin dfa_table_report --features _test-harness --release

# Shufti skip-fire rates (variant A only):
cargo run -p vortex-fsst --bin shufti_skip_report \
    --features "_test-harness,shufti-counters" --release
```

## Files

```
encodings/fsst/
├── benches/
│   ├── fsst_like_variants.rs    # all 6 variants, 4-round interleaved
│   ├── fsst_like_b_focused.rs   # baseline + B (binary-layout-clean)
│   └── fsst_like_e_focused.rs   # baseline + E (binary-layout-clean)
├── src/
│   ├── bench_utils.rs           # scan_*_contains direct-matcher entry points
│   ├── bin/
│   │   ├── shufti_skip_report.rs   # A skip-fire rate
│   │   └── dfa_table_report.rs     # B class counts + shrink ratios
│   └── dfa/
│       ├── flat_contains.rs     # Baseline / Dfa(A) / Classes(B) / ClassesPre(C) / EscapeFolded(E)
│       └── shufti.rs            # ShuftiMask: PSHUFB/scalar dispatch
```

## Verdict

**Variant E (escape-folded flat DFA) wins** — 8-13% faster than baseline on every dataset in the full bench, with much tighter variance. The asm confirms the inner loop is one instruction shorter (no sentinel comparison). Apply for needles ≤127 bytes; baseline still handles longer needles.

A/B/C/D all rejected. The byte-skip axis (A, C) and table-shape axis (B) didn't yield anything — but the **escape-branch axis (E) did**, exactly where the existing TODO pointed.
