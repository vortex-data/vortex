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

## Headline numbers

7 datasets × 5 variants, divan 4-round interleaved, fastest column (excludes LazyLock cold-start outliers):

| Dataset | Baseline | A (shufti) | B (classes) | C (pre) | D (memmem) |
|---|---:|---:|---:|---:|---:|
| short_urls | 2.21 ms | +24% | +6% | +36% | +250% |
| clickbench_urls | 4.24 ms | +12% | +3% | +74% | +513% |
| log_lines | 3.42 ms | +27% | +5% | +84% | +963% |
| json_strings | 4.49 ms | +16% | +4% | +54% | +412% |
| file_paths | 2.27 ms | +26% | +6% | +35% | +345% |
| emails | 2.11 ms | +26% | +5% | +31% | +250% |
| rare_match | 1.76 ms | +73% | +5% | +137% | +1280% |

**Every variant is slower than baseline on every dataset.** No win.

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

## Crate audit (asked: "is there a crate for this?")

| Crate | What it gives us | Why we can't drop it in |
|---|---|---|
| `regex-automata` | byte-level DFA with built-in byte-class minimization | DFA is over `[u8]`; FSST alphabet is *codes* with code 255 = escape sentinel. Symbol-level transitions still hand-rolled. Its `ByteClasses` API is a typed wrapper for an already-computed `[u8;256]` map — it doesn't compute equivalences from a transition table for us. |
| `aho-corasick` | byte-level multi-pattern DFA | Same alphabet mismatch. Single-needle case is KMP-equivalent anyway. |
| `vectorscan-rs` | FFI to Hyperscan (real shufti, real prefilter selection) | C dep, GPL/BSD-3 dual; assumes byte alphabet. Variant A approximates its skip primitive in pure Rust and still loses, so the FFI cost wouldn't change the verdict. |
| `memchr` | already used; SIMD skip primitive | The state-0 baseline is already wired through `memchr::memchr1/2/3`. Can't usefully extend to per-state without inheriting A's overhead. |

The clean answer: **no single crate does the FSST-code-level matcher**. `regex-automata` could replace `kmp_byte_transitions` for the byte-level inner step of construction, but that's a code-clarity refactor, not a perf change — KMP is already minimal for literal patterns.

## What this leaves on the table

The right next direction (already TODO'd in `flat_contains.rs`, references commit `7faf9f36f`):

> **Escape-folded flat DFA** for needles ≤127 bytes: 2N+1 states, no sentinel branch in the inner loop. The escape branch is the per-step cost variant A and C tried to dodge with SIMD; folding it into the state space removes it from the hot loop entirely.

That's the genuine perf opportunity for short/medium needles. The 4 variants tested in this experiment all attacked the *byte-skip* axis or the *table shape* axis; the *escape branch* axis is untouched and is plausibly the largest remaining win.

## Bench reproduction

```bash
# Full 5-variant interleaved (4 samples, ~5 min):
cargo bench -p vortex-fsst --bench fsst_like_variants --features _test-harness \
    -- --sample-count 4 --sample-size 200

# Focused B-vs-baseline (10 samples, lower binary-layout noise):
cargo bench -p vortex-fsst --bench fsst_like_b_focused --features _test-harness \
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
│   ├── fsst_like_variants.rs    # all 5 variants, 4-round interleaved
│   └── fsst_like_b_focused.rs   # baseline + B only (binary-layout-clean)
├── src/
│   ├── bench_utils.rs           # scan_*_contains direct-matcher entry points
│   ├── bin/
│   │   ├── shufti_skip_report.rs   # A skip-fire rate
│   │   └── dfa_table_report.rs     # B class counts + shrink ratios
│   └── dfa/
│       ├── flat_contains.rs     # FlatContainsDfaBaseline / Dfa (A) / Classes (B) / ClassesPre (C)
│       └── shufti.rs            # ShuftiMask: PSHUFB/scalar dispatch
```

## Verdict

The existing `FlatContainsDfaBaseline` (state-0 `SkipStrategy` + flat 256-wide table) is hard to beat on these workloads. The visible perf headroom is in the **escape branch**, not the **table shape** or **per-state SIMD skip**.
