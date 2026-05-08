# Experiment Report: Variant A — Hyperscan-style Shufti Per-state Skip

**Date:** 2026-05-08  
**Worktree:** `/home/user/vortex/.claude/worktrees/agent-ab053c9b87b694a3a`  
**Branch:** `claude/fsst-paper-branch-LJWCh`  
**Base commit:** `f218484a6` (`origin/ji/fsst-like-paper`)

---

## 1. What Was Changed

- **`encodings/fsst/src/dfa/shufti.rs`** (new): `ShuftiMask` struct with
  `lo[u8; 16]` + `hi[u8; 16]` nibble tables. `find_next()` dispatches to
  SSSE3 `find_next_ssse3()` when available, scalar fallback otherwise.
  The SIMD path processes 16 bytes per iteration with 2× `VPSHUFB` + `VPAND`
  + `VPTESTNMB` (compiler upgraded from SSSE3 to AVX-512 on this machine).

- **`encodings/fsst/src/dfa/flat_contains.rs`**: Added `FlatContainsDfaBaseline`
  (original state-0-only skip, preserved verbatim) and rewrote `FlatContainsDfa`
  to build per-state `ShuftiMask` for all DFA states and apply it on every
  state transition. Also added 3 atomic counters (gated behind
  `feature = "shufti-counters"`) for profiling skip fire rates.

- **`encodings/fsst/src/dfa/mod.rs`**: Added `mod shufti`, exposed
  `FlatContainsDfaBaseline` and `FlatContainsDfa` as `pub(crate)`, and
  re-exported counter statics under the feature flag.

- **`encodings/fsst/src/bench_utils.rs`** (new, `_test-harness`): Low-level
  `scan_baseline_contains` / `scan_shufti_contains` helpers that bypass the
  LikeKernel layer. Also exposes `reset_shufti_counters` / `read_shufti_counters`.

- **`encodings/fsst/benches/fsst_like_variants.rs`** (new): divan bench with
  7 datasets × {baseline, shufti} for direct side-by-side comparison.

- **`encodings/fsst/src/bin/shufti_skip_report.rs`** (new): Binary that runs all
  7 datasets and prints skip-call / fired / avg-skip statistics.

- **Tests**: Added `test_shufti_parity_no_symbols` (rstest cases) and
  `test_shufti_parity_exhaustive` confirming shufti agrees with baseline.

---

## 2. Did the Skip Actually Fire?

Run with `cargo run -p vortex-fsst --bin shufti_skip_report --features "_test-harness,shufti-counters" --release`:

| Dataset           | Needle           | Matches | Skip Calls | Fired   | Fire Rate | Avg codes skipped |
|-------------------|------------------|---------|------------|---------|-----------|-------------------|
| `short_urls`      | `google`         | 25,244  | 117,517    | 32,705  | 27.8%     | 3.22              |
| `clickbench_urls` | `yandex`         | 29,891  | 205,542    | 70,315  | 34.2%     | 10.25             |
| `log_lines`       | `Googlebot`      | 16,787  | 207,043    | 49,297  | 23.8%     | 23.54             |
| `json_strings`    | `enterprise`     | 23,427  | 257,518    | 106,778 | 41.5%     | 12.27             |
| `file_paths`      | `target/release` | 18,843  | 127,605    | 28,949  | 22.7%     | 3.90              |
| `emails`          | `gmail`          | 10,194  | 120,906    | 28,524  | 23.6%     | 2.85              |
| `rare_match`      | `xyzzy`          | 1       | 100,004    | 1       | 0.0%      | 37.00             |

The skip **does** fire for 5 of 7 datasets. The `rare_match` dataset essentially
never fires because the needle (`xyzzy`) almost never appears, so the DFA rarely
enters non-zero states. The `log_lines` skip saves the most codes per fire (23.5
codes average), reflecting the long, repetitive log-line structure.

---

## 3. Bench Numbers — 4-round Interleaved

Command:
```
cargo bench -p vortex-fsst --bench fsst_like_variants --features _test-harness -- --sample-count 4
```

All times in milliseconds. "Fastest" is used as the representative number (avoids
cold-start outliers from `LazyLock` dataset initialization in the baseline column).

| Dataset           | Baseline fastest | Shufti fastest | Shufti median | Delta (fastest) |
|-------------------|-----------------|----------------|---------------|-----------------|
| `clickbench_urls` | 4.53 ms         | 4.51 ms        | 4.58 ms       | ≈flat (-0.4%)   |
| `emails`          | 2.26 ms         | 2.40 ms        | 2.45 ms       | +6% slower      |
| `json_strings`    | 4.67 ms         | 4.94 ms        | 4.98 ms       | +6% slower      |
| `log_lines`       | **3.45 ms**     | **4.15 ms**    | 4.22 ms       | **+20% slower** |
| `file_paths`      | 2.46 ms         | 2.66 ms        | 2.69 ms       | +8% slower      |
| `rare_match`      | 1.97 ms         | 2.77 ms        | 2.93 ms       | **+41% slower** |
| `short_urls`      | 2.41 ms         | 2.48 ms        | 2.50 ms       | +3% slower      |

> Note: Baseline "fastest" column excludes the first cold-start iteration; median
> baseline would be ~2–3x higher due to `LazyLock` initialization in the first
> sample. Shufti timings are extremely stable (narrow fastest/slowest band),
> confirming consistent behavior.

**The shufti variant is uniformly slower or flat across all datasets.**

---

## 4. Hot-Path Assembly (FlatContainsDfa::matches)

The compiler (with `target-cpu=native`, which detected AVX-512) generated a tighter
loop than the SSSE3 target specified in `#[target_feature(enable = "ssse3")]`.
The inner 16-byte-at-a-time loop:

```asm
.LBB398_15:
    cmpq    %r8, %rcx                   # 16 bytes available?
    ja      .LBB398_16                  # if not, goto scalar tail
    vmovdqu (%rax,%r10), %xmm3          # load 16 bytes
    vpsrlw  $4, %xmm3, %xmm4           # compute hi nibble (>> 4)
    vpand   %xmm0, %xmm3, %xmm3        # mask lo nibble
    vpshufb %xmm3, %xmm1, %xmm3        # lo_table[lo_nibble]
    vpand   %xmm0, %xmm4, %xmm4        # mask hi nibble
    vpshufb %xmm4, %xmm2, %xmm4        # hi_table[hi_nibble]
    vptestnmb %xmm3, %xmm4, %k0        # AND == 0? (AVX-512)
    addq    $16, %r10                   # advance pos
    addq    $16, %rcx
    addq    $16, %r9
    kortestw %k0, %k0                   # any interesting byte?
    jb      .LBB398_15                  # loop if none
    kmovd   %k0, %ecx                   # extract mask
    notl    %ecx
    tzcntl  %ecx, %ecx                  # find first set bit
    ...
```

That's 14 instructions for 16 bytes = 0.875 instructions/byte in the skip loop.
The full function is 172 lines of assembly including the DFA step, sentinel
handling, and tails.

---

## 5. Honest Verdict

**Does it win? No, not on any dataset tested.**

**Why the skip fires but doesn't help:**

1. **The state-0 baseline is already extremely fast.** The existing
   `SkipStrategy` (memchr for 1-3 interesting codes, bitmap for 4+) at state 0
   is heavily SIMD-accelerated. For most datasets, the DFA spends the vast
   majority of time in state 0, where the baseline already applies the
   equivalent of a SIMD skip. The shufti variant replaces this with a slightly
   more expensive mechanism.

2. **Non-zero states are rare and short.** When the DFA enters states 1, 2,
   …, n-1 (partial match progress), it typically stays there for only a few
   codes before either matching or falling back to state 0. The shufti overhead
   (2× SIMD vector loads, 2× PSHUFB, 1× AND + test) is 4–5 cycles minimum per
   call, which is not amortized over the 3–10 codes saved.

3. **`rare_match` (+41% slower)**: the needle never appears, so the DFA
   almost never leaves state 0. Per-state shufti calls state 0's skip every
   iteration; the baseline's memchr equivalent scans further per call.

4. **`log_lines` (+20% slower despite 23-code average skip):** The long strings
   do benefit from the skip when it fires (avg 23.5 codes skipped), but the
   `Googlebot` needle has many partial-progress opportunities, creating
   frequent state transitions that accumulate shufti overhead.

5. **False-positive rate**: the nibble-based encoding uses only 8 bits per
   byte entry, creating potential false positives for codes that share a nibble
   with interesting codes. These false positives cause unnecessary DFA steps.

**What would actually help:**

- Apply the shufti skip **only at state 0** (which the baseline already does
  via memchr) and for states where there are **many** (>8) uninteresting codes.
- For states with few uninteresting codes (typical for non-zero states near the
  accept), a shufti skip is net overhead.
- The best path for non-zero states would be an escape-folded DFA that handles
  the sentinel in-table (no branch per code) — see the TODO in `flat_contains.rs`
  citing commit `7faf9f36f`.

**What surprised me:**

- The compiler upgraded SSSE3 code to AVX-512 (`VPTESTNMB`) which is cleaner,
  but didn't change the fundamental overhead story.
- The `log_lines` dataset has the highest skip fire rate and largest average
  skip distance, but still loses — showing that raw skip throughput isn't the
  bottleneck. The bottleneck is the per-skip overhead on the short non-zero-state
  segments.
- Shufti timing is extremely stable (narrow fastest/slowest band), making it
  predictable but predictably slower.

---

## Checks Run

- `cargo build -p vortex-fsst` — clean (1 dead-code warning for unused
  `find_anchor_symbol`, pre-existing)
- `cargo test -p vortex-fsst` — **121 passed, 0 failed** (including 10 new
  parity tests)
- `cargo bench -p vortex-fsst --bench fsst_like_variants --features _test-harness -- --sample-count 4` — ran successfully
- `cargo run --bin shufti_skip_report --features "_test-harness,shufti-counters" --release` — skip-fire statistics collected

Did not run `cargo clippy --all-targets` or `./scripts/public-api.sh` because
this is an internal/experimental crate change with no public API surface changes
(all new items are `pub(crate)` or behind `_test-harness`).
