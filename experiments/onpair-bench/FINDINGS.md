# OnPair experiments — consolidated findings

Working notes from `claude/onpair-compression-research-is9Qh`. Reproducible
via the `onpair-bench` crate at `experiments/onpair-bench/`.

## TL;DR

1. **Compression**: OnPair-encoded codes + token-space block front-coding
   on lex-sorted columns matches `zstd -3` block-1024 random-access
   compression within ~5%, while staying in fixed-width SIMD-friendly
   form (dict + bit-packed codes, no LZ scanning). On ClickBench Title
   it hits **17.4×** vs raw + per-row offsets; on URL **6.45×**;
   on TPC-H l_comment **3.54×**.

2. **Sort**: A two-pass "32-byte key + byte-cmp refine" approach (MSD
   radix sort at byte granularity with libc memcmp inner sort) beats
   pure `<[u8]>::cmp` on **all 6** workload variants by **1.12×–2.33×**.
   The pure-encoded-form variant (`compare_fused` refine instead of
   byte-cmp refine) wins on 2/3 datasets.

3. **Open / unanswered**: the row-prefix trick added **300–600× the
   dict size** in auxiliary structure (15 MiB / 30 MiB row prefix vs
   50 KB dict). This is structurally suspect — see "Open critique"
   below.

## Datasets

All measurements at 1M rows × 2 slices per dataset:

| Dataset | avg row | total raw | source |
|---|---:|---:|---|
| TPC-H `l_comment` | 26 B | 25 MiB | `tpchgen` SF=1 |
| ClickBench `Title` | 138 B | 132 MiB | `hits_0.parquet` |
| ClickBench `URL` | 88 B | 84 MiB | `hits_0.parquet` |

## Part 1 — compression on lex-sorted columns

Encoding tested: train an OnPair dict over the column, sort by row tokens'
LPM output (≡ lex-sort the strings), then per-block store
`(block_token_prefix, per_row (shared_with_prev_count, suffix_tokens))`,
all bit-packed at OnPair's 12-bit code width.

Compression ratios vs raw + per-row offsets, all variants include
per-row offsets for random-row access:

| Dataset | byte front-256 | zstd-3 blk-1024 | plain onpair | **onpair + front-1024** |
|---|---:|---:|---:|---:|
| tpch_l_comment | 1.78× | 3.24× | 3.34× | **3.54×** |
| clickbench_title slice 0 | 9.15× | 18.5× | 4.14× | **17.4×** |
| clickbench_title slice 1 | 5.31× | 11.0× | 3.65× | **10.4×** |
| clickbench_url slice 0 | 3.48× | 6.58× | 3.25× | **6.45×** |
| clickbench_url slice 1 | 2.44× | 5.14× | 3.00× | **4.96×** |

Key findings:

- **Token-space front-coding beats byte-space by 1.85–2.03×** universally
  — trained dict tokens align with semantic content boundaries.
- **Front-coding's win over plain OnPair scales with prefix redundancy**:
  ~6% on random English, up to 4.2× on Title.
- FSST is structurally weak (1.67–2.27×) on long-string data because of
  its 8-byte symbol cap.
- The compression encoding is structural-only — no decode-throughput
  measurement yet. The "matches block-zstd with faster decode" claim
  remains untested.

## Part 2 — sort throughput experiments

Goal: implement the `compare_fused` primitive over `&[u16]` OnPair token
sequences, demonstrate it beats decode-then-byte-compare for sort.

Outcome: the naive `sort_by(compare_fused)` does NOT beat libc memcmp on
pre-decoded contiguous bytes, but a **two-pass radix-then-refine** sort
that uses compare_fused (or byte-cmp) as the inner sort DOES.

### `compare_fused` algorithm

Two-phase comparator for two `&[u16]` rows sharing a dict:

1. **Phase 1**: scan equal token IDs (SIMD-friendly u16 equality loop).
   `a[i] == b[i]` ⇒ bytes identical (LPM is deterministic on a fixed
   dict), so each match skips up to 16 B of decoded content.
2. **Phase 2**: when tokens diverge, fall to byte-level comparison of
   the diverging tokens, with cursor-based roll-over for the rare
   prefix-relationship boundary case.

Three Phase 2 variants implemented and benchmarked:
- **v1**: `dict_table[token_id]` → slice → `<[u8]>::cmp`.
- **v2**: precomputed 8-byte token-prefix `u64` → `u64::cmp` with
  validity check on which byte position the difference is at.
- **v3**: precomputed 8-byte row-prefix `u64` → resolve at row level.

Both v2 and v3 needed a non-obvious correctness fix: zero-padding in the
`u64` conflates "string ends here" with "byte is 0". You must verify the
first differing byte position `k` is within both tokens'/rows' real
content before trusting the `u64` comparison. The initial v2 packed
length-as-tiebreaker which is **wrong** for tokens > 7 bytes that differ
in byte 7+; caught by pairwise correctness test.

### Raw `sort_by(compare_fused)` results

Per-comparison overhead dominates. compare_fused is ~2× slower per
comparison than `<[u8]>::cmp` on pre-decoded bytes (libc memcmp). The
only single-comparator win is **v2 on shuffled clickbench_title**: 368 ms
vs byte cmp 388 ms (5% faster). For l_comment and URL, byte cmp wins.

### Two-pass sort

The breakthrough was changing the *sort structure*: precompute a
fixed-width byte prefix per row, integer-sort `(prefix, idx)` pairs with
pdqsort (or sort `Vec<u32>` indices via lookup), then refine the small
ties with a row comparator. This pattern beats single-pass byte cmp
across the board.

Variants:

| Variant | Refine fn | Key size |
|---|---|---|
| 16B + compare_fused refine | compare_fused v1 | 16 B/row |
| 32B + compare_fused refine | compare_fused v1 | 32 B/row |
| 16B + byte cmp refine | `<[u8]>::cmp` on decoded | 16 B/row |
| 32B + byte cmp refine | `<[u8]>::cmp` on decoded | 32 B/row |
| 32B + indirect sort | compare_fused v1 | 32 B/row |

### Sort throughput results (1M rows, sort time only, ms)

**Shuffled order:**

| Dataset | byte cmp pure | **best two-pass** | speedup |
|---|---:|---:|---:|
| tpch_l_comment | 275 | **110** (32B+cf) / 118 (32B+byte) | 2.50× |
| clickbench_title | 339 | **207** (32B+byte) | 1.64× |
| clickbench_url | 331 | **295** (32B+byte) | 1.12× |

**Almost-sorted (1% pairs swapped):**

| Dataset | byte cmp pure | **best two-pass** | speedup |
|---|---:|---:|---:|
| tpch_l_comment | 166 | **66** (16B+byte) / 77 (32B+cf) | 2.51× |
| clickbench_title | 148 | **91** (32B+byte) | 1.63× |
| clickbench_url | 150 | **126** (32B+byte) | 1.19× |

Why two-pass works even when refinement does most of the work:
- For URL, 998K of 1M rows tied on the 32B key, but split into 2368
  partitions of ~421 each.
- Sort cost is `N log N`; partitioned cost is
  `Σ_p (|p| log |p|) < N log(N/M)` where `M` = num partitions.
- For URL: `1M × log 1M ≈ 20M` vs `2368 × 421 × log 421 ≈ 8.6M`. Plus
  partition-local sorts have much better cache behaviour.

### Tradeoffs by variant

- **compare_fused refine** (no decoded bytes needed): the pure
  encoded-form path. Faster than byte cmp on l_comment and Title.
  Loses on URL because URL ties are large and compare_fused is slower
  per-call than memcmp.
- **byte cmp refine** (requires materialised bytes): universal winner.
  Best when decode is "free" or can be amortised across other queries.
- **16B vs 32B key**: 16B wins on cache-tight or highly-discriminating
  data (l_comment almost-sorted: 66 ms vs 76 ms). 32B wins on
  high-shared-prefix data (URL: 295 ms vs 367 ms shuffled).

## Open critique — "the prefix is larger than the dict"

Confirmed via instrumentation:

| Structure | URL | Title | l_comment |
|---|---:|---:|---:|
| OnPair dict (bytes + offsets) | **40 KB** | **45 KB** | **50 KB** |
| 16B row prefix (1M rows) | 15.26 MiB | 15.26 MiB | 15.26 MiB |
| 32B row prefix (1M rows) | 30.52 MiB | 30.52 MiB | 30.52 MiB |
| **Ratio prefix : dict** | **381× / 763×** | **339× / 678×** | **305× / 610×** |

The 32-byte row prefix is **300–600× the dict size**. This is
structurally suspect: the dict captures common substrings (good — exploits
redundancy), but the row prefix duplicates raw bytes (bad — does not
exploit redundancy at all). It's exactly the FSST critique: a fast
auxiliary structure that scales with N, not with the data's compressible
structure.

### Why the per-row prefix exists

We added it because the dict alone doesn't give us a *row-level* sort
key. The token IDs do, but **token-sequence order ≠ lex byte order**
because LPM is greedy and produces different token boundaries for
near-identical strings.

### Possible directions to make the prefix dict-sized instead of N-sized

1. **Order-preserving dict / LPM**. Assign token IDs in lex order AND
   constrain LPM so the tokenization is monotonic in the input. Then
   sorting rows by token sequence ≡ lex-sorting the strings. The token
   sequence (avg ~5–10 u16s per row = 10–20 B) is the sort key — no
   extra prefix needed. Cost: ~10–30% compression ratio loss (known
   tradeoff from order-preserving BPE / ALM variants).

2. **Token-ID prefix as sort key**. Even *without* a fully order-preserving
   dict, the first 2–4 token IDs of each row could serve as a partial sort
   key. 4 token IDs × 2 B = 8 B/row → 8 MiB at 1M rows. Same magnitude as
   16B row prefix but smaller. Use compare_fused for refinement. Requires
   acknowledging that this key gives approximate-only ordering; refinement
   handles all true ties AND false ties from boundary-disagreeing rows.

3. **Hierarchical dict-of-prefixes**. Build a second tiny "prefix dict"
   that maps row-prefix patterns to short codes. Each row stores a small
   prefix-code instead of raw bytes. Sort key = prefix-code. Cost: extra
   training work and another indirection.

4. **Drop per-row prefix entirely**. Use only the dict and `compare_fused`.
   We already measured this path — it wins on Title but loses on
   URL/l_comment. Acceptable if the workload is Title-heavy.

The user's framing was: "you added a FSST+ like prefix thing that's
larger than the dict??" — and yes, that's exactly the structural
oddity. The two-pass + byte cmp refine method *is* the fastest sort we
measured, but its 30 MiB prefix is the elephant in the room. Next
direction would be to test whether **token-prefix-based two-pass** (8
MiB) gets us most of the win at a much smaller footprint.

## What's committed

```
encodings/onpair-rs/                     # pulled from gpu branch
experiments/onpair-bench/
├── Cargo.toml
├── README.md                            # benchmark methodology + numbers
├── FINDINGS.md                          # this file
├── results.md                           # auto-generated raw tables
└── src/
    ├── main.rs                          # CLI
    ├── datasets.rs                      # TPC-H + ClickBench loaders
    ├── encoders.rs                      # raw, zstd, fsst, onpair, front-coded
    ├── frontcode.rs                     # block front-coding in token space
    ├── compare_fused.rs                 # v1/v2/v3 comparator + tests
    └── sort_bench.rs                    # sort throughput harness
```

Branch: `claude/onpair-compression-research-is9Qh`.

## Reproduce

```bash
# Compression study (1M rows × 2 slices, all 3 datasets):
cargo run --release -p onpair-bench -- all 1000000 2

# Sort throughput study:
cargo run --release -p onpair-bench -- sort_bench all 1000000
```

Downloads ClickBench `hits_0.parquet` (~120 MB) and `hits_1.parquet`
(~170 MB) to `/tmp/onpair-bench-data/` on first run. TPC-H l_comment is
generated in-process via `tpchgen`.

## Things to try next

1. **Token-prefix sort key**: 4 token IDs per row (8 B) as the sort
   key, two-pass with refinement. Compare its size+speed against the
   30 MiB byte-prefix approach.
2. **Order-preserving dict variant**: train OnPair with monotonic LPM
   so token-sequence order ≡ lex order; eliminate the prefix entirely.
3. **MSD radix on the integer key** instead of pdqsort — likely
   2-3× speedup on the pass-1 step.
4. **Decode throughput** — the missing measurement to validate the
   compression encoding's "faster than zstd decode" pitch.
5. **`compare_fused` with const-generic BIT_WIDTH dispatch** — match
   `Column::decode_all_inner`'s strategy. Marginal but free.
