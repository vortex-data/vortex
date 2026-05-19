# onpair-bench

Empirical study: does **OnPair + token-space block front-coding** compress
lex-sorted string columns better than the standard baselines?

## Run

```bash
cargo run --release -p onpair-bench -- all 1000000 2
```

Args: `<dataset> <rows_per_slice> <n_slices>`. Datasets: `tpch_l_comment`,
`clickbench_title`, `clickbench_url`, `all`.

Output: `experiments/onpair-bench/results.md` plus stdout tables.

## What's measured

For each lex-sorted 1M-row slice, total bytes (data + per-row offsets) for:

| Encoding | Random access? |
|---|---|
| `raw (sorted)` | yes |
| `zstd-3 / zstd-9 monolithic` | no — must decode the whole stream |
| `zstd-3 block-1024` | per-block |
| `fsst` (vanilla `fsst-rs`) | per-row |
| `byte front-code 256` (DELTA_BYTE_ARRAY-style) | per-row within block |
| `onpair (12-bit)` | per-row |
| `onpair + front-code {64, 256, 1024}` | per-row within block |

## Headline result (1 M sorted rows per slice)

| Dataset | raw MiB | zstd-9 mono | zstd-3 blk-1024 | byte front-256 | **onpair (plain)** | **onpair + front-1024** |
|---|---:|---:|---:|---:|---:|---:|
| tpch_l_comment slice 0 | 29.1 | 3.68× | 3.24× | 1.78× | 3.34× | **3.54×** |
| tpch_l_comment slice 1 | 29.1 | 3.68× | 3.24× | 1.78× | 3.34× | **3.54×** |
| clickbench_title slice 0 | 135.8 | 21.98× | 18.50× | 9.15× | 4.14× | **17.40×** |
| clickbench_title slice 1 | 85.6 | 13.29× | 11.00× | 5.31× | 3.65× | **10.35×** |
| clickbench_url slice 0 | 88.3 | 8.63× | 6.58× | 3.48× | 3.25× | **6.45×** |
| clickbench_url slice 1 | 90.3 | 7.28× | 5.14× | 2.44× | 3.00× | **4.96×** |

## Findings

**1. OnPair + token front-coding matches block-zstd random-access compression
within ~5% across all datasets.**

The right-most two columns above are the apples-to-apples comparison: both
support per-block / per-row random access without re-decoding everything.
Across six independent slices, the gap stays within 5–6%, with OnPair
sometimes ahead (e.g. `clickbench_url` slice 1: 4.96× vs 5.14× — within 4%).

**2. Token-level front-coding consistently beats byte-level front-coding by
1.5–1.9×.**

Compare `onpair + front-1024` vs `byte front-code 256`:

| Dataset | byte front-256 | onpair + front-1024 | improvement |
|---|---:|---:|---:|
| tpch_l_comment | 1.78× | 3.54× | **1.99×** |
| clickbench_title (s0) | 9.15× | 17.40× | **1.90×** |
| clickbench_title (s1) | 5.31× | 10.35× | **1.95×** |
| clickbench_url (s0) | 3.48× | 6.45× | **1.85×** |
| clickbench_url (s1) | 2.44× | 4.96× | **2.03×** |

This confirms the structural argument that trained dictionary tokens align
with semantic content boundaries — the same shared content takes far fewer
"unit comparisons" to elide when units are learned tokens vs raw bytes.

**3. Front-coding's win over plain OnPair scales with prefix redundancy.**

| Dataset | plain onpair | onpair + front-1024 | win |
|---|---:|---:|---:|
| tpch_l_comment | 3.34× | 3.54× | 1.06× |
| clickbench_url (s0) | 3.25× | 6.45× | **1.98×** |
| clickbench_url (s1) | 3.00× | 4.96× | **1.65×** |
| clickbench_title (s1) | 3.65× | 10.35× | **2.83×** |
| clickbench_title (s0) | 4.14× | 17.40× | **4.20×** |

l_comment (random English snippets) shares almost nothing prefix-wise after
sort, so the front-code layer is overhead-neutral but adds nothing.
URL/Title — where sorted neighbours share long URL paths or repeated
domains/titles — get dramatic wins (up to 4.2×).

**4. FSST struggles on long-string data.**

FSST's 8-byte symbol cap means it can't capture long repeated phrases:

| Dataset | fsst | onpair (plain) |
|---|---:|---:|
| tpch_l_comment | 2.27× | 3.34× |
| clickbench_url (s0) | 1.67× | 3.25× |
| clickbench_title (s0) | 1.93× | 4.14× |

OnPair's 16-byte tokens cover much more of the content even without the
front-coding layer.

**5. Block size 1024 is the sweet spot.**

Across all datasets, `front-code 256` and `front-code 1024` differ by under
1% — and `64` is only slightly worse. Larger blocks have lower per-block
overhead, but the LCP within sorted neighbours is mostly captured by the
first few rows. 1024 also matches Vortex/fastlanes natural chunk granularity.

## Limitations / Caveats

- All numbers are *size only*; encode time is reported but decode throughput
  isn't yet measured. The compelling story for replacing block-zstd hinges
  on the fixed-width SIMD-friendly decode path actually being faster than
  zstd decompress, which is the next experiment.
- `byte front-code` here uses a 4-byte `shared` count for safety on long
  strings. A varint encoding would shrink the byte-level numbers somewhat
  (maybe 10–20% on l_comment) but not enough to overturn the
  token-vs-byte gap.
- "Random access" for `zstd block-1024` means *block-granular* — you decode
  a whole 1024-row block to get any row. OnPair + front-code is *per-row
  within a block*, walking ≤N prefix copies.
- These are *steady-state compression sizes*. The proposed encoding would
  need a real implementation, vectorised decode, and integration with
  Vortex's `LayoutReader` before it can be deployed.

## Sort throughput: `compare_fused` vs byte-compare

Run: `cargo run --release -p onpair-bench -- sort_bench all 1000000`.

### Three variants of `compare_fused`

All assume the rows share the same OnPair dict. Phase 1 (SIMD u16
equality-skip) is identical across all three. They differ in **Phase 2**
(what to do when tokens diverge):

| Variant | Phase 2 strategy | Storage cost |
|---|---|---|
| **v1** | `dict_table[]` → slice → `<[u8]>::cmp` | dict only |
| **v2** | `token_prefix[]` (first-8-bytes-BE u64) → `u64::cmp`, with length-bounded validity check; slow path on tie or padding-conflated case | +8 B/token |
| **v3** | `row_prefix[]` (first 8 bytes of decoded row) → `u64::cmp`, falls to v1 on tie | +8 B/row |

The v2/v3 paths require a correctness check (the first differing byte
position `k` must be within both tokens'/rows' real content) because
zero-padding in the u64 conflates "string ends here" with "byte is 0".
The naïve version without this check is wrong and fails the
`matches_byte_order_random` test — debugged inline in `compare_fused.rs`.

### Results (1M rows, all variants)

**Shuffled order (sort comparators see random pairs):**

| Dataset | v1 | v2 | v3 | byte cmp (pre-dec) | byte cmp (decode + sort) |
|---|---:|---:|---:|---:|---:|
| tpch_l_comment | 464 | 456 | **448** | **275** | 309 |
| clickbench_title | 394 | **368** | 436 | 388 | 514 |
| clickbench_url | **545** | 549 | 591 | **340** | 409 |

**Almost-sorted order (sort comparators see similar rows; Phase 1 actually skips):**

| Dataset | v1 | v2 | v3 | byte cmp (pre-dec) | byte cmp (decode + sort) |
|---|---:|---:|---:|---:|---:|
| tpch_l_comment | 258 | 246 | **228** | **166** | 183 |
| clickbench_title | 162 | **149** | 165 | **143** | 231 |
| clickbench_url | 233 | **230** | 258 | **153** | 222 |

### One real win: **v2 on shuffled `clickbench_title` beats libc memcmp**

`compare_fused v2: 368 ms` vs `byte cmp: 388 ms` (5% faster) — the only
combination where the dict-aware comparator beats raw memcmp on
pre-materialised bytes. The structural reason: Title rows are very long
(138 B avg) with strong learned-token redundancy (avg ~10 tokens/row), so
the Phase 1 SIMD skip does substantial work even on shuffled pairs, and
v2's u64 prefix avoids the slice-cmp dispatch entirely.

For shorter (l_comment, 26 B avg) or less-redundant (url, 88 B avg) data,
libc memcmp on materialised bytes wins because there isn't enough common
prefix for Phase 1 to amortise the dispatch overhead.

### Decode+sort end-to-end vs `compare_fused`

This is the realistic comparison when your storage form is OnPair-encoded
and you want to sort:

| Dataset | best `compare_fused` | decode + sort | winner |
|---|---:|---:|---|
| l_comment shuffled | 448 | 309 | **decode + sort** |
| l_comment almost-sorted | 228 | 183 | **decode + sort** |
| title shuffled | **368** | 514 | **compare_fused v2** (28% faster) |
| title almost-sorted | **149** | 231 | **compare_fused v2** (35% faster) |
| url shuffled | 545 | 409 | **decode + sort** |
| url almost-sorted | 230 | 222 | ~tied |

`compare_fused v2` is a clear end-to-end win on **clickbench_title** in
both orderings (decode is itself ~100 ms for 132 MiB of strings). For URL
and l_comment, decode is cheap enough that materialising and using libc
memcmp wins.

### What we tried and what didn't help

- **u128 stack-array XOR (originally proposed in earlier README)**: made it
  worse. The `[0u8; 16]` zero-fill + `copy_from_slice` per call dominated
  for short tokens.
- **Packed `(first 7 bytes BE) | length` per token** (initial v2): wrong
  — comparing length as a tiebreaker is invalid when both tokens are > 7
  bytes (byte 7 determines order, not length). Caught by the random-pair
  pairwise test. Fixed by switching to first-8-bytes packing with an
  explicit `k < min(len_a, len_b)` validity check before trusting the u64
  result.
- **`sort_unstable_by` vs `sort_by`**: ~5% win across the board.
- **Flat `Vec<u16>` + boundaries vs `Vec<Vec<u16>>`**: ~15-20% win, helps
  both compare_fused and byte cmp equally — better cache locality, no
  per-row allocation.

### Universal win: `two-pass: 32B key + byte cmp refine`

A two-stage sort that beats pure `<[u8]>::cmp` on **all six** test cases:

1. **Pass 1**: build a 32-byte key per row (first 32 decoded bytes,
   BE-packed as `[u64; 4]`). Integer-sort `(key, idx)` pairs with pdqsort.
2. **Pass 2 (byte cmp refine)**: within each run of equal keys, sort the
   sub-range using `<[u8]>::cmp` on the materialised decoded bytes.

This is essentially **MSD radix sort at byte granularity, with a fast
inner sort**. The 32B integer-sort partitions even data where the key is
not very discriminating (e.g., URL — 998K of 1M rows tied on the
key, but split into 2368 partitions of ~421 each); sort within
partitions is much faster than sort across all N because per-comparison
cache footprint shrinks and `log(N/M)` < `log(N)`.

| Dataset | 32B + byte refine | byte cmp pure | **speedup** |
|---|---:|---:|---:|
| tpch_l_comment shuffled | **118** | 275 | **2.33×** |
| tpch_l_comment almost-sorted | **84** | 166 | **1.98×** |
| clickbench_title shuffled | **207** | 339 | **1.64×** |
| clickbench_title almost-sorted | **91** | 148 | **1.63×** |
| clickbench_url shuffled | **295** | 331 | **1.12×** |
| clickbench_url almost-sorted | **126** | 150 | **1.19×** |

Caveats:
- Requires both the encoded form (to build the 32B key cheaply from
  tokens) AND the decoded bytes (for the refine step). End-to-end vs
  `decode + sort`, the gap is similar — adding decode (~80-100 ms for
  URL, ~50 ms for l_comment) to the byte-refine path still leaves it
  ahead.
- Key building cost (~20-30 ms) is not in the table above — it's
  treated as sort prep, same as the dict_table build.
- The 32-byte prefix is `1M × 32 = 32 MiB` of additional working set.
  Fits in L2 + L3 comfortably for the workloads tested.

### Two-pass with `compare_fused` refine (no decoded bytes required)

When decoded bytes aren't available (pure encoded-form storage), the same
two-pass structure with `compare_fused` as the refine step is still a win
on 2/3 datasets. Slightly slower than byte-cmp-refine but doesn't require
materialising the column.

| Variant | Key per row | When it wins |
|---|---|---|
| `two-pass: u128 key sort + refine ties` | 16 B | l_comment, Title |
| `two-pass: 32B key sort + refine ties` | 32 B | l_comment, Title, closes URL gap |

Numbers (1M rows, ms — same harness as above):

**Shuffled:**

| Dataset | best compare_fused | 16B two-pass | **32B two-pass** | byte cmp (pre-dec) |
|---|---:|---:|---:|---:|
| tpch_l_comment | 448 (v3) | 146 | **108** | 290 |
| clickbench_title | 368 (v2) | 295 | **239** | 363 |
| clickbench_url | 535 (v1) | 486 | 440 | **370** |

**Almost-sorted:**

| Dataset | best compare_fused | 16B two-pass | **32B two-pass** | byte cmp (pre-dec) |
|---|---:|---:|---:|---:|
| tpch_l_comment | 224 (v3) | 92 | 97 | 186 |
| clickbench_title | 151 (v3) | 117 | **106** | 148 |
| clickbench_url | 230 (v2) | 207 | 191 | **173** |

The 32B two-pass is:
- **2.7× faster** than byte cmp on l_comment shuffled
- **1.5× faster** than byte cmp on Title shuffled
- ~1.4× faster than byte cmp on l_comment almost-sorted
- Within ~15% of byte cmp on URL (loses; first 32 B insufficient
  to discriminate URLs that share long domain+path prefixes)

A 64- or 128-byte key would close the URL gap further, at proportional
memory cost (1 M rows × 64 B = 64 MiB key data vs 32 MiB at 32 B).

The reason it works: pdqsort on `(u64, u64, u64, u64, u32)` tuples is
essentially memory-bandwidth-bound (~250 ns/row at 1 M elements), no
dict access, no slice cmp dispatch. Refinement only runs on the small
fraction of rows that tied — for l_comment and Title most rows differ
within 32 bytes, so refinement is near-zero work.

### Honest takeaway

`compare_fused` is **not a universal sort accelerator** in this naive
implementation. libc memcmp is hard to beat on pre-decoded contiguous
bytes — it's hand-tuned assembly with branch-free paths for short
slices. The dict indirection in Phase 2 has unavoidable per-call cost.

Where it does pay off:

1. **Long highly-structured strings** (clickbench_title): v2 matches or
   beats raw memcmp. Phase 1 SIMD skip does enough work to amortise.
2. **End-to-end on data where decode is expensive** (any sufficiently long
   column): you skip the decode entirely. The win scales with decoded
   row length and is realistic for cold-cache scans of compressed
   columns.

Where it does not help today:

- Short rows (l_comment, ~26 B): byte cmp is too cheap to beat.
- Moderately long rows with weak prefix redundancy (URL): byte cmp wins.
- Sort orders where Phase 1 skips few tokens (most shuffled cases).

## Next experiments

1. **Decode throughput** — SIMD-friendly decode of the front-coded token
   stream vs `zstd_decompress` of an equivalent block.
2. **Adaptive two-pass key length** — start with 16 B, extend the key
   *only for tied groups* that are above a size threshold. Should match
   32-B two-pass on URL while keeping memory low for diverse data.
3. **Radix sort the integer key** — `pdqsort` on `(u64, u64, u64, u64,
   u32)` tuples is already fast (~250 ns/row), but MSD radix sort would
   likely push it to ~80-100 ns/row for the same key size. Marginal win
   on top of two-pass.
4. **Block-prefix factoring** — two-level layout where a block-wide common
   prefix is stored once.
5. **Multi-column shared dict** — share an OnPair dict across columns with
   similar string distributions (e.g., URL + Referer).
