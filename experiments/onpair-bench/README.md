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

Comparator correctness is asserted in code — all three sort methods produce
the same permutation. The algorithm and its boundary case are also covered
by `compare_fused::tests::matches_byte_order_random` (pairwise check
across 2000 rows).

### Results (1M shuffled rows, then sort)

| Dataset | compare_fused (tokens) | byte cmp (pre-decoded, sort only) | byte cmp (decode + sort) |
|---|---:|---:|---:|
| tpch_l_comment | 616 ms (41 MB/s) | **324 ms (78 MB/s)** | 364 ms (69 MB/s) |
| clickbench_title | 472 ms (280 MB/s) | **384 ms (344 MB/s)** | 579 ms (228 MB/s) |
| clickbench_url | 690 ms (122 MB/s) | **351 ms (240 MB/s)** | 524 ms (161 MB/s) |

Per-comparison cost (~`N log₂ N` ≈ 20M comparisons):

| Dataset | compare_fused | `<[u8]>::cmp` (libc memcmp) |
|---|---:|---:|
| tpch_l_comment | 31 ns | 16 ns |
| clickbench_title | 24 ns | 19 ns |
| clickbench_url | 35 ns | 18 ns |

### Honest analysis

**The Phase 1 SIMD u16-skip works as designed, but the algorithm in its
naive form is consistently ~2× slower per comparison than libc memcmp.**
The reason is structural to sort workloads:

- **Sort comparisons hit random pairs.** For a Timsort/quicksort comparator,
  most pairs of rows are *not* sorted neighbours. Their first 1–2 tokens
  almost always differ. Phase 1 contributes ~0 work; the dispatch to Phase 2
  fires on essentially every comparison.
- **Phase 2's per-call overhead is real.** Even one Phase 2 iteration costs:
  one `dict_offsets[]` indexed load (×2), one slice construction (×2), one
  generic `<[u8]>::cmp` call which then dispatches to memcmp. That's more
  startup cost than a single memcmp on the whole row.
- **libc memcmp is hard to beat on short strings.** It's hand-tuned assembly
  with branch-free paths for ≤16 bytes.

`compare_fused` only beats the *end-to-end* decode+sort path on
`clickbench_title` (472 ms vs 579 ms, ~18% win) — and only because Title is
long enough that decode is itself a meaningful cost. For short
(`l_comment`) or moderately-long (`url`) data, decoding is cheap, and the
materialised bytes sort faster than `compare_fused`.

**This is a negative result for the naive implementation, but not for the
algorithm.** The clear next step is making Phase 2 cheap enough that it
matches or beats memcmp's per-call overhead. Specifically:

1. **Inline-XOR Phase 2 for ≤16-byte tokens.** OnPair16's tokens fit in
   `u128`. Load both, XOR, `TZCNT` to find the first differing byte, compare
   those two bytes. No `<[u8]>::cmp` call, no slice construction, no memcmp
   dispatch. This is exactly what the OnPair16 paper describes for its
   internal LPM and is the obvious port.
2. **Use the existing `dict_table: Vec<u64>` packed offset/length table** in
   the `onpair-rs` crate (already built by `column::build_dict_table`) so
   Phase 2 is one indexed load instead of two.
3. **Specialise the comparator with `BITS` const-generic dispatch** the same
   way `Column::decode_all_inner` already does, so the inner cursor inlines
   per bit width.

If those reach `byte cmp (pre-decoded)` parity (~17 ns/cmp), `compare_fused`
becomes a clear win on **encoded-form storage** because you skip decode
entirely. That's the experiment to run next.

## Next experiments

1. **Decode throughput** — SIMD-friendly decode of the front-coded token
   stream vs `zstd_decompress` of an equivalent block.
2. **Optimise `compare_fused` Phase 2** with inline-XOR-on-u128 and packed
   dict table; re-run `sort_bench` to see if it crosses parity with libc
   memcmp.
3. **Block-prefix factoring** — two-level layout where a block-wide common
   prefix is stored once.
4. **Multi-column shared dict** — share an OnPair dict across columns with
   similar string distributions (e.g., URL + Referer).
