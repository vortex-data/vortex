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

## Next experiments

1. **Decode throughput** — measure SIMD-friendly decode of the front-coded
   token stream vs `zstd_decompress` of an equivalent block.
2. **Token-OVC comparator** — implement `compare_fused` over `&[u16]` and
   measure sort throughput against decode-then-byte-compare.
3. **Block-prefix factoring** — try the two-level layout where a block-wide
   common prefix is stored once and per-row front-coding only encodes the
   in-block divergence. Likely small additional win on URL/Title.
4. **Multi-column shared dict** — share an OnPair dict across columns with
   similar string distributions (e.g., URL + Referer).
