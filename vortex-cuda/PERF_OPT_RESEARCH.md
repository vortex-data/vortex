# OnPair AOT-statistics-driven specialization — research synthesis

Companion to `PERF_RESEARCH.md` (which covers the shared-mem staging recipe
already in `onpair_shmem.cu`). Scope here: techniques that exploit
**ahead-of-time dictionary statistics** (`max_len`, `avg_len`, length
distribution, `dict_size_bytes`) to close the 511 → 64-220 GiB/s gap on
real ClickBench / TPC-H columns.

Honest preface: the published GPU dict-decompression literature is small.
None of the four most-relevant papers (GSST, GPU-FSST, FastLanesGPU,
G-ALP) treat AOT statistic-driven kernel selection as a first-class
research question. We synthesize what they do exploit, then propose
where to push beyond them. Numbers are reported when the paper states
them; "—" means not reported.

---

## 1. GSST (Vonk et al., SIGOPS OSR 2025)

[paper](https://dl.acm.org/doi/10.1145/3759441.3759450) ·
[preprint PDF](https://repository.tudelft.nl/file/File_627b50ef-4c9a-4367-bd9c-b640c978edff?preview=1) ·
[thesis](https://repository.tudelft.nl/record/uuid:71c1dddf-3b7d-4c12-a079-d716fad501b2)

**AOT statistics exploited.** FSST guarantees ≤256 symbols of ≤8 bytes
each ⇒ symbol table is **always ≤2 KiB**, fits in shared memory
unconditionally. Max-len-8 is baked into the format, so each thread
reads exactly one `uint64_t` per symbol — no runtime branching on
length. GSST does **not** adapt the kernel to a per-column avg_len or
length histogram; the format constants do all the specialization.

**Speedup.** 191 GB/s on A100 vs nvCOMP LZ4 family (5-60 GB/s on the
same Silesia inputs). Six optimizations stack multiplicatively; the
paper's ablation (Fig 6) names "aligned memory accesses" as the single
biggest jump after shared-memory staging. No per-optimization GB/s
table.

**Applicability to OnPair.** Short tokens: ✓ same regime. Mid: ✓.
Long: partial — OnPair's max_len is per-column and can exceed 8 bytes,
so the `uint64_t`-per-symbol shortcut does not apply uniformly. The
"≤8 B fixed stride" idea **does** generalize: if a column's
`max_len ≤ 8`, we can dispatch a u64-stride variant.

**LOC.** ~30 lines (template the existing `onpair_shmem` on
`STRIDE ∈ {4, 8, 16}`, pick by `max_len`).

---

## 2. GPU-FSST (Anema et al., ADMS/VLDB 2025)

[paper PDF](https://www.vldb.org/2025/Workshops/VLDB-Workshops-2025/ADMS/ADMS25-01.pdf) ·
[repo](https://github.com/timanema/fsst-gpu)

**AOT statistics exploited.** Same FSST format constants as GSST.
Additionally exploits: (a) symbol table is a known small constant ⇒
materialized as a 2-D `result[8][THREAD_COUNT]` shared buffer; (b)
the `match_table` rows are constrained to multiples of 32 with ≤8
columns providing "negligible compression gain" beyond that — i.e.
the format itself prunes the search space the kernel must handle. No
per-column profile-guided variant selection.

**Speedup.** 74 GB/s **compression** on RTX 4090; **decompression
number not headline** — repo example clocks 0.353 GB/s, treated as an
implementation gap.

**Applicability to OnPair.** The encoder-side `result[8][N]` shared
layout is essentially a stride-8 staging buffer — confirms the
stride-by-max-len idea is in use, just on the compression side.

**LOC.** N/A — encoder pattern, not directly portable.

---

## 3. FastLanesGPU / G-ALP (Afroozeh DaMoN 2024; Hepkema 2024)

[FastLanesGPU paper](https://dl.acm.org/doi/10.1145/3662010.3663450) ·
[G-ALP](https://ir.cwi.nl/pub/35205/35205.pdf)

**AOT statistics exploited.** FastLanes encodes 1024 values per data
segment with a fixed lane layout. Each kernel is specialized per
encoding (DICT, FOR, DELTA, RLE, FSST) and per bit-width, generated at
build time. The bit-width *is* the AOT statistic. G-ALP extends this
by integrating decoding into compute kernels (fused decode-then-use),
explicitly minimizing register footprint by emitting **one value at a
time** — relevant when the consumer doesn't need the full token in a
single register.

**Speedup.** FastLanesGPU: 3-4× over Shanbhag's Tile-Based GPU
bit-unpacker on T4/V100 (no A100 number). G-ALP: "highest decode
throughput of all schemes" on V100/RTX 4070 Ti Super (no A100 GB/s
reported).

**Applicability to OnPair.** Short/mid: ✓ — fused decode-into-compute
is the right pattern when the downstream is a GPU scan operator that
doesn't need a materialized byte-packed buffer at all. **Caveat:**
OnPair currently materializes; switching to a fused path is a
cross-API change, not a kernel tweak.

**LOC.** Fused path: 100-200 LOC + downstream API changes. Build-time
bit-width specialization analogue (here: stride specialization):
~50 LOC for the dispatcher.

---

## 4. Tile-Based Lightweight Integer Compression (Shanbhag SIGMOD 2022)

[paper](https://dl.acm.org/doi/10.1145/3514221.3526132)

**AOT statistics exploited.** Per-tile metadata (min, bit-width) is
stored alongside the tile so the kernel reads it once per tile and
specializes the inner decode loop. DICT, FOR, RFOR, DFOR are all
selected per tile based on data characteristics. **Closest published
analogue to per-column profile-guided variant selection** — but
selection is per tile, not per column, and the criterion is
encoding-specific (e.g. run length for RFOR) rather than dict
statistics.

**Speedup.** 2.2× decode, 2.6× end-to-end query vs nvCOMP. No
per-column dict-stat ablation.

**Applicability to OnPair.** Yes — confirms the design point of
embedding statistics with the data and dispatching at tile granularity
is workable. A per-chunk (or per-column) variant pick for OnPair is a
direct port.

**LOC.** ~80 LOC for a runtime dispatcher across 3-4 kernel variants.

---

## 5. BtrBlocks (Kuschewski SIGMOD 2023)

[paper](https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/btrblocks.pdf)

**AOT statistics exploited.** A greedy sample-then-encode algorithm
picks the best encoding **per column chunk** from a fixed catalogue,
recursing on outputs. Identifies "inefficiencies in format designs
when handling [...] GPUs for decoding" but does not propose
GPU-specific variants. **The kernel-selection-per-column-chunk
pattern is exactly the shape we want**, just on a CPU codebase.

**Speedup.** 2.2× scan, 1.8× cheaper vs Parquet (CPU-side). No GPU
numbers.

**Applicability to OnPair.** Pattern: pick a kernel per column based
on sampled statistics. Direct port: at column-open time, classify the
dict into `{tiny ≤4B, short ≤8B, mid ≤16B, mixed}` and launch the
matching kernel.

**LOC.** ~40 LOC for the host-side classifier + dispatcher.

---

## 6. nvCOMP / cuDF / DietGPU — what they do *not* do

[nvCOMP docs](https://docs.nvidia.com/cuda/nvcomp/) ·
[cuDF blog](https://developer.nvidia.com/blog/encoding-and-compression-guide-for-parquet-string-data-using-rapids/) ·
[DietGPU](https://github.com/facebookresearch/dietgpu)

**nvCOMP.** Closed since v2.3. No public evidence of per-column
dict-stat specialization for string decode. Blackwell adds a
fixed-function DE (600 GB/s for Snappy/LZ4/Deflate) which sidesteps
the question on that arch.

**cuDF / libcudf.** Parquet dict decode path uses a generic
variable-length string decoder. Heuristic: dictionary preferred under
~100 K distinct values; ORC limits dict encoding to row counts fitting
`uint16_t`. These are **encoder-side** policies — the decoder is one
kernel.

**DietGPU.** Operates on fixed-width FP/int — no variable-length
output, no dict, so no specialization opportunity exists.

**Net.** No public production GPU library specializes a string-decode
kernel by dict-length statistics. The opportunity is real.

---

## 7. Synthesis: candidate specializations for OnPair

Indicative numbers — none of these have been measured on OnPair; the
estimates are scaled from published ratios on adjacent workloads and
from the bottleneck breakdown in `PERF_RESEARCH.md` §2.

| # | Technique | AOT stat | Est. speedup vs current `onpair_shmem` real-data 100 GiB/s baseline | Short | Mid | Long | LOC |
|---|---|---|---|---|---|---|---|
| S1 | **Stride-by-max_len dispatch.** Compile-time template `STRIDE ∈ {4, 8, 16}` for the `dict_padded` load + per-token store. Eliminates 12 of 16 unrolled per-byte conditional stores when `max_len ≤ 4`. | `max_len` | 1.5-2.5× on ≤8 B columns; 1.0× on full-16B columns | ✓✓ | ✓ | — | 30-50 |
| S2 | **Shared-memory dict cache.** Stage dict into shared mem when `dict_size_bytes ≤ 48 KiB` (well below A100's 164 KiB unified budget). Random L1 sector reads at 10.7 sectors/req → bank-conflict-free shared reads. | `dict_size_bytes` | 1.4-2.0× when `dict_size_bytes ≤ ~32 KiB`; 1.0× otherwise. Sized to fit ~2 K × 16 B entries. | ✓ | ✓ | ✓ | 60-90 |
| S3 | **Per-token vs per-byte work split.** Below `avg_len ≈ 6`, the 32-lane warp produces ~192 output bytes — wide enough for one warp-cooperative store but narrower than the LSU-issue cliff. Above `avg_len ≈ 10`, switch to **2 lanes per token** (lane-pair cooperates on the dict load + store) to halve per-token overhead. | `avg_len` | 1.2-1.6× on `avg_len ≥ 10` columns; neutral on shorter | — | ✓ | ✓✓ | 80-120 (new variant) |
| S4 | **Length-distribution dispatch.** Three pre-built kernel templates: `MAXLEN_LE4`, `MAXLEN_LE8`, `MAXLEN_LE16`. Host inspects dict and picks. Eliminates the 16-deep ladder for ~70% of TPC-H string columns (`l_returnflag`, `l_linestatus`, status codes, country codes — all ≤4 B). | full length histogram | 2-3× on ≤4 B columns specifically | ✓✓ | — | — | 40-60 (templates) + S1 |
| S5 | **Profile-guided kernel selection.** Host-side classifier maps `(max_len, avg_len, dict_size_bytes, p99_len)` to one of `{S1, S2, S3, S4}`. Pre-computed at column-open time, free at decode time. Matches BtrBlocks' per-chunk encoder pattern, applied to decode. | all of the above | Multiplies whichever underlying variant wins | ✓ | ✓ | ✓ | 40-60 (dispatcher) |
| S6 | **Sub-warp grouping.** When `max_len ≤ 4`, pack 4 tokens into one 16-B output stride; the 32-lane warp emits 8 tokens per aligned store. Removes the `__syncwarp` + per-warp prefix sum entirely for the common case. | `max_len ≤ 4` | 2-3× on tiny-token columns. Closest published analogue: GSST's "split parallelism" with split-width=2 ([GSST §4](https://dl.acm.org/doi/10.1145/3719330.3721228)). | ✓✓ | — | — | 100-150 (new kernel) |
| S7 | **Constant-memory dict.** When `dict_size_bytes ≤ 64 KiB` (CC 8.0 constant cache), use `__constant__` storage for the dict. Constant cache is broadcast-optimized — wins when many tokens share a code. | `dict_size_bytes ≤ 64 KiB` + low entropy | 1.1-1.4× when token-code locality is high; 0.9× under uniform codes (constant cache thrashes). | ✓ | ✓ | ✓ | 50-80 |

---

## 8. Honest caveats

1. **None of the seven numbers above are measured.** They are scaled
   from: GSST's 4-7× over per-byte stores ([§2 PERF_RESEARCH.md](./PERF_RESEARCH.md)),
   FastLanes' 3-4× over generic GPU bit-unpacking
   ([Afroozeh DaMoN 2024](https://dl.acm.org/doi/10.1145/3662010.3663450)),
   and Tile-Based's 2.2× decode over nvCOMP
   ([Shanbhag SIGMOD 2022](https://dl.acm.org/doi/10.1145/3514221.3526132)).
   Treat each as "order-of-magnitude believable" not "predicted to
   1 digit".

2. **No published paper reports A100 GB/s for any of S1-S7 specifically.**
   The closest is FastLanesGPU's bit-width-specialized kernel family
   (build-time per-bit-width templates), which is the same shape as
   S1/S4 but for integer bit-widths, not string lengths.

3. **The 64-220 GiB/s real-data range likely has multiple causes** —
   dict cache misses (S2), avg_len overhead (S3, S6), and the 16-deep
   ladder (S1, S4) plausibly all contribute. The right first
   experiment is: pick three ClickBench columns at 64, 120, and
   220 GiB/s, profile each with Nsight Compute, and let the metric
   that dominates pick which of S1-S6 to build first.

4. **S5 (profile-guided dispatch) only pays if at least two of S1-S4
   land individually.** Don't build the dispatcher first.

5. **Cross-paper comparability is poor.** GSST's 191 GB/s is symbol-table
   FSST (not dict-coded short strings); OnPair's 549 GB/s synthetic is
   not directly on the same axis. The real comparator for S1-S7 is
   OnPair-vs-OnPair on the same columns.

## Sources

- [GSST: Vonk et al., SIGOPS OSR 2025](https://dl.acm.org/doi/10.1145/3759441.3759450) /
  [TU Delft preprint](https://repository.tudelft.nl/file/File_627b50ef-4c9a-4367-bd9c-b640c978edff?preview=1) /
  [thesis](https://repository.tudelft.nl/record/uuid:71c1dddf-3b7d-4c12-a079-d716fad501b2)
- [GPU-FSST: Anema et al., ADMS 2025](https://www.vldb.org/2025/Workshops/VLDB-Workshops-2025/ADMS/ADMS25-01.pdf) /
  [repo](https://github.com/timanema/fsst-gpu)
- [FastLanesGPU: Afroozeh et al., DaMoN 2024](https://dl.acm.org/doi/10.1145/3662010.3663450)
- [G-ALP: Hepkema, DaMoN 2025](https://ir.cwi.nl/pub/35205/35205.pdf)
- [Tile-Based: Shanbhag et al., SIGMOD 2022](https://dl.acm.org/doi/10.1145/3514221.3526132)
- [BtrBlocks: Kuschewski et al., SIGMOD 2023](https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/btrblocks.pdf)
- [FSST: Boncz et al., VLDB 2020](https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf)
- [nvCOMP docs](https://docs.nvidia.com/cuda/nvcomp/) /
  [Blackwell DE FAQ](https://docs.nvidia.com/cuda/nvcomp/decompression_engine_faq.html)
- [cuDF Parquet string encoding guide](https://developer.nvidia.com/blog/encoding-and-compression-guide-for-parquet-string-data-using-rapids/)
- [DietGPU README](https://github.com/facebookresearch/dietgpu/blob/main/README.md)
- [CUDA Pro Tip: Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)
