# GPU Decompression State-of-the-Art on NVIDIA A100

A reference for where published GPU decompression throughput sits on the **NVIDIA A100** (CC 8.0, HBM2 peak ~1.555 TB/s ≈ 1.45 TiB/s). All numbers are decompression-side unless otherwise noted; "% peak" is throughput / 1.555 TB/s. Most published figures measure *compressed-bytes-in / decompressed-bytes-out* with input already resident in HBM — cross-paper comparison requires care.

---

## 1. Strings / variable-length byte sequences

LZ-family byte-stream decompressors have control flow poorly suited to SIMT, so the A100 ceiling has historically been well below HBM peak. The GPU-native FSST variant (GSST) recently blew past it.

| Algorithm | Throughput (A100) | Time @ workload | Source |
|---|---|---|---|
| **GSST** (GPU-native FSST variant) | **191 GB/s**, ~2.7-4x compression ratio | not stated; A100 80 GB | [Vonk et al., SIGOPS OSR 2025](https://dl.acm.org/doi/10.1145/3759441.3759450) / [TU Delft preprint](https://repository.tudelft.nl/file/File_627b50ef-4c9a-4367-bd9c-b640c978edff?preview=1) |
| **GPU-FSST** (Anema) | Decode 0.353 GB/s in example run (implementation gap); paper headline is compression at **74 GB/s on RTX 4090** | lineitem-1gb (~990 MB) | [Anema, ADMS 2025 (PDF)](https://www.vldb.org/2025/Workshops/VLDB-Workshops-2025/ADMS/ADMS25-01.pdf), [repo](https://github.com/timanema/fsst-gpu) |
| **FSST** CPU baseline (single core) | ~1-1.5 GB/s/core; comparable to LZ4 | n/a | [Boncz et al., VLDB 2020](https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf) |
| **nvCOMP LZ4 / Snappy / GDeflate** on string data | dominated by GSST in head-to-head Silesia comparison; see §5 for raw LZ4 A100 | — | [Vonk et al., 2025](https://dl.acm.org/doi/10.1145/3759441.3759450) |

Dictionary-coded short-string formats are not separately benchmarked on A100 in the literature. Closest reference points are GSST (symbol-table decode at 191 GB/s) and §2 dict-coded integers. **GSST is the only published A100 string-decode above 100 GB/s.**

---

## 2. Fixed-width integers, FastLanes, FoR, RLE

Lightweight schemes (bit-packing, FoR, RLE, dict) — GPU bottleneck is HBM / shared-memory bandwidth, not control flow.

| Algorithm | Throughput | Time @ workload | Source |
|---|---|---|---|
| **FastLanes bit-unpack on GPU** | **3-4x** faster than the prior Tile-Based GPU bit-unpacker; end-to-end SSB queries up to **2x** faster vs uncompressed. T4 / V100 microbench; **no A100 number**. | not stated | [Afroozeh, DaMoN 2024](https://dl.acm.org/doi/10.1145/3662010.3663450), [PDF](https://ir.cwi.nl/pub/34260/34260.pdf), [repo](https://github.com/cwida/FastLanesGPU) |
| **Tile-Based** lightweight integer | Parity/improvement vs CUB BlockLoad on V100; **no A100 number**. | not stated | [Shanbhag et al., SIGMOD 2022](https://dl.acm.org/doi/10.1145/3514221.3526132) |
| **CODAG RLE v1 / RLE v2 / Deflate** (A100) | **38.07 / 26.87 / 51.96 GB/s** geo-mean (RAPIDS baseline: 2.83 / 4.72 / 44.18) | 7 datasets (Mortgage, NYC Taxi, Criteo, Twitter, HRG) | [Maleki et al., arXiv 2307.03760](https://arxiv.org/html/2307.03760) |
| **nvCOMP LZ4** on Mortgage 2009Q2 col 0 (A100 80 GB) | **312-320 GB/s** (high-level / chunked) | column-aligned integer-like data | [nvCOMP Benchmarks.md](https://github.com/NVIDIA/nvcomp/blob/main/doc/Benchmarks.md) |
| **nvCOMP Cascaded** (RLE+Δ+bit-pack) | "up to 500 GB/s" on numerical analytical workloads (marketing; comp+decomp bracketed) | not stated | [NVIDIA blog](https://developer.nvidia.com/blog/optimizing-data-transfer-using-lossless-compression-with-nvcomp/) |
| **nvCOMP Bitcomp** (proprietary) | Release notes: 7-8x speedup for small files on A100 post-3.0; no raw A100 GB/s in NVIDIA public tables. | not stated | [release notes](https://docs.nvidia.com/cuda/nvcomp/release_notes.html), [native API](https://docs.nvidia.com/cuda/nvcomp/native_api.html) |

> Literature is thin on direct A100 bit-unpack / FoR / dict-int GB/s; FastLanesGPU and Tile-Based both target V100/T4. Vortex has A100-class kernels (`benches/bitpacked_cuda.rs`, `benches/for_cuda.rs`, `benches/dict_cuda.rs`, `benches/runend_cuda.rs`) whose results are not yet published.

---

## 3. Floats — ALP, pco, ndzip, etc.

All numbers below decode to packed `f32` / `f64`.

| Algorithm | Throughput | Time @ workload | Source |
|---|---|---|---|
| **G-ALP** (GPU port of ALP) | "highest decode throughput of all schemes" on V100 / RTX 4070 Ti Super; beats nvCOMP and ndzip in decode + filter queries. **No A100 number in the paper.** | not stated | [Hepkema, G-ALP 2024](https://ir.cwi.nl/pub/35205/35205.pdf) |
| **DietGPU FP codec** | **250-600 GB/s** decode on A100 (range over data sizes / entropy) | "reasonable data sizes" | [DietGPU README](https://github.com/facebookresearch/dietgpu/blob/main/README.md) |
| **ndzip-gpu** | Highest single-precision throughput on Volta/Ampere among lossless FP compressors; no explicit A100 GB/s in the open abstract. | scientific FP | [Knorr et al., SC'21](https://dl.acm.org/doi/10.1145/3458817.3476224) |
| **Falcon** (GPU adaptive FP, 2025) | **12.32 GB/s** avg decode over 12 datasets; **2.4x** vs prior best. Tested on RTX 5080, not A100. | NYC Taxi, gas/temp sensors, etc. | [Li et al., arXiv 2511.04140](https://arxiv.org/abs/2511.04140) |
| **GFC** (legacy double-FP, 2011) | ~90 Gb/s ≈ **~11 GB/s** decompression, Fermi-era | pre-A100 | [O'Neil & Burtscher, GPGPU 2011](https://userweb.cs.txstate.edu/~mb92/papers/gpgpu11.pdf) |
| **pco / pcodec** | CPU only; >1 GiB/s/thread decode. **No GPU port** as of 2026-05. | per-thread | [Loncaric, arXiv 2502.06112](https://arxiv.org/abs/2502.06112) |

DietGPU's 250-600 GB/s on A100 is the strongest published FP number; G-ALP claims "best of class" on smaller GPUs but reports no A100 figure.

---

## 4. Bools / bytebool

Trivially memory-bound — a single warp-cooperative store with no arithmetic decode. Should sit at HBM peak.

| Algorithm | Throughput | Time @ workload | Source |
|---|---|---|---|
| Raw `u8` / packed-bit decode | ~HBM peak (~1.4 TB/s effective on A100) | n/a | No paper specifically benchmarks bool decode in isolation. nvCOMP's "Snappy/LZ4 up to 100 GB/s" gives a lower bound for *compressed* boolean streams. [NVIDIA Blog](https://developer.nvidia.com/blog/optimizing-data-transfer-using-lossless-compression-with-nvcomp/) |

**Literature is thin.** No published A100 number for bit-packed bool decode in isolation. HBM peak is the realistic ceiling; any LZ4/Snappy bool stream is bottlenecked by the LZ decoder, not the bool layout.

---

## 5. General-purpose compression on A100

NVIDIA's public `nvCOMP/doc/Benchmarks.md` includes worked examples only for LZ4; other per-algorithm A100 GB/s come from third-party benchmarks and NVIDIA blog posts.

| Algorithm | Throughput (A100) | Time @ workload | Source |
|---|---|---|---|
| **nvCOMP LZ4** | **312.81 GB/s** (high-level), **320.70 GB/s** (chunked) | Mortgage 2009Q2 col 0 | [nvCOMP Benchmarks.md](https://github.com/NVIDIA/nvcomp/blob/main/doc/Benchmarks.md) |
| **nvCOMP LZ4** (Silesia) | Order-of-magnitude lower than column data: ~5 GB/s in older measurements. nvCOMP 3.0 improved A100 LZ4 decode by **1.4x**. | Silesia.tar 212 MB | [release notes](https://docs.nvidia.com/cuda/nvcomp/release_notes.html) |
| **nvCOMP Snappy** | "up to 100 GB/s" decode on suitable data; **1.9x** A100 speedup in 3.0 | Silesia / arbitrary byte | [NVIDIA blog](https://developer.nvidia.com/blog/optimizing-data-transfer-using-lossless-compression-with-nvcomp/), [release notes](https://docs.nvidia.com/cuda/nvcomp/release_notes.html) |
| **nvCOMP GDeflate** | **152.88 GB/s** on Mortgage 2000Q4 col 12 (RTX A6000, not A100); A100 GDeflate decode improved **2x** in 2.1 | Mortgage column | [nvCOMP Benchmarks.md](https://github.com/NVIDIA/nvcomp/blob/main/doc/Benchmarks.md), [release notes](https://docs.nvidia.com/cuda/nvcomp/release_notes.html) |
| **nvCOMP ZSTD** | **1.5x** A100 speedup in 3.0; no clean public A100 GB/s; slowest GPU codec; NVIDIA cites 15x vs CPU ZSTD | n/a | [release notes](https://docs.nvidia.com/cuda/nvcomp/release_notes.html), [Voltron](https://voltrondata.com/blog/data-analytics-are-faster-on-gpus) |
| **nvCOMP Deflate** | "up to 1.5x" A100 decode improvement in 2.4; no specific Silesia GB/s in NVIDIA tables | n/a | [release notes](https://docs.nvidia.com/cuda/nvcomp/release_notes.html) |
| **nvCOMP Bitcomp** | No published Silesia/Mortgage A100 GB/s; release notes cite 7-8x speedup for small files; community estimates ~300 GB/s on column data | not stated | [release notes](https://docs.nvidia.com/cuda/nvcomp/release_notes.html) |
| **DietGPU ANS** | **250-410 GB/s** decompression on A100 | "reasonable data sizes" | [DietGPU README](https://github.com/facebookresearch/dietgpu/blob/main/README.md) |
| **nvCOMP gANS** | not separately benchmarked in public docs | n/a | [nvCOMP overview](https://github.com/NVIDIA/nvcomp/blob/main/doc/algorithms_overview.md) |
| **nvCOMP Cascaded** | "up to 500 GB/s" (marketing; comp+decomp bracketed) | numerical analytical | [NVIDIA blog](https://developer.nvidia.com/blog/optimizing-data-transfer-using-lossless-compression-with-nvcomp/) |
| **CODAG Deflate** (A100) | 51.96 GB/s geo-mean (1.18x vs RAPIDS) | 7 mixed datasets | [arXiv 2307.03760](https://arxiv.org/html/2307.03760) |

---

## 6. Where 511 GiB/s on OnPair sits

**511 GiB/s ≈ 549 GB/s** of decoded bytes. With 51 M tokens at ~11 B average, that is **~4.65 × 10^10 tokens/s ≈ 46.5 G tokens/s** decoded. The output is byte-packed (no per-token offset writes amortised). Against HBM2 peak of 1.555 TB/s, this is **~35 %** of HBM peak (or ~36 % of TiB-effective). For comparison:

| Reference | Decoded GB/s | % HBM peak | Notes |
|---|---|---|---|
| **OnPair (this work)** | **549 GB/s** (511 GiB/s) | ~35 % | dict-coded short strings, byte-packed output, A100 |
| GSST (Vonk 2025) | 191 GB/s | ~12 % | full FSST-style symbol-table decode, longer strings |
| DietGPU FP codec (FB) | 250-600 GB/s | ~16-39 % | FP-specific, lossless |
| DietGPU ANS | 250-410 GB/s | ~16-26 % | generic ANS entropy |
| nvCOMP LZ4 on Mortgage col | 312-320 GB/s | ~20-21 % | column-aligned integer-like data |
| nvCOMP GDeflate (A6000 ref) | ~153 GB/s | ~10 % (A100-equivalent) | high compression ratio |

OnPair at 549 GB/s sits **above every published GPU string-decompressor on A100**: ~2.9x GSST (the only directly comparable string-decode on the same hardware), and ~1.7x nvCOMP LZ4 on column data (the strongest LZ-byte number NVIDIA publishes). It is in the same band as DietGPU's FP codec — a simpler workload (no symbol table, no variable-length output) — and the upper estimate for Bitcomp. Only the top of DietGPU's FP range (600 GB/s) exceeds it.

For dict-coded short-string decode the bottleneck shifts from *control flow* (the GSST regime) to *output bandwidth* (the OnPair regime). 35 % of HBM peak on byte-packed variable-length output is strong; remaining headroom is likely in offset/length materialization and write coalescing for tokens straddling 16-byte boundaries.

**Bottom line: 511 GiB/s is state-of-the-art for GPU dictionary/short-string decompression on A100, and competitive with the fastest published GPU decompressors of any kind on this hardware.**

---

All sources are inline-linked above.
