# OnPair GPU decode — handover (GH200 → B200)

Handover for continuing the OnPair string-decompression GPU work on a **B200
(Blackwell, sm_100)** machine. All prior measurements were on a single **GH200
(Hopper, sm_90)**. Read this first, then `ONPAIR_GPU_DECISION_TREE.md` (full
optimization tree + verdicts) and `ONPAIR_GPU_FINDINGS.md` (raw NCU). Dataset
result tables: `benchmarks/onpair-bench/GPU_DECODE_SUMMARY.md`.

---

## 1. What this work is

OnPair is a dictionary string codec (FSST-like; tokens are dict codes → 1–16 B
strings). We optimize the **GPU decode kernel**: per token, gather a dict entry
and scatter it to output. Decode is a massively parallel
gather+scatter. Goal: maximize decode throughput (GiB/s), validated byte-exact.

**The bottleneck (GH200):** decode is **L1/TEX cache-request-throughput bound**
on the random dict gather (NCU: L1/TEX 86–93%, DRAM idle ~17%, SM register-capped
at 64 regs → 50% theoretical / ~45% achieved occupancy, "No-Eligible" 65% because
warps stall on the saturated L1/TEX pipe). The dict is L2-resident (~94% hit).
**Every win reduces L1/TEX work; nothing else helps.**

---

## 2. Code map (all synced via mutagen — `kernels/src` is not ignored)

Kernels — `vortex-cuda/kernels/src/`:
- `onpair_shmem_4tpt.cu` — **baseline & default**. 4 tokens/thread, 128 tokens/warp,
  warp prefix-scan via `__shfl`, shared staging buffer, aligned `uint4` body drain
  with `__stcs` (streaming) stores. The kernel everything is measured against.
- `onpair_shmem_4tpt_split8read.cu` — **WIN on bits12** (+4–11%). Reads 8 B (`uint2`)
  from the 32 KB `dict_s8` for the common case; `dict_padded` only for `len>8`.
- `onpair_shmem_4tpt_split4read.cu` — ❌ regressed (32 B sector floor). Kept as evidence.
- `onpair_shmem_4tpt_pdict.cu` / `_vdict.cu` — ❌ dict-in-shared (bank conflicts). Evidence.
- `onpair_shmem_4tpt_lenbucket.cu` — ❌ variable-stride dict (branch divergence). Evidence.
- `onpair_shmem_4tpt_regcache.cu` — ❌ 32-entry `__shfl` hot-code cache. Evidence.
- `onpair_shmem_4tpt_ldcs.cu` — ⟷ neutral; `__ldcs` streaming-load on codes. Evidence.
- (older: `onpair.cu` reference, `_2tpt`, `_s8*`, `_s4l1*`, `_const*`, `_tma`, `_wpb8*`, `_occ*`.)

Bench/host — `vortex-bench/src/onpair_bench.rs`:
- `GPU_KERNELS` registry (KernelVariant list) + `KernelLayout` enum + `launch_variant`.
- `pick_auto_kernel` — **the productized result**: picks `split8read` when dict
  ≤ 4096 entries (bits12) AND token-weighted `frac_le8 ≥ 0.90`; else `4tpt`
  (was a stale `2tpt`).
- `stage_gpu_chunk` — builds the per-chunk GPU buffers; computes `frac_le8`;
  env-gated experiments live here:
  - `ONPAIR_DICT_REORDER=freq|lenbucket` — decode-side dict relabel (byte-exact).
  - `ONPAIR_LEN_HIST=1` — prints token-weighted length histogram to stderr.
  - `ONPAIR_L2_PERSIST=1` — pins dict in L2 via access-policy window (`apply_l2_persist`).
- Bench binary: `vortex-bench/src/bin/onpair-chunk-bench.rs`.

Benchmark orchestration — `benchmarks/onpair-bench/`:
- `columns.py` — dataset registry (incl. `wikipedia` = wikimedia/wikipedia en;
  `fineweb`; `book-reviews`; tpch/tpcds/clickbench/dbtext).
- `run.py` — drives the `bits × chunk × threshold` matrix; default thresholds now
  `[0.2, 0.5]`.

---

## 3. Build & run

```bash
# Build the bench (CUDA). On B200 ensure the CUDA toolkit ≥ 12.8 (Blackwell) and
# that nvcc targets sm_100 — see §6.
cargo build -p vortex-bench --features cuda --bin onpair-chunk-bench --release

# Decode an existing .vortex file/dir (times every registered kernel; validates):
target/release/onpair-chunk-bench gpu-decode-vortex \
  --vortex <file_or_dir> --column <name> --gpu-iters 100 --gpu-validate

# Generate compressed .vortex from a parquet (compression + optional GPU decode):
target/release/onpair-chunk-bench run \
  --parquet <p> --column <c> --dataset-id <d> \
  --bits 12,16 --chunk-bytes 1048576000 --threshold 0.2 \
  --sample-bytes 1000000000 --out-dir vortex-bench/data/onpair-bench

# Or the whole matrix for the registry:
python benchmarks/onpair-bench/run.py            # full
python benchmarks/onpair-bench/run.py --bits 12 --chunk-mb 100,1000   # subset
```

Verification (per repo `CLAUDE.md`): `cargo +nightly fmt --all`,
`cargo clippy --all-targets --all-features`; `./scripts/public-api.sh` only if
public APIs changed. The kernel `.cu` edits and `onpair_bench.rs` are the only
Rust/CUDA surface touched.

---

## 4. ⚠️ Data is NOT synced — regenerate on B200

Mutagen ignores `vortex-bench/data/**`, so the `.vortex` files, downloaded
parquets, and the bits16 dicts are **not on the B200**. Regenerate before
benchmarking:

| dataset | source on B200 | note |
|---|---|---|
| wikipedia | auto-downloads from HF URL in `columns.py` | en shard, ~420 MB |
| fineweb | HF (FINEWEB_URL) or copy `sample_10BT_000_00000.parquet` | |
| **book-reviews** | **no public URL — copy `book_reviews.parquet` manually** | else skipped |
| tpch (ps_comment, l_comment, …) | `onpair-chunk-bench gen-tpch --sf 10` | generated locally |

The columns used most: `fineweb/text`, `wikipedia/text`, `book-reviews/text`
(bits12), `tpch-sf10/ps_comment`, `l_comment`. Generate at least
`chunk1000mb` bits12+bits16 to reproduce the headline tables.

---

## 5. Current results (GH200) — what to expect / reproduce

Decode (chunk1000mb, 100 iters), auto-selected kernel:

| dataset/bits | GiB/s | kernel | vs 4tpt |
|---|---:|---|---:|
| fineweb/b12 | 567 | split8read | +11% |
| wikipedia/b12 | 538 | split8read | +9% |
| book-reviews/b12 | 607 | split8read | +4% |
| ps_comment/b12 | 1117 | 4tpt | — |
| fineweb/b16 | 470 | 4tpt | (s8r −8%) |
| wikipedia/b16 | 538 | 4tpt | — |
| ps_comment/b16 | 866 | 4tpt | (s8r −26%) |

**Two real wins:**
1. `split8read` — auto-selected for bits12 short-token text (fineweb/wiki/book-reviews).
   Mechanism: 8 B not 16 B through the L1/TEX pipe (util 86→73%).
2. **freq-ordered codes** (decode-side validated, *encoder-side to ship*): bits16
   +8–13% (L1 sector hit 35→45%). NOT applied — it needs the OnPair encoder to
   assign low codes to frequent tokens. Top format suggestion.

Everything else tried (dict-in-shared, lenbucket, regcache, L2-persistence,
cp.async, split4, `__ldcs`) **moved work without shrinking it** → neutral/regress.
Full table + reasons in `ONPAIR_GPU_DECISION_TREE.md`.

Selection rule lives in `pick_auto_kernel`: split8read iff (dict ≤4096 entries)
AND (token-weighted `frac_le8 ≥ 0.90`). Crossover validated across 10 columns
(≤8 B fraction: ≥94% → split8read wins; ≤80% → 4tpt).

Compression: bits12 dicts saturate at 4096 entries (~17 KB) within ~10 MB text →
ratio flat across chunk size; bits16 dicts grow as chunks shrink (wiki/b16 dict
456 KB→26 MB, ratio 2.815→2.345 from 1000→10 MB) → prefer large chunks for bits16.

**Threshold: use `0.2` only — do NOT try `0.5`.** OnPair's `threshold` is a
*dynamic frequency* cutoff (smaller ⇒ larger dict), not a string-sampling
fraction. `0.5` was evaluated and dropped; `run.py` defaults to `[0.2]`. All
result tables and the auto-selector are for threshold 0.2. Do not regenerate or
benchmark 0.5 on the B200.

---

## 6. ⭐ B200 (Blackwell) — what to redo and watch

**The GH200 conclusions may not hold on B200 — re-profile first.** Blackwell
changes the exact things our analysis hinged on:

1. **Build for sm_100.** Needs CUDA ≥ 12.8. Check `vortex-cuda/build.rs` / nvcc
   arch flags target Blackwell (or rely on PTX JIT, but compile cubin for sm_100
   for real numbers). Confirm kernels load (the old `8tpt` once failed with
   `CUDA_ERROR_INVALID_PTX`; watch for arch issues).
2. **Re-establish measurement.** GH200 clocks were unlocked (locking blocked by
   policy) → we used **high-iter intra-run ranking** + NCU (clock-robust). On B200,
   lock clocks if permitted; otherwise same protocol. **Always extract kernel
   timings by explicit name, not column position** (a positional bug here once
   mis-attributed split8read's regression to `__ldcs`).
3. **The bottleneck may shift.** B200 has a much larger L2 (~126 MB vs ~50 MB),
   more SMs, bigger/faster HBM3e, and (likely) larger L1. Consequences to re-test:
   - **split8read win may change**: it depends on `dict_s8` (32 KB bits12) fitting
     L1. If B200's L1 is larger, the bits16 `dict_s8` (512 KB) might fit better →
     **split8read could start winning on bits16 too** (it lost there on GH200).
     **Re-run the 10-column analysis sweep and re-fit the `frac_le8`/dict-size
     thresholds in `pick_auto_kernel`.**
   - **freq-sort benefit may shrink**: it fixed bits16 L1 thrashing (35% hit). With
     a bigger L1, less thrashing → smaller gain. Re-measure with `ONPAIR_DICT_REORDER=freq`.
   - **L2-persistence was a no-op on GH200** (dict already L2-resident). Likely
     still a no-op (even bigger L2), but cheap to re-check (`ONPAIR_L2_PERSIST=1`).
   - **Occupancy**: register cap (64 → 50%) is arch-dependent; Blackwell reg file
     per SM may differ. Re-check whether 3 blocks/SM is reachable.
4. **Blackwell-specific opportunities not explored** (could finally beat the wall):
   - 5th-gen TMA / `tcgen05`, larger async-copy, distributed shared memory across
     the cluster (thread-block clusters) — a *cluster-shared dict* could make the
     dict-in-shared idea viable where it failed on Hopper (the GH200 failure was
     per-block bank conflicts; cluster DSMEM changes that calculus).
   - Re-evaluate cp.async/TMA dict prefetch with Blackwell's bigger async pipelines.
5. **Re-profile NCU** SpeedOfLight + MemoryWorkloadAnalysis + SchedulerStats on
   `4tpt` (bits12 and bits16) to find the new limiter before optimizing. The whole
   GH200 story (L1/TEX-request bound, DRAM idle) must be re-confirmed.

---

## 7. Open threads / suggested next steps

- **Done:** `pick_auto_kernel` selects split8read for bits12 short-token columns;
  `run.py` sweeps thresholds 0.2 & 0.5; wikipedia added to the registry.
- **Encoder-side (format) suggestions — not applied** (see DECISION_TREE §"Format"):
  1. **Frequency-ordered code assignment** (highest value; +8–13% bits16, free at encode).
  2. Tunable max symbol length (≤8 B, FSST-style) → split8read needs no fallback,
     halves bits16 dict.
  3. Per-chunk length-class hint (1–2 B) → exact kernel selection without a scan.
  4. Prefer bits12 when ratio permits (decodes much faster).
- **On B200:** re-run §6 items 2–5; re-fit the auto-selector thresholds; explore
  cluster-DSMEM shared dict.

---

## 8. Gotchas (carried from GH200)

- **`gpu-decode-vortex` always runs the slow reference `onpair` kernel + nvCOMP
  recompression** (no flag to skip) → each invocation is slow; many-chunk files
  (10 MB chunks → ~96 chunks) are *very* slow because the reference kernel runs
  per chunk. Prefer chunk1000mb for quick kernel comparisons.
- **nvCOMP hardware ZSTD backend failed on GH200** (`get_decompress_temp_size:
  invalid value`); default backend works but is far slower than OnPair. Re-test on B200.
- **git is broken in this worktree** — its `.git` points at a stale macOS path
  (`/Users/joeisaacs/...`). Use `mv` not `git mv`; run `git add` on a host with a
  working checkout. Mutagen syncs filesystem state regardless.
- **Don't use `pgrep -f "cargo build"` in a wait loop** — it self-matches its own
  command line and deadlocks. (Bit me repeatedly.)
- All `.vortex`/parquet/PTX/.so/.ncu-rep are mutagen-ignored — regenerate data,
  let `build.rs` regenerate PTX from the synced `.cu`.

---

## 9. Doc index

- `ONPAIR_GPU_HANDOVER.md` — this file.
- `ONPAIR_GPU_DECISION_TREE.md` — every optimization axis, verdict, NCU reason,
  format suggestions, kernel-selection policy.
- `ONPAIR_GPU_FINDINGS.md` — chronological NCU findings + raw numbers.
- `benchmarks/onpair-bench/GPU_DECODE_SUMMARY.md` — per-dataset compression + decode tables.
