# Session handoff — B200 OnPair decode + nvCOMP (2026-05-21)

Continuation of the OnPair GPU-decode work on the **B200 (Blackwell, sm_100)** box.
Read this, then the supporting docs it points to. All numbers **PRELIMINARY**: unlocked
clocks (±~5%), single-invocation kernel ranking, NCU + clock-locking **blocked** in this
container. Everything claimed below validated **byte-exact** unless noted.

---

## 0. TL;DR — what got done this session

1. **nvCOMP hardware-engine comparison** (new): Zstd-HW is **unsupported** by NVIDIA's DE;
   Deflate/LZ4/Snappy work. Built a standalone benchmark, characterized Deflate/LZ4 on the
   Blackwell DE with proper presets. OnPair beats them on decode speed; Deflate-hi sometimes
   beats OnPair on ratio.
2. **OnPair decode kernel optimization** (main): made `pick_auto_kernel` **arch-aware**
   (GH200 keeps its kernels, B200 gets new ones) → **+6–12% on most B200 columns, +46% on
   bits12 text**. New winner kernels `b128o12` and `split8read_b128o12`.
3. **`ONPAIR_FAST=1`** infra flag → ~50× faster kernel sweeps.
4. Explained the **dbtext "slowness"** (it's launch-bound tiny columns, not a decode bug).

---

## 1. Environment (unchanged from prior handover; see memory)

- B200, sm_100, 183 GB, CUDA 12.8, nvcomp 5.1.0.21. Memory files (loaded each session):
  `b200-onpair-env-setup` (PATH + `LIBCLANG_PATH=/usr/lib/llvm-18/lib` needed for every
  build), `b200-ncu-perfcounters-blocked`, `nvcomp-hw-zstd-unsupported`, `b200-onpair-kernel-wins`.
- **Every build:** `export PATH="$HOME/.cargo/bin:$PATH"; export LIBCLANG_PATH=/usr/lib/llvm-18/lib`
  then `cargo build -p vortex-bench --features cuda --bin onpair-chunk-bench --release`
  (~5 min). `build.rs` auto-compiles every `.cu` in `kernels/src/` to PTX.
- nightly toolchain now installed (`cargo +nightly fmt --all` works).
- **BLOCKED:** NCU (`ERR_NVGPUCTRPERM`, no `CAP_SYS_ADMIN`) and `nvidia-smi -lgc` clock
  locking (no permission). Both need a host-side container restart with `--cap-add SYS_ADMIN`
  (or `--privileged`). Until then: rank kernels **within one invocation** (shared clock state).

---

## 2. Workstream A — OnPair decode kernel optimization (the main result)

Drove the experiment plan in `B200_VS_GH200_ONPAIR_ANALYSIS.md` track-by-track. Full
results + per-column table in **`vortex-cuda/B200_KERNEL_OPTIMIZATION_RESULTS.md`**.

**Shipped change:** `pick_auto_kernel(chunks, cc_major)` in `vortex-bench/src/onpair_bench.rs`
now branches on compute capability:

- **sm_90 (GH200):** unchanged — `split8read` if small bits12 dict & `frac_le8≥0.90`, else `4tpt`.
- **sm_100 (B200):** `split8read_b128o12` if small bits12 dict & `frac_le8≥0.70`, else `b128o12`.
  (Gate lowered 0.90→0.70 — see §6 #2; captures clickbench/URL bits12 with no regression.)

`cc_major` comes from `device_cc_major()` (cudarc `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`).

**B200 gain over old `4tpt` default (byte-exact):** fineweb/wikipedia bits12 **+46%**,
clickbench/URL bits12 **+18%** (all via `split8read_b128o12`), everything else **+6–12%**
(via `b128o12`).

**New kernels** (all `#define` launch-config variants of existing bodies; no algorithm change):

- `4tpt_b128o12` — 128-thread blocks, `__launch_bounds__(128,12)` → 40 regs, 75% occ, no
  spill. **B200 general default.**
- `4tpt_split8read_b128o12` — split8read body + `(128,12)`. **Standout: +23–26% on bits12
  text.** (8-byte `uint2` reads halve L1/TEX request width; compounds with granularity/occ.)
- Evidence/sweep points: `4tpt_b128`, `_o6`, `_b512o3`, `_b64`, `_b64o24`, `_split8read_occ`.

**Mechanism (decomposed from the evidence-kernel sweep; NCU-blocked so this is the best
available decomposition):** the win is almost entirely **block granularity**, *not*
occupancy. Holding occupancy fixed and varying only block size shows the effect; holding
block size fixed and varying occupancy shows ~nothing:

| comparison                            | what varies                        | B200 effect                               |
|---------------------------------------|------------------------------------|-------------------------------------------|
| `wpb8_occ`→`b128`                     | 256→128-thread block, both 50% occ | **+1 to +4%** (the real lever)            |
| `b128`→`b128o12`                      | 50%→75% occ, both 128-thread       | **~0% (noise)**                           |
| `b128`→`b64`/`b64o24`                 | 128→64-thread block                | **~0% (plateau at 128)**                  |
| `split8read_occ`→`split8read_b128o12` | 256→128-thread split8read          | **+5% → +26%** (granularity 5×'s the win) |

So: **128-thread blocks are the Blackwell lever; forced 75% occupancy buys nothing** (ptxas
cut to 40 regs is harmless but unnecessary — `b128` at 56 regs / 50% occ is equally fast).
Granularity plateaus at 128-thread. This is why the 6 evidence kernels are kept: they encode
this decomposition (the mechanistic story NCU would otherwise provide). **Possible
simplification:** `b128` could replace `b128o12` as the default — equal speed, simpler (no
forced register reduction). The limiter is still presumed occupancy/latency-related (smaller
blocks schedule/drain more evenly across SMs) but is *granularity*-driven, not raw occupancy.

**GH200:** evidence kernels untested this session (no GH200 access). They likely regress or
are neutral there — GH200 was L1/TEX-throughput-bound where granularity/occupancy didn't
help, and `wpb8_occ` itself regressed on GH200. This is exactly why the selector is
arch-gated.

**Dead ends:** Track B′ subsumed (ptxas hits 40 regs no-spill already); Track D freq-order
**forbidden** (interferes with other pipeline parts — per user); Track E hot-dict **dead**
without freq-order (−6–8%, kernel removed); Track B‴ persistent grid low-value for big
columns; Track C NCU blocked.

---

## 3. Workstream B — nvCOMP hardware-engine comparison (new)

Goal was to compare OnPair against nvCOMP on the Blackwell hardware Decompression Engine.

- **Zstd-HW is impossible** — the DE has no Zstd path (`GetTempSize` status 10, GH200 & B200).
  DE supports Deflate / LZ4 / Snappy. (`nvcomp-hw-zstd-unsupported` memory.)
- Built **`benchmarks/onpair-bench/nvcomp_hw_bench.cu`** — standalone, links libnvcomp,
  compresses raw column bytes + times HW decode + compress. Two presets/codec: `hi`
  (Deflate algo=5, max ratio) and `fast` (Deflate algo=0). **GOTCHA: Deflate default
  algo=1 is "low ratio" — always use algo=5 for a fair baseline.** Chunk size is not a
  ratio lever (Deflate window caps at 32 KiB); 256 KiB chunks ≈ optimal DE speed.
- **OnPair vs nvCOMP-HW:** OnPair wins decode by ~3–4×; Deflate-hi sometimes beats OnPair
  on ratio (l_comment 4.56 vs 4.17, clickbench URL 6.44 vs 3.86). Full data:
  `benchmarks/onpair-bench/B200_PRELIMINARY.md` + `b200_results.{csv,json}` (regen via
  `gen_b200_tables.py`). User decided **not** to integrate nvCOMP HW into the Rust bench
  (standalone is enough).

---

## 4. Files created / modified

**Rust (one file):** `vortex-bench/src/onpair_bench.rs`

- arch-aware `pick_auto_kernel` + `device_cc_major()`; sm_100 `frac_le8` gate 0.90→0.70
- `ONPAIR_FAST=1` env (skips reference kernel + nvCOMP in `run_vortex_gpu_decode`)
- 8 new `KernelVariant` registry entries
- `GpuCellResult` now surfaces selector inputs (`frac_le8`, `dict_mean_len`, `dict_max_len`,
  `dict_entries_max`, `small_dict`) in the JSON for gate tuning
- `frac_le8`/`dict_mean_len` field comment tweaks

**New CUDA kernels** (`vortex-cuda/kernels/src/`): `onpair_shmem_4tpt_{b128,b128o12,o6,
b512o3,b64,b64o24,split8read_occ,split8read_b128o12,lenbucket_b128}.cu`. (2tpt.cu
**unchanged**; hotdict.cu was created then **removed**. `lenbucket_b128` is evidence-only —
confirms the 512→128-thread granularity rescue but doesn't beat `b128o12`; not in the selector.)

**Docs / data:** `vortex-cuda/B200_KERNEL_OPTIMIZATION_RESULTS.md`,
`benchmarks/onpair-bench/{B200_PRELIMINARY.md, b200_results.csv, b200_results.json,
gen_b200_tables.py, nvcomp_hw_bench.cu}`.

**Verification done:** `cargo +nightly fmt --all` clean; `cargo clippy -p vortex-bench
--features cuda --all-targets` — **no lints at new code** (29 pre-existing lints are all in
untouched `#[cfg(cuda)]` code that CI never builds — CI doesn't compile `--features cuda`).
No public API changed (`pick_auto_kernel`/`GPU_KERNELS` are private to the bench bin), so
no `public-api.lock` refresh needed.

---

## 5. How to iterate fast (use this!)

```bash
export PATH="$HOME/.cargo/bin:$PATH"
F=vortex-bench/data/onpair-bench/<ds>/<col>/bits12_chunk1000mb_thr0.20/part_0000.vortex
ONPAIR_FAST=1 target/release/onpair-chunk-bench gpu-decode-vortex \
  --vortex "$F" --column <col> --gpu-iters 300 --gpu-validate 2>/dev/null \
  | python3 -c "import sys,json;t=sys.stdin.read();d=json.loads(t[t.index('{'):]);\
[print(k['kernel'],round(k['decode_gib_s'],0),k.get('verified')) for k in d['gpu']['kernels'] if k.get('decode_gib_s')]"
```

- `ONPAIR_FAST=1` → ~12 s/column (vs ~10 min). Compare kernels **within one invocation**.
- Per-kernel JSON key is **`verified`** (not `validated`; `validated` is gpu-level).
- Use 1–2 columns for iteration (l_comment = low-frac/short-token; fineweb = high-frac text).
  Validate the final pick across all 5 big columns + both bits.
- Big columns only: `clickbench/URL, fineweb/text, wikipedia/text, tpch-sf10/{l_comment,
  ps_comment}`. dbtext/s_comment are tiny → launch-bound, ignore their GiB/s.

---

## 6. Open threads / suggested next steps

1. **NCU (Track C) — the one blocked high-value item.** If the box is relaunched with
   `--cap-add SYS_ADMIN`, run SpeedOfLight + MemoryWorkload + Occupancy on `b128o12` vs
   `split8read_b128o12` vs `4tpt`, bits12 & bits16, to *confirm* the occupancy/latency-bound
   story and re-derive `__launch_bounds__`. Also try locking clocks then re-confirm the
   sub-10% deltas (some bits16 gains sit within current ±5% noise).
2. **DONE — clickbench/URL bits12 +5.5% captured.** Lowered the sm_100 `frac_le8` gate
   0.90→0.70. Measured `frac_le8` per column (URL 0.81, l_comment 0.58, ps_comment 0.33;
   fineweb/wikipedia 0.98) and the `split8read_b128o12`−`b128o12` delta: URL +5.5%, l_comment
   −2.6%, ps_comment −4.8% — 0.70 is centered between the win/regression bands. URL bits12 now
   843 GiB/s (+18% over old `4tpt`), all 10 big-column cells re-verified byte-exact, no
   regression. bits16 stays `b128o12` (65 k-entry dicts fail `small_dict`). Selector inputs now
   surfaced in JSON. (Possible future: clickbench has only one large column to confirm against;
   re-check the gate if more low-`frac_le8`-but-`split8read`-favouring columns appear.)
3. **DONE (negative) — length-bucket dict for bits16.** bits16 (slow band, text ~635 GiB/s)
   is walled by its 65 k-entry/~1 MB dict not fitting L1; split8read can't help (dict too big
   for the 32 KB `dict_s8`). Tested the freq-order-*independent* length-bucket layout
   (`ONPAIR_DICT_REORDER=lenbucket`, per-width stride {4,8,12,16}, 2–3× smaller working set) at
   the right granularity via a new `lenbucket_b128` kernel. **Granularity rescue confirmed**
   (512→128-thread = 578→646 GiB/s, +12% on fineweb bits16 — same trap that hid split8read),
   but it only *ties* `b128o12` on high-`frac_le8` text (+0.6–0.9%, within ±5% noise) and
   regresses on long-token columns (l_comment −5.4%, ps_comment −6.1%, URL −1.9%). Kept as an
   evidence kernel, **not** in the selector. bits16 is at the kernel/layout ceiling for these
   columns; remaining headroom is algorithmic (dict-cache needs freq-ordering → forbidden).
5. **Speculative, freq-order-independent:** Track F (per-warp gather dedup) — only helps
   low-cardinality columns (high-card l_comment/URL won't benefit). Cluster-DSMEM (Track G).
6. **Productize beyond the bench:** `pick_auto_kernel` lives in `vortex-bench` (a tool). If
   OnPair decode dispatch in `vortex-cuda` proper should also pick arch-best kernels, that's
   a separate, larger change.
7. **nvCOMP into the bench** — user said standalone is enough; revisit only if wanted.

---

## 7. Gotchas (this session)

- **`KernelLayout` must match the kernel's arg signature.** `split8read*` kernels take
  `dict_s8 + dict_padded` → `KernelLayout::SplitRead8`, NOT `Stride16`. A wrong layout feeds
  wrong buffers → `CUDA_ERROR_MISALIGNED_ADDRESS` / segfault that aborts the whole run.
- `registry block_warps × 32` must equal the kernel's `__launch_bounds__` thread count.
- `gpu-decode-vortex` JSON goes to **stdout**; tracing logs to **stderr**. Strip the log
  prefix before `json.loads` (`t[t.index('{'):]`), or `2>/dev/null`.
- `git` is broken in this worktree (stale macOS `.git` path); use `mv`, not `git mv`.
  Mutagen syncs filesystem state. `vortex-bench/data/**` is mutagen-ignored (regenerate).
- Forcing higher occupancy via `__launch_bounds__(threads, N)` makes ptxas cut registers;
  for the 4tpt body it fits 40 regs at 75% occ with **no spill** (pre-check with
  `ptxas -arch=sm_100 -O3 x.ptx -o /dev/null --verbose`).
