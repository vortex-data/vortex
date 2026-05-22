# Session handoff — B200 OnPair decode + nvCOMP (2026-05-21)

> **Consolidated summary:** see [`B200_ONPAIR_DECODE/`](B200_ONPAIR_DECODE/README.md) — a curated
> folder (findings, experiments-tried ledger, kernel reference, reproduce guide) that indexes this
> handoff and the other detailed docs.

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
- **sm_100 (B200):** `split8read_b128o12` if dict `entries≤16384` & `frac_le8≥0.70`, else
  `b128o12`. (Gate lowered 0.90→0.70 and dict-size raised 4096→16384 — see §6 #2/#11; captures
  clickbench/URL bits12 and fineweb bits14, no regression. `pick_general_blackwell(max_entries,
  frac_le8)`; Hopper selector unchanged at `≤4096`/0.90.)

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

- arch-aware `pick_auto_kernel` + `device_cc_major()`; sm_100 `frac_le8` gate 0.90→0.70.
  The general-case selector is split into two named per-GPU functions —
  `pick_general_blackwell` (sm_100/B200, optimal) and `pick_general_hopper` (sm_90/GH200,
  **verbatim the pre-Blackwell logic**) — dispatched by `cc_major >= 10`, so neither arch's
  tuning can touch the other. Shared early-exits (const1/const2/s4l1/s8) are arch-independent.
- new `KernelLayout::ClusterDsmem` + a dedicated cluster launch branch (compile-time
  `__cluster_dims__`, grid rounded to whole clusters, large dynamic-shared opt-in). Existing
  kernels' launch paths are untouched.
- `ONPAIR_FAST=1` env (skips reference kernel + nvCOMP in `run_vortex_gpu_decode`)
- 8 new `KernelVariant` registry entries
- `GpuCellResult` now surfaces selector inputs (`frac_le8`, `dict_mean_len`, `dict_max_len`,
  `dict_entries_max`, `small_dict`) in the JSON for gate tuning
- `frac_le8`/`dict_mean_len` field comment tweaks

**New CUDA kernels** (`vortex-cuda/kernels/src/`): `onpair_shmem_4tpt_{b128,b128o12,o6,
b512o3,b64,b64o24,split8read_occ,split8read_b128o12,lenbucket_b128,split4read_b128o12,
cluster_dsmem}.cu` + `onpair_shmem_8tpt{,_b128}.cu`. (2tpt.cu/4tpt.cu **unchanged**; hotdict.cu
was created then **removed**. `lenbucket_b128`, `split4read_b128o12`, `cluster_dsmem`, `8tpt`,
`8tpt_b128` are evidence-only — none beats the shipped kernels: lenbucket ties `b128o12` on
bits16, split4read loses to `split8read_b128o12` on bits12 (8 B is the optimal read width),
cluster_dsmem is byte-exact but −80% (occupancy collapse + DSMEM fabric gather), 8tpt loses
−2 to −22% (4tpt is the amortization peak). Not in the selector. `8tpt` reuses the `Stride16`
launch path at `chunk_size=256`.) Plus variable-width kernels
`onpair_shmem_4tpt_{vwidth,vwidth_b128,vwidth4,vwidth4_b128}.cu` (new `KernelLayout::{VWidth,
VWidth4}`): un-pad the 16 B entries; exact 1..16 (`vwidth`) −74% (unaligned `memcpy` loads),
quantized 4-aligned (`vwidth4`) beats `b128o12` +6% on bits12 but −8 to −19% on bits16. Built
from native bytes/offsets (no on-disk change); evidence-only. **Root cause of the bits16 wall now
measured:** `access_top4096_frac` ≈ 0.47 — dict access is near-uniform (99.9% of 65536 entries
used), so there is no hot subset and the full ~1 MB working set is genuinely needed; that's why
every footprint/locality lever fails on bits16.

**Docs / data:** `vortex-cuda/B200_KERNEL_OPTIMIZATION_RESULTS.md`,
`benchmarks/onpair-bench/{B200_PRELIMINARY.md, b200_results.csv, b200_results.json,
gen_b200_tables.py, nvcomp_hw_bench.cu}`.

**Evidence script:** `vortex-cuda/onpair_b200_evidence.py` — runs five controlled comparisons
(granularity-not-occupancy, 8 B optimal read width, bits16 wall, the `frac_le8` gate, tiny
columns launch-bound) and prints one labeled table per claim. `python3
vortex-cuda/onpair_b200_evidence.py` (env: `BIN`, `DATA`, `ITERS`). The cleanest way to
re-demonstrate the findings on a fresh box.

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
   evidence kernel, **not** in the selector. **Also tested L2 persistence** (`ONPAIR_L2_PERSIST`,
   pins the dict in an L2 access-policy window): **no effect** on any bits16 column (637→637,
   621→621, 932→933) — B200's multi-MB L2 already keeps the 1 MB dict resident, so the limiter
   is **L1/TEX gather latency** (dict ≫ L1), not L2 eviction. bits16 is at the kernel/layout/cache
   ceiling for these columns; the only remaining lever (hot codes resident in L1) needs
   freq-ordering → forbidden.
4. **DONE (negative) — split4read (4-byte reads) for the shortest-token bits12 columns.**
   Hypothesis: narrower reads from a 16 KB `dict_s4` (half of split8read's 32 KB) would extend
   the request-narrowing win on fineweb/wikipedia (mean ~4.2 B). Built `split4read_b128o12`.
   Granularity rescue reproduces (512→128-thread: fineweb 522→615, +18%) but it **loses to
   `split8read_b128o12` on every bits12 column** (−8 to −23%). **8 bytes is the optimal read
   width on B200** — halving the padded 16 B read to 8 B cuts transactions, but 4 B is below
   the 32 B sector so it cuts none and adds the >4 B fallback. Confirms the gather is
   transaction/MSHR-bound, not request-width-bound (consistent with the bits16 L2-persist no-op).
   Evidence kernel; not in the selector.
5. **DONE (negative) — Cluster-DSMEM for bits16 (Track G).** Built `cluster_dsmem`: an 8-block
   thread-block cluster shards the ~1 MB dict across its distributed shared memory, per-token
   reads via `map_shared_rank` instead of L2. **Byte-exact but −75 to −80%** (fineweb 637→131,
   l_comment 933→222, URL 952→193 GiB/s). Killed by (1) ~1 block/SM occupancy collapse (128 KB
   dict slice) and (2) remote DSMEM gather saturating the GPC fabric (~7/8 reads remote).
   **Decisive bits16 finding: the limiter is L2 latency hidden by high occupancy, not gather
   bandwidth** — trading occupancy for on-chip staging loses ~5×; `b128o12` at full occupancy is
   near-optimal. (8 warps/block; more recover some but can't close 5×.) Evidence kernel, not in
   the selector. Three independent levers (lenbucket, L2-persist, DSMEM) now all confirm bits16
   is walled. Only Track F (per-warp gather dedup) remains untried — and it only helps
   low-cardinality columns (high-card l_comment/URL won't benefit), so low value here.
6. **DONE (negative) — tokens-per-thread (8tpt).** Amortization sweep was 2tpt < 4tpt; built
   `8tpt`/`8tpt_b128` (256 tokens/warp, halves the epilogue, 8 in-flight loads/thread).
   Byte-exact but slower everywhere (bits12 −22% vs split8read / −4% vs `b128o12`; bits16 −2 to
   −5%). Doubled register pressure cuts occupancy / spills, costing more than the epilogue saved;
   on bits16 the extra per-thread MLP doesn't help → occupancy-bound, not ILP-bound. **4tpt is
   the amortization peak.** Evidence kernels, not in the selector.
7. **DONE — whole-decompress is TRANSFER-bound (the big-picture finding).** Added end-to-end
   measurement (`compressed_bytes`/`h2d_gib_s`/`whole_decompress_gib_s` in JSON): H2D of the
   compressed payload (~10 GiB/s pageable) is 60–100× slower than decode (637–1100 GiB/s), so
   end-to-end is dominated by the copy and `whole/h2d ≈ compression ratio` (1.6–5.9× across
   columns). **The decode-kernel wins (+18–46%) are NOT the end-to-end bottleneck for an
   H2D-then-decode pipeline — compression ratio is.** They matter for **on-device decode** (GPU
   query/scan where the column already lives on the GPU). (Pinned host mem ~3–5× faster H2D, but
   still transfer-bound.) Future: pinned-memory H2D, H2D/decode stream overlap, or batching many
   tiny columns per launch to amortize the ~15 µs launch floor (see the launch-bound note below).
9. **Productize beyond the bench:** `pick_auto_kernel` lives in `vortex-bench` (a tool). If
   OnPair decode dispatch in `vortex-cuda` proper should also pick arch-best kernels, that's
   a separate, larger change.
10. **nvCOMP into the bench** — user said standalone is enough; revisit only if wanted.
11. **DONE — bits14 sweet spot + split8read gate raised to entries≤16384.** Tested a 14-bit dict
    on fineweb (`run --bits 14`): ratio 2.3× (between bits12 1.7× and bits16 2.9×) and
    `split8read_b128o12` = 664 GiB/s, **+9% over `b128o12`** — because the 128 KB `dict_s8` fits
    L1 (split8read wins iff `dict_s8` ≤ L1, i.e. entries ≤ ~32768). Raised the sm_100 gate from
    `entries≤4096`→`≤16384` (kept `frac_le8≥0.70`); fineweb bits14 now auto-selects split8read,
    bits16 (>16384) and low-frac (URL bits14 frac 0.66) stay `b128o12`. Existing kernels handle
    any bit width (codes are u16); only the selector changed. Evidence-script **Demo 7** sweeps
    bits 12/14/16 (compress+bench). fineweb parquet:
    `vortex-bench/data/onpair-bench-src/fineweb/fineweb_10BT_000.parquet`. (Whole-decompress:
    bits16 still wins end-to-end on ratio; bits14 is the pick when on-device decode speed matters.)
    **Full sweep (decode GiB/s):** bits10 674(1.2×), 11 759(1.4×), 12 **802**(1.7×), 14 678(2.3×),
    15 637(2.6×), 16 637(2.9×). Decode peaks at bits12 and flattens at ~637 from bits15→16.
    **bits10/11 are dominated by bits12** (worse ratio, no speed gain). **Half-filling a 16-bit
    dict (bits15, 32768 entries/256 KB dict_s8) gives NO decode gain — 637, same as full bits16** —
    at 256 KB the dict_s8 exactly fills L1; the cache win needs dict_s8 ≤128 KB (bits14). A C++
    "keep top-32768 + re-encode rare tokens" scheme would land on the same flat 637, so it wouldn't
    speed decode. Note: OnPair `threshold` is the training **sample fraction**, not a dict-admission
    cutoff — it does not control fill level. Gate kept at ≤16384.

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
