# OnPair GPU decode — hand-over notes

Branch: `claude/gpu-compression-onpair-ywm65`.
This is a session hand-over for picking the work up on a real CUDA box
(A100 / H100 / GH200) — what landed, what's deferred, what to look for in
the first measurement.

If something here disagrees with the code, the code is right; ping me and
I'll fix the doc.

---

## 1. What this branch does

Adds a CUDA decoder for the **OnPair** short-string compression algorithm
(Gargiulo et al., *OnPair: Short Strings Compression for Fast Random
Access*, arXiv:2508.02280) plus a host-vs-device benchmark.

OnPair is the string-side analogue of FSST: BPE-trained dictionary, 9–16
bit packed codes, longest-prefix-match parsing, ≤16-byte token bodies.
The reference C++ / Rust implementation lives at
`gargiulofrancesco/onpair_rs`. The CPU port in this branch comes from
`origin/claude/port-onpair-cpp-rust-3DK8B` (trimmed; see §3.1).

The scope on this branch is intentionally narrow: **standalone library +
benchmark**, no Vortex `Array` integration. There is no `OnPairArray`,
no VTable, no `CudaExecute` registration, no plugin. This was decided up
front so the patch stays small enough to review and the only thing
exercised is the GPU kernel vs. the CPU baseline.

---

## 2. Where the code lives

```
encodings/onpair-rs/                       CPU port (crate name: onpair-lib)
    src/lib.rs                             pub use surface
    src/{column,parser,trainer,lpm}.rs     compress / parse / train / LPM
    src/{bits,dict,store}.rs               bit-pack reader, dict layout, store
    src/{automaton,kmp,aho_corasick}.rs    compressed-domain predicates
    src/{types,config,tokenize}.rs         types, training config, tokenizer

vortex-cuda/kernels/src/onpair.cu          two kernels x 8 bit-widths = 16 entry points
vortex-cuda/src/kernel/encodings/onpair.rs host pipeline `onpair_gpu_decode_all`
vortex-cuda/benches/onpair_cuda.rs         CPU vs GPU criterion bench
```

Wiring: `Cargo.toml` workspace gains `encodings/onpair-rs` and
`onpair-lib` in `[workspace.dependencies]`. `vortex-cuda/Cargo.toml`
gains `onpair-lib` as a direct dep and a `[[bench]] onpair_cuda` entry.
`vortex-cuda/src/lib.rs` re-exports `onpair_gpu_decode_all` and
`OnPairGpuDecoded`.

---

## 3. Decisions worth knowing

### 3.1 The CPU port was cherry-picked and trimmed
`origin/claude/port-onpair-cpp-rust-3DK8B` contains a three-crate stack:
`onpair-rs` (pure Rust), `onpair-sys` (FFI to C++ reference), `onpair`
(full Vortex array encoding). This branch pulls **only** `onpair-rs`
and drops:
- `tests/cross_impl.rs` (depended on `vortex-onpair-sys`)
- `benches/clickbench.rs` (depended on divan + parquet + arrow + sys)
- The matching `dev-dependencies` (`vortex-onpair-sys`, `divan`,
  `arrow-array`, `arrow-schema`, `parquet`, `rstest`)

The crate stands alone with deps `aho-corasick`, `hashbrown`, `memchr`,
`rand`. **238 in-source tests pass** (`cargo test -p onpair-lib --lib`).

If we later want the full Vortex array integration, pull
`encodings/onpair/` from the same branch — it's a substantial body of
work (compute kernels, file roundtrip, LIKE pushdown). Not needed for
the GPU decode benchmark.

### 3.2 Two kernels, not one

| Pass | Kernel | What it does |
|---|---|---|
| 1 | `onpair_lengths_b<BITS>` | One thread per row. Walks the row's packed token stream, looks up each token's byte length in `dict_table`, writes the sum to `row_lengths[row]`. |
| 2 | `onpair_decode_b<BITS>` | One thread per row. Walks the row's tokens, copies each token's bytes from `dict_bytes` into `output_bytes[output_offsets[row]..]`. |

The CPU baseline (`Column::decode_all`) is itself two-pass — pass 1 sums
total decoded length, pass 2 decodes. So the GPU mirrors the CPU
structure. Moving the bitstream walk off the host CPU is the **single
largest end-to-end win** vs. the v0 design which had it on host.

### 3.3 Why the exclusive scan stays on host

Between the two GPU passes there's a D2H of `row_lengths` (4 MB per 1M
rows), a host exclusive scan, and an H2D of u64 `output_offsets`
(8 MB per 1M rows). A multi-block GPU scan would be the right thing,
but `vortex-cub` doesn't currently expose `cub::DeviceScan` and adding
it is real infrastructure work (new `.cu` wrapper, new bindgen
declarations, new error plumbing). The total round-trip is ~1 ms on
PCIe Gen4 and ~50 µs on NVLink-C2C — acceptable for v1.

If GH200 numbers come back with the host scan dominating, that's the
flag to move it on-device.

### 3.4 16-byte over-copy via `memcpy`, not a manual byte loop

```cuda
memcpy(cur, dict_bytes + off, 16);
cur += len;  // true length, not 16 — the over-copy tail is discarded
```

`dict_bytes` is padded with `MAX_TOKEN_SIZE` (= 16) trailing zeros on
the host before H2D, so the over-copy is safe at the end of the buffer.
`memcpy(dst, src, 16)` is the idiom nvcc recognises and lowers to
whatever store width the runtime alignment proof allows — typically
`st.global.v4.u32` (one 16-byte store) when both pointers are 16-byte
aligned, degrading to narrower stores when they aren't.

A handwritten `#pragma unroll` byte loop was tried first; it compiled to
16× `st.global.u8` and was discarded. **Verify the SASS** on the first
run (see §6).

### 3.5 Why no shared-memory `dict_table` staging

`dict_table` is 32 KB at bits=12 and up to 512 KB at bits=16. At bits=12
it would fit per block; at higher widths it wouldn't. The existing FSST
kernel (whose 2 KB table easily fits in shared) measured shared-memory
staging **slower** at the 1M-row scale because L1/L2 already serves the
table after a few warps warm it. OnPair's table is larger, so L2 is the
right tier (40 MB on A100, ~50 MB on Hopper) — well over any
`dict_table`. Reading via `__ldg` forces the read-only data cache
(texture path), which has different replacement and tends to win when
the working set is reused by many threads.

### 3.6 Variant: OnPair16 only (per scope decision)

The reference impl supports two variants: OnPair (unbounded symbol
length) and OnPair16 (≤16 byte symbols). This branch targets OnPair16
because the 16-byte fixed-width over-copy is the simplest device-side
decode loop. The CPU port supports both, but `decode_all` is
written for the 16-byte over-copy path. If we ever need unbounded
symbols on the GPU, the kernel needs a per-token variable-length
`memcpy` (cudaMemcpyAsync-style) plus alignment care.

---

## 4. How to build and run

CPU side (works without CUDA):
```bash
cargo test -p onpair-lib --lib
# expect: 238 passed
```

GPU side (requires nvcc + CUDA runtime + a device):
```bash
cargo build -p vortex-cuda --benches
cargo bench -p vortex-cuda --bench onpair_cuda
```

Bench reports throughput in **bytes / second of uncompressed output**.
Two benchmarks in the `cuda` group:

* `cuda/onpair/cpu_decode/1M` — CPU `Column::decode_all`
* `cuda/onpair/gpu_decode/1M` — GPU `onpair_gpu_decode_all` end-to-end
  (incl. H2D + lengths + D2H lens + host scan + H2D offs + decode +
  D2H bytes)

The criterion config from `bench_config::cuda_bench_config()` sets
`sample_size = 10`, `warm_up_time = 500 ms`. I bumped
`measurement_time` to 5 seconds for `onpair_cuda` because we're
measuring wall-clock with `Instant::now()`, not CUDA-event time — the
default of 1 ns (used for the CUDA-event-timed benches) would force
`iters = 1`.

If CUDA isn't available at build time, the bench compiles as an empty
`fn main()` thanks to the `#[cuda_available]` / `#[cuda_not_available]`
attribute macros from `vortex-cuda-macros`. So local builds without nvcc
still pass.

---

## 5. Expected numbers (1M ClickBench URLs)

Assumptions: ~100 MB raw, ~25 M tokens at 12 bits → ~37 MB packed
codes, ~30 KB dict, ~100 MB decoded output.

| Stage | A100 (PCIe Gen4, HBM2e 1.55 TB/s) | GH200 (NVLink-C2C 450 GB/s, HBM3 3 TB/s) |
|---|---:|---:|
| **CPU `Column::decode_all`** | **30–50 ms** | 30–50 ms |
| H2D inputs (~42 MB) | 1.5–2 ms | < 0.1 ms |
| `onpair_lengths_b12` | 0.15–0.4 ms | 0.08–0.2 ms |
| D2H lens (4 MB) + host scan + H2D offs (8 MB) | 1.5 ms | 1.1 ms |
| `onpair_decode_b12` | 0.15–0.3 ms | 0.08–0.15 ms |
| D2H decoded bytes (~100 MB) | 4–5 ms | 0.2 ms |
| **GPU total wall-clock** | **~7–9 ms** | **~1.4–1.7 ms** |
| **Speedup vs CPU** | **~4–6×** | **~20–35×** |

The dominant cost is different on the two platforms:
- **A100: PCIe D2H of decoded output is ~50% of wall-clock.** Kernel
  itself is sub-millisecond.
- **GH200: the 1 ms host CPU exclusive scan dominates.** Once that's
  on-device, GH200 should hit ~400–600 µs total → ~60–100× over CPU.

Kernel-only (i.e. if we switched the bench to `TimedLaunchStrategy`
like FSST does, measuring just CUDA event time) is roughly
**100–150× faster than CPU on A100** and similar on H100. The
end-to-end speedup looks smaller because PCIe is in the wall-clock.

---

## 6. First-run checklist

When this finally runs on a CUDA box:

1. **Sanity check correctness first.** Add a temporary integration test
   that calls `column.decode_all()` (CPU) and `onpair_gpu_decode_all`
   (GPU) on the same `Parts` and `assert_eq!`s both `bytes` and
   `offsets`. The bench skips this so the CPU/GPU outputs could diverge
   silently if the kernel has a bug.

2. **Check SASS for the over-copy store width.** `cuobjdump --dump-sass`
   on the generated PTX for `onpair_decode_b12`. Look at the loop body:
   - Good: a single `STG.E.128` (16-byte store) or `STG.E.64 ×2`.
   - Bad: a chain of `STG.E.U8`. If you see this, the `memcpy`
     vectorization didn't fire — likely because nvcc can't prove
     alignment. Fix: either pre-align `dict_bytes` (round every entry's
     start up to 16 — costs ~7 bytes/entry average ≈ 28 KB extra) or
     write a manual store using `*reinterpret_cast<ulonglong2*>` with
     an explicit alignment fallback for the unaligned-output prologue.

3. **Verify A100 PCIe really is the bottleneck**, not the kernel. Run
   the bench, then add a probe that times just the decode kernel via
   `TimedLaunchStrategy` (mirror the FSST bench). If kernel ≈ 0.3 ms
   and end-to-end ≈ 8 ms, PCIe dominates as expected. If the kernel is
   significantly slower than 0.3 ms, something is off (SASS issue, or
   bad occupancy).

4. **Check warp divergence.** Profile with Nsight Compute. The metric
   to watch is "smsp__average_warp_latency_cycles". URLs cluster
   tightly around 80–120 B, so I expect divergence to be moderate. If
   the dataset has bimodal length distribution (some 30-byte rows
   mixed with 500-byte rows in the same warp), divergence will be
   ugly and that's the trigger for §7.1.

---

## 7. Deferred work — in priority order if v1.1 isn't enough

### 7.1 Warp-cooperative decode (largest expected win)
Currently one thread per row; if rows in a warp differ a lot in length,
31 threads idle while one finishes. The fix is one warp per row, where
the 32 lanes each take a different token in the row and do a warp
prefix-scan to figure out where in `output[output_offsets[row]..]` to
write. Likely **3–5× kernel uplift** on URL data. Not done because the
v1 simple version is easier to reason about and probably already beats
CPU comfortably; revisit if measurement shows the kernel itself (not
PCIe) is the limit.

### 7.2 Move exclusive scan on-device
Adds a `scan.cu` / `scan.h` to `vortex-cub/kernels/` wrapping
`cub::DeviceScan::ExclusiveSum<uint32_t,uint64_t>`. Removes the ~1 ms
host scan. Big lever on GH200 (where it's the dominant cost), small
lever on A100 (PCIe still dominates). ~3–4 hours of plumbing.

### 7.3 Pinned host memory for H2D / D2H
The decoded-bytes D2H of 100 MB at PCIe Gen4 currently uses pageable
host memory which DMAs at ~16 GB/s effective. Pinned via
`PinnedByteBufferPool` (already exists at `vortex-cuda/src/pinned.rs`)
runs at ~24 GB/s. ~35% off the D2H step on A100; nearly invisible on
GH200 since NVLink-C2C is already fast. ~2–3 hours.

### 7.4 Pipelined streams
Split the input column into chunks and overlap H2D(chunk N+1) with
kernel(chunk N) with D2H(chunk N-1). On A100 this could roughly halve
wall-clock by hiding PCIe behind compute. ~1 day, including the
correctness work to stitch the per-chunk outputs.

### 7.5 GH200 unified-memory fast path
On Grace-Hopper, `cudaMallocManaged` + `cudaMemAdvise(PreferredLocation
= GPU)` lets the GPU directly access host RAM over NVLink-C2C at
450 GB/s without an explicit H2D. Removes the H2D step entirely on
GH200, but adds a runtime probe for grace-hopper and a separate path.
Probably not worth it — explicit copies on NVLink-C2C are already
near-free and the code stays simpler.

### 7.6 Pre-sort rows by length at compress time
Decode in length-banded chunks; near-zero warp divergence. Costs a
permutation array at compress time. Not started.

---

## 8. Known unknowns

- **Exact CPU baseline numbers** — the OnPair paper reports ~5–7 GB/s
  decode throughput, my estimate above is derived from that plus my
  read of `Column::decode_all`'s two-pass structure. Could be off by
  ±30%.
- **ClickBench URL distribution under the synthetic generator.**
  `vortex_fsst::test_utils::generate_clickbench_urls` produces
  weighted-domain URLs but the avg length and prefix-sharing profile
  may not exactly match real ClickBench `hits.parquet`. If the bench
  numbers look off, swap in a real parquet URL column via the same
  pattern as `onpair-rs`'s deleted `clickbench.rs` bench (the
  `ONPAIR_BENCH_PARQUET` env var path).
- **PTX size at 16 specialisations** (8 bit-widths × 2 kernels). Should
  be < 200 KB so well under the per-module limit, but worth eyeballing
  the `.ptx` size after build.
- **u32 output offset overflow.** `Parts::codes_boundaries` is u32,
  `dict_offsets` is u32, but the **decoded output** can exceed 4 GiB if
  enough rows are decoded together. The host return uses
  `Vec<u32>` offsets; `build_output_offsets` errors out at
  `u32::MAX`. Practical limit is ~4 GiB decoded per call — fine for
  per-chunk decompression, watch out for whole-column decompression
  on huge datasets.

---

## 9. References

- Paper: <https://arxiv.org/abs/2508.02280>
- Reference Rust impl: <https://github.com/gargiulofrancesco/onpair_rs>
- The full Vortex array integration (not on this branch):
  `origin/claude/port-onpair-cpp-rust-3DK8B`
- A heavier research dump (732 lines, file:line citations for every
  claim above) was saved during this session to
  `/tmp/onpair-gpu-research.md` — that's session-local so it won't
  survive once this container is reclaimed, but the content is
  reproducible by running an `Explore` agent over the codebase.
