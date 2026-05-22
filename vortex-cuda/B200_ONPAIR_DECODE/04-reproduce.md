# Reproduce / how-to

## Build

```bash
export PATH="$HOME/.cargo/bin:$PATH"
export LIBCLANG_PATH=/usr/lib/llvm-18/lib
cargo build -p vortex-bench --features cuda --bin onpair-chunk-bench --release   # ~5 min
```

`build.rs` compiles every `vortex-cuda/kernels/src/*.cu` to PTX. If cargo fails with exactly
`sccache: error: Operation not permitted`, prefix with `RUSTC_WRAPPER=`.

> Background builds can be killed when the session pauses — run the build in the **foreground**
> and wait for `Finished`.

## Evidence script (the fastest way to see the findings)

```bash
python3 vortex-cuda/onpair_b200_evidence.py
```

Runs 7 controlled comparisons and prints one labeled table per claim:
1. granularity, not occupancy   2. 8 B optimal read width   3. bits16 wall (L2/DSMEM)
4. the `frac_le8` gate   5. tiny columns are launch-bound   6. whole-decompress is transfer-bound
7. **dict bit-width sweep** (compresses bits 12/14/16 and benchmarks)

Env overrides: `BIN`, `DATA`, `ITERS`, `PARQUET`, `SWEEP_BITS`, `EVIDENCE_COMPRESS=0` (skip
Demo 7's compression).

## Decode an existing `.vortex` file (kernel-only micro-benchmark)

```bash
F=vortex-bench/data/onpair-bench/fineweb/text/bits12_chunk1000mb_thr0.20/part_0000.vortex
ONPAIR_FAST=1 target/release/onpair-chunk-bench gpu-decode-vortex \
  --vortex "$F" --column text --gpu-iters 300 --gpu-validate 2>/dev/null
```

- `ONPAIR_FAST=1` skips the reference kernel + nvCOMP (~50× faster sweeps; ~12 s/big column).
- `--gpu-validate` checks every kernel byte-exact vs CPU decode (per-kernel JSON key `verified`).
- JSON → stdout, logs → stderr (strip the log prefix before the first `{`, or `2>/dev/null`).
- The JSON `gpu` block now also reports: `frac_le8`, `dict_mean_len`, `dict_max_len`,
  `dict_entries_max`, `small_dict`, `distinct_codes`, `access_top4096_frac`, `compressed_bytes`,
  `h2d_gib_s`, `whole_decompress_gib_s`.

## Compress a new dict bit-width (e.g. bits14)

```bash
ONPAIR_FAST=1 target/release/onpair-chunk-bench run \
  --parquet vortex-bench/data/onpair-bench-src/fineweb/fineweb_10BT_000.parquet \
  --column text --dataset-id fineweb \
  --bits 14 --chunk-bytes 1048576000 --threshold 0.2 --sample-bytes 1000000000 \
  --out-dir vortex-bench/data/onpair-bench \
  --gpu-decode --gpu-iters 200 --gpu-validate
```

- `--bits` accepts a comma list (e.g. `12,14,16`) to sweep in one run.
- `MB = 1<<20`, so `--chunk-bytes 1048576000` (= 1000·MB) yields the `chunk1000mb` path name.
- **`--threshold` is the training *sample fraction*** (how much of the data the trainer looks
  at), **not** a dict-admission cutoff — it does not control dict fill level. Dict size is capped
  by `2^bits` and the data.
- Source parquets live under `vortex-bench/data/onpair-bench-src/`. Generated `.vortex` data is
  mutagen-ignored (regenerate as needed).

## Run the ablation NCU-proxy (per-stage cost breakdown)

The ablation kernels are timed in the normal sweep. Extract them:

```bash
ONPAIR_FAST=1 target/release/onpair-chunk-bench gpu-decode-vortex \
  --vortex "$F" --column text --gpu-iters 300 2>/dev/null \
| python3 -c "import sys,json;t=sys.stdin.read();g=json.loads(t[t.index('{'):])['gpu']; \
ks={k['kernel'].replace('onpair_shmem_4tpt_',''):k.get('decode_gib_s') for k in g['kernels'] if k.get('decode_gib_s')}; \
full=ks['ablate']; print('full',round(full)); \
[print(n, round(ks[n]), '(%+d%%)'%round((ks[n]/full-1)*100)) for n in ['ablate_nogather','ablate_noemit','ablate_nodrain','ablate_noscan','ablate_cfree']]"
```

(no `--gpu-validate` — the `_no*`/`_cfree` proxies are intentionally not byte-exact).

## Verification before handing off Rust changes

```bash
cargo +nightly fmt --all                 # or `--all -- --check`
cargo clippy -p vortex-bench --all-targets          # CI builds WITHOUT --features cuda
```

CI does **not** compile `--features cuda`, so the CUDA decode code + GPU kernels aren't built in
CI. `pick_auto_kernel`/`GPU_KERNELS` are private to the bench binary → no `public-api.lock`
refresh needed.

## Blocked / needs infra

- **NCU** (`ERR_NVGPUCTRPERM`) and **clock-locking** (`nvidia-smi -lgc`): need the container
  relaunched with `--cap-add SYS_ADMIN` (or `--privileged`). The top open item — would confirm
  the emit-is-70% diagnosis directly.
