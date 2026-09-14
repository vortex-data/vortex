# cuDF NDS-H Vortex POC

[Plan](../../CUDF_POC_PLAN.md) · [Validation](VALIDATION.md) · [Resume here](PROGRESS.md)

`upstream.patch` adds default-OFF build support, `write_vortex` / `read_vortex`, and
**Q1/Q5/Q6/Q9/Q10 plus projected read-only Parquet/Vortex comparisons** using shared
full-table fixtures and post-read filters. Original Parquet-pushdown benchmarks remain
separate. Generator fixes affect **all** NDS-H consumers, including Vortex OFF;
regenerate fixtures. This is not full TPC-H conformance.

The cumulative patch is refreshed and apply-checked. See [PROGRESS.md](PROGRESS.md)
for the separate staging, generic-performance, and cold-cache patch/test commits. **Current binaries are stale:** the Release build timed out before relinking
cuDF benchmarks and the adapter. There is no fresh runtime validation.

## Current implementation and timing

- All five queries use generic cuDF, not hand-fitted query kernels; `q1_fused` is absent.
  Q1 uses shared sum/count groupby with averages afterward for both formats.
  Q5/Q9/Q10 read independent tables concurrently for both formats.
- Write: 16,777,216-row cuDF chunks → host Arrow → CPU-written CUDA-flat blocks;
  explicit blocks disable byte coalescing and outer layout dictionaries.
- Read: pooled cacheable pinned-host staging → HtoD → GPU decode → retained Arrow
  Device views → one final owning cuDF materialization. Vortex CUDA pool retention
  is 8 GiB. Local files, device 0, flat typed columns; **no GPUDirect Storage**.
- `cache=warm/cold`: before every manual cold callback's timed portion, per-file
  `fdatasync` + `POSIX_FADV_DONTNEED` is followed by required `mincore` residency == 0.
  Cold Vortex data uses `O_DIRECT`, metadata stays buffered, and Parquet uses its
  native path. Lower-level storage caches are not flushed; this is OS-page-cache
  coldness, not guaranteed cold media.
- CPU wall timing includes complete reads/import/materialization, query work where
  selected, destruction, and device completion. Writing, checks, and eviction are
  excluded. RMM peaks exclude Vortex allocations.

Default kernel-event suppression is reverted; no event optimization is retained.

## Apply and build

From the Vortex root, for a **fresh** cuDF checkout:

```sh
git clone https://github.com/NVIDIA/cudf.git build/cudf-ndsh-src
git -C build/cudf-ndsh-src checkout --detach 5339497a1a17d799687cbf189fb113411fb015ca
git -C build/cudf-ndsh-src apply --check ../../benchmarks/cudf-ndsh/upstream.patch
git -C build/cudf-ndsh-src apply ../../benchmarks/cudf-ndsh/upstream.patch
```

The development checkout is already patched; never modify `/home/ubuntu/cudf`.
Build instructions are in patched `cpp/benchmarks/ndsh/VORTEX.md`.
The pinned Release build tree is `build/cudf-ndsh-build`. The last build timed out
at 1200 s; do not retry automatically. See [VALIDATION.md](VALIDATION.md) for the
command, partial-build outcome, and completed checks.

**Local Vortex sources are required:** the retained base pin
`bffdca1109e99e6957ea2fc18f4a7809c88e0a0c` lacks the CUDA-layout edition, device decimal
slicing, bitmap alignment/padding, dictionary export, and projected scan API.
Publish prerequisites and update the immutable pin before removing the gate.

## Run after rebuilding the final source

Benchmarks are named `ndsh_q{1,5,6,9,10}_local`. The default scale-factor axis now
includes SF10; always select axes explicitly for comparable runs. Example SF10 Q10
matrix (the output path is for a future final-source run, not existing evidence):

```sh
build/cudf-ndsh-build/benchmarks/NDSH_Q10_NVBENCH \
  --benchmark ndsh_q10_local --axis scale_factor=10 \
  --axis 'format=[parquet,vortex]' --axis 'workload=[read,q10]' \
  --axis 'cache=[warm,cold]' --min-samples 3 --timeout 30 \
  --json build/cudf-ndsh-build/sf10-q10-final-source.json
```

Use `scale_factor=1` for SF1. Q1/Q5/Q6/Q9 use executables `NDSH_Q01_NVBENCH`,
`NDSH_Q05_NVBENCH`, `NDSH_Q06_NVBENCH`, and `NDSH_Q09_NVBENCH`, respectively, with
matching benchmark/workload names. Q9 also needs explicit
`--axis 'engine=[binaryop,ast,transform]'`. Run GPU benchmarks sequentially. Prefix
future profiler launches with `env -u ANTHROPIC_API_KEY` and follow the
[profile metadata safety rules](VALIDATION.md#profile-evidence-and-safety).
JSON, logs, binaries, Nsight reports, and SQLite exports are ignored, not committed.

## Checks and acceptance status

```sh
python3 -B -m unittest discover -s benchmarks/cudf-ndsh -v
python3 -B -m unittest discover -s vortex-ffi/cmake/tests -v
ruff check benchmarks/cudf-ndsh/test_build_integration.py
ruff format --check benchmarks/cudf-ndsh/test_build_integration.py
```

All 14 current offline tests and refreshed-patch apply checks pass. Earlier runtime
and memcheck results remain historical; the new Release archive alone does not validate
the unrelinked benchmarks/adapter. Full check results are in [VALIDATION.md](VALIDATION.md).

**Latest timings are pre-event-revert diagnostics, not final committed-source
performance evidence.** No new performance runs were made. The ≥2× end-to-end read
**and** query goal across SF1/SF10, warm/cold, is not met. Stabilize those matrices before
SF100; a full Q1 query profile is still pending.
