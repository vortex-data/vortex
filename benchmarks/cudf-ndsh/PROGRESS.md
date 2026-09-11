# Resume: cuDF NDS-H Vortex POC

Checkpoint: 2026-09-11 · branch `ad/cudf-ndsh-build-support`.
[Plan](../../CUDF_POC_PLAN.md) · [Setup](README.md) · [Results](VALIDATION.md)

## Done

- Default-OFF build integration and local-file `write_vortex` / `read_vortex`.
- **Q1, Q5, Q6, Q9, Q10**: Parquet/Vortex × projected read/query; shared full-table
  fixtures, identical projection/post-read filters, independent CPU checks outside timing.
- GPU execution verified at **SF0.01**; all 28 read/query/engine states pass memcheck
  with zero errors. Q1: 52,574 matches/four groups. Q5: 35/four countries. Q6: 554.
  Q9: 538/119 nation-year groups across three amount engines. Q10: 535/70 customers.
- Pinned cuDF **Release** runtime passes all 28 SF1 states and all 20 selected SF10
  states (Q9 uses the representative binary-op engine at SF10). SF1: Q1 5,257,244
  matches/four groups; Q5 5,104/five countries; Q6 56,747 and revenue
  58,942,077.17243248; Q9 64,353/175 groups; Q10 53,866/6,998 customers. SF10:
  Q1 52,600,853/four groups; Q5 51,468/five countries; Q6 568,275; Q9 646,134/175;
  Q10 534,901/69,959. SF1/SF10 benchmark memcheck was not run.
- Q5/Q9 use six-table fixtures; Q10 uses four. `for_each_generated_table`,
  `local_table_files`, and `reference_io.hpp` share generation, files, and bounded CPU
  copies. Synthetic tests independently cover results, predicates, joins, nulls, and empties.
- Generator fixes: independent discount RNG, order year vs month/day RNG, row-aligned
  prices, and `double` supplier scale factors. Before the date fix, each year covered only
  2–3 months and Q10 was empty; afterward all 84 year/month pairs occur at SF0.01.
  Failing-before regressions establish every defect. All NDS-H formats are affected.
- SF1 exposed nonzero scan row counts crossing physical CUDA-flat boundaries and creating
  device-resident `ChunkedArray`s. `batch_rows` is now a maximum using layout-aware
  sub-splitting. Failing-before split-shape and end-to-end Arrow Device tests cover it.
- BenchPress's integration first eliminated default 8,192-row/byte-coalesced blocks and
  double materialization. SF10 Nsight then found that 1,048,576-row blocks still produced
  58 Q1 batches, 46,061 CUDA API calls, and 1,108 kernels with only 32% GPU utilization.
  The benchmark now uses 16,777,216-row physical/read blocks and retains 2 GiB in CUDA's
  default pool. Q1 falls to four batches, 3,472 API calls, and 80 kernels.
- Explicit CUDA blocks now disable outer layout dictionaries. This prevents the SF10
  `orders.o_custkey` dictionary from splitting one requested block into 214 `u16` runs;
  Q10 orders falls to one batch and its profiled read drops 46.9 → 6.2 ms. A
  high-cardinality row-block regression covers this behavior.
- Optimized SF10 Vortex read/query CPU means are Q1 45.607/82.630 ms, Q5
  33.460/37.841, Q6 21.062/23.110, Q9 binary 45.550/53.136, and Q10 49.808/57.635.
  Every state beats Parquet; Vortex reads improve 1.39–2.34× over the prior 1M path.
  Adapter tests pass 15/15 and CUDA FFI tests pass 20/20.

## Next, in order

1. Scale the pinned Release benchmark to **SF100**. Monitor Vortex and RMM allocations
   separately and watch Q10's CPU reference metadata and large string columns.
2. Capture matched-cache SF100 Nsight Systems profiles using the adapter's stage ranges;
   report HtoD traffic/overlap, decode/adapter cost, file size, and peak memory.
3. Publish Vortex prerequisites, replace the retained base pin, and prepare the [POC] PR.

Optional test follow-up: exercise the generator callback API's empty/all-table and
invalid/duplicate-name cases directly; the six-table ordered path is covered by Q5.

## Worktree and build cautions

- **Tracked deliverable:** `benchmarks/cudf-ndsh/upstream.patch`, cumulative against
  cuDF `5339497a1a17d799687cbf189fb113411fb015ca`.
- **Editable cuDF:** `build/cudf-ndsh-src` (ignored, already patched). New source
  files need `git add -N` there before refreshing the cumulative patch:
  `git --no-pager -C build/cudf-ndsh-src diff -- cpp/benchmarks > benchmarks/cudf-ndsh/upstream.patch`.
  On another machine, apply the tracked patch to a fresh pinned checkout per README.
- **Never modify `/home/ubuntu/cudf`**: user's `binary-view-support` checkout. Its
  matching headers/prebuilt cuDF 26.08 Debug library provide supplemental runtime only.
- Local consumer: `build/cudf-q6-prebuilt/CMakeLists.txt`; targets
  `NDSH_Q01_NVBENCH`, `NDSH_Q05_NVBENCH`, `NDSH_Q06_NVBENCH`, `NDSH_Q09_NVBENCH`,
  `NDSH_Q10_NVBENCH`, `NDSH_VORTEX_IO_TEST`, and `NDSH_DATA_GENERATOR_TEST`.
  `utilities_prebuilt.cpp` adds
  fixture APIs; `seeded-compat/` carries generator fixes on the old API. Ignored
  `q09_prebuilt.cpp` and `q10_prebuilt.cpp` differ only for cuDF 26.08 transform/stream APIs.
- Reuse the CUDA-enabled archive at
  `build/cudf-vortex-smoke/build/_deps/vortex-build/ffi/vortex-artifacts/libvortex_ffi.a`.
  Rebuild it after CUDA FFI changes; the smoke CMake tree currently cannot regenerate
  because its minimal project does not define every query target in `vortex.cmake`.
- Pinned Release `libcudf.so`, adapter tests, and all five benchmark executables build and
  run from `build/cudf-ndsh-build`. NVCC13.1 workaround:
  `build/cudf-ndsh-build/access-repro/nvcc131-cudf-hook.cmake`. Link warnings for
  `libnvrtc.so.13`/`libnvJitLink.so.13` are resolved by the executable RPATH. Never link
  old libraries against pinned headers.
- Runtime harness, logs, and build artifacts are **ignored, not committed**. The
  patch and docs preserve the implementation; benchmark JSON and Nsight evidence paths
  are in VALIDATION.md.

## Guardrails

Keep code/docs concise. Run GPU benchmarks sequentially. Do not change predicates or
RNG to force survivors; investigate generator defects with failing-before tests.
Timing includes read/import/copies/concatenation, query work, destruction, and device
completion; excludes writing/checks. Local files only: pinned-host staging → HtoD →
GPU decode, **not GPUDirect Storage**. Q9 preserves the existing benchmark's unrounded
`SUM(amount)` semantics; the separate streaming SQL's two-decimal rounding is out of scope.
