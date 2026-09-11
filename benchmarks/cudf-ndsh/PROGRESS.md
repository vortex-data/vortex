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
- Q5/Q9 use six-table fixtures; Q10 uses four. `for_each_generated_table`,
  `local_table_files`, and `reference_io.hpp` share generation, files, and bounded CPU
  copies. Synthetic tests independently cover results, predicates, joins, nulls, and empties.
- Generator fixes: independent discount RNG, order year vs month/day RNG, row-aligned
  prices, and `double` supplier scale factors. Before the date fix, each year covered only
  2–3 months and Q10 was empty; afterward all 84 year/month pairs occur at SF0.01.
  Failing-before regressions establish every defect. All NDS-H formats are affected.

## Next, in order

1. Validate the **pinned cuDF runtime**. Only compile-only validation is available
   for that revision; do not present supplemental Debug-cuDF timings as upstream evidence.
2. Scale SF1 → SF10 → SF100; monitor Vortex and RMM allocations separately and watch
   Q10's CPU reference metadata at larger scales. Then add NVTX ranges and capture
   Nsight Systems profiles.
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
  `NDSH_Q10_NVBENCH`, and `NDSH_DATA_GENERATOR_TEST`. `utilities_prebuilt.cpp` adds
  fixture APIs; `seeded-compat/` carries generator fixes on the old API. Ignored
  `q09_prebuilt.cpp` and `q10_prebuilt.cpp` differ only for cuDF 26.08 transform/stream APIs.
- Reuse the CUDA-enabled archive at
  `build/cudf-vortex-smoke/build/_deps/vortex-build/ffi/vortex-artifacts/libvortex_ffi.a`.
  No Rust/kernel changes were needed for Q9/Q10.
- Pinned full build previously timed out after 600s at 241/515. Do not automatically
  retry. NVCC13.1 workaround: `build/cudf-ndsh-build/access-repro/nvcc131-cudf-hook.cmake`.
  Narrow compile artifacts are in `build/cudf-seeded-compile/`: Q9/Q10 ON/OFF objects
  plus `commands.json` results for generator/Q6 sources. Never link old libraries against
  pinned headers.
- Runtime harness, logs, and build artifacts are **ignored, not committed**. The
  patch and docs preserve the implementation; local evidence paths are in VALIDATION.md.

## Guardrails

Keep code/docs concise. Run GPU benchmarks sequentially. Do not change predicates or
RNG to force survivors; investigate generator defects with failing-before tests.
Timing includes read/import/copies/concatenation, query work, destruction, and device
completion; excludes writing/checks. Local files only: pinned-host staging → HtoD →
GPU decode, **not GPUDirect Storage**. Q9 preserves the existing benchmark's unrounded
`SUM(amount)` semantics; the separate streaming SQL's two-decimal rounding is out of scope.
