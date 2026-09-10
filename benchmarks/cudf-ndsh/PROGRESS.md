# Resume: cuDF NDS-H Vortex POC

Checkpoint: 2026-09-10 · branch `ad/cudf-ndsh-build-support`.
[Plan](../../CUDF_POC_PLAN.md) · [Setup](README.md) · [Results](VALIDATION.md)

## Done

- Default-OFF build integration and local-file `write_vortex` / `read_vortex`.
- **Q1, Q5, Q6**: Parquet/Vortex × projected read/query; shared full-table fixtures,
  identical projection/post-read filters, independent CPU result checks outside timing.
- GPU execution verified at **SF0.01**, all 12 states pass memcheck with zero errors.
  Q1: 44,973 matches/four groups. Q5: 35 matches/four countries. Q6: 563 matches.
- Q5 adds six-table fixtures, `for_each_generated_table`, `local_table_files`, and
  `q5_reference_builder`; `reference_io.hpp` shares bounded CPU copies with Q1.
  Synthetic tests independently check Q5 CPU/GPU results, dates, joins, and empty outputs.
- Generator fixes: independent discount RNG, row-aligned prices, and `double`
  supplier scale factors (previously SF0.01/0.1 produced zero supplier keys).
  Failing-before regressions establish each defect. All NDS-H formats are affected.

## Next, in order

1. Add **Q9**, then **Q10**, at SF0.01 using the Q5 fixture/reader pattern. Preserve
   original Parquet benchmarks; validate raw projections and nonempty GPU results
   against independent CPU references and hand-written cases. Extend `vortex.cmake`
   and `test_build_integration.py` for each query. Run normal and memcheck states.
2. Validate the **pinned cuDF runtime**. Only compile-only validation is available
   for that revision; do not present supplemental Debug-cuDF timings as upstream evidence.
3. Only after all five queries pass, scale SF1 → SF10 → SF100; monitor Vortex and
   RMM allocations separately. Then add NVTX ranges and capture Nsight Systems profiles.
4. Publish Vortex prerequisites, replace the retained base pin, and prepare the [POC] PR.

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
  `NDSH_Q01_NVBENCH`, `NDSH_Q05_NVBENCH`, `NDSH_Q06_NVBENCH`, `NDSH_DATA_GENERATOR_TEST`.
  `utilities_prebuilt.cpp` includes old matching helpers and adds new fixture APIs;
  `seeded-compat/` carries generator fixes adapted to that old API.
- Reuse the CUDA-enabled archive at
  `build/cudf-vortex-smoke/build/_deps/vortex-build/ffi/vortex-artifacts/libvortex_ffi.a`.
  No Rust/kernel changes are needed for the remaining query wiring.
- Pinned full build previously timed out after 600s at 241/515. Do not automatically
  retry. NVCC13.1 workaround: `build/cudf-ndsh-build/access-repro/nvcc131-cudf-hook.cmake`.
  Narrow compile commands/hashes: `build/cudf-seeded-compile/q5-final-commands.json`
  and `q5-commands.json`. Never link old libraries against pinned headers.
- Runtime harness, logs, and build artifacts are **ignored, not committed**. The
  patch and docs preserve the implementation; local evidence paths are in VALIDATION.md.

## Guardrails

Keep code/docs concise. Run GPU benchmarks sequentially. Do not change predicates or
RNG to force survivors; investigate generator defects with failing-before tests.
Timing includes read/import/copies/concatenation, query work, destruction, and device
completion; excludes writing/checks. Local files only: pinned-host staging → HtoD →
GPU decode, **not GPUDirect Storage**. Q9/Q10 currently remain Parquet-only.
