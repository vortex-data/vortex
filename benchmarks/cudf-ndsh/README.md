# cuDF NDS-H Vortex POC

[Plan](../../CUDF_POC_PLAN.md) · [Validation](VALIDATION.md) · [Resume here](PROGRESS.md)

`upstream.patch` adds default-OFF build support, `write_vortex` / `read_vortex`, and
**Q1/Q5/Q6/Q9/Q10 plus projected read-only Parquet/Vortex comparisons** using shared
full-table fixtures, identical scan projections, and post-read predicates. Original
native Parquet-pushdown benchmarks remain separate. This is not full TPC-H conformance.

The minimal patch touches only `cpp/benchmarks/ndsh/` plus **one include hook** in
`cpp/benchmarks/CMakeLists.txt`; it has no generator changes or generator-test target.
Offline and compile-only validation passed; **binaries remain stale**, with no fresh
minimal-source runtime, memcheck, or performance runs. See [PROGRESS.md](PROGRESS.md)
for the commit breakdown and next steps.

## Approved minimal scope and timing

- Q1 uses original `SUM`/native `MEAN`/`COUNT`; Q5/Q9/Q10 table reads are sequential
  in both formats. All five queries use generic cuDF, not hand-fitted query kernels;
  no `q1_fused` path. Native Parquet benchmarks retain input/intermediate lifetimes
  through output writing.
- Write: 16,777,216-row cuDF chunks → host Arrow → CPU-written CUDA-flat blocks;
  explicit blocks disable byte coalescing and outer layout dictionaries.
- Read: pooled cacheable pinned-host staging → HtoD → GPU decode → retained Arrow
  Device views → one final owning cuDF materialization. These Vortex-internal
  improvements and 8 GiB CUDA pool retention remain. Local files, device 0, flat typed
  columns; **no GPUDirect Storage**.
- `cache=warm/cold`: before every manual cold callback's timed portion, per-file
  `fdatasync` + `POSIX_FADV_DONTNEED` is followed by required `mincore` residency == 0.
  Cold Vortex data uses `O_DIRECT`, metadata stays buffered, and Parquet uses its
  native path. Lower-level storage caches are not flushed; this is OS-page-cache
  coldness, not guaranteed cold media.
- CPU wall timing includes complete reads/import/materialization, query work where
  selected, destruction, and device completion. Writing, checks, and eviction are
  excluded. RMM peaks exclude Vortex allocations.

Default kernel-event suppression is reverted; no event optimization is retained.

## Dataset and correctness

Default `upstream.patch` comparisons use the **original pinned cuDF data generator**,
with identical logical fixtures across formats. The main patch has no generator edits
or `NDSH_DATA_GENERATOR_TEST` target. Original data can produce
empty/degenerate Q6/Q10 and low-SF supplier joins. Exact projection/value and independent
CPU result checks remain even for zero matches, plus synthetic nonempty tests. Explicit
zero match counts disclose degenerate queries, which are **not meaningful full-query
performance evidence**. Q6 checks zero-match `SUM` is NULL and reports revenue as the
string `"NULL"`, not zero. Its CPU reference boundary/sliced/float32 test moved to
`q06.cpp` and gained no-match/empty GPU cases; separating the generator loses no
main-query test coverage. Q9's reference preserves duplicate `partsupp` join multiplicity,
with handwritten matching/unmatched duplicate cases. These GPU cases have not been run
on the minimal source.

Optional `benchmarks/cudf-ndsh/generator-fixes.patch` preserves four independent fixes
and its own regression test across seven files. It is separately reviewable and applies
independently to pinned cuDF; both patch application orders were verified to produce
identical trees. It is not bundled with or automatically applied by `upstream.patch`.
Only the optional patch adds `NDSH_DATA_GENERATOR_TEST`; its changes
affect all NDS-H consumers, including Vortex OFF. After applying it, regenerate both
Parquet and Vortex fixtures and label the dataset as generator-fixed. Restoring the
original generator also requires regenerating both formats; never reuse altered-data
fixtures or historical timings as minimal-patch baselines.

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
The pinned Release build tree is `build/cudf-ndsh-build`. The previous full cuDF build
timed out at 1200 s, reaching 223/635 steps. This checkpoint used compile-only checks,
not CMake regeneration, a full build, relinks, or GPU runs. No broad automatic retry.
See [VALIDATION.md](VALIDATION.md) for current checks, compile artifacts, and the
historical build caveat.

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

Completed at this checkpoint: **17 offline tests**, clang-format for all five query
files, Ruff lint/format, main-patch forward/cached and reverse checks, and optional-patch
forward application to pristine pinned cuDF. Both patch orders produce identical trees.
Compile-only Q1/Q5/Q6/Q9/Q10 checks with Vortex ON/OFF passed **10/10 without warnings**;
the separate generator regression test also compiled, without runtime execution.
Earlier Rust pinned tests/fmt/Clippy were not rerun: no Rust changed this turn.
[VALIDATION.md](VALIDATION.md) records artifacts and historical runtime/memcheck results;
none establishes fresh minimal-source runtime correctness.

**Prior timings are HISTORICAL, not valid baselines for the minimal patch:** they used
altered generator/performance code and event suppression. Source/data changes require
regenerated fixtures and fresh labeled baselines. No new performance runs were made.
The ≥2× end-to-end read **and** query goal across SF1/SF10, warm/cold, is not met;
zero-match timings cannot establish full-query performance. Stabilize nondegenerate
matrices before SF100; a full Q1 query profile is still pending.
