# cuDF NDS-H Vortex POC

[Plan](../../CUDF_POC_PLAN.md) · [Validation](VALIDATION.md) · [Resume here](PROGRESS.md)

`upstream.patch` adds default-OFF build support, `write_vortex` / `read_vortex`,
and **Q1/Q5/Q6/read-only Parquet vs Vortex comparisons** using shared full-table
fixtures, scan-level projection, and shared post-read filters. Original Parquet-pushdown
benchmarks remain separate; Q9/Q10 are still Parquet-only.
Generator fixes separate discount's RNG seed, align prices with rows, and preserve
fractional supplier scale factors. They affect **all** NDS-H consumers, including
Vortex-OFF builds; regenerate fixtures.

- Write: chunked cuDF → host Arrow → CPU-written, CUDA-readable Vortex file.
- Read: pinned-host staging → HtoD → GPU decode → owning cuDF batches → concatenation.
- Scope: local files, device 0, one calling thread, flat typed columns.

## Apply

From the Vortex root, for a **fresh** cuDF checkout:

```sh
git clone https://github.com/NVIDIA/cudf.git build/cudf-ndsh-src
git -C build/cudf-ndsh-src checkout --detach 5339497a1a17d799687cbf189fb113411fb015ca
git -C build/cudf-ndsh-src apply --check ../../benchmarks/cudf-ndsh/upstream.patch
git -C build/cudf-ndsh-src apply ../../benchmarks/cudf-ndsh/upstream.patch
```

The development checkout is already patched; `/home/ubuntu/cudf` is untouched.
Build instructions are in the patched `cpp/benchmarks/ndsh/VORTEX.md`.

**Local Vortex sources are required:** the retained base pin
`bffdca1109e99e6957ea2fc18f4a7809c88e0a0c` lacks the CUDA-layout edition,
device decimal slicing, bitmap alignment/padding, dictionary export, and projected scan API.
Publish these prerequisites and update the immutable pin before removing the gate.

## Checks

```sh
python3 -B -m unittest discover -s benchmarks/cudf-ndsh -v
python3 -B -m unittest discover -s vortex-ffi/cmake/tests -v
ruff check benchmarks/cudf-ndsh/test_build_integration.py
ruff format --check benchmarks/cudf-ndsh/test_build_integration.py
```

At SF0.01, Q1 has 44,973 matches/four groups, Q5 has 35 matches/four countries,
and Q6 has 563 matches. Both GPU formats agree with independent CPU references;
all 12 read/query states pass memcheck (0 errors). Pinned compilation passes;
the full cuDF build timed out. Debug-cuDF timings are supplemental, not full TPC-H
conformance. Next: Q9/Q10 at SF0.01, then pinned runtime, scaling, and Nsight.
