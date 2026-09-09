# cuDF NDS-H Vortex POC

[Plan](../../CUDF_POC_PLAN.md) · [Validation](VALIDATION.md)

`upstream.patch` adds default-OFF build support and benchmark-local
`write_vortex` / `read_vortex` adapters. **Q1/Q6 still read Parquet.**

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
device decimal slicing, bitmap alignment/padding, and dictionary-export fixes.
Publish these prerequisites and update the immutable pin before removing the gate.

## Checks

```sh
python3 -B -m unittest discover -s benchmarks/cudf-ndsh -v
python3 -B -m unittest discover -s vortex-ffi/cmake/tests -v
ruff check benchmarks/cudf-ndsh/test_build_integration.py
ruff format --check benchmarks/cudf-ndsh/test_build_integration.py
```

12 GPU adapter tests pass on prebuilt cuDF 26.08; Compute Sanitizer reports
0 errors. Current pinned cuDF compiles the adapter, but its full library build
is blocked. Q1/Q6 execution, SF100, and performance remain unvalidated.
