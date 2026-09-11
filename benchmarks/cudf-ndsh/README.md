# cuDF NDS-H Vortex POC

[Plan](../../CUDF_POC_PLAN.md) · [Validation](VALIDATION.md) · [Resume here](PROGRESS.md)

`upstream.patch` adds default-OFF build support, `write_vortex` / `read_vortex`,
and **Q1/Q5/Q6/Q9/Q10 plus read-only Parquet vs Vortex comparisons** using shared
full-table fixtures, scan-level projection, and shared post-read filters. Original
Parquet-pushdown benchmarks remain separate.
Generator fixes separate discount and order-date RNG streams, align prices with rows,
and preserve fractional supplier scale factors. They affect **all** NDS-H consumers,
including Vortex-OFF builds; regenerate fixtures.

- Write: 16,777,216-row cuDF chunks → host Arrow → CPU-written CUDA-flat blocks;
  explicit blocks disable byte coalescing and layout dictionaries.
- Read: pooled pinned-host staging → HtoD → GPU decode → retained Arrow Device views →
  one final owning cuDF materialization; CUDA's pool retains up to 2 GiB between reads.
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

At SF0.01, Q1 has 52,574 matches/four groups; Q5 has 35/four countries; Q6 has
554; Q9 has 538/119 nation-year groups; and Q10 has 535/70 customers. Both GPU
formats agree with independent CPU references; all 28 read/query/engine states pass
memcheck (0 errors). All 28 states also pass pinned cuDF Release SF1 execution, and
20 representative SF10 states pass after profile-guided batching/allocator fixes. Vortex is faster in every
optimized pinned Release matched state. Focused SF10 Nsight traces validate that 16M
blocks amortize per-batch CUDA overhead and prevent high-cardinality layout dictionaries
from fragmenting explicit blocks. This is not full TPC-H conformance. Next: SF100 memory
validation and matched-cache Nsight profiles.
