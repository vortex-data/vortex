# cuDF integration: benchmark-only POC

**Goal:** Submit `[POC] Add Vortex reader to libcudf NDS-H benchmarks`, with a
Parquet comparison and Nsight Systems profiles at **SF100**.

- [Upstream proposal and maintainer request](https://github.com/NVIDIA/cudf/issues/23877#issuecomment-5457730105)
- [NDS-H benchmarks](https://github.com/NVIDIA/cudf/tree/main/cpp/benchmarks/ndsh)
- [Vortex CMake integration](lang/cpp/CMakeLists.txt)
- [Existing CUDA C API](vortex-cuda/ffi/cinclude/vortex_cuda.h)

## 1. Add opt-in build integration

- Pin Vortex and embed its existing CMake targets with `VORTEX_ENABLE_CUDA=ON`.
- Link only the participating NDS-H benchmarks; leave default cuDF builds unchanged.
- Validate the RAPIDS toolchain, CUDA architecture, and CUB/nvCOMP runtime dependencies first.

Build scaffolding and offline tests are available in [benchmarks/cudf-ndsh](benchmarks/cudf-ndsh/README.md).
See its validation notes for completed checks and the remaining full cuDF build validation.

## 2. Implement benchmark-local `write_vortex` / `read_vortex`

- **Write:** generated cuDF table → chunked host Arrow export → existing CUDA-compatible
  Vortex writer. This is **CPU writing**, not GPU compression.
- **Read:** existing CUDA file scan → `ArrowDeviceArrayStream` →
  `cudf::from_arrow_device` → owning cuDF table.
- Handle Arrow ownership and cross-stream synchronization explicitly. Include import,
  copies, and concatenation in read timing.

## 3. Make the comparison fair

- Generate identical data and serialize it into both formats.
- Initially use **local files for both**: existing NDS-H Parquet inputs are host buffers,
  whereas Vortex's dedicated CUDA reader currently accepts file paths.
- Add column projection to the Vortex CUDA scan API.
- Start with equivalent projections and post-read cuDF filtering; retain Parquet pushdown
  as a separately labeled baseline.

## 4. Validate and scale

- Add a read-only comparison, then integrate **Q6**, followed by **Q1**.
- Progress through SF0.01 → SF1 → SF10 → **SF100**.
- Verify values, schemas, decimals, nulls, batch boundaries, and query results.
- Bound staging/intermediate memory and measure Vortex allocations separately from RMM.

## 5. Profile and publish

- Add NVTX ranges for reading, import/materialization, query execution, and writing.
- Capture SF100 Nsight Systems profiles with matched hardware and cache policy.
- Report wall-clock latency, file size, HtoD traffic/overlap, decode time, adapter overhead,
  and peak memory.
- Keep fixture generation outside read timing; ensure timing includes completion of all
  contributing Vortex and cuDF GPU work.
- Publish reproducible commands and pinned revisions.

## I/O scope

Vortex currently reads compressed file data into pooled pinned host buffers, transfers it
HtoD, and decodes on GPU; metadata stays on host. This is **not GPUDirect Storage**.

## Out of scope

Public cuDF/Python APIs, GPU writing, general cuDF datasource integration,
remote/device-buffer inputs, and full RMM integration.

## Deliverables

A small Vortex prerequisite PR if needed, followed by the cuDF benchmark POC with
correctness results and SF100 profiles.
