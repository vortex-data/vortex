# cuDF NDS-H Vortex POC

**Goal:** [Benchmark-only upstream POC](https://github.com/NVIDIA/cudf/issues/23877#issuecomment-5457730105)
comparing Vortex with Parquet, with Nsight Systems profiles at **SF100**.

1. **Opt-in build support:** embed CUDA-enabled Vortex; link only Q1/Q6.
   Implemented; default cuDF builds remain unchanged.
2. **Read/write adapters:** chunked host Arrow → CPU Vortex writer;
   CUDA scan → Arrow Device import → owning cuDF tables. Implemented with
   ownership/synchronization tests. Q1/Q6 still read Parquet.
3. **Matched comparison:** generate identical data, use local files for both
   formats, add Vortex projection, and apply equivalent post-read cuDF filters.
   Report Parquet pushdown separately.
4. **Validate and scale:** read-only comparison → Q6 → Q1;
   SF0.01 → SF1 → SF10 → SF100. Check schemas, values, nulls, decimals, batch
   boundaries, and query results. Bound intermediates; track Vortex and RMM memory.
5. **Profile and publish:** add NVTX ranges and capture matched-cache SF100 runs.
   Report read latency, size, HtoD traffic/overlap, decode/adapter cost, and peak
   memory. Time the complete read, including import, copies, concatenation, and
   GPU completion; exclude fixture writing. Publish commands and pinned revisions.

## Status and scope

[Patch and setup](benchmarks/cudf-ndsh/README.md) ·
[Validation](benchmarks/cudf-ndsh/VALIDATION.md)

The adapter passes GPU tests against cuDF 26.08; current pinned cuDF is
compile-only validated. Publish the local Vortex prerequisites and update the
pin before submitting the upstream POC.

I/O uses pooled pinned-host staging → HtoD → GPU decode, with host metadata;
**not GPUDirect Storage**. Public cuDF/Python APIs, GPU writing, general cuDF
datasources, remote/device-buffer inputs, and full RMM integration are out of scope.
