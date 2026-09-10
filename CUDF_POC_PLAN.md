# cuDF NDS-H Vortex POC

**Goal:** [Benchmark-only upstream POC](https://github.com/NVIDIA/cudf/issues/23877#issuecomment-5457730105)
comparing Vortex with Parquet, with Nsight Systems profiles at **SF100**.

1. **Opt-in build support:** embed CUDA-enabled Vortex; link only Q1/Q6.
   Implemented; Vortex adds no dependency when disabled.
2. **Read/write adapters:** chunked host Arrow → CPU Vortex writer;
   CUDA scan → Arrow Device import → owning cuDF tables. Implemented with
   ownership/synchronization tests.
3. **Matched comparison:** implemented for Q6 and projected reads: identical
   local-file fixtures, scan-level projection, and shared post-read cuDF filters.
   The original Parquet-pushdown benchmark remains separate; Q1 is still Parquet-only.
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

Q6/read SF0.01 and 15 adapter tests pass on prebuilt cuDF 26.08, including
Compute Sanitizer. Pinned cuDF is compile-only validated; its full build timed out.
A separate discount RNG seed now gives nonempty Q6 data; both formats are checked
against a CPU reference of the same fixture. Other generator issues, including
price-row ordering, remain. Next: pinned-runtime validation, Q1, and SF100 profiling.
Publish Vortex prerequisites and update the pin before upstream submission.

I/O uses pooled pinned-host staging → HtoD → GPU decode, with host metadata;
**not GPUDirect Storage**. Public cuDF/Python APIs, GPU writing, general cuDF
datasources, remote/device-buffer inputs, and full RMM integration are out of scope.
