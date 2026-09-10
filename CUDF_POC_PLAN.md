# cuDF NDS-H Vortex POC

**Goal:** [Benchmark-only upstream POC](https://github.com/NVIDIA/cudf/issues/23877#issuecomment-5457730105)
comparing Vortex with Parquet, with Nsight Systems profiles at **SF100**.

1. **Opt-in build support:** embed CUDA-enabled Vortex; link only Q1/Q5/Q6.
   Implemented; Vortex adds no dependency when disabled.
2. **Read/write adapters:** chunked host Arrow → CPU Vortex writer;
   CUDA scan → Arrow Device import → owning cuDF tables. Implemented with
   ownership/synchronization tests.
3. **Matched comparison:** implemented for Q1/Q5/Q6 and projected reads: identical
   local-file fixtures, scan-level projection, and shared post-read cuDF filters.
   Original Parquet-pushdown benchmarks remain separate.
4. **Finish query coverage before scaling:** Q1/Q5/Q6 pass at SF0.01. Add
   Q9 and Q10 with matched fixtures and result checks at SF0.01 first. Then
   validate pinned runtime and scale SF1 → SF10 → SF100, checking schemas, nulls,
   decimals, batches, results, and Vortex/RMM memory.
5. **Profile and publish:** add NVTX ranges and capture matched-cache SF100 runs.
   Report read latency, size, HtoD traffic/overlap, decode/adapter cost, and peak
   memory. Time the complete read, including import, copies, concatenation, and
   GPU completion; exclude fixture writing. Publish commands and pinned revisions.

## Status and scope

[Patch and setup](benchmarks/cudf-ndsh/README.md) ·
[Validation](benchmarks/cudf-ndsh/VALIDATION.md) ·
[Resume here](benchmarks/cudf-ndsh/PROGRESS.md)

Q1/Q5/Q6 run on GPU at SF0.01; both formats match independent CPU references.
Pinned cuDF is compile-only validated; its full build timed out. Generator fixes
separate discount RNG, align prices, and preserve fractional supplier scale factors;
other correlations remain. Next: Q9 → Q10 at SF0.01, then pinned runtime and
scaling/profiling. Publish Vortex prerequisites and update the pin before submission.

I/O uses pooled pinned-host staging → HtoD → GPU decode, with host metadata;
**not GPUDirect Storage**. Public cuDF/Python APIs, GPU writing, general cuDF
datasources, remote/device-buffer inputs, and full RMM integration are out of scope.
