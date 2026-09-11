# cuDF NDS-H Vortex POC

**Goal:** [Benchmark-only upstream POC](https://github.com/NVIDIA/cudf/issues/23877#issuecomment-5457730105)
comparing Vortex with Parquet, with Nsight Systems profiles at **SF100**.

1. **Opt-in build support:** embed CUDA-enabled Vortex; link only Q1/Q5/Q6/Q9/Q10.
   Implemented; Vortex adds no dependency when disabled.
2. **Read/write adapters:** chunked host Arrow → CPU Vortex writer;
   CUDA scan → Arrow Device import → owning cuDF tables. Implemented with
   ownership/synchronization tests.
3. **Matched comparison:** implemented for Q1/Q5/Q6/Q9/Q10 and projected reads: identical
   local-file fixtures, scan-level projection, and shared post-read cuDF filters.
   Original Parquet-pushdown benchmarks remain separate.
4. **Finish query coverage before scaling:** Q1/Q5/Q6/Q9/Q10 pass at SF0.01 and
   pinned cuDF Release SF1/SF10 with matched fixtures and result checks. SF10 profiling
   exposed per-batch CUDA API/event overhead and high-cardinality layout dictionaries
   fragmenting explicit blocks. The benchmark now uses 16,777,216-row blocks, disables
   layout dictionaries for explicit CUDA blocks, retains 2 GiB in CUDA's pool, and performs
   one final owning cuDF materialization. Next scale to SF100 while checking schemas,
   nulls, decimals, batches, results, and Vortex/RMM memory.
5. **Profile and publish:** adapter NVTX ranges and focused SF10 profiles are implemented;
   capture matched-cache SF100 runs.
   Report read latency, size, HtoD traffic/overlap, decode/adapter cost, and peak
   memory. Time the complete read, including import, copies, concatenation, and
   GPU completion; exclude fixture writing. Publish commands and pinned revisions.

## Status and scope

[Patch and setup](benchmarks/cudf-ndsh/README.md) ·
[Validation](benchmarks/cudf-ndsh/VALIDATION.md) ·
[Resume here](benchmarks/cudf-ndsh/PROGRESS.md)

Q1/Q5/Q6/Q9/Q10 run on GPU at SF0.01 and pinned cuDF Release SF1/SF10; both
formats match independent CPU references. All 28 SF0.01 states are memcheck-clean, all
28 SF1 states pass normal execution, and all 20 selected SF10 states pass. Vortex is
faster in every optimized pinned Release comparison. Focused Nsight traces validate the
batching and allocator changes; SF100 profiling remains. Generator fixes separate discount
and order-date RNG streams, align prices, and preserve fractional supplier scale factors;
other correlations remain. Next: SF100 and memory accounting, then publish Vortex
prerequisites, update the pin, and prepare the submission.

I/O uses pooled pinned-host staging → HtoD → GPU decode, with host metadata;
**not GPUDirect Storage**. Public cuDF/Python APIs, GPU writing, general cuDF
datasources, remote/device-buffer inputs, and full RMM integration are out of scope.
