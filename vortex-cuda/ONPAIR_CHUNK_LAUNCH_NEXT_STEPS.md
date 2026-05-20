# OnPair Chunk Launch Next Steps

The current real-data OnPair GPU bench launches a kernel per OnPair chunk for
each timed iteration. That is useful for isolated kernel comparison, but it can
make small chunks look worse than they are because CPU launch overhead becomes a
large part of the measured time. This is especially visible for 1 MiB chunks,
where a column can require hundreds of launches per pass.

Before treating any new launch path as a throughput result, add an optional
correctness check that copies output bytes back and compares them with CPU
decode or with a known-good GPU baseline. The current `onpair_shmem_2tpt`
algorithm looks structurally correct, but the benchmark path times it without
validating the materialized output.

## 1. CUDA Graph Replay

Keep the current kernels and launch shape, but record the chunk-launch sequence
once.

Current:

```text
for iter:
  for chunk:
    launch onpair_shmem_2tpt(chunk)
```

Graph version:

```text
build graph:
  for chunk:
    add kernel node onpair_shmem_2tpt(chunk)

instantiate graph

for iter:
  launch graph once
```

What changes:

- No CUDA kernel code changes.
- Host timing path changes in `time_kernel_variant`.
- For each selected variant, create a graph containing one kernel node per
  OnPair chunk.
- Warm up by launching the graph.
- Time repeated graph launches.

Expected benefit:

- Reduces CPU launch overhead.
- Good quick benchmark fix.
- Especially helps 1 MiB chunks with roughly hundreds of launches per pass.
- Directly answers whether the current per-chunk launch loop is hiding kernel
  throughput. In the real-data runs, 1 MiB `l_comment` has 954 OnPair chunks,
  so one timed pass launches 954 kernels per variant. The 100 MiB version has
  only 10 chunks and is much faster, but still launches one kernel per chunk.
- Keeps the same device work and same per-chunk buffers, so any speedup is
  attributable to host launch submission and CUDA scheduling overhead rather
  than a changed decode algorithm.

Limit:

- Still schedules one kernel node per OnPair chunk on device per pass.
- Does not improve per-chunk occupancy or combine small work.
- If device-side scheduling dominates, graph replay will help less than a real
  batched kernel.

Measurement plan:

1. Add an optional graph timing path for a single selected kernel variant.
2. Compare normal launches vs graph replay on the same staged chunks:
   - `l_comment`, `fineweb/text`, `ps_comment`, `o_comment`,
     `book-reviews/text`
   - 1 MiB, 10 MiB, and 100 MiB chunk variants where present
3. Report:
   - chunks per pass
   - decoded bytes per pass
   - normal launch ms / GiB/s
   - graph replay ms / GiB/s
   - graph speedup
4. If graph replay mostly closes the gap to the expected throughput, keep the
   kernels and optimize launch submission first. If graph replay is only a
   small win, prioritize the batched kernel because the bottleneck is more
   likely device-side per-chunk work, dictionary access, or variable-length
   compaction.

## 2. Batched Kernel

Add a new kernel that decodes all OnPair chunks in one launch.

Current:

```text
chunk 0 -> launch kernel
chunk 1 -> launch kernel
...
chunk N -> launch kernel
```

Batched:

```text
one launch:
  grid warps cover all chunks
  each warp decodes one token-block inside one OnPair chunk
```

Descriptor:

```cpp
struct OnPairChunkDesc {
  const uint16_t* codes;
  const uint64_t* chunk_offsets;
  const uint8_t* dict;
  const uint8_t* lens;
  uint8_t* output;
  uint64_t total_tokens;
  uint64_t num_token_chunks;
};
```

Host prepares:

```text
descs[0..num_chunks]
prefix_token_chunks[0..num_chunks + 1]
```

Kernel maps global warp:

```cpp
uint64_t global_warp =
    blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);

// find chunk_idx such that:
// prefix_token_chunks[chunk_idx] <= global_warp < prefix_token_chunks[chunk_idx + 1]

uint64_t local_token_chunk =
    global_warp - prefix_token_chunks[chunk_idx];

decode descs[chunk_idx], local_token_chunk
```

What changes:

- Add `onpair_shmem_batched`, likely one per tuned variant:
  - `onpair_shmem_batched`
  - `onpair_shmem_2tpt_batched`
  - `onpair_shmem_4tpt_batched`
  - later `s8`, `s4`, and constant-length variants
- Add host staging for descriptor arrays.
- Add timing path that launches one batched kernel per iteration.

Expected benefit:

- Turns `chunks * iters` launches into `iters` launches.
- For a column with 954 chunks and 10 timed iterations, 9540 timed launches
  becomes 10.
- Much better measurement of true GPU throughput.

Limit:

- Needs device-side chunk mapping.
- Binary search over `prefix_token_chunks` per warp could add overhead.
- That overhead can be reduced later with one block-group per file chunk or a
  precomputed warp-to-chunk map.
- More code and more validation needed than CUDA Graphs.

## Recommendation

Implement CUDA Graph replay first if the goal is a quick answer on launch
overhead.

Implement the batched kernel if the goal is for the benchmark to represent real
GPU decode throughput rather than launch overhead.
