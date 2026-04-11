# vector-search-bench

Brute-force cosine-similarity benchmark for Vortex on public VectorDBBench
embedding corpora.

## What it measures

For each `(dataset, format)` pair, the benchmark records four numbers:

1. **Size** — compressed storage footprint in bytes. For the Vortex variants
   that round-trip through `.vortex` files today (uncompressed & BtrBlocks
   default) this is the real on-disk size. For `vortex-turboquant` it is
   the in-memory `.nbytes()` footprint, because the `L2Denorm` scalar-fn
   array does not yet have a concrete `serialize_metadata` implementation.
2. **Full-scan decode time** — wall time to materialize the whole `Vector<dim, f32>`
   column into a `FixedSizeListArray<f32>`.
3. **Cosine-similarity execute time** — wall time for
   `CosineSimilarity(data, const_query)` executed to a materialized f32 array.
4. **Cosine-filter execute time** — wall time for the full
   `Binary(Gt, [CosineSimilarity, threshold])` expression tree executed to
   a `BoolArray`.

The TurboQuant variant additionally reports **Recall@10** against the
uncompressed Vortex scan as local ground truth. Lossless variants are trivially
1.0 so they are not re-measured.

## Formats

- `parquet` — Parquet file read via `parquet::arrow` into an Arrow
  `FixedSizeListArray<f32>`, then a hand-rolled Rust cosine loop. This is the
  "what you'd do without Vortex" external floor.
- `vortex-uncompressed` — Raw `Vector<dim, f32>` extension array, no
  encoding-level compression applied.
- `vortex-default` — `BtrBlocksCompressor::default()` applied to the FSL
  storage child. Generic lossless Vortex compression for float vectors.
- `vortex-turboquant` — The full
  `L2Denorm(SorfTransform(FSL(Dict(codes, centroids))), norms)` pipeline.
  Lossy; recall@10 is reported alongside throughput.

## Datasets

The first dataset wired up is **Cohere-100K** (`cohere-small`): 100K rows ×
768 dims, cosine metric, ~150 MB zstd-parquet. This is the smallest
VectorDBBench-supplied embedding corpus and sits comfortably inside a CI
time / bandwidth budget.

The upstream URL is
`https://assets.zilliz.com/benchmark/cohere_small_100k/train.parquet`. The
public Zilliz bucket is anonymous-readable so the code _can_ hit it directly.

## Running locally

```bash
cargo run -p vector-search-bench --release -- \
    --datasets cohere-small \
    --formats parquet,vortex-uncompressed,vortex-default,vortex-turboquant \
    --iterations 5 \
    -d table
```

The first run downloads the parquet file into
`vortex-bench/data/cohere-small/cohere-small.parquet` and caches it
idempotently for subsequent runs.

## CI note: dataset mirror

CI runs after every develop-branch merge. Hitting `assets.zilliz.com`
from every merge would create recurring egress traffic on a third-party
bucket — the same courtesy reason `RPlace` / `AirQuality` are excluded
from CI in `compress-bench`.

Before enabling the `vector-search-bench` entry in `.github/workflows/bench.yml`
on a fork, either:

1. **Mirror the file into an internal bucket** and swap the URL in
   `vortex-bench/src/vector_dataset.rs::VectorDataset::parquet_url`, or
2. **Accept the upstream egress cost** and leave the URL as-is.

The mirror step is a one-off `aws s3 cp` and is documented here rather
than automated in the build because the destination bucket is
organization-specific.
