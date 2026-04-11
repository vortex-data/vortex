# vector-search-bench

Brute-force cosine-similarity benchmark for Vortex on public VectorDBBench
embedding corpora.

## What it measures

For each `(dataset, format)` pair, the benchmark records:

1. **`nbytes`** — in-memory footprint of the variant's array tree, in bytes.
   Reporting the in-memory `.nbytes()` instead of an on-disk file size is
   deliberate: the Vortex default write path runs BtrBlocks on every tree
   regardless of whether it's already compressed, so "on-disk size" would
   collapse `vortex-uncompressed` and `vortex-default` to the same bytes
   even though their in-memory trees are different. The `nbytes()`
   number is consistent with what the *compute* measurements actually
   operate on.
   - The `handrolled` baseline reports the canonical parquet file size
     on disk — that's the only encoded representation it has.
2. **Compress time** — wall time to build the variant tree from the
   materialized uncompressed source. ~0 for `vortex-uncompressed` (identity),
   meaningful for the two compressed variants.
3. **Decompress time** — wall time to execute the variant tree all the way
   back into a canonical `FixedSizeListArray<f32>` with a materialized f32
   element buffer. For `vortex-uncompressed` this is a no-op; for
   `vortex-default` it includes ALP-RD bit-unpacking; for
   `vortex-turboquant` it includes the inverse SORF rotation and
   dictionary lookup.
4. **Cosine-similarity time** — `CosineSimilarity(data, const_query)`
   executed to a materialized f32 array.
5. **Cosine-filter time** — `Binary(Gt, [CosineSimilarity, threshold])`
   executed to a `BoolArray`.
6. **Recall@10** (TurboQuant only) — the fraction of the exact top-10
   nearest neighbours that TurboQuant recovers, using the uncompressed
   Vortex scan as local ground truth.

Before any timing starts, the benchmark runs a **correctness verification
pass**: cosine scores for a single query are computed against every
variant and compared to the uncompressed baseline. Lossless variants must
match within `1e-4` max-abs-diff; TurboQuant must stay within `0.2`. A
mismatch bails the run — you cannot publish throughput numbers for a
variant that returns wrong answers.

## Formats

- `handrolled` — Hand-rolled Rust scalar cosine loop over a flat
  `Vec<f32>` that was decoded from the canonical parquet file via
  `parquet-rs` / `arrow-rs`. The **decompress** phase does the parquet
  read, downcasts to `Float32Array`, and memcpies into a plain `Vec<f32>`.
  The **compute** phase is a plain scalar loop over `&[f32]` — no Arrow
  compute kernels, no scalar-function dispatch, no SIMD annotations.

  This is a **compute-cost floor**, not a realistic parquet-on-DBMS
  baseline. It answers the question "what's the minimum cost you could
  get away with if you wrote a vector-search scan by hand with no query
  engine?" Real parquet users would pay substantially more (DuckDB
  `list_cosine_similarity`, DataFusion with a vector UDF, etc.) —
  adding those as additional baselines is a natural v2 direction.
- `vortex-uncompressed` — Raw `Vector<dim, f32>` extension array, no
  encoding-level compression applied.
- `vortex-default` — `BtrBlocksCompressor::default()` applied to the FSL
  storage child. On float vectors this typically finds ~15% lossless
  savings via ALP-RD (mantissa/exponent split + bitpacking).
- `vortex-turboquant` — The full
  `L2Denorm(SorfTransform(FSL(Dict(codes, centroids))), norms)` pipeline.
  Lossy; recall@10 is reported alongside throughput. At the default 8-bit
  config this typically gives ~3× storage reduction at >90% top-10
  recall.

## Datasets

The smallest built-in dataset is **Cohere-100K** (`cohere-small`): 100K
rows × 768 dims, cosine metric, ~150 MB zstd-parquet. It's the smallest
VectorDBBench-supplied corpus that still exercises every encoding path.
Larger variants (`cohere-medium`, `openai-small`, `openai-medium`,
`bioasq-medium`, `glove-medium`) are wired up for local / on-demand
experiments; see `vortex-bench/src/vector_dataset.rs` for the full list.

The upstream URL for Cohere-100K is
`https://assets.zilliz.com/benchmark/cohere_small_100k/train.parquet`.
The public Zilliz bucket is anonymous-readable so the code can hit it
directly.

## Running locally

```bash
cargo run -p vector-search-bench --release -- \
    --datasets cohere-small \
    --formats handrolled,vortex-uncompressed,vortex-default,vortex-turboquant \
    --iterations 5 \
    -d table
```

The first run downloads the parquet file into
`vortex-bench/data/cohere-small/cohere-small.parquet` and caches it
idempotently for subsequent runs.

### Running without network access

The `gen_synthetic_dataset` helper writes a VectorDBBench-shape parquet
file (`id: int64` + `emb: list<float32>`, zstd-compressed) at any path.
Use it to populate the dataset cache so the benchmark's idempotent
download step skips the HTTP fetch:

```bash
cargo run -p vector-search-bench --bin gen_synthetic_dataset --release -- \
    --num-rows 5000 \
    --dim 768 \
    --out vortex-bench/data/cohere-small/cohere-small.parquet
```

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
