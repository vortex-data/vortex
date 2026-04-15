# vector-search-bench

On-disk cosine-similarity scan benchmark for Vortex on public VectorDBBench
embedding corpora. The benchmark writes one `.vortex` file per train shard per
flavor and then issues filtered scans against the resulting files, so the
numbers reflect realistic out-of-memory workloads — not in-memory `ArrayRef`
manipulation.

## Quick start

```bash
cargo run -p vector-search-bench --release -- \
    --dataset cohere-small-100k \
    --flavors vortex-uncompressed,vortex-turboquant,handrolled \
    --iterations 3 \
    --threshold 0.8
```

The first run downloads the parquet shards into
`vortex-bench/data/vector-search/<dataset>/<layout>/train/...`, ingests them
into per-flavor `.vortex` files in sibling directories, samples a query row
from `test.parquet`, and runs the timed scan loop.

A datasets that publishes more than one layout (e.g. `cohere-large-10m`
hosts both `partitioned` and `partitioned-shuffled`) requires `--layout` to
disambiguate.

## What it measures

Per `(dataset, flavor)`:

| Metric              | What it is                                              |
|---------------------|---------------------------------------------------------|
| compress wall       | Sum of per-shard write time (parquet → `.vortex`).      |
| input bytes         | Sum of input parquet shard sizes.                       |
| output bytes        | Sum of output `.vortex` shard sizes.                    |
| compression ratio   | input bytes / output bytes.                             |
| scan wall (best)    | Best-of-N wall-clock for the per-iteration scan.        |
| scan wall (median)  | Median wall-clock for the per-iteration scan.           |
| matches             | Rows that survived `cosine(emb, query) > threshold`.    |
| rows scanned        | Total rows in the `.vortex` files (sanity check).       |
| rows / sec          | rows scanned / scan wall (best).                        |
| recall@K (mean/p05) | Only emitted when `--recall` is passed (lossy flavors). |

## Flavors

- **`vortex-uncompressed`** — `BtrBlocksCompressorBuilder::empty()`. Vortex
  framing with no compression schemes registered, so the `emb` column lands
  as canonical `FixedSizeList<f32>` on disk. Lossless ceiling on the size
  axis.
- **`vortex-turboquant`** — `BtrBlocksCompressorBuilder::empty().with_turboquant()`.
  Only the TurboQuant scheme is registered, so the `emb` column ends up
  wrapped as `L2Denorm(SorfTransform(FixedSizeList(Dict)))`. Lossy; significant
  size win.
- **`handrolled`** — Sequential parquet scan + 4-way unrolled scalar cosine
  loop over a flat `Vec<f32>` (decoded via `parquet-rs` / `arrow-rs`). This
  is a *compute-cost floor*, not a realistic parquet-on-DBMS baseline. Real
  parquet users would pay substantially more (DuckDB
  `list_cosine_similarity`, DataFusion with a vector UDF, etc.) — adding
  those as additional baselines is a natural future direction.

The benchmark always operates in `f32`. The ingest pipeline casts `f64`
sources (e.g. OpenAI corpora) to `f32` once at write time, so all downstream
code is uniformly `f32`.

## Datasets

All 16 published VectorDBBench corpora are wired into the catalog, with
explicit declarations of which train-split layouts upstream actually hosts.
See `vortex-bench/src/vector_dataset/catalog.rs` for the full table. CLI
helpfully lists choices when run with `--help`.

| Dataset            | dim  | rows | layouts                                     |
|--------------------|------|------|---------------------------------------------|
| cohere-small-100k  | 768  | 100K | single, single-shuffled                     |
| cohere-medium-1m   | 768  | 1M   | single, single-shuffled                     |
| cohere-large-10m   | 768  | 10M  | partitioned (10), partitioned-shuffled (10) |
| openai-small-50k   | 1536 | 50K  | single, single-shuffled                     |
| openai-medium-500k | 1536 | 500K | single, single-shuffled                     |
| openai-large-5m    | 1536 | 5M   | partitioned (10), partitioned-shuffled (10) |
| bioasq-medium-1m   | 1024 | 1M   | single-shuffled                             |
| bioasq-large-10m   | 1024 | 10M  | partitioned-shuffled (10)                   |
| glove-{small,medium}, gist-{small,medium} | varies | varies | single only |
| sift-small-500k    | 128  | 500K | single                                      |
| sift-medium-5m     | 128  | 5M   | single                                      |
| sift-large-50m     | 128  | 50M  | partitioned (50)                            |
| laion-large-100m   | 768  | 100M | partitioned (100)                           |

## Recall@K

Pass `--recall --recall-k 10 --recall-queries 100` to measure recall against
`neighbors.parquet`. The lossless `vortex-uncompressed` flavor is skipped
because its recall is 1.0 by construction; only `vortex-turboquant` is
measured. Datasets that don't host `neighbors.parquet` (sift, glove, gist)
bail out when `--recall` is set.

## Future work

1. Native `f64` flavor — drop the prepare-time downcast for OpenAI datasets.
2. `--decompress-only` mode — project + drain, no filter — for pure decode
   timing.
3. Filtered scans via `scalar_labels` (already projected through the ingest
   pipeline; the `neighbors_int_*p.parquet` and `neighbors_labels_*.parquet`
   ground-truth files exist for verification).
4. DuckDB / DataFusion parquet baselines — real engines, not just hand-rolled.
5. MSE-vs-ground-truth correctness mode (catches "right top-K, wrong scores").
6. Promote the cosine-filter expression helpers from `expression.rs` into
   `vortex-tensor::vector_search` if a second caller materializes.
