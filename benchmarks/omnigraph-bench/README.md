# omnigraph-bench

How would [Omnigraph](https://github.com/ModernRelay/omnigraph) (an object-storage graph database
with git-style branching) fare on Vortex instead of Lance? Omnigraph keeps each node type and edge
type of a graph in its own Lance dataset (`<graph>.omni/nodes/<id>`, `<graph>.omni/edges/<id>`).
This tool takes those tables **exactly as Omnigraph writes them** and compares them with the same
rows written as **Vortex with its default write options**:

- `convert` writes each table as Vortex and reports on-disk sizes. For reference it also writes
  Parquet (zstd level 3) and a Lance copy with zstd forced on every column.
- `query` runs Omnigraph's own benchmark queries (`benchmarks/workloads/e2e_query_*.gqt` and
  `e2e_traversal_*.gqt`), translated from GQ to SQL, in DataFusion over Omnigraph's Lance tables,
  the zstd Lance copy and Vortex. It checks that every format returns the same rows before it
  records timings.

This compares storage formats. Omnigraph's engine still runs on Lance: its versioning,
branches, transactions and its vector and full-text indexes are all Lance features.

## Reproduce

```bash
# Omnigraph 0.15 (needs protoc)
git clone https://github.com/ModernRelay/omnigraph && cd omnigraph
cargo build --release -p omnigraph-cli
OG=$PWD/target/release/omnigraph

# The schema of Omnigraph's e2e_query_shapes fixture, scaled to 200k people / 1.92M edges
python3 benchmarks/omnigraph-bench/data/generate.py --people 200000 --out graph.jsonl
$OG init --schema benchmarks/omnigraph-bench/data/schema.pg graph.omni

# Default load limits: keyed loads take at most 8,192 entities and 32 MiB, so load in chunks.
mkdir chunks
grep '"type":"Person"' graph.jsonl | split -l 8000 -d -a 4 - chunks/p_
grep '"edge":"Knows"'  graph.jsonl | split -l 8000 -d -a 4 - chunks/e_
mode=overwrite; for f in chunks/p_* chunks/e_*; do
  $OG load --data $f --mode $mode --target graph.omni; mode=append
done

# Omnigraph's own maintenance, with its defaults: compact fragments, drop old versions.
$OG optimize --target graph.omni
$OG cleanup --keep 1 --older-than 0s --confirm --target graph.omni

cargo run --release -p omnigraph-bench -- convert \
  --table person=graph.omni/nodes/<id> --table knows=graph.omni/edges/<id> --out out
cargo run --release -p omnigraph-bench -- query --data out --iterations 5
```

## Results

200,000 `Person` nodes (`name @key, age, team, bio, embedding: Vector(32)`) and 1,919,160 `Knows`
edges. Measured on a 4-vCPU, 16 GB cloud VM, with files on local disk and warm in the page cache.

### Size

| table | rows | Omnigraph table dir (data + manifests + indexes) | Omnigraph Lance data files | Lance + forced zstd | Parquet zstd(3) | Vortex (default) | Vortex vs Omnigraph Lance |
|---|---:|---:|---:|---:|---:|---:|---:|
| person | 200,000 | 44.02 MiB | 32.49 MiB | 33.30 MiB | 29.14 MiB | 25.76 MiB | 1.26x smaller |
| knows | 1,919,160 | 110.76 MiB | 54.13 MiB | 42.35 MiB | 29.30 MiB | 44.72 MiB | 1.21x smaller |
| **total** | | 154.78 MiB | 86.62 MiB | 75.65 MiB | 58.44 MiB | 70.48 MiB | 1.23x smaller |

`knows` is mostly Omnigraph's generated 26-character `__id` strings plus the endpoint keys.
Parquet with zstd compresses that table better than default Vortex does.

### Runtime

Median of 5 runs after one warm-up, DataFusion, same SQL for every format.

| Omnigraph query | rows | Omnigraph Lance (ms) | Lance + zstd (ms) | Vortex default (ms) | Vortex speedup vs Omnigraph Lance |
|---|---:|---:|---:|---:|---:|
| count_all | 1 | 1.02 | 0.98 | 1.24 | 0.83x |
| scan | 200,000 | 5.42 | 8.35 | 5.34 | 1.01x |
| wide_scan | 200,000 | 33.35 | 68.04 | 16.85 | 1.98x |
| lookup | 1 | 7.50 | 11.37 | 4.29 | 1.75x |
| filter | 24,539 | 11.39 | 13.12 | 4.60 | 2.48x |
| top_people | 20 | 11.12 | 14.56 | 5.43 | 2.05x |
| grouped | 64 | 8.80 | 9.83 | 5.80 | 1.52x |
| friends | 1,919,160 | 268.60 | 273.17 | 169.10 | 1.59x |
| destination_projection | 1,919,160 | 248.69 | 301.55 | 203.23 | 1.22x |
| filtered_friends | 236,462 | 168.95 | 189.98 | 62.74 | 2.69x |
| grouped_fanout | 64 | 297.45 | 310.31 | 170.15 | 1.75x |
| no_friends | 0 | 198.36 | 230.72 | 93.65 | 2.12x |
| destination_search | 12,588 | 161.88 | 239.21 | 62.61 | 2.59x |
| hop1 | 1,919,160 | 264.83 | 282.76 | 146.20 | 1.81x |
| hop2 (count) | 1 | 1029.73 | 1073.22 | 943.83 | 1.09x |
| hop3 (count) | 1 | 6604.16 | 6522.03 | 6947.65 | 0.95x |
| selective | 26 | 150.76 | 218.51 | 60.37 | 2.50x |
| **geomean** | | | | | **1.66x** |

For scale, some of the same queries through Omnigraph's own engine and CLI on the same graph
(best of 5, including process start and JSON output): count_all 34 ms, lookup 47 ms, filter 112 ms,
top_people 105 ms, grouped 149 ms, filtered_friends 518 ms, selective 63 ms.

### Caveats

- Lance 14 pins DataFusion 54 and Vortex uses DataFusion 55, so the two sides run on different
  DataFusion versions. Some of the gap may come from DataFusion itself.
- The SQL translation follows Omnigraph's semantics: an edge pattern requires both endpoint nodes
  to exist. It does not use Omnigraph's planner or its CSR traversal index.
- The `nearest` and `rrf` workloads need Lance vector indexes and are left out.
  `destination_search` runs as a `LIKE` substring filter on every format, not as a full-text
  index search.
- Full-graph `hop2` and `hop3` return `count(*)`: the full path lists would not fit in memory.
  These two are dominated by join work, not by the scan.
- `no_friends` returns 0 rows because the generator gives every person at least one outgoing edge.
