<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Indexed vector SQL benchmark

Run both engines against the same seeded dataset and 21 query vectors.

```sh
cargo run --profile bench -p lance-bench --bin vector-bench -- --large
```

Q1 retrieves ten nearest embeddings by cosine distance. Q4 does the same search over rows with `price >= 90`. The SQL in [queries/vbench/indexed](queries/vbench/indexed) calls a table function registered for each engine. DataFusion computes the final cosine distances and orders the returned candidates.

The large fixture has 400,000 rows and 128 dimensions. Omit `--large` for 20,000 rows and 32 dimensions. Use `--engine lance --query q4 --rounds 1` for a quick filtered run. The runner writes temporary data and indexes under `--work-dir`, which defaults to the current directory.

Each engine builds its index once per invocation. After one warmup pass per query, the runner times each SQL request from parsing through result collection and alternates which engine runs first. The default five passes yield 105 latency samples per engine and query. Recall@10 measures overlap with an exhaustive cosine search over the eligible rows. Ground-truth computation, index construction, and validation are outside the measured interval.

The develop benchmark workflow runs the large fixture with eight IVF probes. It sends median latency and individual samples to the SQL benchmark ingest pipeline. The workflow summary and the per-commit results artifact include p95 and recall@10. To produce the same files locally, pass `--report vector-quality.json --ingest-jsonl results.ingest.jsonl`.
