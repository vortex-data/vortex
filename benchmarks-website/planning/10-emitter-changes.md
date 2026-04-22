<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 10 - `vortex-bench` emitter changes

v3 moves the "structured meaning" of a benchmark measurement out of the
consumer (v2's `server.js::getGroup` regex stack) and into the emitter.
Each `vortex-bench` measurement type learns to emit a v3-shape JSON record
directly, so the website's `/api/ingest` handler is a serde-validated
passthrough with no classification logic.

This doc is the scope + migration plan for the emitter work. It's peer to
[`07-ingestion.md`](./07-ingestion.md) (server side) and
[`06-migration.md`](./06-migration.md) (historical data).

## Why

Today (v2):

```text
vortex-bench emits: name = "tpch_q01/datafusion:vortex-file-compressed"
                    + a few other unstructured fields
                            |
                            v
               server's getGroup() regex stack unpacks name
                            |
                            v
                      UI renders groups
```

v3 with a classifier in the server:

```text
vortex-bench emits v2-shape         server's classifier unpacks
             |                              |
             v                              v
     /api/ingest payload     →     classify() → measurements table
```

v3 with emitter rewrite (this doc):

```text
vortex-bench emits v3-shape
             |
             v
     /api/ingest payload  →  serde parse  →  measurements table
```

The third is the cleanest endpoint. The emitter already knows every
dimensional field at emission time (`self.benchmark_dataset`,
`self.target.engine`, `self.target.format`, `self.query_idx`,
`self.storage`, ...). It currently joins them into a slash-delimited `name`
string that v2's server then parses back out. v3 cuts out the round trip.

## Scope

### New output format

Add a `gh-json-v3` variant to `DisplayFormat` in `vortex-bench/src/display.rs`:

```rust
#[derive(ValueEnum, Default, Clone, Debug)]
pub enum DisplayFormat {
    #[default]
    Table,
    GhJson,     // legacy, v2-shape, deleted post-cutover
    GhJsonV3,   // new, v3-shape, emitted alongside GhJson during dual-write
}
```

Wire `print_measurements_json_v3` through `runner.rs` the same way
`print_measurements_json` is wired today.

### New emission trait (or method)

Each measurement type learns a `to_v3_json()` method (or a `ToV3Json`
trait - pick what reads cleanest; same information either way). The output
matches the `measurements` table shape from [`05-schema.md`](./05-schema.md).

```jsonc
// QueryMeasurement::to_v3_json output (tpch, random access, etc.):
{
  "metric_kind":     "query_time",
  "dataset":         "tpch",
  "scale_factor":    "10.0",
  "dataset_variant": null,
  "query_idx":       1,
  "storage":         "nvme",
  "engine":          "datafusion",
  "format":          "vortex-file-compressed",
  "value_ns":        1234567,
  "all_runtimes_ns": [1234567, 1251121, ...],
  "env_triple":      "x86_64-linux-gnu",
  "commit_sha":      "<40-hex>",
  "data_descriptor": null
}
```

One row per measurement, same structured shape regardless of the source
measurement type. No more `name = "..."` smuggling.

### Per-measurement-type mapping

Fill in the new shape for each existing `impl ToJson for X` in
`vortex-bench/src/measurements.rs`:

| Current type | `metric_kind` | Notes |
|---|---|---|
| `QueryMeasurement` | `"query_time"` | `dataset` / `scale_factor` / `dataset_variant` from the tagged `BenchmarkDataset`; `query_idx` already stored structurally. |
| `MemoryMeasurement` | `"query_memory"` | Same dimensions as query_time plus the four memory fields. |
| `TimingMeasurement` (random-access, etc.) | `"random_access"` | Parse the existing `name` field's structure into dimensions in the emitter - the random-access runner has the dataset/pattern/format at hand. |
| `CompressionTimingMeasurement` | `"compression_time"` | Already carries `format`; map "compress" vs "decompress" into the `metric_kind` (split into `"compression_encode"` / `"compression_decode"`) or keep as a single kind with a field in `data_descriptor`. Pick one; former is cleaner. |
| `CustomUnitMeasurement` | varies (ratios/sizes/etc.) | The one place where the existing format is genuinely free-form. Emitters that use this carry context the base type doesn't - plumb more dimensions through to keep the v3 output structured. |

The file-size measurements (Shape D from [`03-raw-data-schema.md`](./03-raw-data-schema.md))
become `metric_kind = "compression_size"` with `value_bytes` populated. No
more separate `file-sizes-*.json.gz` output stream - they get folded into
the main `results.json` emitted by each benchmark binary.

### vector-search-bench gets promoted

`benchmarks/vector-search-bench/` currently has its own `display.rs` with
a bespoke tabled table renderer and **does not emit `gh-json` at all**. As
part of this work we add v3-shape emission to it too, and wire it into
CI. Records land as:

```jsonc
{
  "metric_kind":     "vector_search_time",   // or "vector_search_count",
  "dataset":         "cohere-large-10m",     // "vector_search_bytes"
  "scale_factor":    null,
  "dataset_variant": null,
  "storage":         "nvme",
  "engine":          "vortex",
  "format":          "vortex-turboquant",
  "value_ns":        485000000,
  "data_descriptor": {
    "layout":         "partitioned",
    "threshold":      0.85,
    "n_dimensions":   1024,
    "n_rows":         10000000,
    "iterations":     5,
    "query_seed":     42
  },
  ...
}
```

Multiple records per run (one per metric: wall time mean, wall time
median, matches, rows scanned, bytes scanned) all share the same
`data_descriptor` so they can be grouped in the UI if desired. This is
the "extensibility proof" for `data_descriptor`.

### CI wiring (dual-write window)

For each existing bench workflow (`bench.yml`, `sql-benchmarks.yml`) and
each benchmark binary invocation, emit **both** formats during the
transition:

```bash
# Old path, stays alive for v2.
bash scripts/bench-taskset.sh target/release_debug/${{ matrix.benchmark.id }} \
    --formats ${{ matrix.benchmark.formats }} \
    -d gh-json -o results.json
bash scripts/cat-s3.sh vortex-ci-benchmark-results data.json.gz results.json

# New path, feeds v3.
bash scripts/bench-taskset.sh target/release_debug/${{ matrix.benchmark.id }} \
    --formats ${{ matrix.benchmark.formats }} \
    -d gh-json-v3 -o results.v3.json
python3 scripts/post-ingest.py \
    --server  https://bench-preview.vortex.dev \
    --commit-sha "$GITHUB_SHA" \
    --benchmark-id "${{ matrix.benchmark.id }}" \
    --results results.v3.json \
    --token   "$INGEST_BEARER_TOKEN" \
    --spool   "s3://vortex-ci-benchmark-results/outbox/"
```

Running each benchmark twice for dual output isn't necessary - the
benchmark runs once, only the serialization / upload doubles. That's
pennies of extra CI time.

## What stays, what goes

### Stays (in `vortex-bench` and elsewhere)

- The measurement types (`QueryMeasurement`, `MemoryMeasurement`, etc.) -
  unchanged.
- The runner logic (`runner.rs`) - unchanged except for a new
  output-format branch.
- The table renderer (`display::render_table`) - unchanged. Developers
  running benchmarks locally still get human-readable tables.
- `-d gh-json` output - **kept during dual-write**, deleted post-cutover.

### Goes (post-cutover)

- The `GhJson` variant of `DisplayFormat` and all `ToJson` impls - deleted.
  `ToV3Json` is the only JSON emission path going forward.
- `scripts/cat-s3.sh` - deleted.
- `scripts/commit-json.sh` - deleted (its output now goes straight into
  the `/api/ingest` payload, not a separate JSON blob).
- `file-sizes-*.json.gz` S3 writes - deleted. Size records are part of the
  main POST.

## Server-side simplification

With emitter rewrite, `/api/ingest` reduces to:

```rust
async fn ingest(
    State(ctx): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<IngestPayload>,
) -> Result<Json<IngestResponse>> {
    authenticate(&headers)?;
    let tx = ctx.db.transaction()?;
    upsert_commit(&tx, &payload.commit)?;
    let mut counts = Counts::default();
    for record in payload.records {
        // record is ALREADY a ClassifiedMeasurement shape; no classification.
        let id = hash_measurement_id(&record);
        match tx.execute("INSERT INTO measurements ... ON CONFLICT ... DO UPDATE", ...) {
            Ok(Rows(1)) => counts.inserted += 1,
            Ok(_)       => counts.updated  += 1,
            Err(e)      => { /* log, bump counts.errored, don't abort */ }
        }
    }
    tx.commit()?;
    Ok(Json(IngestResponse::from(counts)))
}
```

No `classify()` function. No v2-parity regex stack in the main repo. The
serde parse of the incoming JSON is the only "validation" step.

## Migrator still has a classifier

The historical migrator (see [`06-migration.md`](./06-migration.md)) must
still parse v2-shape `data.json.gz`. It contains years of records emitted
by the old `ToJson` path - we can't re-run history.

So the migrator binary carries its own one-shot classifier that mimics
v2's `server.js::getGroup`. The classifier lives **only** in the migrator
binary, ships only on the feature branch, runs once at cutover, and gets
deleted along with the migrator binary afterward.

The classifier module **never lands on the main branch** of the repo.

## Rollout plan

1. **Add `-d gh-json-v3` emission** to every measurement type in
   `vortex-bench`. Leave `-d gh-json` alone. Tests compare old vs. new
   output to confirm the new path carries the same data, just structured.
2. **Add v3 emission to `vector-search-bench`**. Wire the benchmark into
   `.github/workflows/vector-bench.yml` (new workflow) so it runs in CI
   alongside the rest.
3. **Stand up the preview v3 server + migrator** (work described in
   [`06-migration.md`](./06-migration.md)).
4. **Dual-write CI**: add the new POST step to `bench.yml` +
   `sql-benchmarks.yml`. Both paths alive simultaneously.
5. **Compare**: preview site and live v2 site should show equivalent
   numbers for every chart.
6. **Cutover**: flip DNS, remove `-d gh-json`, remove `cat-s3.sh`,
   remove `commit-json.sh`, delete the migrator binary.

## Estimated scope

- Emitter extensions in `vortex-bench`: ~2-3 days. Mechanical; write a
  `to_v3_json()` per measurement type + a couple of field-plumbing fixes
  for `CustomUnitMeasurement`.
- vector-search-bench v3 emission + CI wiring: ~2 days.
- CI dual-write yaml changes: ~0.5 days.

Total: about a week of focused work. Dual-write window can run for as
long as we want (weeks is fine); cutover is a yaml deletion + DNS flip.
