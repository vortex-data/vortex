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

### On-wire / on-disk format

`-d gh-json-v3` writes **newline-delimited JSON**, one `ClassifiedMeasurement`
per line. Same line-per-record convention as today's `-d gh-json`, just
the record shape changes.

The file is **not** an `IngestPayload`. Wrapping records into
`IngestPayload { run_meta, commit, records }` is the job of the CI wrapper
(`scripts/post-ingest.py`), not the Rust emitter. The emitter has no
commit metadata or run metadata of its own - it emits records; the
wrapper adds the envelope before POSTing to `/api/ingest`.

That split keeps the emitter dependency-light (still just `serde` +
`serde_json`) and matches the existing JSONL convention, so local
developers can `cat results.v3.json` and reason about one record per line.

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
| `TimingMeasurement` (random-access, etc.) | `"random_access"` | Extend the struct with structured dimensions (dataset, pattern, `storage: Storage` enum) rather than re-parsing the `name` string. The random-access runner has the dataset/pattern/format at hand today; it's just flattening them into `name` on emission. |
| `CompressionTimingMeasurement` | `"compression_encode"` / `"compression_decode"` | Already carries `format`. Split "compress" vs "decompress" into two distinct `metric_kind` values (not one kind with an `op` field in `data_descriptor`) - cleaner SQL for the "compress time over time" chart. Decision pinned in [`11-implementation-kickoff.md`](./11-implementation-kickoff.md). |
| `CustomUnitMeasurement` (raw file sizes from `compress/mod.rs::benchmark_compress`) | `"compression_size"` | Emit as a `compression_size` record with `value_bytes` populated. Consider routing through a dedicated `CompressionSizeMeasurement` struct rather than re-using `CustomUnitMeasurement`'s `to_v3_json`, since the v3 shape wants structured dimensions (dataset/format). |
| `CustomUnitMeasurement` (cross-format ratios from `calculate_ratios`) | **not emitted** | `vortex:parquet size`, `vortex:lance ratio compress time`, etc. are fully derivable from the stored `compression_size` + `compression_encode` + `compression_decode` records. They become **DuckDB views** in v3 (see [`05-schema.md`](./05-schema.md) and AGENTS.md "Should we store ratios as rows?" = No). The emitter drops them. |

The file-size measurements (Shape D from [`03-raw-data-schema.md`](./03-raw-data-schema.md))
become `metric_kind = "compression_size"` with `value_bytes` populated. No
more separate `file-sizes-*.json.gz` output stream - they get folded into
the main `results.json` emitted by each benchmark binary.

Note on `CustomUnitMeasurement`: `vortex-bench`'s only in-tree consumer is
`compress-bench` (via `compress/mod.rs`). The struct itself may be kept
for future free-form extensibility - vector-search-bench-style metrics
that carry auxiliary fields in `data_descriptor` - but it has **no direct
`to_v3_json()` emission in step 1**. Both the raw-bytes path and the
ratios path are handled explicitly (structured struct and DB view
respectively).

### vector-search-bench: deferred to post-launch

Originally this doc proposed adding v3-shape emission to
`benchmarks/vector-search-bench/` (which has its own `display.rs` and
doesn't emit `gh-json` today) in the same pass as the rest of the
emitter work. **Decision: deferred.** Getting vector-search results
into v3 is a post-launch follow-up, not a launch requirement.

The hooks we keep now so we don't paint ourselves into a corner:

- `CustomUnitMeasurement` is retained (no direct `to_v3_json()` in
  step 1, but not deleted) as the forward-compat surface for
  future-vector-search-style free-form metrics.
- `data_descriptor` on `ClassifiedMeasurement` is designed for
  auxiliary fields like `{layout, threshold, n_dimensions, n_rows,
  iterations, query_seed}` that vector search will need.
- `MetricKind::VectorSearchTime / VectorSearchCount / VectorSearchBytes`
  exist in the enum as reserved variants; the server accepts them
  but no benchmark emits them at launch.

What we do **not** do in this project: add `--format=gh-json-v3` to
`vector-search-bench`, add `.github/workflows/vector-bench.yml`, or
build `BenchmarkGroupFilter::VectorSearch` routing in the server.
Those all wait until vector-search-bench is ready to graduate.

### CLI surface

Existing `-d <format> -o <path>` is replaced by **one output flag per
format**, each taking its own path. Multiple output flags may be
supplied in the same invocation; the benchmark runs once and writes
every requested output from the already-collected measurements.

```text
--table                   emit a human-readable table to stdout (default if no other output given)
--gh-json     <path>      emit legacy v2-shape JSONL to <path>    (retired post-cutover)
--gh-json-v3  <path>      emit v3-shape JSONL to <path>
```

Designed in a vacuum - backwards compatibility with `-d`/`-o` is
deliberately not preserved. `-d` was always a mode selector pretending
to be a flag; swapping to format-named flags lets us emit several
formats from one run without overloading `clap` or inventing a
`format=path` micro-syntax.

All existing callers that pass `-d gh-json -o results.json` get updated
in the same pass. Post-cutover, `--gh-json` is removed and the help
text collapses to just `--table` and `--gh-json-v3`.

### CI wiring (dual-write window)

For each existing bench workflow (`bench.yml`, `sql-benchmarks.yml`),
emit **both** formats from a **single benchmark run** during the
transition. The expensive part is running the benchmark loop;
serializing the already-collected measurements twice is free.

```bash
# Single benchmark run; emits v2 legacy JSONL and v3 JSONL side by side.
bash scripts/bench-taskset.sh target/release_debug/${{ matrix.benchmark.id }} \
    --formats ${{ matrix.benchmark.formats }} \
    --gh-json    results.json \
    --gh-json-v3 results.v3.json

# v2 upload path, stays alive through cutover.
bash scripts/cat-s3.sh vortex-ci-benchmark-results data.json.gz results.json

# v3 ingest path.
python3 scripts/post-ingest.py \
    --server  https://bench-preview.vortex.dev \
    --commit-sha "$GITHUB_SHA" \
    --benchmark-id "${{ matrix.benchmark.id }}" \
    --results results.v3.json \
    --token   "$INGEST_BEARER_TOKEN" \
    --spool   "s3://vortex-ci-benchmark-results/outbox/"
```

Post-cutover, `--gh-json` is removed and this collapses to a single
`--gh-json-v3 results.v3.json`.

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

1. **Add `--gh-json-v3` emission** to every measurement type in
   `vortex-bench` (and convert the CLI from `-d`/`-o` to format-named
   output flags in the same pass). Leave `--gh-json` alive. Tests
   compare old vs. new output to confirm the new path carries the same
   data, just structured.
2. **Stand up the preview v3 server + migrator** (work described in
   [`06-migration.md`](./06-migration.md)).
3. **Dual-write CI**: update `bench.yml` + `sql-benchmarks.yml` to the
   single-run two-flag pattern, add the POST step. Both paths alive
   simultaneously.
4. **Compare**: preview site and live v2 site should show equivalent
   numbers for every chart.
5. **Cutover**: flip DNS, remove `--gh-json`, remove `cat-s3.sh`,
   remove `commit-json.sh`, delete the migrator binary.

Vector-search-bench v3 emission is **not** part of this rollout. It's
a post-launch follow-up (see "vector-search-bench: deferred to
post-launch" above).

## Estimated scope

- Emitter extensions in `vortex-bench`: ~2-3 days. Mechanical; write a
  `to_v3_json()` per measurement type, add a `CompressionSizeMeasurement`
  for the compress-bench bytes path, extend `TimingMeasurement` with
  structured random-access dimensions, and convert every caller to the
  new `--gh-json` / `--gh-json-v3` flag names.
- CI dual-write yaml changes: ~0.5 days.

Total: roughly half a week for step 1. Dual-write window can run for
as long as we want (weeks is fine); cutover is a yaml deletion + DNS
flip.
