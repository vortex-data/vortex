<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 03 - Raw data schema (today)

This doc describes the shape of the data currently stored in
`s3://vortex-ci-benchmark-results/data.json.gz` and `commits.json`. It exists so
that anyone writing the v3 ingester can understand the input without having to
reverse-engineer `vortex-bench::measurements`.

## `commits.json` (JSONL)

One line per commit to `develop`, emitted by `scripts/commit-json.sh`.

```jsonc
{
  "author":    { "email": "...", "name": "..." },
  "committer": { "email": "...", "name": "..." },
  "id":        "<40-char hex sha1>",
  "message":   "<first line of commit message>",
  "timestamp": "2026-01-15T12:34:56+00:00",   // iso-strict
  "tree_id":   "<40-char hex>",
  "url":       "https://github.com/vortex-data/vortex/commit/<id>"
}
```

This is clean. It maps 1:1 to a `commits` table in DuckDB.

## `data.json.gz` (gzipped JSONL)

One line per measurement, emitted by `vortex-bench::display::print_measurements_json`
via one of several `impl ToJson for ...` implementations in
`vortex-bench/src/measurements.rs`. **The schema is not uniform.** There are
four shapes in the wild.

### Shape A: "generic" (`JsonValue`)

Used for `CompressionTimingMeasurement`, `CustomUnitMeasurement`, and the
random-access / compress-bench outputs.

```jsonc
{
  "name":       "<arbitrary slash-delimited string>",
  "storage":    "nvme" | "s3" | null,       // usually absent
  "unit":       "ns" | "bytes" | "ratio" | "MiB/s" | null,
  "value":      <number>,
  "target":     { "engine": "vortex"|"arrow"|"datafusion"|"duckdb"|...,
                  "format": "vortex-file-compressed"|"parquet"|"lance"|... },
  "time":       <u128 ns> | null,           // only for throughputs
  "bytes":      <u64> | null,               // only for throughputs
  "commit_id":  "<40-char hex sha1>"
}
```

### Shape B: `QueryMeasurementJson` (SQL suites)

Used by `QueryMeasurement` for TPC-H, TPC-DS, Clickbench, StatPopGen,
PolarSignals, Fineweb query runtimes.

```jsonc
{
  "name":         "tpch_q01/datafusion:vortex-file-compressed",  // <dataset>_q<NN>/<engine>:<format>
  "storage":      "s3" | "nvme",
  "dataset":      { "tpch": { "scale_factor": "10.0" } },        // tagged enum
  "unit":         "ns",
  "value":        <u128 median ns>,
  "all_runtimes": [<u128 ns>, ...],
  "target":       { "engine": "...", "format": "..." },
  "commit_id":    "<40-char hex sha1>",
  "env_triple":   { "architecture": "x86_64",
                    "operating_system": "linux",
                    "environment": "gnu" }
}
```

`dataset` is a serde-tagged enum. Known variants (from
`vortex-bench/src/datasets/mod.rs::BenchmarkDataset`):

- `{ "tpch":         { "scale_factor": "<str>" } }`
- `{ "tpcds":        { "scale_factor": "<str>" } }`
- `{ "clickbench":   { "flavor": "partitioned" | "single" } }`
- `{ "public-bi":    { "name": "<str>" } }`
- `{ "statpopgen":   { "n_rows": <u64> } }`
- `{ "polarsignals": { "n_rows": <usize> } }`
- `{ "fineweb":      {} }` (unit variant, may be bare `"fineweb"`)
- `{ "gharchive":    {} }`

### Shape C: `MemoryMeasurementJson`

Same as Shape B but with four memory fields replacing the timing fields:

```jsonc
{
  "name":                   "tpch_q01_memory/...",
  "physical_memory_delta":  <i64>,
  "virtual_memory_delta":   <i64>,
  "peak_physical_memory":   <u64>,
  "peak_virtual_memory":    <u64>,
  ...same envelope as Shape B
}
```

### Shape D: size measurements

Appended by the SQL workflow as `file-sizes-*.json.gz` files (separate from the
main `data.json.gz`, but also read by v2's server). Roughly Shape A with
`unit: "bytes"` and a `name` like `"<dataset>:<format> size/<something>"` or
`"<format> size/<dataset>"`. The exact forms v2's `getGroup` recognizes include:

- `"vortex size/..."`, `"parquet size/..."`, `"lance size/..."`
- `"vortex-file-compressed size/..."`
- `"<dataset>:raw size/..."`, `"<dataset>:parquet-zstd size/..."`,
  `"<dataset>:lance size/..."`

## How v2's server parses `name`

This logic lives in `benchmarks-website/server.js`. It is a stack of
hand-written rules. We have to reproduce its classifications in the v3 ingester,
then not have to reproduce them ever again.

Roughly:

```text
name = lowercase(b.name)

if starts_with("random-access/") or "random access/":
    group = "Random Access"
    if name has 4 /-segments: chart = "<dataset>/<pattern>", series = <format>
    elif 2 segments:          chart = "RANDOM ACCESS",       series = <format>

elif any of {"vortex size/", "parquet size/", "lance size/", ":raw size/",
             ":parquet-zstd size/", ":lance size/", "vortex-file-compressed size/"}:
    group = "Compression Size"

elif starts_with any of {"compress time/", "decompress time/",
                         "parquet_rs-zstd (de)compress", "lance (de)compress",
                         "vortex:lance ratio", "vortex:parquet-zstd ratio",
                         "vortex:raw ratio"}:
    group = "Compression"

elif matches "<suite_prefix>_q<NN>" or "<suite_prefix>/..." for any suite:
    if suite.fanOut:
        group = "<displayName> (<storage>) (SF=<scale>)"
    else:
        group = suite.displayName
    chart   = formatQuery(part_0)          // e.g. "TPC-H Q1"
    series  = rename(part_1 or "default")  // engine:format, renamed

else:
    drop the record.
```

Unit conversions:

- `ns` → rendered as `ms` (divide by 1e6 for `value`, display unit = `"ms/iter"`)
- `bytes` → rendered as `MiB` (divide by 1024**2, display unit = `"MiB"`)
- `ratio` → passthrough

Plus various `ENGINE_RENAMES`:

- `"datafusion:vortex-file-compressed"` → `"datafusion:vortex"`
- `"vortex-tokio-local-disk"` → `"vortex-nvme"`
- `"lance-tokio-local-disk"` → `"lance-nvme"`
- ...

## What's broken about this

1. **`name` is overloaded.** Depending on shape it packs `dataset`, `query_idx`,
   `engine`, `format`, `storage`, and `metric_kind` all into one
   slash-delimited string. When the client parses it, it is re-inferring what
   the emitter already knew.

2. **Partial structured fields exist but are ignored.** Shapes B and C already
   carry a `dataset` tagged-enum and a `storage` field and a
   `target: {engine, format}` field. The v2 frontend mostly re-derives them
   from `name` anyway, because not every record has them.

3. **No record-level schema version.** If we change the emitter, old records
   still exist on S3 with the old shape, and the reader has no way to know.

4. **No run-level metadata.** The `env_triple` is repeated on every record (Shape
   B and C only). Hardware info (CPU model, NUMA topology, kernel version) is
   not captured at all.

5. **"ratio" is a first-class value.** Records like `"vortex:parquet-zstd ratio
   compress time/<dataset>"` are derived metrics computed at emit time. In v3
   these should be SQL views over the raw measurements, not stored records.

6. **Some groups drop records on the floor.** `v2 server.js` silently drops
   records whose name has `"parquet-unc"` or includes `" throughput"`. That
   logic has to be re-expressed somewhere.

## What v3 stores instead

See [`05-schema.md`](./05-schema.md). Short version: the ingester owns all of
the "parse `name` into structured columns" logic. The DuckDB schema has named
dimension columns (`dataset`, `scale_factor`, `storage`, `query_idx`, `engine`,
`format`, `metric_kind`) and a single `measurements` fact table. Derived
measurements (ratios, geomeans) are SQL views.
