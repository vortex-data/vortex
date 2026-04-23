<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 11 - Implementation kickoff

This doc pins the concrete contracts and a handful of decisions that the
earlier planning docs left open. **Read this after docs 00-10.** If you're
about to write code and you find yourself thinking "but what exactly does
the Rust struct look like?" - it should be here. If it isn't, that's a gap
and you should flag it before inventing something.

Everything here is **binding**: the emitter-side and server-side code must
agree on these contracts, or dual-write breaks. Don't drift during
implementation without updating this doc.

## Crates and directory layout

Two new Cargo workspace members. Add both to the root `Cargo.toml`'s
`[workspace] members`.

```text
benchmarks-website/
  server/                    <- NEW crate: `benchmarks-website-server`
    Cargo.toml
    src/
      main.rs                 <- axum bootstrap
      config.rs               <- env var parsing
      db.rs                   <- DuckDB handle, schema_meta check, seed
      routes/
        mod.rs
        ingest.rs             <- POST /api/ingest
        group.rs              <- GET /group/:slug, /api/groups
        chart.rs              <- GET /chart/:slug, /api/chart/:slug
        commit.rs             <- GET /commit/:sha
        health.rs             <- GET /health
      templates/              <- maud macros or askama .html files
        layout.rs
        group.rs
        chart.rs
        ...
      model.rs                <- IngestPayload, IngestResponse, etc.
      groups.rs               <- BenchmarkGroupFilter enum + SQL builder
      seed/
        known.sql             <- known_engines/formats/datasets seed
      migrations/
        001_initial.sql
        002_...
    static/
      chart.js                <- vanilla-JS glue for Chart.js
      style.css

  migrator/                  <- NEW crate: `benchmarks-website-migrator`
    Cargo.toml                  (deleted post-cutover)
    src/
      main.rs                 <- CLI entry
      classifier.rs           <- v2 getGroup/formatQuery port (deleted post-cutover)
      raw.rs                  <- RawMeasurement enum (v2 shapes A-D)
      io.rs                   <- S3 reader + JSONL streaming
      verify.rs               <- diff against v2's /api/metadata
```

The migrator depends on `benchmarks-website-server`'s `model.rs` so both
use the same `ClassifiedMeasurement` struct - that's the only shared
type. It does **not** depend on server routes/templates/DB handle.

## Binding Rust contracts

### Core record shape (`server::model::ClassifiedMeasurement`)

This is what `/api/ingest` accepts and what `vortex-bench`'s `-d gh-json-v3`
emits. It matches the `measurements` table from [`05-schema.md`](./05-schema.md)
1:1.

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]   // catches version skew loudly
pub struct ClassifiedMeasurement {
    pub commit_sha: String,              // 40-hex lowercase

    pub metric_kind: MetricKind,
    pub dataset: Option<String>,
    pub scale_factor: Option<String>,
    pub dataset_variant: Option<String>,
    pub query_idx: Option<i32>,
    pub storage: Option<Storage>,
    pub engine: Option<String>,
    pub format: Option<String>,

    pub value_ns: Option<i64>,
    pub value_bytes: Option<i64>,
    pub value_unitless: Option<f64>,

    pub peak_physical: Option<i64>,
    pub peak_virtual: Option<i64>,
    pub physical_delta: Option<i64>,
    pub virtual_delta: Option<i64>,

    #[serde(default)]
    pub all_runtimes_ns: Vec<i64>,

    pub env_triple: Option<String>,

    /// Free-form; see [`05-schema.md`] "Extensibility notes" for example shapes.
    #[serde(default)]
    pub data_descriptor: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricKind {
    RandomAccess,
    CompressionEncode,   // was "compress time" in v2
    CompressionDecode,   // was "decompress time" in v2
    CompressionSize,
    QueryTime,
    QueryMemory,
    VectorSearchTime,
    VectorSearchCount,   // matches count
    VectorSearchBytes,   // rows / bytes scanned
    Microbench,          // reserved for future
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Storage {
    Nvme,
    S3,
}
```

**Decision on `CompressionTimingMeasurement`**: split into
`CompressionEncode` and `CompressionDecode` as separate `metric_kind`
values (not one kind with an `op` field in `data_descriptor`). Cleaner SQL
for the common "compress time over time" chart.

**Decision on `Storage` enum membership**: closed to `{Nvme, S3}`. A
stale doc comment on `TimingMeasurement::storage` mentions
`"One of: s3, gcs, nvme"`, but `gcs` is not a real target and is not
emitted by any current benchmark. Drop the mention when migrating to the
`Storage` enum. If a historical record in the v2 JSONL has `"gcs"`, the
migrator should error loudly rather than silently ingest it.

**Decision on `commit_sha` emission**: the emitter emits `GIT_COMMIT_ID`
as-is with no validation. Local developer runs may produce a short or
dirty SHA; that's fine, local runs don't flow into `/api/ingest`. The
server's `/api/ingest` is responsible for rejecting payloads whose
`commit_sha` is not 40-hex lowercase.

**Decision on `env_triple` emission**: the emitter flattens today's
structured `TripleJson { architecture, operating_system, environment }`
into `format!("{arch}-{os}-{env}")` (e.g. `"x86_64-linux-gnu"`). No
need to preserve the pre-flattened struct on the wire - if a consumer
needs the components, it parses the string.

**Decision on `scale_factor` format**: emit whatever stringified form
`BenchmarkDataset` already carries for scale factor (no forced decimal
normalization). Consistency is the emitter's responsibility - the same
`(dataset, scale_factor)` tuple must produce the same string every time,
or `measurement_id` hashing breaks. This is enforced by a snapshot test,
not by a format rule.

**Decision on output file format**: emitter writes **JSONL of bare
`ClassifiedMeasurement` records** (one per line), not an `IngestPayload`.
See [`10-emitter-changes.md`](./10-emitter-changes.md) §"On-wire / on-disk
format". The `IngestPayload` envelope (with `run_meta` + `commit`) is
assembled by the CI wrapper (`scripts/post-ingest.py`) before POSTing,
not by the Rust emitter.

### Ingest payload (`server::model`)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IngestPayload {
    pub run_meta: RunMeta,
    pub commit: CommitMetadata,            // full commit info, upserted into `commits`
    pub records: Vec<ClassifiedMeasurement>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunMeta {
    pub benchmark_id: String,              // e.g. "random-access-bench"
    pub schema_version: u32,               // 1 for launch
    pub started_at: chrono::DateTime<chrono::Utc>,
    pub hardware_class: Option<String>,    // from runs-on runner spec
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommitMetadata {
    pub sha: String,                       // 40-hex lowercase
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub message: String,                   // first line of commit msg
    pub author_name: String,
    pub author_email: String,
    pub committer_name: String,
    pub committer_email: String,
    pub tree_sha: String,
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestResponse {
    pub inserted: usize,
    pub updated: usize,
    pub unclassified: usize,
    pub warnings: Vec<String>,
}
```

### Group definition enum (`server::groups`)

```rust
/// Closed set of benchmark groups. Adding a new group kind is a code
/// change, not a data change. Route handlers match on this to build SQL
/// WHERE clauses with bound parameters (never string concatenation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkGroupFilter {
    RandomAccess,
    Compression,                                    // encode + decode
    CompressionSize,
    QuerySuite { dataset: &'static str },           // single-scale suites: clickbench, statpopgen, polarsignals
    FanOut {                                        // multi-scale suites: tpch, tpcds
        dataset: &'static str,
        storage: Storage,
        scale_factor: &'static str,
    },
    VectorSearch,                                   // added when vector-search-bench lands in CI
    // Microbench { ... }                           // future
}

/// Every group the site knows about, ordered for rendering.
pub const ALL_GROUPS: &[BenchmarkGroupFilter] = &[
    BenchmarkGroupFilter::RandomAccess,
    BenchmarkGroupFilter::Compression,
    BenchmarkGroupFilter::CompressionSize,
    BenchmarkGroupFilter::QuerySuite { dataset: "clickbench" },
    BenchmarkGroupFilter::QuerySuite { dataset: "statpopgen" },
    BenchmarkGroupFilter::QuerySuite { dataset: "polarsignals" },
    BenchmarkGroupFilter::FanOut { dataset: "tpch",  storage: Storage::Nvme, scale_factor: "1"    },
    BenchmarkGroupFilter::FanOut { dataset: "tpch",  storage: Storage::S3,   scale_factor: "1"    },
    BenchmarkGroupFilter::FanOut { dataset: "tpch",  storage: Storage::Nvme, scale_factor: "10"   },
    // ...port the full list from benchmarks-website/src/config.js::FAN_OUT_GROUPS
];

impl BenchmarkGroupFilter {
    /// URL slug. Stable across deploys; bookmark-safe.
    pub fn slug(&self) -> String { ... }

    /// Human-readable display name for the page header.
    pub fn display_name(&self) -> String { ... }

    /// Build a (SQL fragment, bound params) pair suitable for the
    /// `measurements` table. Never returns a string with user data interpolated.
    pub fn where_clause(&self) -> (&'static str, Vec<duckdb::Value>) { ... }
}
```

Port the full list from `benchmarks-website/src/config.js` (see
[`reference/v2-config-top.js`](./reference/v2-config-top.js) for the
snapshot).

## Pinned decisions

### Hash algorithm for `measurement_id`

**xxhash64** via the `xxhash-rust` crate (`xxhash-rust = { version = "0.8",
features = ["xxh64"] }`). Canonicalization:

```rust
pub fn measurement_id(m: &ClassifiedMeasurement) -> i64 {
    let mut h = xxhash_rust::xxh64::Xxh64::new(0);
    h.update(m.commit_sha.as_bytes());
    h.update(&[0u8]);                  // unambiguous field separator
    h.update(serde_json::to_string(&m.metric_kind).unwrap().as_bytes());
    h.update(&[0u8]);
    write_opt(&mut h, m.dataset.as_deref());
    write_opt(&mut h, m.scale_factor.as_deref());
    write_opt(&mut h, m.dataset_variant.as_deref());
    write_opt_i32(&mut h, m.query_idx);
    write_opt(&mut h, m.storage.map(storage_str));
    write_opt(&mut h, m.engine.as_deref());
    write_opt(&mut h, m.format.as_deref());
    // data_descriptor canonicalized: sorted keys, compact, UTF-8
    if let Some(d) = &m.data_descriptor {
        h.update(&[1u8]);              // "present" marker
        h.update(canonical_json(d).as_bytes());
    } else {
        h.update(&[0u8]);              // "absent" marker (distinct from empty string)
    }
    h.digest() as i64                  // DuckDB BIGINT is i64
}

fn write_opt(h: &mut Xxh64, s: Option<&str>) {
    match s {
        Some(v) => { h.update(&[1u8]); h.update(v.as_bytes()); h.update(&[0u8]); }
        None    => { h.update(&[0u8]); }
    }
}
```

The "present" / "absent" marker byte is what makes NULL distinguishable
from an empty string in the hash input without rejecting legitimate empty
values.

`canonical_json` serializes with sorted keys and no whitespace. Implement
it once; use it everywhere that hashes JSON.

### DuckDB crate

**`duckdb` (the `duckdb-rs` crate)**, version >=1.0. This is the upstream
Rust binding maintained by the DuckDB team, not the third-party `duckdb-rs`
from 2021. Pull the latest stable.

### DuckDB concurrency model

One `Connection` wrapped in a `tokio::sync::Mutex` for writes; separate
read-only `Connection` per request handler via a connection pool.

```rust
pub struct AppState {
    writer: Arc<Mutex<duckdb::Connection>>,        // one, guarded
    reader_pool: deadpool::Pool<duckdb::Connection>, // many, read-only handles
}
```

Write path acquires the writer mutex; read path pulls from the pool.
DuckDB supports multiple read-only handles on the same file concurrently
with the writer handle.

### Transaction / error behavior in `/api/ingest`

One transaction per POST. Records the handler accepts but that fail at
INSERT (e.g. FK violation on `commit_sha` - shouldn't happen since we
upsert the commit first, but) go into `unclassified_records`. **Never
abort the whole transaction for one bad record.** Return 200 with
`{unclassified: N, warnings: [...]}` even when some records failed. The
CI logs the response; the operator notices if `unclassified > 0`.

HTTP status matrix:

| Condition | Status | Body |
|---|---|---|
| Happy path (even with some unclassified) | 200 | `IngestResponse` |
| Malformed JSON at outer level | 400 | `{error: "invalid json"}` |
| Missing/invalid bearer token | 401 | `{error: "unauthorized"}` |
| Schema version newer than server expects | 409 | `{error: "upgrade server", expected: N, got: M}` |
| DB unreachable / write lock timeout | 503 | `{error: "retry"}` |
| Other server error | 500 | `{error: "..."}` |

CI retries on 4xx (token refresh), 503, and network errors. Does not
retry on 400 (bad data - would just fail again).

### Seed SQL bootstrap

Start with the v2 `ENGINE_RENAMES` + `SERIES_COLOR_MAP` tables from
`benchmarks-website/src/config.js` (see
[`reference/v2-config-top.js`](./reference/v2-config-top.js)). Port every
entry to `known_engines` / `known_formats` rows with the same
display-name and color values.

```sql
-- benchmarks-website/server/src/seed/known.sql
INSERT INTO known_engines (name, display_name, color_hex) VALUES
  ('datafusion',           'DataFusion',           NULL),
  ('duckdb',               'DuckDB',               NULL),
  ('vortex',               'Vortex',               NULL)
ON CONFLICT (name) DO UPDATE SET
  display_name = excluded.display_name,
  color_hex    = excluded.color_hex;

INSERT INTO known_formats (name, display_name, color_hex) VALUES
  ('vortex-file-compressed', 'vortex',        '#19a508'),
  ('parquet',                'parquet',       '#ef7f1d'),
  ('lance',                  'lance',         '#3B82F6'),
  ('vortex-turboquant',      'vortex-turboquant', '#15850a')
-- ...etc, port the full SERIES_COLOR_MAP
ON CONFLICT (name) DO UPDATE SET ...;
```

Colors like `'#19a508'` come straight from
`benchmarks-website/src/config.js::SERIES_COLOR_MAP`. Port them 1:1;
don't invent new colors.

Idempotent upserts let the server safely re-run the seed on every boot.

### v2 classifier reference

The migrator's one-shot classifier must reproduce v2's
`benchmarks-website/server.js::getGroup` + `formatQuery` +
`normalizeChartName` bug-for-bug. Full verbatim snapshot committed at
[`reference/v2-classifier.js`](./reference/v2-classifier.js) - port
line-by-line, not from memory or from the prose description in
[`03-raw-data-schema.md`](./03-raw-data-schema.md).

### `/api/chart/:slug` response shape

Sketch (refine during implementation):

```rust
#[derive(Debug, Clone, Serialize)]
pub struct ChartResponse {
    pub slug: String,
    pub display_name: String,
    pub unit: String,                  // "ms" | "MiB" | "ratio" | ...
    pub commits: Vec<CommitStub>,      // length N, sorted by timestamp
    pub series: HashMap<String, Vec<Option<f64>>>,  // key = series name (e.g. "datafusion:vortex"); value aligned with commits
}

#[derive(Debug, Clone, Serialize)]
pub struct CommitStub {
    pub sha: String,                   // short 7-char + full elsewhere
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub message_first_line: String,
    pub url: String,
}
```

The initial page render embeds this JSON inline via a `<script
type="application/json" id="chart-data-<slug>">` tag. Chart.js reads it.
Zoom/pan triggers a fetch to `GET /api/chart/:slug?start=<ts>&end=<ts>`
returning the same shape, with `commits` + `series` sliced to the range.

### Vector-search-bench wiring

`benchmarks/vector-search-bench/` currently uses its own runner/display
and does not go through `vortex-bench::runner`. When adding
`-d gh-json-v3` to it:

1. Teach `vector-search-bench::main` to accept the `--format=gh-json-v3`
   CLI flag (mirror `DisplayFormat::GhJsonV3`).
2. The scan phase already produces a `ScanTiming` plus per-flavor metrics.
   Convert each into one or more `ClassifiedMeasurement` records with
   `metric_kind = VectorSearchTime / VectorSearchCount / VectorSearchBytes`.
3. Add a new CI workflow `.github/workflows/vector-bench.yml` that runs
   the binary on merge to `develop`, POSTs to `/api/ingest`. Not gated on
   launch, but nice to add in the same pass.

## Review checklist before merging emitter changes

- [ ] For every benchmark in v2's `data.json.gz`, the new emitter
      produces a `ClassifiedMeasurement` that, when inserted, matches
      what the migrator produced for the same historical record.
- [ ] No new emitter uses the `name` string to smuggle dimensions.
- [ ] Every `data_descriptor` value is documented by an example in
      `05-schema.md`.
- [ ] `to_v3_json()` output compares clean against an `insta` golden
      snapshot for one record of each measurement type. Snapshots
      scrub environment-dependent fields (`commit_sha`, `env_triple`)
      via redactions so they're reproducible across machines.
- [ ] No cross-format ratio records are emitted (`vortex:parquet size`,
      `vortex:lance ratio compress time`, etc.) - ratios are DuckDB
      views, not rows.
- [ ] `CompressionTimingMeasurement` emits `compression_encode` or
      `compression_decode` (not `compression_time`); each
      `compress/decompress time/<name>` today maps to exactly one
      metric_kind.
- [ ] Raw file-size `CustomUnitMeasurement` records emit as
      `compression_size` with `value_bytes` set; the
      `compressed_size as f64` roundtrip is gone (we store the bytes
      as an `i64`, not a float).

## Review checklist before merging `/api/ingest`

- [ ] Unknown JSON field → serde error → HTTP 400 (not silent ignore).
- [ ] Duplicate POST → `updated == N`, `inserted == 0`.
- [ ] Malformed commit SHA → 400 with pointer to offending field.
- [ ] Schema version mismatch → 409 with both versions in the body.
- [ ] Bearer token comparison is constant-time.
- [ ] Integration test round-trips a fixture payload via the actual
      serde codepath.
