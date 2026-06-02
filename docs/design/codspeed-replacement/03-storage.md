<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# 03 — Storage for microbenchmark runs

Bullet: *"A nice storage for each run, impl here."*

This slice designs how the microbenchmark results currently locked in CodSpeed's
SaaS get stored **alongside** the existing macro-benchmark results, reusing the
project's own self-hosted stack (`benchmarks-website/server`, DuckDB, the S3
result bucket, and the `GitHubBenchmarkRole` OIDC role). It is grounded entirely
in components that already exist in this repo; new code is additive.

---

## 1. The CURRENT storage (what already exists)

There are two generations of storage in the repo. The **v3** stack is the live
target; v2 is legacy and being migrated away from.

### 1.1 v3 DB — DuckDB, server-owned (source of truth)

`benchmarks-website/server` (crate `vortex-bench-server`) owns a single DuckDB
file (`VORTEX_BENCH_DB`, canonically `/var/lib/vortex-bench/bench.duckdb`).
`server/src/db.rs::open` creates one root `duckdb::Connection`, applies
`COMMITS_DDL`, then iterates `family::FAMILIES` applying each family's
`schema_ddl`. All DB work runs in `spawn_blocking` with a 4-permit read
semaphore (`READ_CONCURRENCY_LIMIT`).

Schema (verbatim from `server/src/schema.rs`):

- **`commits`** (dim, the only dim table). PK `commit_sha TEXT`. Columns:
  `timestamp TIMESTAMPTZ NOT NULL`, `message`, `author_name`, `author_email`,
  `committer_name`, `committer_email` (all nullable), `tree_sha TEXT NOT NULL`,
  `url TEXT NOT NULL`. Upserted on every ingest from the envelope `commit` block.
- **`query_measurements`** — fact. PK `measurement_id BIGINT`. Natural key
  `(commit_sha, dataset, dataset_variant, scale_factor, query_idx, storage,
  engine, format)`. Value cols `value_ns BIGINT`, `all_runtimes_ns BIGINT[]`,
  memory quartet `peak_physical/peak_virtual/physical_delta/virtual_delta`
  (all-four-or-none), `env_triple TEXT`.
- **`compression_times`** — natural key `(commit_sha, dataset, dataset_variant,
  format, op)` where `op ∈ {encode, decode}`; `value_ns`, `all_runtimes_ns`.
- **`compression_sizes`** — natural key `(commit_sha, dataset, dataset_variant,
  format)`; `value_bytes BIGINT`. Ratios computed at read time, not stored.
- **`random_access_times`** — natural key `(commit_sha, dataset, format)`;
  `value_ns`, `all_runtimes_ns`.
- **`vector_search_runs`** — natural key `(commit_sha, dataset, layout, flavor,
  threshold)`; `value_ns`, `all_runtimes_ns`, side counters `matches`,
  `rows_scanned`, `bytes_scanned`, `iterations`.

`measurement_id` is **server-internal**: `db.rs::measurement_id_*` xxhash64 over
a per-table `table_name` seed (`hasher_for`) plus `commit_sha` plus the dim
tuple. It never crosses the wire; re-emitting the same `(commit, dim)` pair is
the upsert case (`ON CONFLICT (measurement_id) DO UPDATE`). `SCHEMA_VERSION = 1`.

The per-table registry in `server/src/family.rs` (`struct Family`) is the
**spine**: each entry ties `table_name`, `chart_slug_prefix`,
`group_slug_prefix`, `schema_ddl`, `measurement_id`, `apply_record`,
`collect_chart_for_key`, `collect_groups`, `row_count`. `FAMILIES` drives DDL
apply, ingest dispatch, chart/group collection, `/health` row counts, and
`schema::TABLES` (= `commits` + every family). **Adding a sixth fact table is one
new `const Family` + one `FAMILIES` entry + one DDL const + one `records` struct
+ one `Record` variant** — the compiler enforces every hook is populated.

### 1.2 v3 ingest API

`POST /api/ingest` (`server/src/ingest.rs`), bearer-gated by
`auth::require_bearer` (`INGEST_BEARER_TOKEN`). Accepts one `Envelope`
(`records.rs`): `run_meta { benchmark_id, schema_version, started_at }` +
`commit` (CommitInfo, `sha` renamed from `commit_sha`) + `records[]`
(`#[serde(tag = "kind")]`, `deny_unknown_fields`). One DuckDB transaction per
POST, all-or-nothing, returns `{ inserted, updated }`. 64 MiB body limit. Write
conflicts retried up to 128×. On success: `cache.invalidate()` +
`read_store.schedule_rebuild()`.

### 1.3 Read path

`server/ARCHITECTURE.md`: a materialized `ReadGeneration` built from one DuckDB
snapshot serves `/api/groups`, `/api/chart/{slug}` (latest-100 default),
`/api/group/{slug}`, and versioned shard artifacts, each pre-encoded identity /
gzip / brotli with ETag. `?n=all`/non-default windows fall back to DB through
`QueryCache` single-flight + read semaphore. Ingest schedules a background
rebuild; `RETAINED_PREVIOUS_GENERATIONS = 8` keeps stale shard URLs resolving.
(`src/api.js` + `src/config.js` are the **legacy v2** React frontend hitting
`/api/metadata` + `/api/data`; the v3 server renders its own HTML/charts.)

### 1.4 S3 layout

Two distinct buckets:

- **`vortex-ci-benchmark-results`** (v2 raw, public, `--no-sign-request`):
  - `data.json.gz` — append-only gzipped JSONL of all macro measurements
    (concatenated by `scripts/cat-s3.sh` using S3 `If-Match`/`If-None-Match`
    optimistic concurrency on the object ETag).
  - `commits.json` — append-only JSONL of commit metadata
    (`scripts/commit-json.sh` builds one record; `cat-s3.sh` appends it).
  - `file-sizes-*.json.gz` — per-suite size dumps.
  - Written by `bench.yml`'s `commit-metadata` and `Upload Benchmark Results`
    steps after assuming `arn:aws:iam::245040174862:role/GitHubBenchmarkRole`
    (OIDC, `us-east-1`). PR runs read `data.json.gz` for the comparison base
    (`bench-pr.yml` → `s3-download.py` + `grep $base_commit_sha`).
- **`vortex-benchmark-results-database`** (v3 backups, private): hourly
  snapshots under `v3-backups/<ts>.tar.gz`, each a `tar czf` of a snapshot dir
  holding `schema.sql` + one `<table>.vortex` per table
  (`COPY ... TO ... (FORMAT vortex)` via `POST /api/admin/snapshot`,
  `ops/backup.sh`, `VortexBenchServerRole`, 7-day lifecycle). Restore =
  `read_vortex()` per table (`ops/BOOTSTRAP.md` Phase 8).

### 1.5 Producer (today)

`vortex-bench` emits macro results: `-d gh-json -o results.json` (legacy v2
JSON, → S3) and `--gh-json-v3 results.v3.jsonl` (bare v3 records → wrapped by
`scripts/post-ingest.py` → `/api/ingest`). `vortex-bench/src/measurements.rs`
and `display.rs` produce the measurement shapes; `src/v3.rs` maps each
measurement type 1:1 to a `kind`. `src/output.rs` writes to
`target/vortex-bench/<benchmark-id>/results.json`.

### 1.6 Microbenchmarks today (the gap)

`.github/workflows/codspeed.yml` builds Criterion-style benches
(`*/benches/*.rs`, ~hundreds across `vortex-array`, `vortex`, encodings, etc.)
with `cargo codspeed` and uploads to CodSpeed SaaS. Two regimes:

- **`bench-codspeed`**: 8 shards (`matrix.shard` 1..8, each a `packages` set),
  `mode: "simulation"` — CodSpeed's CPU simulator producing **instruction
  counts** (deterministic, low-noise).
- **`bench-codspeed-cuda`**: 3 shards, `mode: "walltime"` — real **wall-clock /
  cycle** timings on `g5` GPU runners.

So each micro result is keyed by `(commit, arch, runner, shard, benchmark_id,
metric)` where `metric ∈ {instructions, walltime_ns}` — and **none of this is in
our DB**; it lives only in CodSpeed.

---

## 2. Data model for microbenchmark results

A microbenchmark is a different *(dim shape, value shape)* from the five
macro families, so per `schema.rs` "one fact table per (dim shape, value shape)"
principle it gets **its own family**, not a discriminator column shoved into an
existing table. One row per `(commit, benchmark, env, metric)`.

### 2.1 Dimensions

| Dim                | Source                                                        | Notes |
|--------------------|---------------------------------------------------------------|-------|
| `commit_sha`       | `${{ github.sha }}` (already the `commits` PK)                | FK to `commits` |
| `benchmark_id`     | Criterion bench id, e.g. `take_primitive/dict/1024`           | full nested id incl. parameters |
| `crate`            | `matrix.packages` member, e.g. `vortex-array`                 | enables per-crate grouping/sharding |
| `bench_group`      | leading segment of the bench id, e.g. `take_primitive`        | chart grouping key |
| `metric`           | `instructions` \| `walltime_ns` \| `cycles`                   | the discriminator **within** this family's value shape |
| `arch`             | `x86_64` / `aarch64` (from `Triple::host()`, see below)       | x86 simulation vs gpu walltime |
| `runner`           | `amd64-medium` / `g5` (runs-on tag)                           | distinguishes hardware class |
| `shard`            | `matrix.shard`                                                | provenance / debugging; not in the dim hash |
| `env_triple`       | `arch-os-env` (already a column convention)                   | reused verbatim |

`metric` is the one place a discriminator is justified: instruction counts and
walltime are the *same value shape* (a single scalar + an optional sample
vector) measured on the *same dim tuple*, so collapsing them into one table with
a `metric` discriminator avoids a 1:N split — consistent with how
`compression_times` keeps `op ∈ {encode,decode}` in one table.

### 2.2 SQL DDL (new family — extends the existing DB)

New const in `server/src/schema.rs`:

```sql
-- MICROBENCHMARKS_DDL
CREATE TABLE IF NOT EXISTS microbenchmarks (
    measurement_id   BIGINT      PRIMARY KEY NOT NULL,  -- server-internal xxhash64
    commit_sha       TEXT        NOT NULL,              -- FK -> commits.commit_sha
    crate            TEXT        NOT NULL,              -- e.g. 'vortex-array'
    bench_group      TEXT        NOT NULL,              -- e.g. 'take_primitive'
    benchmark_id     TEXT        NOT NULL,              -- full criterion id incl. params
    metric           TEXT        NOT NULL,              -- 'instructions'|'walltime_ns'|'cycles'
    arch             TEXT        NOT NULL,              -- 'x86_64'|'aarch64'
    runner           TEXT        NOT NULL,              -- runs-on hardware class
    value            BIGINT      NOT NULL,              -- median: instr count or ns
    all_samples      BIGINT[]    NOT NULL,              -- per-iteration samples (cold storage, like all_runtimes_ns)
    sample_count     INTEGER     NOT NULL,              -- iterations measured (side count, not in hash)
    shard            INTEGER,                           -- provenance; nullable, not in hash
    env_triple       TEXT                               -- 'x86_64-linux-gnu'
);
```

Natural key (the dim hash, in `db.rs::measurement_id_microbenchmark`):
`(commit_sha, crate, bench_group, benchmark_id, metric, arch, runner)`. `shard`,
`sample_count`, `all_samples` are deliberately **not** in the hash (side
counts / cold storage), matching `vector_search_runs.iterations` precedent. The
`metric` column being in the key is what lets one `benchmark_id` carry both an
`instructions` row and a `walltime_ns` row without collision.

`value` is `BIGINT` because both metrics are integral (instruction counts;
nanoseconds). If CodSpeed-style fractional walltime is ever needed, add a
sibling `value_double DOUBLE` nullable column (cheap, per principle 3) rather
than widening `value`.

### 2.3 Registry wiring (the only server edits)

- `schema.rs`: add `MICROBENCHMARKS_DDL` const (above).
- `records.rs`: add `Microbenchmark` struct (`deny_unknown_fields`) + a
  `Record::Microbenchmark(Microbenchmark)` variant with `kind =
  "microbenchmark"`; extend `Record::commit_sha()`/`kind()`.
- `ingest.rs`: add `insert_microbenchmark` (validate `metric` against
  `{instructions, walltime_ns, cycles}`, mirror the `ON CONFLICT DO UPDATE`).
- `db.rs`: add `measurement_id_microbenchmark` (`hasher_for("microbenchmarks")`
  + the natural-key tuple).
- `family.rs`: add `const MICROBENCHMARKS: Family { table_name:
  "microbenchmarks", chart_slug_prefix: "mb", group_slug_prefix: "mbg", ... }`
  and append to `FAMILIES`. Slug prefixes `mb`/`mbg` are unused (the
  `slug_prefixes_are_distinct` test will guard this).
- `api/charts.rs` + `api/groups.rs`: add `collect_microbenchmark_chart` /
  `collect_microbenchmark_groups`. A natural chart = one `(crate, bench_group,
  benchmark_id, metric, arch)` time series over commits; a natural group =
  per-`crate` (or per-`bench_group`) collection.

`schema::TABLES`, `/health` row counts, snapshot, and backup completeness check
all pick the new table up automatically because they derive from `FAMILIES` —
**except** `ops/backup.sh`'s hardcoded `required_files` array, which must gain
`microbenchmarks.vortex` (see §4) and the `BOOTSTRAP.md` restore `INSERT INTO`
list.

### 2.4 S3 raw-artifact layout (full blobs + profiles)

The DB stores the reduced scalar series for charts. Full per-run blobs (raw
Criterion/CodSpeed estimates JSON, and any profiles) are bulky and rarely read,
so they go to S3 keyed for direct lookup, **not** into DuckDB:

```
s3://vortex-ci-benchmark-results/micro/<commit_sha>/<arch>/<crate>/<shard>/estimates.json.gz
s3://vortex-ci-benchmark-results/micro/<commit_sha>/<arch>/<crate>/<shard>/raw.jsonl.gz   # the v3 micro records actually ingested
s3://vortex-ci-benchmark-results/micro/<commit_sha>/<arch>/<crate>/<shard>/profile.pb.gz  # optional pprof, if captured
```

Reuse the existing bucket + `GitHubBenchmarkRole` (already has write there).
Keys are content-addressed by `(commit, arch, crate, shard)` so they are
immutable and idempotent on re-run — no `cat-s3.sh` concat needed for these
(unlike `data.json.gz`, micro artifacts are one-object-per-run, written with a
plain `aws s3 cp`). A `micro/index.jsonl` is unnecessary because the DB row's
dim tuple deterministically reconstructs the key prefix.

---

## 3. Ingestion path

**Chosen: S3-then-ingest is NOT needed; direct ingest via the existing
`/api/ingest`** is the primary path, with S3 used only for the raw-blob archive.
This reuses every existing pattern with zero new server endpoints.

Per CodSpeed shard job, after `cargo codspeed run`:

1. **Convert** CodSpeed/Criterion output → bare v3 `microbenchmark` JSONL. A new
   small converter (a `--micro-json-v3` mode in `vortex-bench`, or a
   `scripts/codspeed-to-v3.py`) reads the `target/codspeed/...` estimates and
   emits one bare record per `(benchmark_id, metric)`:
   ```json
   {"kind":"microbenchmark","commit_sha":"<sha>","crate":"vortex-array",
    "bench_group":"take_primitive","benchmark_id":"take_primitive/dict/1024",
    "metric":"instructions","arch":"x86_64","runner":"amd64-medium",
    "value":418923,"all_samples":[...],"sample_count":50,"shard":2,
    "env_triple":"x86_64-linux-gnu"}
   ```
   `arch`/`env_triple` come from `Triple::host()` exactly as `measurements.rs`
   already does for `QueryMeasurementJson::env_triple`.
2. **Archive raw blobs to S3** (`aws s3 cp` to the `micro/<...>` keys in §2.4)
   after assuming `GitHubBenchmarkRole` — the same `configure-aws-credentials`
   step the bench jobs already use.
3. **Ingest to the v3 server** by reusing `scripts/post-ingest.py` verbatim
   (it is record-kind-agnostic — it only wraps bare records in an envelope and
   POSTs):
   ```bash
   python3 scripts/post-ingest.py micro.v3.jsonl \
     --server "${{ vars.V3_INGEST_URL }}" \
     --commit-sha "${{ github.sha }}" \
     --benchmark-id "codspeed-shard-${{ matrix.shard }}" \
     --repo-url "${{ github.server_url }}/${{ github.repository }}"
   ```
   `INGEST_BEARER_TOKEN` is already wired into the bench environment. The
   server's `ON CONFLICT (measurement_id) DO UPDATE` makes shard re-runs and
   retries idempotent.

Why direct ingest over S3-then-ingest: there is no batch reducer step (each
shard's records are already final scalars), the envelope fits well under 64 MiB
(hundreds of benches × few metrics × ~200 B ≈ low single-digit MiB), and it
avoids standing up a new "ingest from S3" job. The S3 copy is purely an archive
of the raw evidence, decoupled from the chart-serving path.

The CodSpeed `CodSpeedHQ/action` step is replaced by these three steps; the
`cargo codspeed build`/`run` steps stay (they produce the measurements we
convert). `bench-taskset.sh` pinning is preserved.

---

## 4. Retention, indexing, baseline vs PR

**Indexing for "history of benchmark X on arch Y".** The hot query is
`SELECT c.timestamp, m.value FROM microbenchmarks m JOIN commits c USING
(commit_sha) WHERE benchmark_id = ? AND metric = ? AND arch = ? ORDER BY
c.timestamp`. DuckDB is columnar and the table is small (hundreds of benches ×
~daily commits × few metrics × 1–2 archs), so a full scan is already fast and
the materialized latest-100 read path makes the common case zero-SQL. If/when
needed, add a DuckDB ART index: `CREATE INDEX micro_series ON microbenchmarks
(benchmark_id, metric, arch, commit_sha)`. The `measurement_id` PK already
indexes the upsert path.

**Retention.** Same model as the macro families: the DuckDB file is the durable
source of truth and is kept indefinitely (rows are tiny — a scalar + a short
sample vector). The S3 raw blobs in `micro/` are the bulky part; add an S3
lifecycle rule (mirroring `v3-backups-7d`) to expire `micro/` raw artifacts
after e.g. 90 days while the DB keeps the chartable scalars forever. Hourly
`ops/backup.sh` snapshots now include `microbenchmarks.vortex` (add it to the
`required_files` completeness array and the `BOOTSTRAP.md` restore list).

**Baseline/develop vs PR runs.** The DB has no branch column today and the
`commits` table is the only dim — develop commits and PR-head commits are both
just `commit_sha`s. The existing scheme distinguishes them implicitly: only
`develop` push runs ingest (`bench.yml` is `on: push: branches: [develop]`),
while PR runs (`bench-pr.yml`) **compare against a base and comment, but do not
ingest**. We keep this exactly:

- **develop micro runs** (the new steps added to `codspeed.yml`'s `push:
  develop` trigger) ingest to the DB → they become the chart time series.
- **PR micro runs** do **not** ingest. Instead they fetch the develop baseline
  for the relevant `benchmark_id`s (either via the v3 read API
  `/api/chart/{slug}?last=1`, or by `grep`ing the S3 `micro/<base_sha>/...`
  blob like `bench-pr.yml` does today with `data.json.gz`) and post a PR
  comment with the delta. This preserves "develop is the baseline" without a
  branch column. The base develop SHA is found exactly as `bench-pr.yml` does:
  query `bench.yml` (here: `codspeed.yml`) successful runs on `develop` for the
  latest `head_sha`.

If a branch dimension is ever wanted in the DB (e.g. to chart PR experiments),
add a nullable `branch TEXT` column rather than a new table — but that is out of
scope for cutover.

---

## 5. Migration / coexistence with CodSpeed

**Start fresh, no backfill of historical instruction counts.** CodSpeed's
historical series live in its SaaS and are not exportable into our exact dim
model cheaply; instruction counts are only meaningful relative to a stable
toolchain/runner, so a clean baseline from cutover is acceptable. (Contrast with
the macro v2→v3 path, which had a public S3 JSON dump and a bug-for-bug
`migrate/classifier` port — there is no analogous public dump for CodSpeed.)

**Coexistence during transition** (dual-write, lowest risk):

1. Phase A — land the `microbenchmarks` family (server + `SCHEMA_VERSION` stays
   `1`; adding a new `kind` is additive and old envelopes still validate). Add
   the converter + the three new steps to `codspeed.yml` **in addition to** the
   existing `CodSpeedHQ/action` step. Both stores receive every develop run.
   Verify the DB series and the self-hosted charts match CodSpeed for a few
   weeks.
2. Phase B — flip PR gating: keep CodSpeed's PR comment as the source of truth
   while our PR-delta comment is validated side by side.
3. Phase C — remove the `CodSpeedHQ/action` steps and `CODSPEED_TOKEN`; drop the
   CUDA `walltime` runs into the same `microbenchmarks` table (`metric =
   walltime_ns`, `arch = x86_64`/`runner = g5`).

Rollback at any phase is trivial: re-enable the CodSpeed action; our additive
rows and S3 keys are harmless if unused. No `SCHEMA_VERSION` bump is required at
any point, so `post-ingest.py`'s hardcoded constant stays in sync untouched.

---

## 6. Open questions / decisions for the user

1. **Converter home**: add a `--micro-json-v3` emitter to `vortex-bench` (Rust,
   reuses `Triple::host()` + `GIT_COMMIT_ID`, type-checked against the wire
   shape) **or** a standalone `scripts/codspeed-to-v3.py` (stdlib-only, like
   `post-ingest.py`)? Recommend the Rust emitter for the wire-shape coupling
   guarantee, but it means parsing `cargo codspeed`/Criterion output.
2. **Metric set**: store only `instructions` for the simulation shards and
   `walltime_ns` for CUDA, or also derive `cycles`? CodSpeed exposes more than
   instruction count — confirm which metrics matter for regression gating.
   (Coordinate with bullet 2's measurement plan.)
3. **`shard` permanence**: keep `shard` as a stored provenance column, or is it
   purely a CI sharding detail that should be dropped once `crate` is recorded?
   (It is already excluded from the dim hash, so it is safe to keep or drop.)
4. **Arch coverage**: are we committing to a single x86_64 runner class for
   instruction counts, or will aarch64 be added? The `arch`/`runner` columns are
   in the key now so multi-arch is free, but the chart UX (one series per arch)
   needs a product decision.
5. **Backfill stance**: confirm "start fresh" is acceptable, i.e. we lose the
   pre-cutover CodSpeed history rather than attempting an export.
6. **Raw-blob retention window**: 90 days for `micro/` S3 artifacts, or longer?
   The DB keeps scalars forever regardless; this only affects how far back the
   full per-iteration sample blobs and profiles are retrievable.
7. **Profiles**: do we want to capture pprof/Polar Signals profiles per micro
   run into `micro/.../profile.pb.gz` now, or defer (the bench/PR macro jobs
   already wire Polar Signals; micro does not yet)?
