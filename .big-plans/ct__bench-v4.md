# benchmarks-website migration to hosted Postgres + Next.js on Vercel — big-plans plan

## Current State

```yaml
status: executing
branch: ct/bench-v4
planning_sub_flow: null
current_phase: "Phase 1: RDS + schema + hash port"
phase_index: 1
current_pr: null
pr_index: 3
outstanding_must_fix: 0
deferred_items_total: 12
last_user_touchpoint: 2026-05-28T14:10:00Z
last_user_touchpoint_what: "PR-1.2 complete (confidence: high, deferred: 4)"
subagent_invocations_this_pr: 0
subagent_invocations_total: 10
review_cycles_this_pr: 0
phase_entry_sha: d8f12ebbb
phase_end_cycle: 0
phase_end_reject_cycles: 0
last_phase_end_verdict: null
current_pr_is_ci_reopen: null
last_commit: bdd53140c
last_cycle_commits: []
```

## Context

The `benchmarks-website/` subsystem is the public face of Vortex's continuous-benchmark numbers. The current implementation is a Rust/Axum server with an embedded DuckDB database running on an EC2 instance, with a custom systemd-driven deploy + hourly S3 backup pipeline, bearer-token-authenticated CI ingestion at `POST /api/ingest`, an in-process precomputed-artifact read cache (`read_model.rs`), and a Vite/React static SPA. CI runs (one writer on `ubuntu-latest`, two on `bench-dedicated`, eleven from the SQL bench matrix on `bench-dedicated`) fan in to ~14 parallel envelope POSTs per push to `develop`. The subsystem just completed a v2→v3 cutover (dual-write soak, single deletion commit `7efbcacd2`); the rewrite was earned through one failed attempt (`cc06c6022` revert) and is well-tested operationally.

The migration target is the same data model on AWS RDS Postgres `db.t4g.micro` (account `245040174862`, region `us-east-1`, single-AZ) fronted by RDS Proxy, with CI writing directly via GitHub OIDC → AWS IAM → RDS-IAM-auth tokens (replacing the Axum POST and the bearer token), and a stateless Next.js 15 read service on Vercel using server components + `unstable_cache` + `revalidateTag` against the same DB. Edge CDN replaces the in-process artifact cache. Decommission targets are `benchmarks-website/server/`, `benchmarks-website/ops/`, the bearer tokens, `/api/admin/*`, the custom backup pipeline, and systemd timers. Success is: every existing chart renders byte-equivalently against the new substrate; the next CI run after cutover idempotently upserts existing `measurement_id` rows; the EC2 instance is decommissioned and the runbook is replaced by managed-Postgres + Vercel.

**Work shape**: migration — substrate change (DuckDB → Postgres, Axum → Next.js, EC2 → Vercel) with bit-exact preservation of `measurement_id` (xxhash64), v3 envelope wire format, 6-table schema shape, `ON CONFLICT (measurement_id) DO UPDATE` upsert semantics, captured metrics, and chart URL shape. The highest-leverage insight is **Behavior-preservation**: every preserved invariant must appear as an explicit acceptance criterion and be pinned by a test.

The orchestrating branch for this work is `ct/bench-v4`. Per user direction at planning kickoff, all child PRs land on `ct/bench-v4` rather than `develop`; the final integration of `ct/bench-v4` into `develop` happens once via squash-merge after the full migration ships.

**Orchestration note** (read on resume): the big-plans state machine operates on `ct/bench-v4` throughout — plan-edits, implementation commits, inner-loop gauntlet, and phase-end gauntlet all happen on this branch. At each phase end, after gauntlet's verdict is `accept`, the assistant cherry-picks the phase's commits (range `phase_entry_sha..HEAD`) onto a fresh `claude/phase-N-<slug>` branch created at `phase_entry_sha`, pushes it, and opens a review-only GH PR (`claude/phase-N-<slug>` → `ct/bench-v4`) labeled "review only — do not merge". This preserves per-phase reviewability on GitHub without bifurcating the working branch. The single squash-merge of `ct/bench-v4` → `develop` happens at end-of-Phase-5 via a final GH PR.

## Out of scope

- Changing the v3 envelope wire format in `vortex-bench/src/v3.rs` (this stays).
- Changing the 6-table schema shape (column order, nullability, dim-tuple membership).
- Changing the xxhash64 `measurement_id` algorithm in `benchmarks-website/server/src/db.rs:162-257` — the new ingest writer reproduces it bit-identically.
- Adding materialized views or DB triggers.
- Changing the `commits` upsert semantics (`ON CONFLICT (commit_sha) DO UPDATE SET <all-but-PK>`).
- Adding a non-Postgres read path (e.g., S3 + JSON files) during steady state.
- A multi-region Postgres setup; single region is sufficient.
- Authenticated user accounts on the read side; everything stays public-read.
- Backfilling rows older than the current v3 DB (cutover seeds from the live DuckDB snapshot only).
- Editing the v2 React/Vite frontend (`src/`, `index.html`, `vite.config.js`, `package.json`, top-level `Dockerfile`, `docker-compose.yml`, `publish-benchmarks-website.yml`) outside the dedicated final-cutover PR.
- Re-introducing best-effort (`continue-on-error: true`) on v3/v4 CI ingest steps.
- Persisting `measurement_id` on the wire — it stays server-internal.
- The `/api/admin/sql` operator console (replaced by managed-Postgres console / psql access).

## Prior art / external references

- **`/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/AGENTS.md`** — subsystem playbook including "Footguns we have already hit"; the `SCHEMA_VERSION` lockstep rule and wire-shape coupling across `records.rs`, `v3.rs`, `classifier.rs` carry over.
- **`/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/README.md`** — the v2→v3 cutover playbook; same dual-write → soak → single deletion-commit pattern applies to v3→Postgres.
- **`/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/migrate/src/verify.rs`** — `VerifyReport { matched_groups, only_in_v2, only_in_v3, ChartDiff }` structural diff between substrates. Direct template for the dual-write verification harness.
- **`/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/migrate/src/migrate/mod.rs:31-35`** — uses `vortex_bench_server::db::measurement_id_*` to keep hash compatibility across the v2→v3 boundary. The same dependency edge applies to the new ingest writer.
- **`/Users/connor/spiral/vortex-data/vortex4/.github/workflows/compat-gen-upload.yml`** — dry-run + `environment: compat-upload` (manual-approval gate) + OIDC role assumption. Structural template for any approval-gated migration step (schema deploy, prod-data ingest, decommission).
- **`/Users/connor/spiral/vortex-data/vortex4/.github/workflows/bench.yml:9-11,21-24,98-101`** — production GH-OIDC → AWS IAM pattern reference (`GitHubBenchmarkRole` in account `245040174862`, used for benchmark-S3-cache uploads today). Mirror the trust-policy SHAPE (sub-claim wildcard, `aws-actions/configure-aws-credentials@d979d5b3a71173a29b74b5b88418bfda9437d885 # v6`) for the NEW `GitHubBenchmarkSchemaRole` + future `GitHubBenchmarkIngestRole` created by PR-1.1's `provision.sh` in the same account `245040174862`.
- **`/Users/connor/spiral/spiraldb/.github/workflows/deploy-spiraldb.yml:37-89`** — `deploy-migrations` job blueprint (OIDC → cloud identity → managed Postgres → `prisma migrate deploy`). GCP flavor; translate step-by-step to AWS.
- **`/Users/connor/spiral/spiraldb/.github/workflows/prisma-isolation.yml`** — 49-line workflow rejecting PRs that mix migration files with code files. The schema-isolation discipline benchmarks-website will need post-cutover; copy verbatim with a different migration directory.
- **`/Users/connor/spiral/spiraldb/spiraldb/`** — production Rust/Axum + `sea-orm` + `sqlx-postgres` + Prisma-managed schema service. Reference for connection-pool config + the sidecar-proxy operational pattern. No `sqlx::migrate!` in the Spiral ecosystem; choosing it for this migration is new ground.
- **`vortex-bench/src/v3.rs:32-53`** — per-binary→`V3Record` mapping; the bench harnesses that produce envelopes are the immutable producer interface.

## Architecture

```
                          ┌─────────────────────────────────────┐
  ~14 CI writers          │   GitHub Actions OIDC → AWS STS     │
  per `develop` push      │   AssumeRoleWithWebIdentity         │
  (bench.yml, sql-        │   GitHubBenchmarksIngestRole        │
   benchmarks.yml,        │   (or GitHubBenchmarkRole + policy) │
   v3-commit-metadata)    └────────────┬────────────────────────┘
                                       │ IAM-auth token
                                       ▼
                           ┌────────────────────────┐    pooler
                           │ AWS RDS Postgres       │◀───  RDS Proxy
                           │  (db.t4g.micro)        │      (IAM-auth)
                           │ 6 tables, BIGINT[]     │
                           │ composite indexes      │
                           │  (dim_tuple,           │
                           │   commit_timestamp     │
                           │   DESC)                │
                           └─────────┬──────────────┘
                                     │ read-replica or
                                     │ same primary
                                     ▼
                         ┌─────────────────────────────┐
                         │ Next.js on Vercel           │
                         │  server components +        │
                         │  unstable_cache +           │
                         │  revalidateTag              │
                         │  edge-CDN response cache    │
                         └─────────────────────────────┘
                                     │
                                     ▼
                              Public read users
```

The system splits cleanly into three independent surfaces. The **writer** is a small (script or binary) ingest tool that produces bit-identical `measurement_id` xxhash64 values from a v3-envelope JSONL input and executes `INSERT ... ON CONFLICT (measurement_id) DO UPDATE` against Postgres. It is invoked from each CI workflow in place of `python3 scripts/post-ingest.py ... --server $V3_INGEST_URL`. The **reader** is a Next.js application whose server components issue parameterized SQL through a connection pool, wrap responses in `unstable_cache` with per-chart tags, and serve from the edge CDN. The **schema** lives in a single source of truth (the chosen migration tool) and is deployed to Postgres via a CI workflow gated by `prisma-isolation`-style discipline.

The migration's load-bearing invariant is the `measurement_id`: it is a server-internal xxhash64 over `(table_tag || 0x00 || commit_sha || dim_fields...)` with length-prefixed strings (little-endian u64), `Option<String>` as `0x00`/`0x01+write_str`, `i32` as 4 LE bytes, `f64` as `to_bits()` LE u64, finished `as i64`. Existing rows seeded from the DuckDB dump carry these IDs; every subsequent CI write must hash to the same bytes to upsert correctly. Producing those bytes from the new writer is the migration's central correctness obligation.

The read path's bimodal access pattern (small `LIMIT n` windows up to 1000, plus `?n=all` whole-history downsampled to ~1000 buckets client-side via LTTB) is well-served by composite indexes alone — no materialized views needed. Cold-start cost on stateless Vercel is the main read-side budget concern; the in-process `read_model.rs` artifact regeneration was only 100% materialized on EC2 because the process was long-lived. The Next.js layer needs ISR / `revalidateTag` semantics so the warm path is cache-only.

CI writers and the Next.js reader connect through a pooler — RDS Proxy if AWS RDS, provider-managed pooler if Neon/Supabase/Crunchy, pgbouncer self-hosted if none. The pooler choice is downstream of the Postgres flavor choice. Connection budget is bounded by Vercel's serverless concurrency × pool-per-invocation × a small connections-per-pool (≤4) constant.

CI network reach is straightforwardly "public + IAM" — the existing OIDC setup is on `ubuntu-latest` (public) and `bench-dedicated` (AWS-hosted via `runs-on`) runners, both with public egress. VPC + self-hosted runners would lose the `v3-commit-metadata.yml` job's `ubuntu-latest` slot unless it moves to self-hosted; that's a large operational expansion for marginal security gain.

Cutover style follows the v2→v3 template that just shipped: dual-write for ~1 week, soak under real CI traffic, structural-diff verification via a `migrate/verify.rs`-shaped tool, then one PR that deletes `benchmarks-website/server/`, `benchmarks-website/ops/`, the bearer token, the v2 React frontend, and the systemd-driven publishing workflow.

## Key decisions

| Decision | Choice | Rationale |
|---|---|---|
| Branching / merge target | Per-phase child branches → squash-merge into `ct/bench-v4`; final `ct/bench-v4` → `develop` once at end | User pick (Q1). Per-phase GH PRs give review granularity at GitHub level without per-logical-PR overhead. Mechanics: big-plans state machine runs on the phase's child branch; phase-end Step 3.5 Proceed opens child→ct/bench-v4 PR and awaits squash-merge before the next phase's child branch is spawned. |
| Postgres flavor | **RDS Postgres on `db.t4g.micro`** in AWS account `245040174862`, region `us-east-1`, single-AZ | User pick (Q2). ~$13/mo floor + storage + IO. Latest Postgres (no Aurora version lag). Always-on (no cold-start). IAM auth native. Aurora's storage/replication/throughput benefits are unused at this scale. Account `245040174862` is the existing bench-S3 / `GitHubBenchmarkRole` account; CI already authenticates to it via OIDC. Operator uses an SSO `bench` profile to act against it (`aws sso login --profile bench`). Cross-account implication for Phase 3: v3 EC2 (DuckDB source) lives in personal account `375504701696/us-east-2`; the one-shot load needs operator creds for both accounts (or DuckDB snapshot via cross-account S3 bucket policy). See Risks #10. |
| Connection pooler | **RDS Proxy** (front of RDS) with IAM-auth pass-through | Locked by Q2. Raises effective connection ceiling from ~100 (direct) to several thousand. Single pooler covers both CI writers and Vercel reader. |
| Ingest writer language | **Pure Python** — extend `scripts/post-ingest.py`; port xxhash64 to Python with golden-vector tests against the existing Rust source-of-truth | User pick (Q4). Avoids adding sqlx + aws-sdk-rds Rust deps. Hash port is bounded by tests; Rust impl in `server/src/db.rs` stays the source of truth. Uses `psycopg[binary]` + `boto3.client('rds').generate_db_auth_token`. |
| Postgres schema deploy tool | **In-house `scripts/migrate-schema.py`** (~30-50 LOC) + plain SQL files under `migrations/` | User pick (Q5a). Tracks via `_applied_migrations` table; applies pending `00N_name.sql` in name order; CI workflow invokes with OIDC + IAM. Zero new tools / languages. |
| One-shot historical data load | **Retarget `benchmarks-website/migrate/`** (existing Rust crate) for DuckDB→Postgres bulk load | Q5b — natural reuse. The crate already reads DuckDB via the `duckdb` crate and reuses `vortex_bench_server::db::measurement_id_*`. Add a `--postgres-target` mode + a Postgres bulk-insert path. Deleted post-cutover per AGENTS.md throwaway-migrator pattern. |
| CI network reach | **Public + IAM** — RDS Proxy public endpoint, security group `0.0.0.0/0` because IAM is the gate, sslmode=verify-full | User pick (Q6). All 14 current writers continue to work. Matches the existing CI-to-AWS-S3 operational model. |
| Cutover style | **Short dual-write window** (CI writes to BOTH v3 EC2 AND Postgres) for ~3-7 days of soak; then drop v3-write; then DNS flip from v2 directly to v4; then decommission v3 | User pick (Q7). v3 EC2 is a stepping stone that never goes live (DNS stays on v2 until the v4 flip). Dual-write gives benchmark-data-loss safety net during the v4 verification window. |
| Read service framework | **Next.js 15 + App Router + React Server Components + `unstable_cache` + `revalidateTag`** at `benchmarks-website/web/` | User pick (Q8). Server components fetch directly from Postgres; per-chart cache tags invalidated from the Python writer's CLI via a Vercel revalidation endpoint. Latest stable Next.js. Pages Router avoided. |
| Operator SQL replacement | **`scripts/psql-bench.sh`** — tiny helper that runs `aws rds generate-db-auth-token` and pipes into psql with IAM creds | User pick (Q9). Replaces `/api/admin/sql`. No bearer tokens, no Lambda. Documented in benchmarks-website/web/README.md. RDS PITR (35-day) replaces `/api/admin/snapshot`. |
| Composite index definition strategy | Net-new in initial schema migration (`migrations/001_initial_schema.sql`). Indexes on `(dim_tuple..., commit_timestamp DESC)` per-fact-table to serve both LIMIT-N and full-history read modes. | Forward-looking — none exist today. Designed alongside the schema in Phase 1. |
| v3 EC2 final disposition | **Decommissioned at end of Phase 5** (single deletion PR removes `benchmarks-website/server/`, `benchmarks-website/ops/`, `benchmarks-website/migrate/`, top-level v2 files, `publish-benchmarks-website.yml`, `INGEST_BEARER_TOKEN`/`ADMIN_BEARER_TOKEN` secrets). EC2 instance terminated by hand after PR merges. | v3 never goes live; Q7 cutover model goes v2→v4 directly. |

## Project-specific BANS

**General (apply regardless of language/shape):**
- Behavioral drift without test coverage: any change to observable behavior requires an added or updated test.
- Silent error swallowing: `let _ = fallible_op()`, `.ok()`, bare `try { } catch {}` without explicit rationale.
- Long `// TODO`, `// FIXME`, `// PORT NOTE`, or any `// SAFETY:` >100 chars explaining why a hack is OK (justification-comment reward-hacking).

**Migration / behavior-preservation BANS:**
- Do not change the byte layout of any `measurement_id_*` xxhash64 function in `benchmarks-website/server/src/db.rs:162-257`. (Why: existing rows in the v3 DuckDB and any Postgres rows seeded from it use these IDs as PKs; any change silently produces duplicate rows on next ingest instead of upserting.)
- Do not stop including `commit_sha` in the dim-tuple hash, and do not add `iterations` (or any side-counter) to the `vector_search_runs` hash. (Why: removing `commit_sha` collapses every commit's timing into one row; adding side counters splits one (commit, dim) tuple across rows.)
- Do not change the `ON CONFLICT (measurement_id) DO UPDATE SET ...` column lists in the new ingest path without auditing every column. (Why: each `DO UPDATE SET` enumerates only value columns; a dim column accidentally entering the SET list drifts PK and content, and an omitted value column silently keeps stale data.)
- Do not skip `commits` upsert before fact-table inserts. (Why: no FK exists, so an orphan fact row renders as a phantom point with no commit metadata.)
- Do not bump `SCHEMA_VERSION` in only one of (`server/src/schema.rs`, `vortex-bench/src/v3.rs`, `scripts/post-ingest.py` or its successor). (Why: server validation rejects mismatched envelopes; lockstep bump in one PR is required.)
- Do not remove `#[serde(deny_unknown_fields)]` from any envelope/record struct. (Why: unknown fields are how version skew surfaces loudly; relaxing tolerance silently drops producer data.)
- Do not introduce a code path that writes only to Postgres or only to DuckDB during the dual-write window. (Why: the rollback story requires each substrate to be restorable on its own.)
- Do not delete bearer-token middleware while any `/api/ingest` or `/api/admin/*` route still exists, and do not delete those routes while any caller still depends on them. (Why: half-decommission either creates an unauthenticated public write endpoint or breaks a dead caller silently.)
- Do not put `measurement_id` on the wire (request body, response body, HTML). (Why: it is server-internal by design; exposing it makes the hash a public API and freezes byte-layout migration forever.)
- Do not raise or remove `MAX_NUMERIC_COMMIT_WINDOW` on the numeric `?n=` path, and do not re-introduce a server-side cap on `?n=all`. (Why: documented DoS floor + uncapped escape hatch; reviewers have reverted this before.)
- Do not edit top-level v2 files (`server.js`, `src/`, `index.html`, `vite.config.js`, `package.json`, top-level `Dockerfile`, `docker-compose.yml`, `publish-benchmarks-website.yml`) outside the dedicated cutover PR. (Why: v2 is production until DNS flips.)
- Do not re-add `continue-on-error: true` to the new v4 ingest steps. (Why: ingest is intentionally no-longer-best-effort; silently swallowing failures masks divergence between bench run and served data.)

**Rust / clippy carryover BANS (apply to any new Rust ingest code):**
- Do not use `std::collections::HashMap`, `std::collections::HashSet`, `std::sync::Mutex`, `std::sync::RwLock`. (Why: workspace `clippy.toml` bans them; use `vortex_utils::aliases` + `parking_lot`.)
- Do not call `std::thread::available_parallelism`. (Why: banned; use `vortex_utils::parallelism::get_available_parallelism`.)
- Do not `unwrap`/`expect` outside `#[cfg(test)]`; do not invent ad-hoc error enums when existing `thiserror` variants cover the case.

**Next.js / TypeScript BANS (carry from `vortex-web/`):**
- Do not introduce a TS file without the SPDX `Apache-2.0` + `Copyright the Vortex contributors` header pair.
- Do not disable `strict`, `noUnusedLocals`, `noUnusedParameters`, `noFallthroughCasesInSwitch`, `noUncheckedSideEffectImports` in `tsconfig*.json`.
- Do not introduce a second formatter config — reuse the existing `vortex-web/.prettierrc.json` shape (`singleQuote: true`, `trailingComma: "all"`, `printWidth: 100`, `tabWidth: 2`).
- Do not reach for WASM in the new web layer.

**benchmarks-website-specific UI BANS (carry to the Next.js rewrite):**
- Do not flip the per-row delta predecessor walk to `idx + 1`. (Why: `payload.commits[]` is oldest-first; predecessor is `idx - 1`. A "fix" in this direction has been reverted.)
- Do not set `pointer-events: auto` on the tooltip host. (Why: known cursor-tracking flicker loop.)
- Do not bind UI handlers to `change` on range sliders; use throttled `input`. (Why: `change` fires only on release.)
- Do not refetch chart payloads on pan / zoom / slider / range-strip change. (Why: in-memory LTTB over cached payload; the only exception is the one-shot `?n=all` lazy hop.)

**Work-shape highest-leverage insight**: **Behavior-preservation under test** — every preserved invariant (xxhash64 byte layout, `ON CONFLICT` SET-column membership, 6-table schema shape, v3 envelope `deny_unknown_fields`, `SCHEMA_VERSION` triple-site coupling, all-four-or-none memory-quartet validation, `IS NOT DISTINCT FROM` NULL semantics on dim equality) must appear as an explicit acceptance criterion in the relevant PR's row and must be pinned by an automated test. Reviewers verify against this list; phase-end review re-checks every preserved invariant against the diff.

## Phases and PRs

### Phase summary

| Phase | Name | Scope (one line) | Exit criteria (machine-checkable) | PR count | Review-count |
|---|---|---|---|---|---|
| 1 | RDS + schema + hash port | Provision RDS, write schema-deploy script + initial DDL, port xxhash64 to Python with golden vectors | `aws rds describe-db-instances --db-instance-identifier vortex-bench-prod` returns `available`; `python scripts/migrate-schema.py status` clean; `pytest scripts/test_post_ingest_hash.py` all green; Rust golden-vector test in `vortex-bench-server` matches Python output bit-exactly | 5 | 3-vote |
| 2 | Python writer + dual-write CI | Extend post-ingest.py with `--postgres` mode + IAM-auth; flip 3 CI workflows to dual-write; verification harness | `bench.yml` PR CI green with `--postgres` flag set; verification harness reports `only_in_postgres = []` and `only_in_duckdb = []` after a test run | 4 | 3-vote |
| 3 | Historical data load (DuckDB → Postgres) | Extend `benchmarks-website/migrate/` with `--postgres-target`; run one-shot load; verify | `migrate --postgres-target --verify` reports `matched_rows == duckdb_rows AND only_in_postgres == []` against a live DuckDB snapshot | 3 | 3-vote |
| 4 | Next.js read service on Vercel | Scaffold `benchmarks-website/web/`; connection lib + `unstable_cache`; port all read endpoints; deploy to Vercel preview; revalidation API + token-plumbing from Python writer | `vercel deploy --target=preview` produces URL serving all chart slugs; `curl preview-url/api/groups | jq '.[].slug' \| sort` matches the family registry; preview lighthouse score ≥ 90 | 6 | 3-vote |
| 5 | Cutover + decommission | Drop v3-write from CI; DNS flip v2 → v4; delete v2 frontend + server/ + ops/ + migrate/ + publish-benchmarks-website.yml + bearer-token secrets | `git grep -n INGEST_BEARER_TOKEN` returns 0; `gh workflow list` does not include `publish-benchmarks-website`; production DNS resolves to Vercel; v3 EC2 terminated | 3 | 4-vote (final) |

Total: **5 phases, 21 PRs**.

### PR enumeration

| PR | Phase | Scope (one line) | Files touched (expected) | Acceptance (specific, testable) |
|---|---|---|---|---|
| PR-1.1 | 1 | Provision RDS Postgres `db.t4g.micro` + RDS Proxy + GitHub OIDC schema role via aws-cli script; document in `benchmarks-website/infra/README.md`; capture endpoint via post-run `gh variable set RDS_BENCH_ENDPOINT` | `benchmarks-website/infra/provision.sh`, `benchmarks-website/infra/README.md`, `.github/workflows/schema-deploy.yml` (skeleton) | `aws rds describe-db-instances --db-instance-identifier vortex-bench-prod` returns `available` + `iam_database_authentication_enabled: true`; `aws rds describe-db-proxies --db-proxy-name vortex-bench-proxy` returns `Endpoint`. |
| PR-1.2 | 1 | Write `scripts/migrate-schema.py` (~80-180 LOC; original ~30-50 estimate was for the bare runner — status/drift, autocommit txn discipline, typed exceptions, and recovery commentary brought it closer to ~180) — applies `migrations/*.sql` in name order, tracks via `_applied_migrations` table, idempotent | `scripts/migrate-schema.py`, `scripts/test_migrate_schema.py`, `migrations/` (dir created + README), `pyproject.toml` (psycopg + testcontainers dev deps) | Unit test: applies a fresh schema to testcontainers Postgres, re-runs idempotently (0 rows changed second time), inserts and re-applies a v2 migration in order. `apply` survives a failing later migration without losing earlier ones (subprocess test). `status` exits non-zero on drift and does not DDL. |
| PR-1.3 | 1 | Write `migrations/001_initial_schema.sql` (the 6 tables + composite indexes per Table B); `migrations/002_iam_db_user.sql` (CREATE ROLE for IAM auth) | `migrations/001_initial_schema.sql`, `migrations/002_iam_db_user.sql`, `scripts/test_migrate_schema.py` (extended) | `python scripts/migrate-schema.py --target=$RDS_DSN apply` succeeds; `\dt` shows the 6 tables; `\di` shows the composite indexes; `\du` shows the IAM-auth role with `rds_iam` group. |
| PR-1.4 | 1 | Wire `.github/workflows/schema-deploy.yml` — OIDC → `GitHubBenchmarkSchemaRole` → `python scripts/migrate-schema.py apply` against RDS Proxy; gated by `environment: schema-deploy` (manual approval) | `.github/workflows/schema-deploy.yml`, IAM role doc in `web/ops/README.md` | Workflow runs to completion on `develop` push touching `migrations/`; manual-approval gate fires; `migrate-schema.py status` reports clean post-apply. |
| PR-1.5 | 1 | Port xxhash64 to Python in `scripts/_measurement_id.py`; mirror per-table tag + write_str/write_opt_str/write_i32/write_f64 encoding; golden-vector test against Rust source-of-truth (new test in `vortex-bench-server`) | `scripts/_measurement_id.py`, `scripts/test_measurement_id.py`, `benchmarks-website/server/src/db.rs` (golden-vector test added) | `pytest scripts/test_measurement_id.py` all green; for 100 fixture (commit, dim-tuple) inputs the Python output matches the Rust output bit-exactly. |
| PR-2.1 | 2 | Extend `scripts/post-ingest.py` with `--postgres $RDS_DSN` mode: parse JSONL, compute measurement_id via `_measurement_id.py`, generate IAM token via boto3, INSERT … ON CONFLICT against Postgres | `scripts/post-ingest.py`, `scripts/test_post_ingest_postgres.py`, `pyproject.toml` (psycopg, boto3) | Integration test (testcontainers Postgres): POST a v3 envelope, verify N rows inserted; POST same envelope, verify 0 inserted + N updated; verify all measurement_id values match. |
| PR-2.2 | 2 | Flip `.github/workflows/bench.yml`, `sql-benchmarks.yml`, `v3-commit-metadata.yml` to dual-write — invoke `post-ingest.py` once with `--server` (v3) then once with `--postgres` (v4); both must succeed | The 3 workflow files; `scripts/post-ingest.py` (minor `--mode` flag) | All 3 workflows green on a test PR push; CloudWatch RDS metrics show ~14 writes per push from the matrix. |
| PR-2.3 | 2 | Verification harness: `scripts/verify-substrates.py` that connects to both DuckDB EC2 (via /api/admin/sql) and Postgres, runs the same SELECT shapes, asserts row-for-row equivalence | `scripts/verify-substrates.py`, `scripts/test_verify_substrates.py` | Tool reports `only_in_postgres=[]` and `only_in_duckdb=[]` and `mismatched_rows=[]` on a seeded fixture in both substrates. |
| PR-2.4 | 2 | Add `.github/workflows/dual-write-verify.yml` — runs `verify-substrates.py` after every `develop` push; alerts via incident.io webhook on non-zero diff | `.github/workflows/dual-write-verify.yml` | Workflow runs on simulated divergence (manually insert a mismatch) and fires the alert; runs on clean state and exits 0. |
| PR-3.1 | 3 | Extend `benchmarks-website/migrate/` with `--postgres-target $DSN`; add Postgres bulk-insert via `COPY FROM STDIN` for each table; reuse existing `measurement_id_*` for hash compat | `benchmarks-website/migrate/Cargo.toml` (sqlx-postgres + aws-sdk-rds), `benchmarks-website/migrate/src/postgres.rs`, `benchmarks-website/migrate/src/main.rs` (CLI flag) | `cargo run -p vortex-bench-migrate -- --duckdb <snapshot.duckdb> --postgres-target $DSN` completes; row counts match per table. |
| PR-3.2 | 3 | Extend `migrate/src/verify.rs` to support Postgres target (`VerifyReport` per Q5b) | `benchmarks-website/migrate/src/verify.rs` | `--verify` exits 0 on identical substrates; reports each diff row otherwise. |
| PR-3.3 | 3 | Run the one-shot load against the live v3 DuckDB; capture row counts in `Implementation status`; verify | (no code; operational PR with logs) | Production RDS has the full v3 history; `verify` reports clean against a fresh DuckDB snapshot taken post-load. |
| PR-4.1 | 4 | Scaffold `benchmarks-website/web/` Next.js 15 project: `package.json`, `tsconfig.json` (matching `vortex-web/`'s strict flags), `next.config.js`, `.gitignore`, SPDX headers everywhere | `benchmarks-website/web/{package.json,tsconfig.json,next.config.js,.gitignore}`, `app/layout.tsx`, `app/page.tsx` (stub) | `pnpm install && pnpm build` succeeds; lints clean. |
| PR-4.2 | 4 | Connection lib at `web/lib/db.ts`: `pg.Pool` + `@aws-sdk/rds-signer` for IAM token gen + token-refresh-before-expiry; expose `sql` tagged-template helper | `web/lib/db.ts`, `web/lib/db.test.ts` | Integration test (testcontainers Postgres): pool connects via password (test fixture); pool roundtrips a SELECT. IAM-auth path mocked. |
| PR-4.3 | 4 | Port read endpoints to App Router routes + server components — `/api/groups`, `/api/chart/{slug}`, `/api/group/{slug}`, `/api/artifacts/{generation}/groups/{slug}/shards/{i}`, `/health`; wrap with `unstable_cache` per-chart tags | `web/app/api/**/*`, `web/lib/queries.ts` | Each route returns the same shape as the current Axum server (insta-style snapshot test against fixtures). |
| PR-4.4 | 4 | Port HTML SSR + UI from existing React (chart rendering, filter bar, range strip, tooltip) to Next.js server components + client islands; preserve all UI BANS | `web/app/{layout,page,chart/[slug]/page}.tsx`, `web/components/*` | Visual diff against current v2 site for 5 representative chart slugs ≤ 5% pixel delta. |
| PR-4.5 | 4 | Vercel deploy config (`vercel.json`); GitHub Action `web-deploy.yml` for preview-per-PR + production deploy on merge | `vercel.json`, `.github/workflows/web-deploy.yml` | PR opens trigger preview deploy; merging to ct/bench-v4 triggers prod deploy (still behind dev-only Vercel domain at this stage). |
| PR-4.6 | 4 | Revalidate API endpoint at `web/app/api/revalidate/route.ts`; HMAC-signed token (REVALIDATE_SECRET); call from `post-ingest.py` after successful inserts | `web/app/api/revalidate/route.ts`, `scripts/post-ingest.py` (revalidate hook) | After an ingest, the relevant `chart:<slug>` tags are invalidated; next reader pageload re-fetches; manual test passes. |
| PR-5.1 | 5 | Drop v3-write from 3 CI workflows; `post-ingest.py` runs only with `--postgres` | `.github/workflows/{bench.yml,sql-benchmarks.yml,v3-commit-metadata.yml}`, `scripts/post-ingest.py` (remove --server mode) | Workflows green; verify-substrates.yml decommissioned; CloudWatch shows zero traffic to v3 EC2. |
| PR-5.2 | 5 | DNS flip v2 → v4 (update Cloudflare/Route53 records for `benchmarks.vortex.dev` to Vercel) | (operational PR; possibly `dns/` configs if checked in) | `dig benchmarks.vortex.dev` resolves to Vercel; site loads in browser; Cloudflare/Vercel SSL chain verified. |
| PR-5.3 | 5 | Delete: `benchmarks-website/server/`, `benchmarks-website/ops/`, `benchmarks-website/migrate/`, top-level `server.js`, `src/`, `index.html`, `vite.config.js`, `package.json` (root), `package-lock.json` (root), `Dockerfile`, `docker-compose.yml`, `.github/workflows/publish-benchmarks-website.yml`; remove `INGEST_BEARER_TOKEN` + `ADMIN_BEARER_TOKEN` repo secrets; terminate v3 EC2 by hand post-merge | ~30 files deleted | `git grep -n INGEST_BEARER_TOKEN` returns 0; `gh workflow list` does not include `publish-benchmarks-website`; `aws ec2 describe-instances --instance-ids <v3-instance>` returns `terminated`. |

(Each PR targets 1-3 logical commits; some — like PR-1.3 (DDL + IAM role) and PR-4.3 (3 endpoints) — are at the upper end and may split during execution per the mid-flight expansion protocol.)

## Reference tables

**REQUIRED for migration work shape.** Mechanical, exhaustive lookups that downstream PRs grep into.

### Table A — `measurement_id_*` xxhash64 algorithm specification

The new ingest writer reproduces this byte-for-byte. Source: `benchmarks-website/server/src/db.rs:162-257`.

| Step | Operation | Encoding |
|---|---|---|
| 1 | Construct hasher | `XxHash64::with_seed(0)` (no per-call seeding) |
| 2 | Write per-table tag | `hasher.write(tag.as_bytes())` then `hasher.write_u8(0)` |
| 3 | `write_str(s)` | `hasher.write_u64(s.len() as u64)` THEN `hasher.write(s.as_bytes())`. No NUL terminator. Length is LE u64 (twox-hash writes LE without byte-swap). |
| 4 | `write_opt_str(opt)` | `Some` → `hasher.write_u8(0x01)` then `write_str`. `None` → `hasher.write_u8(0x00)` and stop. |
| 5 | `write_i32(v)` | `hasher.write_i32(v)` — 4 LE bytes. |
| 6 | `write_f64(v)` | `hasher.write_u64(v.to_bits())` — IEEE 754 bits as LE u64. |
| 7 | Finalize | `hasher.finish() as i64` (bit-cast `u64 → i64`). |

### Table B — per-table dim-tuple field order (drives the hash)

| Table | Tag (literal) | Field order |
|---|---|---|
| `query_measurements` | `"query_measurements"` | `commit_sha` → `dataset` → opt `dataset_variant` → opt `scale_factor` → `query_idx` (i32) → `storage` → `engine` → `format` |
| `compression_times` | `"compression_times"` | `commit_sha` → `dataset` → opt `dataset_variant` → `format` → `op` |
| `compression_sizes` | `"compression_sizes"` | `commit_sha` → `dataset` → opt `dataset_variant` → `format` |
| `random_access_times` | `"random_access_times"` | `commit_sha` → `dataset` → `format` (NO `dataset_variant`) |
| `vector_search_runs` | `"vector_search_runs"` | `commit_sha` → `dataset` → `layout` → `flavor` → `threshold` (f64 bits). `iterations` intentionally EXCLUDED. |

### Table C — DuckDB → Postgres type/syntax map

| Column shape | DuckDB | Postgres |
|---|---|---|
| `BIGINT[]` (`all_runtimes_ns`) | `[1,2,3]` literal + `CAST(? AS BIGINT[])` | Native `BIGINT[]` bind via `&[i64]` |
| `TIMESTAMPTZ` (`commits.timestamp`) | `CAST(? AS TIMESTAMPTZ)` from RFC 3339 | Native `TIMESTAMPTZ` from `OffsetDateTime` |
| `IS NOT DISTINCT FROM` (NULL-aware eq) | Supported | Supported (native) |
| Upsert | `ON CONFLICT (measurement_id) DO UPDATE SET <cols> = excluded.<cols>` | Identical syntax |
| Write-conflict retry | 128-attempt retry loop (DuckDB MVCC) | Postgres `SQLSTATE 40001` retry; design afresh |

### Table D — `SCHEMA_VERSION = 1` lockstep sites

Every bump touches all rows in this table in one PR (or CI ingest 400/409s).

| File | Line / symbol | Form |
|---|---|---|
| `benchmarks-website/server/src/schema.rs` | `SCHEMA_VERSION = 1` (line 223) | Rust `const i32` |
| `vortex-bench/src/v3.rs` | (struct field default + tests) | Rust |
| `scripts/post-ingest.py` | `SCHEMA_VERSION = 1` (line 52) | Python literal |
| `scripts/post-ingest.py` (extended) | `SCHEMA_VERSION = 1` (existing line 52) | Python `const int` |
| `scripts/_measurement_id.py` (new, PR-1.5) | Imports SCHEMA_VERSION from `post-ingest.py` | Python re-export to keep one site |
| `benchmarks-website/web/lib/schema-version.ts` (new, PR-4.2 or 4.3) | `export const SCHEMA_VERSION = 1` | TypeScript const — read-path version-gate (returns 4xx if envelope mismatch surfaces in any out-of-band write) |
| `benchmarks-website/migrate/src/lib.rs` (existing) | `pub const SCHEMA_VERSION` | Rust constant referenced by the v3→v4 one-shot migrator |

### Table E — CI ingest call sites

The new ingest writer replaces these `python3 scripts/post-ingest.py ...` invocations. Each gets a feature-flag branch during dual-write.

| Workflow | Step location | Trigger | Concurrent writers per run |
|---|---|---|---|
| `.github/workflows/bench.yml` | lines 108-118 | `push: develop` | 2 (matrix: random-access-bench, compress-bench) |
| `.github/workflows/sql-benchmarks.yml` | lines 527-537 | `workflow_call` from bench.yml + nightly + bench-pr (PR mode skips) | 11 (matrix entries) |
| `.github/workflows/v3-commit-metadata.yml` | lines 23-34 | `push: develop`, `workflow_dispatch` | 1 |

Total per `develop` push: **~14 parallel writers**.

### Table F — Decommission inventory

Items deleted in the final cutover PR.

| Path | Decommission rationale |
|---|---|
| `benchmarks-website/server/` (Rust crate) | Replaced by Postgres + Next.js |
| `benchmarks-website/ops/` (systemd + backup + deploy) | Replaced by managed Postgres + Vercel |
| `benchmarks-website/server/src/auth.rs` (bearer middleware) | No `/api/*` routes remain |
| `INGEST_BEARER_TOKEN` references in `bench.yml`, `sql-benchmarks.yml`, `v3-commit-metadata.yml` | Replaced by OIDC + IAM-auth role |
| `ADMIN_BEARER_TOKEN` env + secret | Admin endpoints retired |
| `benchmarks-website/server/src/admin.rs` | `/api/admin/*` retired (psql replaces it) |
| Top-level v2: `server.js`, `src/`, `index.html`, `vite.config.js`, `package.json`, `package-lock.json`, `Dockerfile`, `docker-compose.yml` | Replaced by Next.js 15 app under `benchmarks-website/web/` |
| `.github/workflows/publish-benchmarks-website.yml` | Vercel deploy replaces GHCR push |
| `benchmarks-website/migrate/` (one-shot v2→v3 migrator) | One-shot; survived v2→v3 only because of dual-write soak. **Decision in interview**: retire alongside v3→v4 migration, or retain as a long-tail tool? |

## Critical files

- `/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/server/src/schema.rs` — 6-table DDL constants; `SCHEMA_VERSION`; the table-order applied at boot.
- `/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/server/src/db.rs:162-257` — `measurement_id_*` xxhash64; load-bearing for migration.
- `/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/server/src/ingest.rs` — `POST /api/ingest` handler; envelope validation; per-record upsert; transaction boundaries.
- `/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/server/src/records.rs` — server-side envelope/record struct with `deny_unknown_fields`.
- `/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/server/src/api/charts.rs` — per-chart SQL via `seeded_commits_in_window` two-pass pattern; `IS NOT DISTINCT FROM` for NULL-aware dim equality.
- `/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/server/src/api/window.rs` — `CommitWindow` enum; `MAX_NUMERIC_COMMIT_WINDOW = 1_000`; numeric vs `all` semantics.
- `/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/server/src/read_model.rs` — `ReadGeneration` with precompressed gzip/brotli artifacts. Maps onto Next.js `unstable_cache` + edge CDN.
- `/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/migrate/src/verify.rs` — `VerifyReport` structural diff template.
- `/Users/connor/spiral/vortex-data/vortex4/vortex-bench/src/v3.rs` — v3 envelope and `V3Record` producer side; immutable across this migration.
- `/Users/connor/spiral/vortex-data/vortex4/scripts/post-ingest.py` — current CI envelope-poster; replaced by the new ingest writer.
- `/Users/connor/spiral/vortex-data/vortex4/.github/workflows/bench.yml` / `sql-benchmarks.yml` / `v3-commit-metadata.yml` — CI workflows that POST the envelope; sites of the dual-write modification.
- `/Users/connor/spiral/vortex-data/vortex4/.github/workflows/compat-gen-upload.yml` — structural template for any approval-gated migration step.
- `/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/AGENTS.md` — subsystem playbook; carry rules from v2→v3 cutover.
- `/Users/connor/spiral/vortex-data/vortex4/benchmarks-website/server/tests/common/mod.rs` — `Server` harness pattern; carry to Postgres testcontainer harness.
- `/Users/connor/spiral/spiraldb/.github/workflows/deploy-spiraldb.yml:37-89` — cross-codebase deploy-migrations blueprint.
- `/Users/connor/spiral/spiraldb/.github/workflows/prisma-isolation.yml` — schema-isolation guardrail template.

## Risks

1. **Hash-byte-layout drift**: P=med (every reviewer initially distrusts a private byte-encoded PK); impact=severe (silently produces duplicate rows on next CI write, breaking all historical chart continuity). Mitigation: Phase 1 produces a golden-vector unit test from a `(commit, dim) → i64` snapshot of the live DuckDB; every subsequent PR runs the test; reviewers cross-reference Table A/B before approving.
2. **`SCHEMA_VERSION` lockstep slip**: P=med (4-5 sites now, including a new Rust writer and a new TS read layer); impact=moderate (CI ingest 400/409s after a partial bump). Mitigation: introduce a single source-of-truth file (`SCHEMA_VERSION` in a one-line file or shared crate), grep guard in CI.
3. **Cold-start cost on Vercel**: P=med (read path was 100% in-process materialized; cold function rebuilds artifacts from Postgres). Mitigation: aggressive `unstable_cache` + edge CDN; `revalidateTag` from the ingest writer's CLI invoking Vercel deploy hook. Phase 4 has a lighthouse-score acceptance criterion.
4. **Dual-write divergence under matrix concurrency**: P=med (~14 parallel CI writers hitting both substrates per push); impact=moderate (the verification harness has to handle "in-flight" rows). Mitigation: `verify.rs`-shaped tool runs at PR end, not mid-run; settle window before assertion.
5. **Connection-budget exhaustion**: P=low-med (Vercel serverless × matrix concurrency × CI ingest = many pools); impact=severe (Postgres rejects writers). Mitigation: pooler choice in Phase 2; explicit `PgPool::max_connections = 4`; budget calculation in plan.
6. **Decommission half-rollback**: P=low (the v2→v3 cutover went smoothly); impact=severe (a partial deletion leaves orphan systemd units or unauthenticated routes). Mitigation: single decommission PR (Phase 5) gated behind a verification harness; explicit `grep -r INGEST_BEARER_TOKEN` exit criterion.
7. **Phase scope drift on the Next.js side**: P=med (no in-house Next.js + DB precedent); impact=moderate (Phase 4 turns into a multi-week spike). Mitigation: pick stack early (decision in interview); use `context7` for current Vercel docs; allow a planning re-scope if Phase 4 exceeds 5 PRs.
8. **Operational gaps in admin replacement**: P=low-med (managed Postgres console + Vercel logs replace `/api/admin/sql` + systemd journal); impact=moderate (operator can't introspect prod when needed). Mitigation: write runbook before decommission; confirm console access; capture explicit operator workflows in Phase 5 description.
9. **History migration losing rows**: P=low (DuckDB → Postgres COPY/INSERT is well-supported); impact=severe (lost benchmark data is unrecoverable for prior commits). Mitigation: pre-cutover snapshot of DuckDB via `ops/backup.sh`-equivalent; post-cutover row-count assertion; keep DuckDB snapshot for at least 90 days post-cutover.
10. **Cross-account one-shot migration**: P=low-med; impact=moderate. v4 RDS lives in `245040174862/us-east-1` (bench account); v3 EC2 + DuckDB live in `375504701696/us-east-2` (personal account). Phase 3 one-shot DuckDB→Postgres load crosses BOTH account and region boundaries. Mitigation options: (a) write a DuckDB snapshot from `375504701696/us-east-2` to a cross-account-accessible S3 bucket (either in `245040174862` with a bucket policy permitting `375504701696` to PutObject, or vice versa), then download from `245040174862` and load locally to RDS; (b) operator runs the migrator with both profiles configured (`AWS_PROFILE` switching for read vs write); (c) temporarily grant `375504701696` IAM identity an `rds-db:connect` role in `245040174862` for direct write. Steady-state CI ingest after Phase 3 is unaffected (GH Actions OIDC into `245040174862` already works).

## Verification

**Phase 1 verification:**

```sh
cargo test -p vortex-bench-server hash_stability
# expected: all golden vectors match
<migration-tool> migrate status
# expected: clean against a fresh Postgres
```

**Phase 2 verification:**

```sh
# Integration test against testcontainers Postgres
cargo test -p <new-ingest-crate> --test integration
# Compare result of replaying a v3 envelope JSONL against Postgres vs DuckDB
cargo run -p <new-ingest-crate> --bin verify-roundtrip -- \
  --duckdb benchmarks-website/server/fixtures/snapshot.duckdb \
  --postgres postgres://localhost/test
# expected: zero diff
```

**Phase 3 verification:**

```sh
gh workflow run bench.yml -f mode=develop
# observe both EC2 ingest + Postgres ingest succeed
cargo run -p <verify-crate> -- --against-both
# expected: only_in_v3 = []
```

**Phase 4 verification:**

```sh
# Vercel preview URL
curl -sS https://<preview>.vercel.app/api/groups | jq '.[].slug' | sort > preview-slugs.txt
curl -sS https://benchmarks.vortex.dev/api/groups | jq '.[].slug' | sort > prod-slugs.txt
diff preview-slugs.txt prod-slugs.txt
# expected: empty
```

**Phase 5 (cutover) verification:**

```sh
git grep -n INGEST_BEARER_TOKEN
# expected: zero hits in any tracked file
git grep -n V3_INGEST_URL
# expected: zero hits
gh run list --workflow=publish-benchmarks-website.yml
# expected: workflow does not exist
```

**Full verification (end-state):**

```sh
# 1. Hash stability still passes
cargo test -p <ingest-crate> hash_stability
# 2. Read-path renders all chart slugs
curl -sS https://benchmarks.vortex.dev/api/groups | jq '.[].slug' | wc -l
# expected: matches the family count from src/family.rs
# 3. Idempotent re-ingest
cargo run -p <ingest-crate> -- --envelope <recent-envelope.jsonl> --dry-run
# expected: every record reports updated=1, inserted=0
```

## Implementation status

### PR-1.1: Provision RDS Postgres + RDS Proxy + GitHub OIDC schema role  (11 code commits, ending at 2336d48c1)

- **Scope shipped**: AWS infrastructure bootstrap script (`benchmarks-website/infra/provision.sh`) + operator runbook (`benchmarks-website/infra/README.md`) + GitHub Actions schema-deploy workflow skeleton (`.github/workflows/schema-deploy.yml`). Provisions RDS Postgres `db.t4g.micro` + RDS Proxy + GitHub OIDC provider + `GitHubBenchmarkSchemaRole` IAM role in account `245040174862` / `us-east-1`. Idempotent re-run; existence-checked.
- **Acceptance criteria — verified by operator**:
  - `aws rds describe-db-instances --db-instance-identifier vortex-bench-prod` returns `available` + `IAMDatabaseAuthenticationEnabled: true` ✓
  - `aws rds describe-db-proxies --db-proxy-name vortex-bench-proxy` returns `available` + Endpoint `vortex-bench-proxy.proxy-c4f8qygk4xdp.us-east-1.rds.amazonaws.com` ✓
- **Repo vars set** (verified): `RDS_BENCH_ENDPOINT`, `RDS_BENCH_REGION`, `RDS_BENCH_DB_NAME`, `GH_BENCH_SCHEMA_ROLE_ARN`.
- **Resource identities** (for downstream PRs to reference):
  - RDS instance: `vortex-bench-prod` (resource-id `db-4VPTDACTRQHOS24WEIR3TNC2M4`)
  - Master secret: `arn:aws:secretsmanager:us-east-1:245040174862:secret:rds!db-23f1d9f9-ce44-4dc9-ac97-d3a5afaef690-egkQgW` (RDS-managed)
  - Proxy: `vortex-bench-proxy.proxy-c4f8qygk4xdp.us-east-1.rds.amazonaws.com:5432`
  - Default VPC: `vpc-03cbb254d7f018995` (6 subnets discovered)
  - Schema role: `arn:aws:iam::245040174862:role/GitHubBenchmarkSchemaRole`
- **Tests added**: none (acceptance is operator-runs-script + describe-db-* probes; no unit-test surface).
- **Review**: 2-vote (preset=pr-2, fresh+correctness) / cycle 1 reject → 7 fix commits → operator execution surfaced 3 additional bugs → 4 more fix commits → operator accept. No cycle-2 static gauntlet (deferred; operator-execution provided higher-signal verification than static re-review would).
- **Confidence**: high (acceptance criteria pass under real AWS; all gauntlet must-fix items addressed; bug discovery shifted from static review to operator execution as expected for infra scripts).
- **Deferred items**: 8 should-fixes (see Deferred work table below).
- **Surprises during implementation**:
  - Operator only had SSO access to personal account `375504701696`; the actual bench account is `245040174862` (Linux Foundation org, no SSO for operator yet). Resolution: operator logged into 245040174862 root via signin.aws.amazon.com and ran provision.sh from AWS CloudShell (no local AWS creds needed). Documented as the supported operator path.
  - Three operator-found bugs not surfaced by cycle-1 static gauntlet: (1) `mktemp -t` template missing X's on GNU mktemp (CloudShell), (2) `aws rds wait db-proxy-available` is not a real AWS CLI v2 waiter, (3) `environment: schema-deploy` reference required GitHub repo-admin to create the environment, which operator lacks. All three fixed via additional cycle-2 fix-commits (674668406, d633735d9, 2336d48c1).
  - Pattern: static reviewers can't catch what running the script catches. Acknowledged as work-shape characteristic; future infra-script PRs should plan operator-execution as part of the acceptance flow, not gauntlet-cycle-2-only.

### PR-1.2: Postgres migration runner + tests + migrations/ scaffolding  (4 code commits + 2 doc-fix commits + 3 gauntlet cycles, ending at f6999d6d3)

- **Scope shipped**: `scripts/migrate-schema.py` (substrate-agnostic forward-only Postgres migration runner with `apply` and `status` commands; libpq env vars OR `--target=<dsn>`; PEP 723 inline metadata for standalone `uv run`); `scripts/test_migrate_schema.py` (16 pytest+testcontainers tests covering apply / idempotency / sequential order / subprocess close-on-exception rollback / status drift detection / empty-file rejection / case-insensitive discover / subdirectory-with-.sql-suffix filtering / non-default search_path); `migrations/` directory with `README.md` documenting naming convention + authoring rules + transactional-only invariant; `pyproject.toml` workspace dev deps gain `psycopg[binary]>=3.2` + `testcontainers[postgres]>=4.9`; `uv.lock` regenerated.
- **Acceptance criteria — verified against postgres:16-alpine testcontainer**:
  - Apply applies pending migrations idempotently (`test_apply_creates_..._and_runs_migrations`, `test_apply_is_idempotent`) ✓
  - Apply applies new migration in name order without re-applying earlier ones (`test_apply_applies_new_migration_in_order`) ✓
  - Apply survives a failing later migration without losing earlier ones — subprocess test mirroring production close-on-exception (`test_apply_rolls_back_on_failure_subprocess`) ✓
  - Status exits non-zero on drift and does not DDL (`test_status_reports_applied_and_pending`, `test_status_clean_returns_zero`, `test_status_flags_orphaned_applied_files`, `test_status_does_not_create_table`) ✓
  - Apply rejects empty + whitespace-only .sql files (`test_apply_rejects_empty_sql_file`, `test_apply_rejects_whitespace_only_sql_file`) ✓
  - Apply + Status agree on ledger location under non-default search_path (`test_apply_uses_public_schema_under_custom_search_path`) ✓
  - Discover ignores subdirectories with .sql suffix (`test_discover_ignores_subdirectory_with_sql_suffix`) ✓
- **Tests added**: 16 pytest tests (4 pure-function, 12 testcontainer-backed). Local runs all pass against `postgres:16-alpine`. CI execution of this test suite is a deferred item (see below).
- **Review**: 2-vote (preset=pr-2, fresh+correctness, Claude executor) / cycle 1 reject (2 must-fix: implicit-outer-txn savepoint bug + masked test) → fix bundle (txn model via autocommit + subprocess test + 6 should-fix items + plan-edit) → cycle 2 reject (2 new must-fix: empty-file replay bug exposed by carry-forward + schema-qualification divergence under non-default search_path) → fix bundle (empty-file rejection + schema-qualify all ledger refs + applied_set→_applied_set rename + is_file() guard + subprocess timeout + new regression tests) → cycle 3 accept (zero must-fix; both reviewers ACCEPT; remaining 9 should-fix + 3 nit triaged as docstring drift fixed inline or deferred).
- **Confidence**: high (3-cycle gauntlet with monotonically improving verdicts; 16/16 tests green against testcontainers; substrate-agnostic runner depends only on libpq env vars; clean separation between writer-DDL path and read-only status path; per-migration autocommit-True transactions correctly survive partial failure; schema-qualification pins ledger to `public._applied_migrations` regardless of role search_path).
- **Deferred items**: 4 should-fixes for PR-1.2 (concurrency advisory lock, autocommit toggle precondition for library callers, ledger fingerprint for edit-after-apply drift, CI pytest runner — all in Deferred work table).
- **Surprises during implementation**:
  - Cycle 1 surfaced a non-obvious psycopg3 transaction-model bug: `CREATE TABLE IF NOT EXISTS` outside a `with conn.transaction()` block lazily opens an implicit outer transaction; subsequent `conn.transaction()` blocks become SAVEPOINTs rather than top-level transactions, so a failing later migration triggers the connection-level rollback that discards all prior savepoint-released work. The original test masked this by catching the exception inside the test body, so the fixture's `with psycopg.connect()` exited cleanly via commit instead of with-exception via rollback. Fix: `conn.autocommit = True` at the top of apply()/status() forces each `conn.transaction()` to be a real top-level transaction; subprocess-based regression test exercises the production close-on-exception path.
  - Cycle 2 surfaced a second non-obvious bug: schema-qualification divergence between `applied_set` / `apply` (unqualified `_applied_migrations`) and `_read_applied_filenames` (`public._applied_migrations` via `to_regclass`). Under default search_path both paths happen to find the same table; under a non-default search_path (very plausible for the PR-1.3 `migrator` IAM role under standard `rds_iam` least-privilege patterns) writer and reader would silently transact with different ledger tables. Fix: canonicalize every ledger reference to `public._applied_migrations` everywhere (DDL + INSERT + SELECT). Test connects with `options='-c search_path=foo,public'` and verifies writer-reader agreement.
  - Pattern: per-cycle gauntlet caught two correctness ship-blockers neither reviewer would have spotted with a single pass — different prompts catch different bugs (the spec's load-bearing principle). The cycle-2 fix-commit's `Shared fix-commit attention` block at cycle 3 surfaced H1 docstring drift the fix introduced and let me clean it up before completion.
  - Spec deviation: cycle-2 fix-commit bundled both must-fix items into a single fix-commit (rather than the per-must-fix split Step 2.4 prescribes), because the edits on the apply()/applied_set/ledger boundary genuinely overlap. Documented in the commit body; followed by one plan-edit decrementing by 2.

## Deferred work

| Source | File:line | Severity | Description | Deferral rationale |
|---|---|---|---|---|
| PR-1.1 cycle-1 gauntlet | `provision.sh:478-487` (security group rule check) | should-fix | `ensure_security_group` checks `FromPort==5432` but not CIDR; an existing rule from `10.0.0.0/8` would silently leave SG more restrictive. | Low operational impact for first-run; revisit if drift surfaces. |
| PR-1.1 cycle-1 gauntlet | `provision.sh:429-446` (subnet group reconciliation) | should-fix | `ensure_db_subnet_group` doesn't reconcile subnet membership on existing group; stale group could prevent new-AZ scheduling. | Same; revisit on drift. |
| PR-1.1 cycle-1 gauntlet | `provision.sh:432, 497, 590, 554, 677, 642` (`2>&1` redirects) | should-fix | Permission errors masked by `2>&1` in existence-check redirects; missing IAM perms surface as "doesn't exist". | Affects operator debugging only; low priority. |
| PR-1.1 cycle-1 gauntlet | `provision.sh:590-605` (proxy auth reconciliation) | should-fix | Proxy auth config not reconciled on existing proxy (master secret ARN, IAMAuth setting). | One-shot script; secret ARN doesn't change post-creation. |
| PR-1.1 cycle-1 gauntlet | `README.md:264-276` (tear-down sequence) | should-fix | Tear-down missing `aws rds wait db-proxy-deleted` + `aws rds wait db-instance-deleted` between async deletes. | Tear-down is off the critical path; add to ops runbook if/when needed. |
| PR-1.1 cycle-1 gauntlet | `provision.sh:357` (engine version pin `16.4`) | should-fix | Pinned minor version may be deprecated by AWS over time. | Works today; revisit when AWS deprecates 16.4. |
| PR-1.1 cycle-1 gauntlet | `provision.sh:649` (OIDC thumbprint hardcoded) | should-fix | AWS no longer enforces the thumbprint for GitHub OIDC; cosmetic. | Add comment when next touching the file. |
| PR-1.1 cycle-1 gauntlet | `README.md:183` ("IAM auth is the gate") | should-fix | Phrasing overstates what PR-1.1 alone accomplishes; password auth on `postgres` master remains until PR-1.3's migration 002. | Update README in PR-1.3 alongside the rds_iam grant migration. |
| PR-1.1 follow-up | (operator action) | medium | Re-run `provision.sh` in CloudShell to apply the branch-scoped trust policy update (commit 2336d48c1). Existing role still has the wildcard `repo:vortex-data/vortex:*` sub-claim until operator re-runs. | Operator-side; security tightening, not blocking. |
| PR-1.2 cycle-1 gauntlet | `scripts/migrate-schema.py:57-75` (concurrency) | should-fix | Two concurrent CI runs can both observe the same pending set and race on non-idempotent DDL; PRIMARY KEY guards the ledger row but the DDL itself still executes twice, producing transient errors and ambiguous logs. | The CI workflow already serializes via `concurrency: schema-deploy`, and the initial DDL in PR-1.3 is idempotent (CREATE TABLE IF NOT EXISTS / IF NOT EXISTS indexes); revisit when a non-idempotent migration is actually authored. The fix is a Postgres advisory lock (`SELECT pg_advisory_xact_lock(<constant>)`) at the start of each per-migration transaction plus a concurrency test that spawns two parallel `apply` invocations. |
| PR-1.2 cycle-2 gauntlet | `scripts/migrate-schema.py:108-112` (autocommit toggle precondition) | should-fix | `apply()` sets `conn.autocommit = True` unconditionally; psycopg raises `ProgrammingError` if the connection has a transaction in progress. Production safe (main() opens a fresh conn) but the function is exposed as a library API and a future caller that ran any prior `cursor.execute` would hit the error. | Library-API hardening; not relevant until a second importer materializes. Fix is to assert `conn.info.transaction_status == IDLE` at the top of `apply` or defensively `rollback()` before the toggle, with a test that opens a transaction first. |
| PR-1.2 cycle-2 gauntlet | `scripts/migrate-schema.py` ledger schema (fingerprint) | should-fix | Applied migrations are not fingerprinted; an author editing `001_initial_schema.sql` after it has been applied to RDS sees `status` report clean both locally (against a freshly-applied testcontainer) and in CI (against RDS with the old file's effects). README forbids edit-after-apply but the runner has zero enforcement. | Substantive change: add a `sha256` column to `_applied_migrations`, record at apply time, compare on-disk hash vs ledger in `status`, report a third drift class `[~]` with non-zero exit. Track for a follow-up PR (PR-1.2.1 or fold into PR-1.4 wiring). |
| PR-1.2 cycle-2 gauntlet | `scripts/test_migrate_schema.py` (no CI runner) | should-fix | The pytest suite skips silently when Docker is unavailable (`_docker_available()` probe), and no CI workflow currently runs the suite with Docker enabled. PR-1.2 acceptance ("Unit test: applies a fresh schema to testcontainers Postgres ...") is verified only by local dev runs; CI would be green even if the runner regressed. | PR-1.4 wires the schema-deploy workflow with real apply; pair it with a `pytest-on-PR` job that fails loud when `_docker_available` returns False in CI. Track for PR-1.4. |

## Accepted tradeoffs / r1 traps

(Empty. Populated as the user explicitly accepts a reviewer-flagged item.)
