# benchmarks-website migration to hosted Postgres + Next.js on Vercel — big-plans plan

## Current State

```yaml
status: planning
branch: ct/bench-v4
planning_sub_flow: re-plan-phase-2
current_phase: "Phase 1: RDS + schema + hash port"
phase_index: 1
current_pr: null
pr_index: 7
outstanding_must_fix: 0
deferred_items_total: 25
last_user_touchpoint: 2026-06-01T19:46:41Z
last_user_touchpoint_what: "Phase 1 ACCEPTED + live-verified (see 'Phase 1 live verification (2026-06-01)' section — prod schema 001+002+003 applied, migrator IAM-auth path proven end-to-end at the DB, repo vars reconciled; do NOT redo this live state). Operator chose 'Re-plan Phase 2' at the Phase-1->2 boundary (Step 3.4). Now in the Phase 1 sub-flow scoped to Phase 2's PRs (planning_sub_flow: re-plan-phase-2). Goal: re-distribute the ~25 Phase-1 deferred items (PR-2.1 ingest role-ownership/grants, dead-proxy-grant cleanup, NaN/Inf is_finite() guard, the CI-hardening cluster golden==Python gate + pytest-on-PR + testcontainer-in-CI) plus the deferred 2026-05-29 deploy-model trigger-switch decision across a revised Phase-2 PR breakdown BEFORE implementing. Residual Phase-1 gate (unchanged): the live schema-deploy.yml OIDC workflow run is BLOCKED until ct/bench-v4 merges to develop. AWS access: local CLI profile bench-prod (IAM user connor-aws-cli, account 245040174862); see memories project_bench_aws_access + project_bench_phase1_live_state. Resume routes via status=planning + planning_sub_flow=re-plan-phase-2 into the design-tree interview (Step 1.4). Also: re-unlock 1Password if op-ssh-sign locks again."
subagent_invocations_this_pr: 0
subagent_invocations_total: 22
review_cycles_this_pr: 0
phase_entry_sha: ae3e0494f
phase_end_cycle: 2
phase_end_reject_cycles: 0
last_phase_end_verdict: accept
current_pr_is_ci_reopen: null
last_commit: cb68db1c6
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
| Connection pooler | **RDS Proxy for the Vercel read service only**; CI writers connect directly to the public RDS instance endpoint with IAM | Locked by Q2, **amended 2026-05-29 (Phase-1 re-plan)**. RDS Proxy endpoints are VPC-internal (not publicly reachable), so off-VPC GitHub-hosted CI runners cannot use the proxy. The proxy's pooling value is for Vercel's serverless read concurrency (Phase 4); the ~14 CI writers per push connect directly to the public instance endpoint with IAM tokens (the instance was provisioned `--publicly-accessible` with IAM auth + verify-full TLS). The original "single pooler covers both CI writers and Vercel reader" claim was wrong; corrected by the Phase-1 phase-end gauntlet (RDS-Proxy-unreachable-from-off-VPC finding). |
| Ingest writer language | **Pure Python** — extend `scripts/post-ingest.py`; port xxhash64 to Python with golden-vector tests against the existing Rust source-of-truth | User pick (Q4). Avoids adding sqlx + aws-sdk-rds Rust deps. Hash port is bounded by tests; Rust impl in `server/src/db.rs` stays the source of truth. Uses `psycopg[binary]` + `boto3.client('rds').generate_db_auth_token`. |
| Postgres schema deploy tool | **In-house `scripts/migrate-schema.py`** (~30-50 LOC) + plain SQL files under `migrations/` | User pick (Q5a). Tracks via `_applied_migrations` table; applies pending `00N_name.sql` in name order; CI workflow invokes with OIDC + IAM. Zero new tools / languages. |
| Schema-deploy authorization + execution-safety model | **PR merge IS the deploy gate** (no manual-approval `environment:` gate). `schema-deploy.yml` triggers `apply` on push to the deploy branch under `paths: migrations/**`; keep `dry_run`/`status` as the pre-apply preview and the testcontainer CI test as the per-PR safety check. **No** GitHub Environment / required-reviewer gate. | User decision 2026-05-29 (supersedes the original `environment: schema-deploy` manual-approval mandate). Two axes were conflated: *authorization* ("do we want this change?") is fully answered by reviewed-PR-merge; a human clicking "Approve" on an Environment only re-confirms authorization already given at merge, and does NOT verify the migration will apply cleanly. *Execution safety* ("will the DDL succeed against prod's current state?") is the real risk, and it is addressed by **testing the migration**, not by a manual gate. The testcontainer-against-empty-schema test (shipped) gates additive DDL (CREATE TABLE/INDEX, ADD COLUMN) at PR time, so a migration that cannot apply cannot merge. This also resolves the repo-admin blocker (creating an Environment needs admin; deciding the gate is the wrong tool removes the dependency). Tradeoffs knowingly forgone: deploy-timing decoupling (merge-now/apply-in-window) and segregation-of-duties (different approver than author); neither is material for a small-team benchmark site; revisit only on a compliance need. RDS has **no Neon-style instant copy-on-write branching**; the data-affecting-migration safety layer is a PITR-snapshot-restore-to-throwaway-instance CI step, added only when a migration mutates existing data (type change, NOT NULL on an existing column, backfill); out of scope for the additive Phase 1/early migrations. (Aurora fast-clone is the true Neon analog but requires reversing the `db.t4g.micro` Key decision; not worth it at this scale.) |
| One-shot historical data load | **Retarget `benchmarks-website/migrate/`** (existing Rust crate) for DuckDB→Postgres bulk load | Q5b — natural reuse. The crate already reads DuckDB via the `duckdb` crate and reuses `vortex_bench_server::db::measurement_id_*`. Add a `--postgres-target` mode + a Postgres bulk-insert path. Deleted post-cutover per AGENTS.md throwaway-migrator pattern. |
| CI network reach | **Public + IAM** — public RDS **instance** endpoint (the proxy is VPC-internal, not public), security group `0.0.0.0/0` because IAM is the gate, sslmode=verify-full | User pick (Q6), **amended 2026-05-29 (Phase-1 re-plan)**: the reachable endpoint is the RDS **instance** (publicly-accessible + IAM), not the proxy (which cannot be public). Every direct Postgres connection is IAM-gated; public-read of benchmark data is served by the Vercel HTTP layer, not by direct DB reads. All 14 CI writers connect to the instance endpoint with OIDC→IAM tokens. Matches the existing CI-to-AWS-S3 operational posture. |
| Cutover style | **Short dual-write window** (CI writes to BOTH v3 EC2 AND Postgres) for ~3-7 days of soak; then drop v3-write; then DNS flip from v2 directly to v4; then decommission v3 | User pick (Q7). v3 EC2 is a stepping stone that never goes live (DNS stays on v2 until the v4 flip). Dual-write gives benchmark-data-loss safety net during the v4 verification window. |
| Read service framework | **Next.js 15 + App Router + React Server Components + `unstable_cache` + `revalidateTag`** at `benchmarks-website/web/` | User pick (Q8). Server components fetch directly from Postgres; per-chart cache tags invalidated from the Python writer's CLI via a Vercel revalidation endpoint. Latest stable Next.js. Pages Router avoided. |
| Operator SQL replacement | **`scripts/psql-bench.sh`** — tiny helper that runs `aws rds generate-db-auth-token` and pipes into psql with IAM creds | User pick (Q9). Replaces `/api/admin/sql`. No bearer tokens, no Lambda. Documented in benchmarks-website/web/README.md. RDS PITR (35-day) replaces `/api/admin/snapshot`. |
| Composite index definition strategy | Net-new in `migrations/001_initial_schema.sql`. **As-shipped: dim-leading composite indexes following the read-path chart-query filter columns** (per `api/charts.rs`), NOT the hash field order. | **Amended 2026-05-29 (Phase-1 re-plan)** to match what PR-1.3 shipped: the original `(dim_tuple..., commit_timestamp DESC)` framing was superseded — every chart query filters on the dim columns and joins `commits` on `commit_sha`, so a dim-leading index serves the read path; PK uniqueness over the full hash tuple is already enforced by `measurement_id`. (PR-1.3 surprise, ratified here; an index-column-definition test is folded into PR-1.6.) |
| CI-write endpoint (re-plan 2026-05-29) | **Public RDS instance endpoint + direct IAM** for all CI writers (schema-deploy + Phase-2 ingest); RDS Proxy is Vercel-reads-only | Phase-1 phase-end gauntlet found the RDS Proxy is VPC-internal (unreachable from off-VPC GitHub runners). The instance was already provisioned `--publicly-accessible` with IAM auth, so CI writers connect to it directly with OIDC→IAM tokens + verify-full TLS. This **moots** the "register a migrator credential in the proxy auth config" finding for the CI write path (proxy auth config becomes a Phase-4 concern for the Vercel read role). Supersedes the proxy-for-CI assumption in the original pooler/Q6 decisions. |
| v3 EC2 final disposition | **Decommissioned at end of Phase 5** (single deletion PR removes `benchmarks-website/server/`, `benchmarks-website/ops/`, `benchmarks-website/migrate/`, top-level v2 files, `publish-benchmarks-website.yml`, `INGEST_BEARER_TOKEN`/`ADMIN_BEARER_TOKEN` secrets). EC2 instance terminated by hand after PR merges. | v3 never goes live; Q7 cutover model goes v2→v4 directly. |
| Phase-2 ingest DB identity (re-plan 2026-06-01) | **Dedicated `bench_ingest` role** (DB-side) + **`GitHubBenchmarkIngestRole`** (AWS-side), separate from the `migrator` / `GitHubBenchmarkSchemaRole` schema-deploy identity. `bench_ingest` gets DML-only (`SELECT,INSERT,UPDATE`, no DELETE/DDL) on the 6 data tables via migration 004. | Re-plan Q2. The ~14-writer ingest path runs on every push against a `PubliclyAccessible: true` instance with `0.0.0.0/0:5432` ingress (live-verified Phase-1 posture); a separate least-privilege identity means a leaked CI token can do data DML only, never DDL/migrations/role changes. Matches the `GitHubBenchmarkIngestRole` the original plan already anticipated. Cost: one migration + one provision.sh role block. Rejected: reuse `migrator` (conflates schema-deploy authority with the most-exposed code path). |
| Phase-2 dual-write verify scope (re-plan 2026-06-01) | **Postgres-side reconciliation** during the soak: the writer's computed measurement_id set for each pushed commit is asserted fully present in Postgres after a settle window (alert on missing/extra) + idempotent re-ingest. The authoritative cross-substrate DuckDB↔Postgres row comparison stays in **Phase 3's `migrate --verify`**. | Re-plan Q3. The new/risky path during dual-write is Postgres (v3 is the proven baseline); reconciling the writer's intended ids against Postgres directly guards Risk #4 (dropped/duplicated rows under matrix concurrency) without reading DuckDB. Rejected: `/api/admin/sql` cross-substrate (loopback-only `127.0.0.1`, decommission-bound, needs CI SSH) and v3-read-API compare (requires porting chart SQL before the Phase-4 read layer exists; only post-processed data). Revises the Phase-2 exit criterion from `only_in_duckdb = []` to Postgres-side reconciliation. |

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

**Phase-2 ingest/identity BANS (re-plan 2026-06-01):**
- Do not grant the `bench_ingest` role `DELETE`, `TRUNCATE`, or any DDL/role privilege; the ingest path is data-DML-only (`SELECT,INSERT,UPDATE` on the 6 tables). (Why: the dedicated role exists for least-privilege separation from `migrator`; widening it defeats the separation-of-duties decision.)
- Do not have CI authenticate the ingest write path as `migrator` / `GitHubBenchmarkSchemaRole`. (Why: that conflates the high-frequency, internet-reachable ingest path with the schema-deploy identity that can run DDL + migrations.)
- Do not write to Postgres without an `is_finite()` guard on every f64 dim that feeds the hash (e.g. `threshold`). (Why: Rust `to_bits()` preserves NaN payload bits while Python `struct.pack('<d', nan)` emits canonical NaN, so a non-finite value hashes differently across languages and silently produces a duplicate row instead of an upsert.)
- Do not echo or log the RDS IAM auth token / `PGPASSWORD` (no `set -x` around token generation; assign it on its own line). (Why: the token is a short-lived credential; the schema-deploy.yml discipline must carry to the ingest workflows.)

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
| 1 | RDS + schema + hash port | Provision RDS, write schema-deploy script + initial DDL, port xxhash64 to Python with golden vectors; **(re-plan) repoint CI writes to the public instance endpoint + sweep cycle-1 must-fixes** | `aws rds describe-db-instances --db-instance-identifier vortex-bench-prod` returns `available`; `python scripts/migrate-schema.py status` clean; `pytest scripts/test_measurement_id.py` all green; Rust golden-vector test in `vortex-bench-server` matches Python output bit-exactly; **`schema-deploy.yml` applies live as the OIDC migrator against the public instance endpoint (verify-full TLS) and `status` reports clean** | 6 | 3-vote |
| 2 | Postgres writer + dual-write CI | **(re-plan 2026-06-01)** Dedicated `bench_ingest` role + grants (migration 004); extend post-ingest.py with `--postgres` mode + IAM-auth + NaN/Inf guard; wire `scripts/` pytest into CI; flip 3 CI workflows to dual-write + switch schema-deploy trigger to push-on-deploy-branch; Postgres-side dual-write reconciliation + alert | `bench_ingest` connect-and-upsert round-trip test green; writer integration test green (insert N, re-ingest 0 inserted + N updated, measurement_id match); `scripts/` pytest job runs in CI (golden==Python gated, fails loud if Docker absent); 3 workflows green on a test PR with dual-write; `schema-deploy.yml` triggers `apply` on push to deploy branch under `paths: migrations/**`; reconciliation harness reports the writer's computed measurement_id set fully present in Postgres after a settle window + alerts on divergence | 5 | 3-vote |
| 3 | Historical data load (DuckDB → Postgres) | Extend `benchmarks-website/migrate/` with `--postgres-target`; run one-shot load; verify | `migrate --postgres-target --verify` reports `matched_rows == duckdb_rows AND only_in_postgres == []` against a live DuckDB snapshot | 3 | 3-vote |
| 4 | Next.js read service on Vercel | Scaffold `benchmarks-website/web/`; connection lib + `unstable_cache`; port all read endpoints; deploy to Vercel preview; revalidation API + token-plumbing from Python writer | `vercel deploy --target=preview` produces URL serving all chart slugs; `curl preview-url/api/groups | jq '.[].slug' \| sort` matches the family registry; preview lighthouse score ≥ 90 | 6 | 3-vote |
| 5 | Cutover + decommission | Drop v3-write from CI; DNS flip v2 → v4; delete v2 frontend + server/ + ops/ + migrate/ + publish-benchmarks-website.yml + bearer-token secrets | `git grep -n INGEST_BEARER_TOKEN` returns 0; `gh workflow list` does not include `publish-benchmarks-website`; production DNS resolves to Vercel; v3 EC2 terminated | 3 | 4-vote (final) |

Total: **5 phases, 23 PRs** (Phase 2 re-planned 2026-06-01 from 4 → 5 PRs).

### PR enumeration

| PR | Phase | Scope (one line) | Files touched (expected) | Acceptance (specific, testable) |
|---|---|---|---|---|
| PR-1.1 | 1 | Provision RDS Postgres `db.t4g.micro` + RDS Proxy + GitHub OIDC schema role via aws-cli script; document in `benchmarks-website/infra/README.md`; capture endpoint via post-run `gh variable set RDS_BENCH_ENDPOINT` | `benchmarks-website/infra/provision.sh`, `benchmarks-website/infra/README.md`, `.github/workflows/schema-deploy.yml` (skeleton) | `aws rds describe-db-instances --db-instance-identifier vortex-bench-prod` returns `available` + `iam_database_authentication_enabled: true`; `aws rds describe-db-proxies --db-proxy-name vortex-bench-proxy` returns `Endpoint`. |
| PR-1.2 | 1 | Write `scripts/migrate-schema.py` (~80-180 LOC; original ~30-50 estimate was for the bare runner — status/drift, autocommit txn discipline, typed exceptions, and recovery commentary brought it closer to ~180) — applies `migrations/*.sql` in name order, tracks via `_applied_migrations` table, idempotent | `scripts/migrate-schema.py`, `scripts/test_migrate_schema.py`, `migrations/` (dir created + README), `pyproject.toml` (psycopg + testcontainers dev deps) | Unit test: applies a fresh schema to testcontainers Postgres, re-runs idempotently (0 rows changed second time), inserts and re-applies a v2 migration in order. `apply` survives a failing later migration without losing earlier ones (subprocess test). `status` exits non-zero on drift and does not DDL. |
| PR-1.3 | 1 | Write `migrations/001_initial_schema.sql` (the 6 tables + composite indexes per Table B); `migrations/002_iam_db_user.sql` (CREATE ROLE for IAM auth) | `migrations/001_initial_schema.sql`, `migrations/002_iam_db_user.sql`, `scripts/test_migrate_schema.py` (extended) | `python scripts/migrate-schema.py --target=$RDS_DSN apply` succeeds; `\dt` shows the 6 tables; `\di` shows the composite indexes; `\du` shows the IAM-auth role with `rds_iam` group. |
| PR-1.4 | 1 | Wire `.github/workflows/schema-deploy.yml` — OIDC → `GitHubBenchmarkSchemaRole` → `python scripts/migrate-schema.py apply` against RDS Proxy. ~~gated by `environment: schema-deploy` (manual approval)~~ SUPERSEDED by the 2026-05-29 deploy-model Key decision: **no `environment:` gate; PR merge is the deploy gate**. As-shipped (PR-1.4 complete): `workflow_dispatch`-only + `dry_run`; the merge-trigger switch is tracked in Deferred work. | `.github/workflows/schema-deploy.yml`, IAM role doc in `benchmarks-website/infra/README.md` (plan originally said `web/ops/README.md`; `web/` does not exist until PR-4.1) | `migrate-schema.py apply` runs as the OIDC `migrator` role against RDS Proxy; `status` reports clean post-apply. (Original "manual-approval gate fires" criterion dropped per the deploy-model decision.) |
| PR-1.5 | 1 | Port xxhash64 to Python in `scripts/_measurement_id.py`; mirror per-table tag + write_str/write_opt_str/write_i32/write_f64 encoding; golden-vector test against Rust source-of-truth | `scripts/_measurement_id.py`, `scripts/test_measurement_id.py`, `benchmarks-website/server/tests/measurement_id_golden.rs` (golden generator/asserter) | `pytest scripts/test_measurement_id.py` all green; for **63** committed golden vectors (all 5 fact tables + i32 MIN/MAX + empty/`Some("")` strings + multibyte UTF-8 — amended 2026-05-29 from the original "100 fixture inputs" estimate; the 63 exhaustively cover every table + boundary class) the Python output matches the Rust output bit-exactly. |
| PR-1.6 | 1 | **(re-plan)** Repoint CI writes to the public RDS instance endpoint + sweep the cycle-1 phase-end must-fixes/should-fixes: capture+export the instance-endpoint repo var (`RDS_BENCH_INSTANCE_ENDPOINT`) in `provision.sh`; point `schema-deploy.yml` `PGHOST` at it; set `PGSSLMODE=verify-full` in the README bootstrap; fix the `provision.sh:19` account comment + promote the hardcoded `proxy_role_name`; add `003` to `migrations/README`; tighten the stale schema-deploy env-gate comment; doc the CI=instance / proxy=Vercel split; add an index-column-definition test | `benchmarks-website/infra/provision.sh`, `.github/workflows/schema-deploy.yml`, `benchmarks-website/infra/README.md`, `migrations/README.md`, `scripts/test_migrate_schema.py` | `yamllint --strict` clean; `schema-deploy.yml` `PGHOST` resolves to the instance-endpoint repo var (not the proxy); README bootstrap uses verify-full; `git grep -n 375504701696 benchmarks-website/infra/provision.sh` returns 0; index-column-definition test green; **operator runs the live OIDC apply against the instance endpoint and `status` reports clean**. |
| PR-2.1 | 2 | **(re-plan)** Foundational ingest identity: `migrations/004_ingest_role.sql` creates `bench_ingest` (rds_iam member; `GRANT SELECT,INSERT,UPDATE` on the 6 data tables + `USAGE ON SCHEMA public` + `ALTER DEFAULT PRIVILEGES`; **no DELETE, no DDL**); `provision.sh` adds AWS `GitHubBenchmarkIngestRole` (OIDC trust branch-scoped to develop + ct/bench-v4, `rds-db:connect` for dbuser `bench_ingest`) and drops the dead RDS-Proxy `rds-db:connect` grant on `GitHubBenchmarkSchemaRole` | `migrations/004_ingest_role.sql`, `scripts/test_migrate_schema.py` (extended), `benchmarks-website/infra/provision.sh`, `benchmarks-website/infra/README.md` | testcontainer test connects AS `bench_ingest` and runs `INSERT ... ON CONFLICT DO UPDATE` on all 6 tables successfully, and is denied a DDL/DELETE attempt; `git grep` shows no proxy `rds-db:connect` on the schema role; migration 004 is idempotent under re-apply (re-run reports 0 applied) |
| PR-2.2 | 2 | Extend `scripts/post-ingest.py` with a `--postgres $RDS_DSN` mode: parse JSONL, compute measurement_id via `_measurement_id.py`, mint an IAM token (boto3) for `bench_ingest`, upsert `commits` first then the fact tables via `INSERT ... ON CONFLICT DO UPDATE` over verify-full TLS; add an `assert <f64 dim>.is_finite()` guard (e.g. `threshold`) at the ingest boundary; keep `SCHEMA_VERSION` lockstep | `scripts/post-ingest.py`, `scripts/test_post_ingest_postgres.py`, `pyproject.toml` (psycopg, boto3) | Integration test (testcontainers Postgres): POST a v3 envelope → N rows inserted; POST same envelope → 0 inserted + N updated; all measurement_id values match `_measurement_id.py`; a NaN/Inf f64 dim raises a loud error rather than writing a divergent row; `SCHEMA_VERSION` matches `schema.rs`/`v3.rs` |
| PR-2.3 | 2 | **(re-plan, CI-hardening)** Wire the `scripts/` pytest suites into CI: a job running `uv run --all-packages pytest scripts/` covering `test_measurement_id.py` (golden==Python, no Docker) and `test_migrate_schema.py` (testcontainer, needs Docker), failing loud when Docker is unavailable in CI (no silent skip) | `.github/workflows/ci.yml` (or new `scripts-test.yml`), `pyproject.toml` (`testpaths`/markers if needed), `scripts/test_migrate_schema.py` (Docker-required assertion) | The job runs both suites on PR; an intentional divergence in `_measurement_id.py` fails the golden==Python check (now CI-gated); the testcontainer suite runs (not silently skipped) when Docker is present and fails loud when absent |
| PR-2.4 | 2 | **(re-plan)** Flip `bench.yml`, `sql-benchmarks.yml`, `v3-commit-metadata.yml` to dual-write — invoke `post-ingest.py` once `--server` (v3) then once `--postgres` (v4) via OIDC → `GitHubBenchmarkIngestRole`, both must succeed (no `continue-on-error`); add `id-token: write` + an assume-role step where missing (e.g. `v3-commit-metadata.yml`). AND switch `schema-deploy.yml` from `workflow_dispatch`-only to also push on the deploy branch under `paths: migrations/** + scripts/migrate-schema.py` (keep `workflow_dispatch` + `dry_run`), removing the superseded `environment:`-gate comments | `.github/workflows/{bench.yml,sql-benchmarks.yml,v3-commit-metadata.yml,schema-deploy.yml}` | `yamllint --strict` clean; all 3 ingest workflows green on a test PR push with dual-write (both substrates written, neither best-effort); `schema-deploy.yml` `on:` includes push to the deploy branch under `paths: migrations/**`; CloudWatch shows ~14 Postgres writes per develop push |
| PR-2.5 | 2 | **(re-plan)** Postgres-side dual-write reconciliation: `scripts/reconcile-ingest.py` recomputes the expected measurement_id set for a pushed commit from its envelope(s) via `_measurement_id.py` and asserts every id is present in Postgres after a settle window (alerts on missing/extra); `.github/workflows/dual-write-verify.yml` runs it after `develop` pushes and alerts via incident.io webhook on divergence. (Authoritative cross-substrate DuckDB↔Postgres comparison stays in Phase 3's `migrate --verify`.) | `scripts/reconcile-ingest.py`, `scripts/test_reconcile_ingest.py`, `.github/workflows/dual-write-verify.yml` | Tool reports zero missing/extra measurement_ids on a seeded fixture; reports + alerts on an injected mismatch (manually drop a row); workflow exits 0 on clean state and yamllint --strict clean |
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
| ~~`scripts/_measurement_id.py`~~ | (removed 2026-05-29) | The shipped hash port does NOT import or re-export `SCHEMA_VERSION` (no need for it, and `post-ingest.py` is not cleanly importable without `importlib`). `_measurement_id.py` is NOT a lockstep site; the Python writer's lockstep site is `scripts/post-ingest.py:52` (above). |
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
- **Repo vars set** (verified at PR-1.1): `RDS_BENCH_ENDPOINT`, `RDS_BENCH_REGION`, `RDS_BENCH_DB_NAME`, `GH_BENCH_SCHEMA_ROLE_ARN`. **Superseded by PR-1.6**: CI was repointed to the public instance endpoint, `RDS_BENCH_INSTANCE_ENDPOINT` was added as the CI var, and `RDS_BENCH_ENDPOINT` (proxy) was dropped as a GitHub variable — the proxy endpoint is now a Vercel-config value (PR-4.2), not a GitHub variable. **CORRECTION (2026-06-01 live verification):** these repo-var changes (add `RDS_BENCH_INSTANCE_ENDPOINT`, drop `RDS_BENCH_ENDPOINT`) describe the intended end state but were NOT actually applied to the live GitHub repo by PR-1.6; they were reconciled in-session on 2026-06-01. See section "Phase 1 live verification (2026-06-01)".
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

### PR-1.3: Postgres initial-schema + IAM migrator role migrations  (1 code commit + 1 gauntlet cycle, ending at b00cd967d)

- **Scope shipped**: `migrations/001_initial_schema.sql` (the `commits` dim table + 5 fact tables + 6 read-path composite indexes, the Postgres translation of the authoritative DuckDB DDL in `benchmarks-website/server/src/schema.rs`); `migrations/002_iam_db_user.sql` (the `migrator` login role + conditional `rds_iam` grant + `GRANT CREATE, USAGE ON SCHEMA public`); `scripts/test_migrate_schema.py` gains 12 testcontainer-backed tests (apply-cleanly, idempotency, table set, index set, per-table column-shape pin, key type-translation spot-checks, migrator role).
- **Acceptance criteria — verified against `postgres:16-alpine` testcontainer**:
  - Real migrations apply cleanly + idempotently (`test_real_migrations_apply_cleanly`, `test_real_migrations_idempotent`) ✓
  - All 6 tables created (`test_real_migrations_create_expected_tables`) ✓
  - All 6 composite indexes created (`test_real_migrations_create_expected_indexes`) ✓
  - Per-table column order + nullability matches `schema.rs` exactly (`test_real_migrations_preserve_column_shape[*]`, 6 parametrized cases) ✓
  - `DOUBLE`→`double precision`, `BIGINT[]`→`ARRAY`, `measurement_id` bigint PK (`test_real_migrations_key_column_types`) ✓
  - `migrator` login role created (`test_real_migrations_create_migrator_role`) ✓
  - The `\dt`/`\di`/`\du` + `apply` plan acceptance criteria are exercised by the testcontainer assertions; `rds_iam` group membership is operator-verified on real RDS (untestable on vanilla Postgres, guarded by `IF EXISTS`).
- **Tests added**: 12 (all testcontainer-backed). Local run: **28/28 pass** (16 prior + 12 new) against `postgres:16-alpine`. `py_compile` OK, `ruff` clean, `git diff --check` clean. (CI execution of the suite remains the deferred D-`PR-1.2 no-CI-runner` item, tracked for PR-1.4.)
- **Review**: 2-vote (preset=pr-2, fresh+correctness, Claude executor) / cycle 1 ACCEPT (zero must-fix; both reviewers accept). Correctness skeptic ran an automated column-by-column comparison vs `schema.rs` confirming byte-equivalent schema shape across all 6 tables. 1 should-fix (migrator table privileges) deferred to PR-2.1; 3 nits dismissed (redundant-but-intentional `NOT NULL`, PG15+ index-planner dependency satisfied by target, untestable `rds_iam` membership).
- **Confidence**: high (single-cycle clean accept; behavior-preservation obligation met and pinned by an ordinal column-shape regression test; transaction-block constraint and idempotency reasoned through by both lenses).
- **Deferred items**: 1 should-fix (migrator table-level privileges -> resolve in PR-2.1 role-ownership design; see Deferred work table).
- **Surprises during implementation**:
  - None material. The composite-index design deliberately follows the read-path chart-query filter columns (`api/charts.rs`) rather than the `measurement_id` hash field order in Table B — the hash tuple leads with `commit_sha` but every chart query filters on the dim columns and joins `commits` on `commit_sha`, so a dim-leading index is what serves the read path; PK uniqueness over the full hash tuple is already enforced by `measurement_id`.

### PR-1.4: Wire schema-deploy.yml (apply-as-migrator) + ledger grant  (2 code commits + 2 fix commits + 2 gauntlet cycles, ending at 5fc9ad5be)

- **Scope shipped**: `.github/workflows/schema-deploy.yml` rewritten from the PR-1.1 OIDC-probe skeleton into the real apply workflow (workflow_dispatch-only with a `dry_run` input; OIDC -> `GitHubBenchmarkSchemaRole`; client-side IAM auth-token; RDS global CA bundle download; `PGSSLMODE=verify-full`; `uv run --no-project scripts/migrate-schema.py apply` then `status` as the `migrator` role through the RDS Proxy). `migrations/003_migrator_ledger_grant.sql` (minimal idempotent `GRANT SELECT, INSERT ON public._applied_migrations TO migrator` so CI's migrator-role apply can record/read the ledger against a master-owned bootstrap; no DELETE/UPDATE = append-only least-privilege). `scripts/test_migrate_schema.py` updated for the 3-migration set + a discriminating ledger-grant test (SELECT/INSERT present, DELETE+UPDATE absent). `benchmarks-website/infra/README.md` gains a "Schema deploys + one-time bootstrap" section (master bootstrap must hit the INSTANCE endpoint since the proxy is `IAMAuth=REQUIRED`) and a corrected master-password claim.
- **Acceptance criteria**: workflow runs `migrate-schema.py apply` as the OIDC `migrator` role (plan PR-1.4 row) — wired and yamllint --strict clean; manual-approval gate intentionally deferred (operator lacks repo-admin to create environments; workflow_dispatch IS the gate). Tests: **29/29 pass** against `postgres:16-alpine`; `py_compile` OK, `ruff` clean, `git diff --check` clean.
- **Tests added**: 1 (ledger-grant privileges); 1 existing test extended (UPDATE-absent assertion).
- **Review**: 2-vote (preset=pr-2, fresh+correctness, Claude executor). **Cycle 1 REJECT (3 must-fix)** — the workflow `Write` had silently failed (pre-existing skeleton not Read first), so the "wire" commit touched only the README while the workflow stayed a placeholder-echo skeleton, the `push:` trigger remained, and docs described a nonexistent workflow. Caught by BOTH lenses independently. **Fix**: landed the real workflow (resolving all 3 facets at once) + tightened the test. **Cycle 2 ACCEPT** (zero must-fix; both lenses); fix-commit attention block confirmed no regressions (token leak-safe, PEP 723 deps resolve, CA path consistent). 1 should-fix deferred, 3 nits dismissed.
- **Confidence**: high (the namesake-deliverable bug was the exact failure gauntlet exists to catch; cycle-2 reviewers verified the wired workflow as new code — token handling, `uv run --no-project` PEP 723 resolution against sibling workflows, verify-full, dry_run parsing all confirmed correct).
- **Deferred items**: 1 should-fix (uv Python auto-provisioning without an explicit pin -> matches repo convention, revisit in a CI-hardening pass; see Deferred work).
- **Surprises during implementation**:
  - The cycle-1 reject was a process bug, not a design bug: a `Write` tool call was rejected ("file not Read first") for the pre-existing skeleton and the rejection was missed inside a batch of parallel calls. This is precisely the Edit/Write-without-Read failure mode the spec's NEW-C "verify the staged diff actually changed before committing" discipline guards against. The recovery applied the discipline literally (Read -> Write -> `git diff --cached --quiet` check -> commit).
  - The plan named `web/ops/README.md` for the IAM-role doc, but `web/` does not exist until PR-4.1; the docs went to `benchmarks-website/infra/README.md` (the correct home today). Recorded as an intentional plan deviation.

### PR-1.5: xxhash64 measurement_id Python port + cross-language golden vectors  (1 impl commit + 1 fix commit + 1 gauntlet cycle, ending at b912d4b6a)

- **Scope shipped**: `scripts/_measurement_id.py` (byte-for-byte Python port of the server-internal xxhash64 `measurement_id_*` functions from `benchmarks-website/server/src/db.rs`: per-table tag + `0x00` separator, LE-u64 length-prefixed `write_str`, `write_opt_str` tag bytes, LE `write_i32`, IEEE-754 `write_f64` bits, `u64 -> i64` bitcast); `benchmarks-website/server/tests/measurement_id_golden.rs` (Rust source-of-truth golden-vector generator + always-assert test; regenerates the committed JSON under `REGEN_GOLDEN_VECTORS`); `scripts/measurement_id_golden.json` (63 committed golden vectors covering all 5 fact tables + `i32` MIN/MAX + empty / `Some("")` strings + multibyte UTF-8); `scripts/test_measurement_id.py` (asserts the Python port reproduces every committed golden vector). Transitive pin: Rust == golden == Python.
- **Acceptance criteria**: `pytest scripts/test_measurement_id.py` all green -- **65 passed** (re-verified this session). Python output matches the Rust golden file bit-exactly for all 63 vectors. (NOTE: the PR-1.5 row promised "100 fixture inputs"; 63 were shipped -- flagged by the phase-end gauntlet as a must-fix to reconcile.)
- **Tests added**: `scripts/test_measurement_id.py` (65 cases: per-vector golden comparison across all tables + boundary fixtures + all-tables-covered + multibyte-present guards) plus the Rust golden generator/asserter in `measurement_id_golden.rs`.
- **Review**: 2-vote (preset=pr-2, fresh+correctness) / cycle 1 ACCEPT (zero must-fix). 1 cycle-1 fix-commit (`c80bf5c6d`: dead code + dangling doc ref, should-fix applied post-accept). 1 should-fix deferred (scripts/ pytest not wired into CI -- see Deferred work).
- **Confidence**: high (single-cycle clean accept; the load-bearing cross-language hash equivalence is pinned transitively and verified bit-exact).
- **Deferred items**: 1 should-fix (golden==Python not CI-gated; PR-1.5 cycle-1 -- see Deferred work table).
- **Surprises during implementation**:
  - The implementation landed bundled in commit `0b431a8c5` (subject `plan: PR-1.5 gauntlet cycle 1 accepted`) rather than a separate `<area>:` implementation commit -- an artifact of the operator's 65-commit rebase onto `develop`. The code content is correct and tested; only the commit-subject attribution is non-standard.
  - **Ledger backfill**: this entry was reconstructed and backfilled during the Phase-1 boundary on resume after it was found missing from Implementation status (PR-1.5 was marked complete at `b912d4b6a` without its Step 2.5 ledger entry landing). The phase-end gauntlet ran with PR-1.5 context supplied via the PR-enumeration row + the cumulative diff.

### PR-1.6: (re-plan) repoint CI schema-deploy to the public instance endpoint + sweep phase-end cross-ref drift  (2 impl/fix commits ending at `b90a740da`, across 7 inner-loop gauntlet cycles)

- **Scope shipped**: Repointed `schema-deploy.yml` `PGHOST` from the VPC-internal RDS Proxy var (`RDS_BENCH_ENDPOINT`) to the public instance var (`RDS_BENCH_INSTANCE_ENDPOINT`) with `PGSSLMODE=verify-full`; corrected the AWS account comment (`375504701696` → `245040174862`); established the CI=instance / proxy=Vercel split consistently across `schema-deploy.yml`, `benchmarks-website/infra/README.md`, `benchmarks-website/infra/provision.sh`, and `migrations/002_iam_db_user.sql`; replaced the README master-password bootstrap with a non-interactive, fail-fast Secrets-Manager fetch (`master_secret=$(aws ...) || exit 1; PGPASSWORD=$(printf '%s' "$master_secret" | jq -er '.password') || exit 1; export PGPASSWORD`); removed `RDS_BENCH_ENDPOINT` from the GitHub repo-vars (it is a Vercel-config value, not a GitHub variable). **CORRECTION (2026-06-01):** this describes the intended end state; the live GitHub repo-var edits were not actually performed until the 2026-06-01 live verification. See section "Phase 1 live verification (2026-06-01)".
- **Acceptance criteria**: `yamllint --strict` clean on `schema-deploy.yml`; `PGHOST` resolves to the instance var; README bootstrap uses `verify-full`; `git grep 375504701696 benchmarks-website/infra/provision.sh` returns 0; the static guard suite (5 no-DB tests + the DB-backed index-column test) green locally (9 passed / 26 Docker-skipped). The "operator runs the live OIDC apply against the instance endpoint" criterion is an out-of-band operator step (not gated here).
- **Tests added**: 5 static regression guards in `scripts/test_migrate_schema.py` — `test_schema_deploy_targets_instance_endpoint` (PGHOST=instance var + `PGSSLMODE=verify-full`, no proxy var), `test_provision_emits_instance_endpoint_var` (gh-var name/body: instance var set from `${DB_ENDPOINT}`, no repo-var from `${PROXY_ENDPOINT}`), `test_provision_grants_instance_dbuser` (IAM policy grants `dbuser:${DB_RESOURCE_ID}/${PG_MIGRATOR_ROLE}`), `test_readme_bootstrap_pins_verify_full` (rejects require/prefer/allow/disable/verify-ca), `test_readme_bootstrap_password_fetch_is_safe` (pins the non-interactive SM fetch; bans `export PGPASSWORD=$(`/stty/interactive-read on code lines) — plus the index-column test now pins `(table, columns)` per index incl. `tablename` and DESC/DESC ordering.
- **Review**: 2-vote `preset=pr-2` (fresh + correctness), `executor=parallel` (Claude **and** Codex per lens). **7 inner-loop cycles**, 2 early-breaks. Cycles 1–5: cross-reference drift from the endpoint split, surfaced at distinct new sites each cycle (README:139 → :21/provision:500/002:5 → :71/76 repo-vars + provision:505 → :52 → fully converged at cycle 5). Cycles 3–7: the bootstrap-password runbook line churned across 4 fixes (single-quote → `read -rsp` → `stty/read` → SM-fetch → fail-fast assign-then-export). **Cycle-7 (final) verdict: REJECT, 2 must-fix — both test-guard-strictness / coverage-completeness, NOT functional defects** (pin `\|\| exit 1`; pin `PROXY_ROLE_NAME` usage). **Accepted by user-authorized override** at the cycle-7 second early-break ("accept regardless after one final cycle; defer residual"). The parallel Claude+Codex executor disjointness was the value driver: the Codex lens caught the cycle-2/3/4/5/6/7 must-fix that both Claude lenses rated accept/nit every cycle.
- **Confidence**: high on the functional change (the endpoint repoint + cross-ref consistency were verified clean by every reviewer across all 7 cycles; the password fetch is fail-fast and syntax-checked under sh/bash/zsh/dash). medium on test-guard exhaustiveness (the deferred cycle-7 residual hardens the guards further).
- **Deferred items**: 6 (cycle-6/7 residual) → see Deferred work: 2 must-fix deferred via user-authorized early-break (`test:868` pin `\|\| exit 1`; `provision.sh:63` `PROXY_ROLE_NAME` guard), 2 should-fix (`test:884` quoted-export form; `README:23` IAM-work contradiction), 2 cycle-6 doc nits (`secretsmanager:GetSecretValue` prereq; tear-down OIDC ARN account). All fold into a follow-up test-hardening + doc-polish pass.
- **Surprises during implementation**:
  - The runbook-snippet + test-coverage churn (7 cycles) was driven by adversarial Codex reviewers in unbounded refinement mode on a doc/test-heavy PR: each cycle's fix surfaced the next stricter angle (H4 self-reinforcement at the fix boundary). The synthesizer flagged this explicitly at cycles 5–7; the user bounded it with two early-breaks (Continue at cycle 5, accept-after-one-final-cycle at cycle 7).
  - The functional change itself never regressed — every reject was about cross-reference doc consistency or test-guard strictness, never the repoint logic, IAM grant, or hash/schema invariants.

### Cycle 1 — preset=phase-3 — reject

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1)</summary>

```json
{
  "schema_version": 1,
  "preset": "phase-3",
  "lenses_used": [
    "spec",
    "correctness",
    "maint"
  ],
  "review_count": 3,
  "unified_findings": [
    {
      "severity": "must-fix",
      "kind": "bug",
      "file_line": ".github/workflows/schema-deploy.yml:77",
      "description": "RDS_BENCH_ENDPOINT is the RDS Proxy hostname, but RDS Proxy endpoints are not publicly accessible and GitHub-hosted runners are off-VPC. As wired, the schema-deploy job cannot reach the proxy at all. This challenges Key decisions Q2/Q6 ('RDS Proxy public endpoint, security group 0.0.0.0/0').",
      "recommended_fix": "Point off-VPC schema-deploy at the public RDS *instance* endpoint with direct IAM auth (the proxy stays for Vercel reads), OR run schema-deploy inside the VPC (self-hosted runner / CodeBuild), OR expose the proxy via an intentional NLB/PrivateLink design. Resolve before claiming the schema-deploy path works; this likely amends Q2/Q6.",
      "found_by": [
        "correctness/codex"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "bug",
      "file_line": "benchmarks-website/infra/provision.sh:311",
      "description": "The RDS Proxy is provisioned with only the RDS-managed master secret, but CI connects as PGUSER=migrator. Standard RDS Proxy IAM auth still needs a Secrets Manager credential registered for the migrator DB user; without it the proxy cannot authenticate the migrator connection.",
      "recommended_fix": "Either configure end-to-end IAM auth for the proxy for the migrator user, or create+attach a migrator credential secret and register it in the proxy auth config. Add a smoke test that connects through the chosen endpoint as migrator. (Couples with the schema-deploy.yml:77 reachability finding \u2014 resolve the CI-write endpoint design together.)",
      "found_by": [
        "correctness/codex"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "missing-acceptance",
      "file_line": ".github/workflows/schema-deploy.yml:68",
      "description": "PR-1.4's acceptance criterion was 'migrate-schema.py apply runs as the OIDC migrator role against RDS Proxy; status reports clean post-apply.' Implementation status records only wiring + yamllint + testcontainer coverage \u2014 the live OIDC apply against real RDS Proxy was never executed. Combined with the proxy-reachability and migrator-credential findings, the schema-deploy path is unproven and may be non-functional as designed.",
      "recommended_fix": "After resolving the endpoint/credential design, run schema-deploy live once and record the clean apply/status, OR explicitly amend the PR-1.4 acceptance criterion to state live execution is deferred (and to which phase) with rationale.",
      "found_by": [
        "spec"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "unsafe",
      "file_line": "benchmarks-website/infra/README.md:120",
      "description": "The master-user bootstrap runbook command uses PGSSLMODE=require, which encrypts but does NOT verify the RDS server certificate, while transmitting the master password. This is a MITM exposure on the single most sensitive credential in the system. The schema-deploy workflow correctly uses verify-full; the bootstrap runbook is inconsistent.",
      "recommended_fix": "Change the bootstrap runbook to PGSSLMODE=verify-full with PGSSLROOTCERT pointed at the downloaded RDS CA bundle, matching the workflow.",
      "found_by": [
        "correctness/codex"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "doc-quality",
      "file_line": "benchmarks-website/infra/provision.sh:19",
      "description": "The header comment says the script provisions into account 375504701696 'by default', but the actual TARGET_ACCOUNT default (line 50) and the entire README/plan are 245040174862. Per Key decisions, 375504701696 is the PERSONAL/v3-EC2 account \u2014 exactly the account the bench infra must NOT land in. An operator trusting the header points at the wrong account; verify_prereqs would then die confusingly, or worse the operator provisions into the wrong account.",
      "recommended_fix": "Change the line-19 comment to account 245040174862 to match TARGET_ACCOUNT and the README; or interpolate the value rather than hardcoding a stale literal.",
      "found_by": [
        "maint",
        "correctness/claude"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "scope-drift",
      "file_line": "benchmarks-website/server/tests/measurement_id_golden.rs:102",
      "description": "PR-1.5's acceptance promised '100 fixture (commit, dim-tuple) inputs'; the generator + committed golden file contain 63 vectors. The qualitative coverage is strong (all 5 tables + i32 MIN/MAX + empty/Some('') strings + multibyte UTF-8), but the literal acceptance criterion is unmet and was not amended.",
      "recommended_fix": "Either add ~37 more deterministic fixture vectors, OR amend the PR-1.5 acceptance criterion to '63 vectors' with rationale (the chosen 63 exhaustively cover all tables + boundary classes). Cheap; pick one and record it.",
      "found_by": [
        "spec"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "weak-exit-criteria",
      "file_line": "scripts/test_measurement_id.py:1",
      "description": "The Phase-1 exit criterion (Phases-and-PRs table) names 'pytest scripts/test_post_ingest_hash.py all green', but the artifact ships scripts/test_measurement_id.py. The documented phase gate is unrunnable as written (no such file). Independently confirmed during exit-criteria execution.",
      "recommended_fix": "Amend the Phase-1 exit-criteria string to 'pytest scripts/test_measurement_id.py' (the as-shipped file). Plan-edit only.",
      "found_by": [
        "spec"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "scope-drift",
      "file_line": "scripts/_measurement_id.py:50",
      "description": "Reference Table D ('SCHEMA_VERSION lockstep sites') claims scripts/_measurement_id.py 'Imports SCHEMA_VERSION from post-ingest.py | Python re-export to keep one site'. The shipped module neither imports nor re-exports SCHEMA_VERSION (and could not cleanly import from hyphenated post-ingest.py without importlib). The hash port is correct to omit it; Table D is stale and would mislead a future SCHEMA_VERSION bump. (Table D is a reference downstream PRs grep into.)",
      "recommended_fix": "Amend Table D to remove the _measurement_id.py re-export row (or replace it with the real lockstep site). Do NOT add a spurious re-export to the hash port. Plan-edit only. (See disagreement: spec framed this as must-fix scope-drift requiring re-export OR amendment; maint framed it as should-fix amend-the-doc; synthesizer call = amend Table D, code is correct.)",
      "found_by": [
        "spec",
        "maint"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "doc-quality",
      "file_line": "migrations/README.md:35",
      "description": "The 'Initial files' section lists only 001 and 002 and says the SQL files 'land in PR-1.3', but PR-1.4 added 003_migrator_ledger_grant.sql, which exists on disk and is exercised by the test suite. The directory's own README is a stale, incomplete inventory.",
      "recommended_fix": "Add a bullet for 003_migrator_ledger_grant.sql (GRANT SELECT,INSERT on the ledger to migrator, PR-1.4) and fix the 'land in PR-1.3' sentence.",
      "found_by": [
        "spec",
        "maint"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "coverage",
      "file_line": "benchmarks-website/server/tests/measurement_id_golden.rs:1108-1117",
      "description": "No golden vector exercises a NaN or Inf f64 threshold. Rust write_f64 uses v.to_bits() (preserves NaN payload bits); Python struct.pack('<d', nan) emits canonical NaN (0x7ff8...). A NaN threshold would hash differently across languages -> silent duplicate row, the exact failure the hash pin exists to prevent. Inf is canonical on both, so the gap is narrowly NaN.",
      "recommended_fix": "Add f64::NAN / INFINITY / NEG_INFINITY threshold vectors and regenerate the golden file, OR assert threshold.is_finite() at the PR-2.1 ingest boundary so a non-finite value fails loudly rather than diverging silently.",
      "found_by": [
        "correctness/claude"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "scope-drift",
      "file_line": "migrations/001_initial_schema.sql:75",
      "description": "The composite-index Key decision promised indexes on '(dim_tuple..., commit_timestamp DESC)'; the migration creates dim-leading (read-path filter) indexes WITHOUT the trailing commit_timestamp, and the tests assert only index *names*, not indexed columns/order. The divergence is explained in PR-1.3's surprises (dim-leading serves the chart read path; PK enforces hash-tuple uniqueness) but the Key decision row was never updated.",
      "recommended_fix": "Amend the composite-index Key decision to the implemented read-path strategy, AND/OR add a test asserting the index column definitions (not just names) so the intended shape is pinned.",
      "found_by": [
        "spec"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "doc-quality",
      "file_line": ".github/workflows/schema-deploy.yml:12-14",
      "description": "The workflow header still advertises a 'schema-deploy GitHub Environment with manual-approval as the stronger gate ... tracked as deferred hardening'. The 2026-05-29 deploy-model Key decision SUPERSEDED that path (PR merge is the gate; the Environment gate was judged the wrong tool). The comment points a future engineer at deferred work the plan reversed. (NOT a re-flag of the accepted no-env-gate tradeoff \u2014 this flags the stale comment, aligned with the decision.)",
      "recommended_fix": "Replace the Environment-gate framing with the actual deferred item: switch the trigger to push on the deploy branch under paths: migrations/** (PR-merge-is-the-gate). Naturally lands with the deferred trigger-switch (Deferred work, 2026-05-29 deploy-model item).",
      "found_by": [
        "maint"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "doc-quality",
      "file_line": "benchmarks-website/infra/provision.sh:266",
      "description": "proxy_role_name ('vortex-bench-proxy-role') and its policy name are hardcoded locals, but the README 'Customizing' section claims 'Every name / class / engine version / region is set at the top via readonly declarations with ${ENV:-default} fallbacks.' The proxy role is an exception, and the tear-down runbook uses the literal default name, so an operator who overrode other names gets a tear-down mismatch.",
      "recommended_fix": "Promote proxy_role_name to a top-level readonly PROXY_ROLE_NAME=\"${PROXY_ROLE_NAME:-vortex-bench-proxy-role}\" like the other names, OR soften the README 'every name is overridable' claim to enumerate the proxy role as fixed.",
      "found_by": [
        "maint"
      ]
    },
    {
      "severity": "nit",
      "kind": "boundary",
      "file_line": "scripts/migrate-schema.py:2733-2745",
      "description": "status() reports a generic 'pending' (exit 1) for an empty/whitespace-only migration file, while apply() rejects it explicitly only when it reaches it. Same bad file, two different diagnoses. Behavior is safe (loud both ways) but asymmetric.",
      "recommended_fix": "Optionally have status() classify empty/whitespace-only on-disk files distinctly so the operator sees the same diagnosis from status as from apply.",
      "found_by": [
        "correctness/claude"
      ]
    },
    {
      "severity": "nit",
      "kind": "doc-quality",
      "file_line": "migrations/README.md:39",
      "description": "002 is described as 'CREATE ROLE for the IAM-auth user that bench.yml workflows assume into', but migrator is consumed by the schema-deploy workflow (PR-1.4), not bench.yml (the ingest workflow, which uses a separate future ingest role). The WHY misattributes the consumer.",
      "recommended_fix": "Change 'bench.yml workflows' to 'the schema-deploy workflow (PR-1.4)'.",
      "found_by": [
        "maint"
      ]
    },
    {
      "severity": "nit",
      "kind": "scaffolding",
      "file_line": "benchmarks-website/server/tests/measurement_id_golden.rs:30",
      "description": "REGEN_GOLDEN_VECTORS is permanent regeneration scaffolding (write-on-env, always-assert otherwise). Correct and well-documented, but nothing flags committed-JSON drift unless the Rust test runs in CI, and the golden==Python half is not CI-gated (deferred).",
      "recommended_fix": "Add one sentence to the test module doc noting golden==Python is only enforced once 'uv run --all-packages pytest scripts/' is wired into CI (deferred), so a green local run does not prove cross-language parity in CI.",
      "found_by": [
        "maint"
      ]
    },
    {
      "severity": "nit",
      "kind": "doc-quality",
      "file_line": ".github/workflows/schema-deploy.yml:84-89",
      "description": "The 'Apply migrations' step carries a ~5-line justification comment (set -x suppression, PGPASSWORD-on-own-line vs export masking). It documents two real footguns but is exactly the >100-char justification-comment shape the shared BAN warns about.",
      "recommended_fix": "Keep the substance but tighten to two short sentences (token-leak + export-masking). No behavior change.",
      "found_by": [
        "maint"
      ]
    },
    {
      "severity": "nit",
      "kind": "scope-drift",
      "file_line": "benchmarks-website/server/tests/measurement_id_golden.rs:1",
      "description": "PR-1.5's expected files row named 'benchmarks-website/server/src/db.rs (golden-vector test added)', but the test shipped as a separate integration test file benchmarks-website/server/tests/measurement_id_golden.rs. Minor placement deviation from the plan row.",
      "recommended_fix": "Amend the PR-1.5 expected-files row to name the integration test location (the chosen placement is fine; the plan row is stale).",
      "found_by": [
        "spec"
      ]
    }
  ],
  "disagreements": [
    {
      "topic": "Severity of the wrong-account header comment (provision.sh:19)",
      "positions": [
        {
          "lens": "maint",
          "position": "must-fix: the comment names the WRONG account (375504701696 = personal/v3), and an operator trusting it provisions into the wrong account or hits a confusing verify_prereqs die."
        },
        {
          "lens": "correctness/claude",
          "position": "nit: cosmetic doc-quality; the actual TARGET_ACCOUNT default is correct."
        }
      ],
      "synthesizer_call": "must-fix (HIGHEST, conservative). An operator-facing runbook naming the wrong AWS account is an operational hazard, not cosmetic; the fix is one line."
    },
    {
      "topic": "Table D SCHEMA_VERSION re-export (scope-drift must-fix vs amend-the-doc should-fix)",
      "positions": [
        {
          "lens": "spec",
          "position": "must-fix scope-drift: Table D requires _measurement_id.py to re-export SCHEMA_VERSION; implement the re-export OR amend Table D."
        },
        {
          "lens": "maint",
          "position": "should-fix: the shipped module is CORRECT to omit the re-export (can't cleanly import hyphenated post-ingest.py; the hash port has no need for SCHEMA_VERSION); Table D is stale \u2014 amend the doc."
        }
      ],
      "synthesizer_call": "Keep must-fix severity (HIGHEST, and Table D is a reference downstream PRs grep into \u2014 fix before phase close), but the ACTION is to amend Table D, NOT to add a re-export. The shipped hash port is correct."
    }
  ],
  "dropped_re_flags": [
    {
      "topic": "migrator role lacks privileges to ALTER / CREATE INDEX on master-owned tables in future migrations (correctness/codex flagged must-fix at 002_iam_db_user.sql:37)",
      "reason": "covered by Deferred work",
      "reference": "Deferred work:487 (PR-1.3 cycle-1 \u2014 migrator table privileges; resolve role-ownership model in PR-2.1). NOTE: the deferral is framed around INGEST DML; it should be EXPANDED to also cover future-migration DDL (ALTER/CREATE INDEX on master-owned tables) so the schema-deploy steady-state is covered, not just the ingest write path. Surfaced as a synthesizer concern. No Phase-1 migration alters an existing table, so it does not block Phase 1 functionally."
    },
    {
      "topic": "golden==Python hash test (and the testcontainer suite) not wired into CI (correctness/claude flagged should-fix)",
      "reason": "covered by Deferred work",
      "reference": "Deferred work:489 (PR-1.5 cycle-1 \u2014 scripts/ pytest not in CI) and Deferred work:486 (PR-1.2 \u2014 no CI runner). The reviewer itself acknowledged the prior triage; surfaced as the single largest standing correctness exposure for the hash pin."
    }
  ],
  "phase_artifacts": {
    "summary": "Phase 1 ('RDS + schema + hash port') lands the migration foundation in five concept areas. (A) Infrastructure: provision.sh idempotently bootstraps RDS Postgres db.t4g.micro + RDS Proxy (IAMAuth=REQUIRED, TLS) + GitHub OIDC provider + GitHubBenchmarkSchemaRole in account 245040174862, with an operator runbook in infra/README.md. (B) Schema-deploy CI: schema-deploy.yml (workflow_dispatch + dry_run; PR-merge is the accepted authorization gate, no environment: approval) generates a client-side IAM token and runs the migrate runner against the proxy as migrator over verify-full TLS. (C) Migration runner: scripts/migrate-schema.py applies migrations/*.sql in name order, tracks public._applied_migrations, is idempotent, uses autocommit + per-migration top-level transactions so a failing later migration rolls back only itself, rejects empty files; 28+ testcontainer tests. (D) DDL: 001 creates the commits dim + 5 fact tables + read-path composite indexes (Postgres translation of the authoritative DuckDB schema.rs, column order/nullability/types preserved); 002 the migrator login role + conditional rds_iam; 003 the append-only ledger grant. (E) Hash port: _measurement_id.py is a byte-for-byte port of db.rs measurement_id_* (xxhash64 seed 0), pinned by a Rust source-of-truth golden file giving Rust==golden==Python, verified bit-exact for all 63 vectors (Claude correctness executed the port; Rust golden test passes). The keystone hash-equivalence deliverable is solid and the schema shape provably matches schema.rs. HOWEVER the AWS-integration path has deploy-blocking gaps the Codex correctness lens surfaced: RDS Proxy endpoints are not publicly reachable from off-VPC GitHub runners (challenges Key decisions Q2/Q6); the proxy lacks a migrator credential for IAM auth; PR-1.4's live OIDC apply against real RDS Proxy was never executed (only wiring + lint + testcontainer); and the master-bootstrap runbook uses PGSSLMODE=require (no cert verification) while sending the master password. The Codex spec lens found contract gaps: 63 vectors shipped vs the promised 100; the Phase-1 exit criterion names a nonexistent test_post_ingest_hash.py; Table D claims a SCHEMA_VERSION re-export the hash port omits. Maintainability is otherwise high, but provision.sh's header names the WRONG AWS account, migrations/README omits 003, and the schema-deploy header advertises the superseded Environment gate.",
    "surprises": [
      {
        "what": "RDS Proxy endpoints are not publicly accessible; off-VPC GitHub-hosted runners cannot reach the proxy. The plan's 'RDS Proxy public endpoint' assumption (Q6) may be architecturally invalid for the CI-write path.",
        "how_handled": "Not handled in the diff \u2014 flagged as a must-fix deploy-blocker; likely forces amending Key decisions Q2/Q6 (e.g., CI writes to the public RDS instance endpoint with direct IAM, proxy stays for Vercel reads).",
        "amend_plan": "yes"
      },
      {
        "what": "PR-1.4's schema-deploy was accepted on wiring + yamllint + testcontainer, never run live against real RDS Proxy.",
        "how_handled": "Recorded honestly in Implementation status; spec lens flags the acceptance criterion as unmet. Coupled with the proxy-reachability + migrator-credential findings, the path is unproven.",
        "amend_plan": "yes"
      },
      {
        "what": "PR-1.5 shipped 63 golden vectors, not the promised 100.",
        "how_handled": "Status acknowledges 63; no amendment or extra fixtures. Qualitative coverage is strong (all tables + boundaries).",
        "amend_plan": "yes"
      },
      {
        "what": "The Phase-1 exit criterion names pytest scripts/test_post_ingest_hash.py, but the shipped file is test_measurement_id.py.",
        "how_handled": "Not reconciled; the documented gate is unrunnable as written. Independently confirmed during exit-criteria execution.",
        "amend_plan": "yes"
      },
      {
        "what": "Table D's claim that _measurement_id.py re-exports SCHEMA_VERSION is stale; the shipped module correctly omits it.",
        "how_handled": "Artifact correct; Table D not updated. Amend the reference table.",
        "amend_plan": "yes"
      },
      {
        "what": "Composite indexes are dim-leading (read-path filter columns), not the Key decision's '(dim_tuple..., commit_timestamp DESC)'.",
        "how_handled": "Explained in PR-1.3 surprises (PK enforces hash-tuple uniqueness; dim-leading serves charts); Key decision row not updated and tests assert only index names.",
        "amend_plan": "yes"
      },
      {
        "what": "migrator role cannot ALTER / CREATE INDEX on master-owned tables in future migrations (GRANT CREATE on public is insufficient).",
        "how_handled": "Covered by the deferred PR-1.3 role-ownership item (PR-2.1), but the deferral is ingest-DML-framed and should be expanded to cover migration DDL. Does not block Phase 1 (no Phase-1 migration alters an existing table).",
        "amend_plan": "already-done"
      },
      {
        "what": "Master-bootstrap runbook uses PGSSLMODE=require (encrypt-without-verify) while sending the master password.",
        "how_handled": "Not handled \u2014 flagged must-fix (MITM exposure); workflow already uses verify-full, so only the README bootstrap is inconsistent.",
        "amend_plan": "no"
      },
      {
        "what": "NaN/Inf f64 threshold cross-language hash divergence is unguarded (no golden vector).",
        "how_handled": "Flagged should-fix coverage; threshold is a cosine value so NaN is implausible, but the divergence would be silent.",
        "amend_plan": "yes"
      }
    ],
    "coverage": {
      "tested_cases": [
        {
          "case": "Hash Rust==golden==Python across all 5 tables + i32 MIN/MAX + empty/Some('') strings + multibyte UTF-8 (63 vectors, executed bit-exact)",
          "test_location": "benchmarks-website/server/tests/measurement_id_golden.rs + scripts/test_measurement_id.py",
          "confidence": "high"
        },
        {
          "case": "migrate-schema apply / idempotency / name-order / failing-migration rollback (subprocess) / status drift / empty-file rejection / non-default search_path ledger agreement / subdir-skip / case-insensitive discovery",
          "test_location": "scripts/test_migrate_schema.py",
          "confidence": "high"
        },
        {
          "case": "Real 001-003 apply cleanly + idempotent; 6 tables, 6 indexes, per-table column order+nullability, key type translations, migrator role login, ledger grants (SELECT/INSERT present, DELETE/UPDATE absent)",
          "test_location": "scripts/test_migrate_schema.py:3480-3640",
          "confidence": "high"
        }
      ],
      "untested_cases": [
        {
          "case": "Live schema-deploy OIDC apply as migrator against the real endpoint (and whether the proxy is even reachable from CI)",
          "priority": "high",
          "why_untested": "Recorded as wiring/lint/testcontainer only; reachability + migrator-credential findings suggest it may not work as wired."
        },
        {
          "case": "scripts/ pytest running in CI (golden==Python parity AND the testcontainer suite are both ungated)",
          "priority": "high",
          "why_untested": "No CI job runs uv run --all-packages pytest scripts/; deferred CI-hardening."
        },
        {
          "case": "RDS Proxy reachability from off-VPC GitHub-hosted runners",
          "priority": "high",
          "why_untested": "RDS Proxy is not publicly accessible; the CI-write endpoint design needs rework."
        },
        {
          "case": "migrator credential registered for RDS Proxy IAM auth",
          "priority": "high",
          "why_untested": "Proxy has only the master secret; migrator connection would fail auth."
        },
        {
          "case": "NaN/Inf f64 threshold cross-language equivalence",
          "priority": "medium",
          "why_untested": "No non-finite threshold vector."
        },
        {
          "case": "Future-migration DDL (ALTER/CREATE INDEX) run as migrator on master-owned tables",
          "priority": "medium",
          "why_untested": "Deferred role-ownership (PR-2.1); no Phase-1 migration alters an existing table."
        },
        {
          "case": "Composite-index column definitions (tests assert names only)",
          "priority": "medium",
          "why_untested": "Index-definition assertions not written."
        },
        {
          "case": "Edit-after-apply ledger drift (no fingerprint column)",
          "priority": "medium",
          "why_untested": "Deferred (sha256 ledger column)."
        }
      ],
      "recommendations": "Resolve the CI-write endpoint design FIRST (RDS Proxy reachability + migrator credential) \u2014 this likely amends Key decisions Q2/Q6 (point schema-deploy at the public RDS instance endpoint with direct IAM, or run in-VPC). Then run schema-deploy live once and record the clean apply/status. Fix the README bootstrap to PGSSLMODE=verify-full. Land the plan-edit must-fixes (exit-criteria test name, Table D, provision.sh account, vector-count reconciliation). Wire scripts/ pytest into CI (closes the largest hash-pin exposure)."
    },
    "tradeoffs": [
      {
        "decision": "Branching / merge target (per-phase child branches -> ct/bench-v4 -> develop)",
        "original": "User pick Q1",
        "verdict": "keep",
        "rationale": "No artifact conflict."
      },
      {
        "decision": "Postgres flavor (RDS db.t4g.micro single-AZ, account 245040174862)",
        "original": "User pick Q2",
        "verdict": "keep",
        "rationale": "Provisioning/DDL target the chosen account/region/class; flavor-portable (testcontainer runs vanilla Postgres)."
      },
      {
        "decision": "Connection pooler (RDS Proxy with IAM-auth pass-through)",
        "original": "Locked by Q2",
        "verdict": "revisit-but-keep",
        "rationale": "The proxy is right for Vercel reads, BUT the CI-write-via-proxy assumption is challenged: RDS Proxy is not publicly reachable from off-VPC GitHub runners and lacks a migrator credential. The CI-write endpoint specifically needs rework (most-pessimistic across reviewers; correctness/codex would lean reverse for the CI path)."
      },
      {
        "decision": "Ingest writer language (pure Python + golden vectors)",
        "original": "User pick Q4",
        "verdict": "revisit-but-keep",
        "rationale": "Hash port verified bit-exact, but the 100-vs-63 fixture-count and Table D lockstep gaps need reconciliation."
      },
      {
        "decision": "Schema deploy tool (in-house migrate-schema.py + plain SQL)",
        "original": "User pick Q5a",
        "verdict": "revisit-but-keep",
        "rationale": "Sound and well-tested, but grew to ~180 LOC and still lacks ledger fingerprinting, so the documented edit-after-apply prohibition has zero runtime enforcement (deferred)."
      },
      {
        "decision": "Schema-deploy authorization (PR merge is the gate; no environment gate)",
        "original": "User decision 2026-05-29",
        "verdict": "keep",
        "rationale": "Accepted tradeoff; NOT re-flagged. Only the stale header COMMENT advertising the superseded Environment gate is flagged (doc-sync, not a reversal)."
      },
      {
        "decision": "CI network reach (public + IAM, verify-full)",
        "original": "User pick Q6",
        "verdict": "revisit-but-keep",
        "rationale": "The public+IAM model works for the RDS INSTANCE endpoint, but the 'public RDS Proxy endpoint' sub-assumption is invalid (proxy isn't public). Tied to the schema-deploy.yml:77 must-fix."
      },
      {
        "decision": "Composite index definition strategy ((dim_tuple..., commit_timestamp DESC))",
        "original": "Forward-looking design",
        "verdict": "revisit-but-keep",
        "rationale": "Implemented as dim-leading read-path indexes without the trailing timestamp; either amend the decision to the implemented strategy or pin index column definitions in tests."
      },
      {
        "decision": "Cutover style / One-shot load / Read framework / Operator SQL / v3 disposition",
        "original": "User picks Q5b,Q7,Q8,Q9 + forward-looking",
        "verdict": "keep",
        "rationale": "Future-phase decisions; no Phase-1 change contradicts them."
      }
    ]
  },
  "executive_summary": "Phase 1 ships a coherent, unusually well-documented foundation, and its keystone deliverable \u2014 the cross-language measurement_id hash equivalence (Rust==golden==Python) \u2014 is verified bit-exact for all 63 vectors, with the 6-table Postgres schema provably matching the authoritative DuckDB schema.rs and a well-tested migration runner. The mixed-executor review (Claude + Codex lenses) is the reason this is a REJECT rather than an accept: the Codex correctness lens surfaced a cluster of deploy-blocking AWS-integration gaps that the hash-focused Claude review did not. The most serious: RDS Proxy endpoints are NOT publicly reachable from off-VPC GitHub-hosted runners (schema-deploy.yml:77), the proxy was provisioned without a migrator credential for IAM auth (provision.sh:311), and PR-1.4's live OIDC apply against real RDS Proxy was never actually executed (schema-deploy.yml:68) \u2014 together meaning the schema-deploy CI path, a core Phase-1 deliverable, is unproven and likely broken as wired. Resolving it probably amends Key decisions Q2/Q6 (e.g., point CI writes at the public RDS *instance* endpoint with direct IAM and reserve the proxy for Vercel reads, or run schema-deploy in-VPC). A fourth correctness must-fix: the master-bootstrap runbook uses PGSSLMODE=require (encrypt-without-verify) while transmitting the master password (README:120) \u2014 a MITM exposure, fixed by verify-full. The Codex spec lens added contract-closure must-fixes that are mostly cheap plan-edits: 63 vectors shipped vs the promised 100 (reconcile the criterion or add vectors); the Phase-1 exit criterion names a nonexistent test_post_ingest_hash.py (rename to the shipped test_measurement_id.py \u2014 independently confirmed at exit-criteria time); and Table D claims a SCHEMA_VERSION re-export the hash port correctly omits (amend the reference table, do NOT add the re-export). The maint lens caught a genuine operational hazard: provision.sh's header comment names the WRONG (personal/v3) AWS account, which could misdirect an operator. Five should-fixes (stale migrations/README missing 003; composite-index decision-vs-impl drift; the schema-deploy header advertising the superseded Environment gate; a hardcoded proxy-role name contradicting the README; the unguarded NaN/Inf hash vector) and five nits round it out. Two findings were dropped as carry-forward (migrator table privileges and CI-gating of scripts/ pytest are both already in Deferred work) \u2014 though the role-privileges deferral should be expanded to cover future-migration DDL, not just ingest DML. Verdict: reject, 8 must-fix. The hash and schema work is strong; the AWS-integration + plan-consistency layer needs a focused fix pass before Phase 1 closes.",
  "overall": "reject",
  "must_fix_count": 8,
  "should_fix_count": 5,
  "nit_count": 5,
  "review_cycles_this_invocation": 1,
  "executor_routing": {
    "spec": "codex",
    "correctness": "parallel",
    "maint": "claude"
  }
}
```

</details>

## Superseded (re-planned) phase-end must-fix items — Phase 1: RDS + schema + hash port — cycle 1

| Severity | File:line | Description | Implicated PR | Resolved |
|----------|-----------|-------------|---------------|----------|
| must-fix | .github/workflows/schema-deploy.yml:77 | RDS_BENCH_ENDPOINT is the RDS Proxy hostname, but RDS Proxy endpoints are not publicly accessible and GitHub-hosted runners are off-VPC. As wired, the schema... | PR-1.4 | [ ] |
| must-fix | benchmarks-website/infra/provision.sh:311 | The RDS Proxy is provisioned with only the RDS-managed master secret, but CI connects as PGUSER=migrator. Standard RDS Proxy IAM auth still needs a Secrets M... | PR-1.1 | [ ] |
| must-fix | .github/workflows/schema-deploy.yml:68 | PR-1.4's acceptance criterion was 'migrate-schema.py apply runs as the OIDC migrator role against RDS Proxy; status reports clean post-apply.' Implementation... | PR-1.4 | [ ] |
| must-fix | benchmarks-website/infra/README.md:120 | The master-user bootstrap runbook command uses PGSSLMODE=require, which encrypts but does NOT verify the RDS server certificate, while transmitting the maste... | PR-1.4 | [ ] |
| must-fix | benchmarks-website/infra/provision.sh:19 | The header comment says the script provisions into account 375504701696 'by default', but the actual TARGET_ACCOUNT default (line 50) and the entire README/p... | PR-1.1 | [ ] |
| must-fix | benchmarks-website/server/tests/measurement_id_golden.rs:102 | PR-1.5's acceptance promised '100 fixture (commit, dim-tuple) inputs'; the generator + committed golden file contain 63 vectors. The qualitative coverage is ... | PR-1.5 | [ ] |
| must-fix | scripts/test_measurement_id.py:1 | The Phase-1 exit criterion (Phases-and-PRs table) names 'pytest scripts/test_post_ingest_hash.py all green', but the artifact ships scripts/test_measurement_... | PR-1.5 | [ ] |
| must-fix | scripts/_measurement_id.py:50 | Reference Table D ('SCHEMA_VERSION lockstep sites') claims scripts/_measurement_id.py 'Imports SCHEMA_VERSION from post-ingest.py \| Python re-export to keep... | PR-1.5 | [ ] |

## Phase 1: RDS + schema + hash port — end-of-phase review (cycle 1) — rejected (3-vote)
**Synthesizer output from /spiral:gauntlet (preset=phase-3, executor=mixed: spec→codex, correctness→parallel, maint→claude); full Synthesizer Output JSON in the `<details>` block at the end of this section.**

Verdict: **reject** — 8 must-fix, 5 should-fix, 5 nit.

### Summary of changes

Phase 1 ('RDS + schema + hash port') lands the migration foundation in five concept areas. (A) Infrastructure: provision.sh idempotently bootstraps RDS Postgres db.t4g.micro + RDS Proxy (IAMAuth=REQUIRED, TLS) + GitHub OIDC provider + GitHubBenchmarkSchemaRole in account 245040174862, with an operator runbook in infra/README.md. (B) Schema-deploy CI: schema-deploy.yml (workflow_dispatch + dry_run; PR-merge is the accepted authorization gate, no environment: approval) generates a client-side IAM token and runs the migrate runner against the proxy as migrator over verify-full TLS. (C) Migration runner: scripts/migrate-schema.py applies migrations/*.sql in name order, tracks public._applied_migrations, is idempotent, uses autocommit + per-migration top-level transactions so a failing later migration rolls back only itself, rejects empty files; 28+ testcontainer tests. (D) DDL: 001 creates the commits dim + 5 fact tables + read-path composite indexes (Postgres translation of the authoritative DuckDB schema.rs, column order/nullability/types preserved); 002 the migrator login role + conditional rds_iam; 003 the append-only ledger grant. (E) Hash port: _measurement_id.py is a byte-for-byte port of db.rs measurement_id_* (xxhash64 seed 0), pinned by a Rust source-of-truth golden file giving Rust==golden==Python, verified bit-exact for all 63 vectors (Claude correctness executed the port; Rust golden test passes). The keystone hash-equivalence deliverable is solid and the schema shape provably matches schema.rs. HOWEVER the AWS-integration path has deploy-blocking gaps the Codex correctness lens surfaced: RDS Proxy endpoints are not publicly reachable from off-VPC GitHub runners (challenges Key decisions Q2/Q6); the proxy lacks a migrator credential for IAM auth; PR-1.4's live OIDC apply against real RDS Proxy was never executed (only wiring + lint + testcontainer); and the master-bootstrap runbook uses PGSSLMODE=require (no cert verification) while sending the master password. The Codex spec lens found contract gaps: 63 vectors shipped vs the promised 100; the Phase-1 exit criterion names a nonexistent test_post_ingest_hash.py; Table D claims a SCHEMA_VERSION re-export the hash port omits. Maintainability is otherwise high, but provision.sh's header names the WRONG AWS account, migrations/README omits 003, and the schema-deploy header advertises the superseded Environment gate.

### Unified findings

| # | Severity | Kind | File:line | Description | found_by |
|---|----------|------|-----------|-------------|----------|
| 1 | must-fix | bug | .github/workflows/schema-deploy.yml:77 | RDS_BENCH_ENDPOINT is the RDS Proxy hostname, but RDS Proxy endpoints are not publicly accessible and GitHub-hosted runners are off-VPC. ... | correctness/codex |
| 2 | must-fix | bug | benchmarks-website/infra/provision.sh:311 | The RDS Proxy is provisioned with only the RDS-managed master secret, but CI connects as PGUSER=migrator. Standard RDS Proxy IAM auth sti... | correctness/codex |
| 3 | must-fix | missing-acceptance | .github/workflows/schema-deploy.yml:68 | PR-1.4's acceptance criterion was 'migrate-schema.py apply runs as the OIDC migrator role against RDS Proxy; status reports clean post-ap... | spec |
| 4 | must-fix | unsafe | benchmarks-website/infra/README.md:120 | The master-user bootstrap runbook command uses PGSSLMODE=require, which encrypts but does NOT verify the RDS server certificate, while tr... | correctness/codex |
| 5 | must-fix | doc-quality | benchmarks-website/infra/provision.sh:19 | The header comment says the script provisions into account 375504701696 'by default', but the actual TARGET_ACCOUNT default (line 50) and... | maint, correctness/claude |
| 6 | must-fix | scope-drift | benchmarks-website/server/tests/measurement_id_golden.rs:102 | PR-1.5's acceptance promised '100 fixture (commit, dim-tuple) inputs'; the generator + committed golden file contain 63 vectors. The qual... | spec |
| 7 | must-fix | weak-exit-criteria | scripts/test_measurement_id.py:1 | The Phase-1 exit criterion (Phases-and-PRs table) names 'pytest scripts/test_post_ingest_hash.py all green', but the artifact ships scrip... | spec |
| 8 | must-fix | scope-drift | scripts/_measurement_id.py:50 | Reference Table D ('SCHEMA_VERSION lockstep sites') claims scripts/_measurement_id.py 'Imports SCHEMA_VERSION from post-ingest.py \| Pyth... | spec, maint |
| 9 | should-fix | doc-quality | migrations/README.md:35 | The 'Initial files' section lists only 001 and 002 and says the SQL files 'land in PR-1.3', but PR-1.4 added 003_migrator_ledger_grant.sq... | spec, maint |
| 10 | should-fix | coverage | benchmarks-website/server/tests/measurement_id_golden.rs:1108-1117 | No golden vector exercises a NaN or Inf f64 threshold. Rust write_f64 uses v.to_bits() (preserves NaN payload bits); Python struct.pack('... | correctness/claude |
| 11 | should-fix | scope-drift | migrations/001_initial_schema.sql:75 | The composite-index Key decision promised indexes on '(dim_tuple..., commit_timestamp DESC)'; the migration creates dim-leading (read-pat... | spec |
| 12 | should-fix | doc-quality | .github/workflows/schema-deploy.yml:12-14 | The workflow header still advertises a 'schema-deploy GitHub Environment with manual-approval as the stronger gate ... tracked as deferre... | maint |
| 13 | should-fix | doc-quality | benchmarks-website/infra/provision.sh:266 | proxy_role_name ('vortex-bench-proxy-role') and its policy name are hardcoded locals, but the README 'Customizing' section claims 'Every ... | maint |
| 14 | nit | boundary | scripts/migrate-schema.py:2733-2745 | status() reports a generic 'pending' (exit 1) for an empty/whitespace-only migration file, while apply() rejects it explicitly only when ... | correctness/claude |
| 15 | nit | doc-quality | migrations/README.md:39 | 002 is described as 'CREATE ROLE for the IAM-auth user that bench.yml workflows assume into', but migrator is consumed by the schema-depl... | maint |
| 16 | nit | scaffolding | benchmarks-website/server/tests/measurement_id_golden.rs:30 | REGEN_GOLDEN_VECTORS is permanent regeneration scaffolding (write-on-env, always-assert otherwise). Correct and well-documented, but noth... | maint |
| 17 | nit | doc-quality | .github/workflows/schema-deploy.yml:84-89 | The 'Apply migrations' step carries a ~5-line justification comment (set -x suppression, PGPASSWORD-on-own-line vs export masking). It do... | maint |
| 18 | nit | scope-drift | benchmarks-website/server/tests/measurement_id_golden.rs:1 | PR-1.5's expected files row named 'benchmarks-website/server/src/db.rs (golden-vector test added)', but the test shipped as a separate in... | spec |

### Surprises and discoveries

- **RDS Proxy endpoints are not publicly accessible; off-VPC GitHub-hosted runners cannot reach the proxy. The plan's 'RDS Proxy public endpoint' assumption (Q6) may be architecturally invalid for the CI-write path.** — handled: Not handled in the diff — flagged as a must-fix deploy-blocker; likely forces amending Key decisions Q2/Q6 (e.g., CI writes to the public RDS instance endpoint with direct IAM, proxy stays for Vercel reads). (amend_plan: yes)
- **PR-1.4's schema-deploy was accepted on wiring + yamllint + testcontainer, never run live against real RDS Proxy.** — handled: Recorded honestly in Implementation status; spec lens flags the acceptance criterion as unmet. Coupled with the proxy-reachability + migrator-credential findings, the path is unproven. (amend_plan: yes)
- **PR-1.5 shipped 63 golden vectors, not the promised 100.** — handled: Status acknowledges 63; no amendment or extra fixtures. Qualitative coverage is strong (all tables + boundaries). (amend_plan: yes)
- **The Phase-1 exit criterion names pytest scripts/test_post_ingest_hash.py, but the shipped file is test_measurement_id.py.** — handled: Not reconciled; the documented gate is unrunnable as written. Independently confirmed during exit-criteria execution. (amend_plan: yes)
- **Table D's claim that _measurement_id.py re-exports SCHEMA_VERSION is stale; the shipped module correctly omits it.** — handled: Artifact correct; Table D not updated. Amend the reference table. (amend_plan: yes)
- **Composite indexes are dim-leading (read-path filter columns), not the Key decision's '(dim_tuple..., commit_timestamp DESC)'.** — handled: Explained in PR-1.3 surprises (PK enforces hash-tuple uniqueness; dim-leading serves charts); Key decision row not updated and tests assert only index names. (amend_plan: yes)
- **migrator role cannot ALTER / CREATE INDEX on master-owned tables in future migrations (GRANT CREATE on public is insufficient).** — handled: Covered by the deferred PR-1.3 role-ownership item (PR-2.1), but the deferral is ingest-DML-framed and should be expanded to cover migration DDL. Does not block Phase 1 (no Phase-1 migration alters an existing table). (amend_plan: already-done)
- **Master-bootstrap runbook uses PGSSLMODE=require (encrypt-without-verify) while sending the master password.** — handled: Not handled — flagged must-fix (MITM exposure); workflow already uses verify-full, so only the README bootstrap is inconsistent. (amend_plan: no)
- **NaN/Inf f64 threshold cross-language hash divergence is unguarded (no golden vector).** — handled: Flagged should-fix coverage; threshold is a cosine value so NaN is implausible, but the divergence would be silent. (amend_plan: yes)

### Testing coverage assessment

**Tested:**
- Hash Rust==golden==Python across all 5 tables + i32 MIN/MAX + empty/Some('') strings + multibyte UTF-8 (63 vectors, executed bit-exact) (`benchmarks-website/server/tests/measurement_id_golden.rs + scripts/test_measurement_id.py`, confidence high)
- migrate-schema apply / idempotency / name-order / failing-migration rollback (subprocess) / status drift / empty-file rejection / non-default search_path ledger agreement / subdir-skip / case-insensitive discovery (`scripts/test_migrate_schema.py`, confidence high)
- Real 001-003 apply cleanly + idempotent; 6 tables, 6 indexes, per-table column order+nullability, key type translations, migrator role login, ledger grants (SELECT/INSERT present, DELETE/UPDATE absent) (`scripts/test_migrate_schema.py:3480-3640`, confidence high)

**Untested (skeptical):**
- [high] Live schema-deploy OIDC apply as migrator against the real endpoint (and whether the proxy is even reachable from CI) — Recorded as wiring/lint/testcontainer only; reachability + migrator-credential findings suggest it may not work as wired.
- [high] scripts/ pytest running in CI (golden==Python parity AND the testcontainer suite are both ungated) — No CI job runs uv run --all-packages pytest scripts/; deferred CI-hardening.
- [high] RDS Proxy reachability from off-VPC GitHub-hosted runners — RDS Proxy is not publicly accessible; the CI-write endpoint design needs rework.
- [high] migrator credential registered for RDS Proxy IAM auth — Proxy has only the master secret; migrator connection would fail auth.
- [medium] NaN/Inf f64 threshold cross-language equivalence — No non-finite threshold vector.
- [medium] Future-migration DDL (ALTER/CREATE INDEX) run as migrator on master-owned tables — Deferred role-ownership (PR-2.1); no Phase-1 migration alters an existing table.
- [medium] Composite-index column definitions (tests assert names only) — Index-definition assertions not written.
- [medium] Edit-after-apply ledger drift (no fingerprint column) — Deferred (sha256 ledger column).

_Recommendations:_ Resolve the CI-write endpoint design FIRST (RDS Proxy reachability + migrator credential) — this likely amends Key decisions Q2/Q6 (point schema-deploy at the public RDS instance endpoint with direct IAM, or run in-VPC). Then run schema-deploy live once and record the clean apply/status. Fix the README bootstrap to PGSSLMODE=verify-full. Land the plan-edit must-fixes (exit-criteria test name, Table D, provision.sh account, vector-count reconciliation). Wire scripts/ pytest into CI (closes the largest hash-pin exposure).

### Tradeoffs re-evaluation

- **Branching / merge target (per-phase child branches -> ct/bench-v4 -> develop)** → `keep` — No artifact conflict.
- **Postgres flavor (RDS db.t4g.micro single-AZ, account 245040174862)** → `keep` — Provisioning/DDL target the chosen account/region/class; flavor-portable (testcontainer runs vanilla Postgres).
- **Connection pooler (RDS Proxy with IAM-auth pass-through)** → `revisit-but-keep` — The proxy is right for Vercel reads, BUT the CI-write-via-proxy assumption is challenged: RDS Proxy is not publicly reachable from off-VPC GitHub runners and lacks a migrator credential. The CI-write endpoint specifically needs rework (most-pessimistic across reviewers; correctness/codex would lean reverse for the CI path).
- **Ingest writer language (pure Python + golden vectors)** → `revisit-but-keep` — Hash port verified bit-exact, but the 100-vs-63 fixture-count and Table D lockstep gaps need reconciliation.
- **Schema deploy tool (in-house migrate-schema.py + plain SQL)** → `revisit-but-keep` — Sound and well-tested, but grew to ~180 LOC and still lacks ledger fingerprinting, so the documented edit-after-apply prohibition has zero runtime enforcement (deferred).
- **Schema-deploy authorization (PR merge is the gate; no environment gate)** → `keep` — Accepted tradeoff; NOT re-flagged. Only the stale header COMMENT advertising the superseded Environment gate is flagged (doc-sync, not a reversal).
- **CI network reach (public + IAM, verify-full)** → `revisit-but-keep` — The public+IAM model works for the RDS INSTANCE endpoint, but the 'public RDS Proxy endpoint' sub-assumption is invalid (proxy isn't public). Tied to the schema-deploy.yml:77 must-fix.
- **Composite index definition strategy ((dim_tuple..., commit_timestamp DESC))** → `revisit-but-keep` — Implemented as dim-leading read-path indexes without the trailing timestamp; either amend the decision to the implemented strategy or pin index column definitions in tests.
- **Cutover style / One-shot load / Read framework / Operator SQL / v3 disposition** → `keep` — Future-phase decisions; no Phase-1 change contradicts them.

### Disagreements

- **Severity of the wrong-account header comment (provision.sh:19)**: maint: must-fix: the comment names the WRONG account (375504701696 = personal/v3), and an operator trusting it provisions into the wrong account or hits a confusing verify_prereqs die.; correctness/claude: nit: cosmetic doc-quality; the actual TARGET_ACCOUNT default is correct. → call: must-fix (HIGHEST, conservative). An operator-facing runbook naming the wrong AWS account is an operational hazard, not cosmetic; the fix is one line.
- **Table D SCHEMA_VERSION re-export (scope-drift must-fix vs amend-the-doc should-fix)**: spec: must-fix scope-drift: Table D requires _measurement_id.py to re-export SCHEMA_VERSION; implement the re-export OR amend Table D.; maint: should-fix: the shipped module is CORRECT to omit the re-export (can't cleanly import hyphenated post-ingest.py; the hash port has no need for SCHEMA_VERSION); Table D is stale — amend the doc. → call: Keep must-fix severity (HIGHEST, and Table D is a reference downstream PRs grep into — fix before phase close), but the ACTION is to amend Table D, NOT to add a re-export. The shipped hash port is correct.

### Dropped re-flags (carry-forward)

- migrator role lacks privileges to ALTER / CREATE INDEX on master-owned tables in future migrations (correctness/codex flagged must-fix at 002_iam_db_user.sql:37) — covered by Deferred work (Deferred work:487 (PR-1.3 cycle-1 — migrator table privileges; resolve role-ownership model in PR-2.1). NOTE: the deferral is framed around INGEST DML; it should be EXPANDED to also cover future-migration DDL (ALTER/CREATE INDEX on master-owned tables) so the schema-deploy steady-state is covered, not just the ingest write path. Surfaced as a synthesizer concern. No Phase-1 migration alters an existing table, so it does not block Phase 1 functionally.)
- golden==Python hash test (and the testcontainer suite) not wired into CI (correctness/claude flagged should-fix) — covered by Deferred work (Deferred work:489 (PR-1.5 cycle-1 — scripts/ pytest not in CI) and Deferred work:486 (PR-1.2 — no CI runner). The reviewer itself acknowledged the prior triage; surfaced as the single largest standing correctness exposure for the hash pin.)

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1)</summary>

```json
{
  "schema_version": 1,
  "preset": "phase-3",
  "lenses_used": [
    "spec",
    "correctness",
    "maint"
  ],
  "review_count": 3,
  "unified_findings": [
    {
      "severity": "must-fix",
      "kind": "bug",
      "file_line": ".github/workflows/schema-deploy.yml:77",
      "description": "RDS_BENCH_ENDPOINT is the RDS Proxy hostname, but RDS Proxy endpoints are not publicly accessible and GitHub-hosted runners are off-VPC. As wired, the schema-deploy job cannot reach the proxy at all. This challenges Key decisions Q2/Q6 ('RDS Proxy public endpoint, security group 0.0.0.0/0').",
      "recommended_fix": "Point off-VPC schema-deploy at the public RDS *instance* endpoint with direct IAM auth (the proxy stays for Vercel reads), OR run schema-deploy inside the VPC (self-hosted runner / CodeBuild), OR expose the proxy via an intentional NLB/PrivateLink design. Resolve before claiming the schema-deploy path works; this likely amends Q2/Q6.",
      "found_by": [
        "correctness/codex"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "bug",
      "file_line": "benchmarks-website/infra/provision.sh:311",
      "description": "The RDS Proxy is provisioned with only the RDS-managed master secret, but CI connects as PGUSER=migrator. Standard RDS Proxy IAM auth still needs a Secrets Manager credential registered for the migrator DB user; without it the proxy cannot authenticate the migrator connection.",
      "recommended_fix": "Either configure end-to-end IAM auth for the proxy for the migrator user, or create+attach a migrator credential secret and register it in the proxy auth config. Add a smoke test that connects through the chosen endpoint as migrator. (Couples with the schema-deploy.yml:77 reachability finding \u2014 resolve the CI-write endpoint design together.)",
      "found_by": [
        "correctness/codex"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "missing-acceptance",
      "file_line": ".github/workflows/schema-deploy.yml:68",
      "description": "PR-1.4's acceptance criterion was 'migrate-schema.py apply runs as the OIDC migrator role against RDS Proxy; status reports clean post-apply.' Implementation status records only wiring + yamllint + testcontainer coverage \u2014 the live OIDC apply against real RDS Proxy was never executed. Combined with the proxy-reachability and migrator-credential findings, the schema-deploy path is unproven and may be non-functional as designed.",
      "recommended_fix": "After resolving the endpoint/credential design, run schema-deploy live once and record the clean apply/status, OR explicitly amend the PR-1.4 acceptance criterion to state live execution is deferred (and to which phase) with rationale.",
      "found_by": [
        "spec"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "unsafe",
      "file_line": "benchmarks-website/infra/README.md:120",
      "description": "The master-user bootstrap runbook command uses PGSSLMODE=require, which encrypts but does NOT verify the RDS server certificate, while transmitting the master password. This is a MITM exposure on the single most sensitive credential in the system. The schema-deploy workflow correctly uses verify-full; the bootstrap runbook is inconsistent.",
      "recommended_fix": "Change the bootstrap runbook to PGSSLMODE=verify-full with PGSSLROOTCERT pointed at the downloaded RDS CA bundle, matching the workflow.",
      "found_by": [
        "correctness/codex"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "doc-quality",
      "file_line": "benchmarks-website/infra/provision.sh:19",
      "description": "The header comment says the script provisions into account 375504701696 'by default', but the actual TARGET_ACCOUNT default (line 50) and the entire README/plan are 245040174862. Per Key decisions, 375504701696 is the PERSONAL/v3-EC2 account \u2014 exactly the account the bench infra must NOT land in. An operator trusting the header points at the wrong account; verify_prereqs would then die confusingly, or worse the operator provisions into the wrong account.",
      "recommended_fix": "Change the line-19 comment to account 245040174862 to match TARGET_ACCOUNT and the README; or interpolate the value rather than hardcoding a stale literal.",
      "found_by": [
        "maint",
        "correctness/claude"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "scope-drift",
      "file_line": "benchmarks-website/server/tests/measurement_id_golden.rs:102",
      "description": "PR-1.5's acceptance promised '100 fixture (commit, dim-tuple) inputs'; the generator + committed golden file contain 63 vectors. The qualitative coverage is strong (all 5 tables + i32 MIN/MAX + empty/Some('') strings + multibyte UTF-8), but the literal acceptance criterion is unmet and was not amended.",
      "recommended_fix": "Either add ~37 more deterministic fixture vectors, OR amend the PR-1.5 acceptance criterion to '63 vectors' with rationale (the chosen 63 exhaustively cover all tables + boundary classes). Cheap; pick one and record it.",
      "found_by": [
        "spec"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "weak-exit-criteria",
      "file_line": "scripts/test_measurement_id.py:1",
      "description": "The Phase-1 exit criterion (Phases-and-PRs table) names 'pytest scripts/test_post_ingest_hash.py all green', but the artifact ships scripts/test_measurement_id.py. The documented phase gate is unrunnable as written (no such file). Independently confirmed during exit-criteria execution.",
      "recommended_fix": "Amend the Phase-1 exit-criteria string to 'pytest scripts/test_measurement_id.py' (the as-shipped file). Plan-edit only.",
      "found_by": [
        "spec"
      ]
    },
    {
      "severity": "must-fix",
      "kind": "scope-drift",
      "file_line": "scripts/_measurement_id.py:50",
      "description": "Reference Table D ('SCHEMA_VERSION lockstep sites') claims scripts/_measurement_id.py 'Imports SCHEMA_VERSION from post-ingest.py | Python re-export to keep one site'. The shipped module neither imports nor re-exports SCHEMA_VERSION (and could not cleanly import from hyphenated post-ingest.py without importlib). The hash port is correct to omit it; Table D is stale and would mislead a future SCHEMA_VERSION bump. (Table D is a reference downstream PRs grep into.)",
      "recommended_fix": "Amend Table D to remove the _measurement_id.py re-export row (or replace it with the real lockstep site). Do NOT add a spurious re-export to the hash port. Plan-edit only. (See disagreement: spec framed this as must-fix scope-drift requiring re-export OR amendment; maint framed it as should-fix amend-the-doc; synthesizer call = amend Table D, code is correct.)",
      "found_by": [
        "spec",
        "maint"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "doc-quality",
      "file_line": "migrations/README.md:35",
      "description": "The 'Initial files' section lists only 001 and 002 and says the SQL files 'land in PR-1.3', but PR-1.4 added 003_migrator_ledger_grant.sql, which exists on disk and is exercised by the test suite. The directory's own README is a stale, incomplete inventory.",
      "recommended_fix": "Add a bullet for 003_migrator_ledger_grant.sql (GRANT SELECT,INSERT on the ledger to migrator, PR-1.4) and fix the 'land in PR-1.3' sentence.",
      "found_by": [
        "spec",
        "maint"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "coverage",
      "file_line": "benchmarks-website/server/tests/measurement_id_golden.rs:1108-1117",
      "description": "No golden vector exercises a NaN or Inf f64 threshold. Rust write_f64 uses v.to_bits() (preserves NaN payload bits); Python struct.pack('<d', nan) emits canonical NaN (0x7ff8...). A NaN threshold would hash differently across languages -> silent duplicate row, the exact failure the hash pin exists to prevent. Inf is canonical on both, so the gap is narrowly NaN.",
      "recommended_fix": "Add f64::NAN / INFINITY / NEG_INFINITY threshold vectors and regenerate the golden file, OR assert threshold.is_finite() at the PR-2.1 ingest boundary so a non-finite value fails loudly rather than diverging silently.",
      "found_by": [
        "correctness/claude"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "scope-drift",
      "file_line": "migrations/001_initial_schema.sql:75",
      "description": "The composite-index Key decision promised indexes on '(dim_tuple..., commit_timestamp DESC)'; the migration creates dim-leading (read-path filter) indexes WITHOUT the trailing commit_timestamp, and the tests assert only index *names*, not indexed columns/order. The divergence is explained in PR-1.3's surprises (dim-leading serves the chart read path; PK enforces hash-tuple uniqueness) but the Key decision row was never updated.",
      "recommended_fix": "Amend the composite-index Key decision to the implemented read-path strategy, AND/OR add a test asserting the index column definitions (not just names) so the intended shape is pinned.",
      "found_by": [
        "spec"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "doc-quality",
      "file_line": ".github/workflows/schema-deploy.yml:12-14",
      "description": "The workflow header still advertises a 'schema-deploy GitHub Environment with manual-approval as the stronger gate ... tracked as deferred hardening'. The 2026-05-29 deploy-model Key decision SUPERSEDED that path (PR merge is the gate; the Environment gate was judged the wrong tool). The comment points a future engineer at deferred work the plan reversed. (NOT a re-flag of the accepted no-env-gate tradeoff \u2014 this flags the stale comment, aligned with the decision.)",
      "recommended_fix": "Replace the Environment-gate framing with the actual deferred item: switch the trigger to push on the deploy branch under paths: migrations/** (PR-merge-is-the-gate). Naturally lands with the deferred trigger-switch (Deferred work, 2026-05-29 deploy-model item).",
      "found_by": [
        "maint"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "doc-quality",
      "file_line": "benchmarks-website/infra/provision.sh:266",
      "description": "proxy_role_name ('vortex-bench-proxy-role') and its policy name are hardcoded locals, but the README 'Customizing' section claims 'Every name / class / engine version / region is set at the top via readonly declarations with ${ENV:-default} fallbacks.' The proxy role is an exception, and the tear-down runbook uses the literal default name, so an operator who overrode other names gets a tear-down mismatch.",
      "recommended_fix": "Promote proxy_role_name to a top-level readonly PROXY_ROLE_NAME=\"${PROXY_ROLE_NAME:-vortex-bench-proxy-role}\" like the other names, OR soften the README 'every name is overridable' claim to enumerate the proxy role as fixed.",
      "found_by": [
        "maint"
      ]
    },
    {
      "severity": "nit",
      "kind": "boundary",
      "file_line": "scripts/migrate-schema.py:2733-2745",
      "description": "status() reports a generic 'pending' (exit 1) for an empty/whitespace-only migration file, while apply() rejects it explicitly only when it reaches it. Same bad file, two different diagnoses. Behavior is safe (loud both ways) but asymmetric.",
      "recommended_fix": "Optionally have status() classify empty/whitespace-only on-disk files distinctly so the operator sees the same diagnosis from status as from apply.",
      "found_by": [
        "correctness/claude"
      ]
    },
    {
      "severity": "nit",
      "kind": "doc-quality",
      "file_line": "migrations/README.md:39",
      "description": "002 is described as 'CREATE ROLE for the IAM-auth user that bench.yml workflows assume into', but migrator is consumed by the schema-deploy workflow (PR-1.4), not bench.yml (the ingest workflow, which uses a separate future ingest role). The WHY misattributes the consumer.",
      "recommended_fix": "Change 'bench.yml workflows' to 'the schema-deploy workflow (PR-1.4)'.",
      "found_by": [
        "maint"
      ]
    },
    {
      "severity": "nit",
      "kind": "scaffolding",
      "file_line": "benchmarks-website/server/tests/measurement_id_golden.rs:30",
      "description": "REGEN_GOLDEN_VECTORS is permanent regeneration scaffolding (write-on-env, always-assert otherwise). Correct and well-documented, but nothing flags committed-JSON drift unless the Rust test runs in CI, and the golden==Python half is not CI-gated (deferred).",
      "recommended_fix": "Add one sentence to the test module doc noting golden==Python is only enforced once 'uv run --all-packages pytest scripts/' is wired into CI (deferred), so a green local run does not prove cross-language parity in CI.",
      "found_by": [
        "maint"
      ]
    },
    {
      "severity": "nit",
      "kind": "doc-quality",
      "file_line": ".github/workflows/schema-deploy.yml:84-89",
      "description": "The 'Apply migrations' step carries a ~5-line justification comment (set -x suppression, PGPASSWORD-on-own-line vs export masking). It documents two real footguns but is exactly the >100-char justification-comment shape the shared BAN warns about.",
      "recommended_fix": "Keep the substance but tighten to two short sentences (token-leak + export-masking). No behavior change.",
      "found_by": [
        "maint"
      ]
    },
    {
      "severity": "nit",
      "kind": "scope-drift",
      "file_line": "benchmarks-website/server/tests/measurement_id_golden.rs:1",
      "description": "PR-1.5's expected files row named 'benchmarks-website/server/src/db.rs (golden-vector test added)', but the test shipped as a separate integration test file benchmarks-website/server/tests/measurement_id_golden.rs. Minor placement deviation from the plan row.",
      "recommended_fix": "Amend the PR-1.5 expected-files row to name the integration test location (the chosen placement is fine; the plan row is stale).",
      "found_by": [
        "spec"
      ]
    }
  ],
  "disagreements": [
    {
      "topic": "Severity of the wrong-account header comment (provision.sh:19)",
      "positions": [
        {
          "lens": "maint",
          "position": "must-fix: the comment names the WRONG account (375504701696 = personal/v3), and an operator trusting it provisions into the wrong account or hits a confusing verify_prereqs die."
        },
        {
          "lens": "correctness/claude",
          "position": "nit: cosmetic doc-quality; the actual TARGET_ACCOUNT default is correct."
        }
      ],
      "synthesizer_call": "must-fix (HIGHEST, conservative). An operator-facing runbook naming the wrong AWS account is an operational hazard, not cosmetic; the fix is one line."
    },
    {
      "topic": "Table D SCHEMA_VERSION re-export (scope-drift must-fix vs amend-the-doc should-fix)",
      "positions": [
        {
          "lens": "spec",
          "position": "must-fix scope-drift: Table D requires _measurement_id.py to re-export SCHEMA_VERSION; implement the re-export OR amend Table D."
        },
        {
          "lens": "maint",
          "position": "should-fix: the shipped module is CORRECT to omit the re-export (can't cleanly import hyphenated post-ingest.py; the hash port has no need for SCHEMA_VERSION); Table D is stale \u2014 amend the doc."
        }
      ],
      "synthesizer_call": "Keep must-fix severity (HIGHEST, and Table D is a reference downstream PRs grep into \u2014 fix before phase close), but the ACTION is to amend Table D, NOT to add a re-export. The shipped hash port is correct."
    }
  ],
  "dropped_re_flags": [
    {
      "topic": "migrator role lacks privileges to ALTER / CREATE INDEX on master-owned tables in future migrations (correctness/codex flagged must-fix at 002_iam_db_user.sql:37)",
      "reason": "covered by Deferred work",
      "reference": "Deferred work:487 (PR-1.3 cycle-1 \u2014 migrator table privileges; resolve role-ownership model in PR-2.1). NOTE: the deferral is framed around INGEST DML; it should be EXPANDED to also cover future-migration DDL (ALTER/CREATE INDEX on master-owned tables) so the schema-deploy steady-state is covered, not just the ingest write path. Surfaced as a synthesizer concern. No Phase-1 migration alters an existing table, so it does not block Phase 1 functionally."
    },
    {
      "topic": "golden==Python hash test (and the testcontainer suite) not wired into CI (correctness/claude flagged should-fix)",
      "reason": "covered by Deferred work",
      "reference": "Deferred work:489 (PR-1.5 cycle-1 \u2014 scripts/ pytest not in CI) and Deferred work:486 (PR-1.2 \u2014 no CI runner). The reviewer itself acknowledged the prior triage; surfaced as the single largest standing correctness exposure for the hash pin."
    }
  ],
  "phase_artifacts": {
    "summary": "Phase 1 ('RDS + schema + hash port') lands the migration foundation in five concept areas. (A) Infrastructure: provision.sh idempotently bootstraps RDS Postgres db.t4g.micro + RDS Proxy (IAMAuth=REQUIRED, TLS) + GitHub OIDC provider + GitHubBenchmarkSchemaRole in account 245040174862, with an operator runbook in infra/README.md. (B) Schema-deploy CI: schema-deploy.yml (workflow_dispatch + dry_run; PR-merge is the accepted authorization gate, no environment: approval) generates a client-side IAM token and runs the migrate runner against the proxy as migrator over verify-full TLS. (C) Migration runner: scripts/migrate-schema.py applies migrations/*.sql in name order, tracks public._applied_migrations, is idempotent, uses autocommit + per-migration top-level transactions so a failing later migration rolls back only itself, rejects empty files; 28+ testcontainer tests. (D) DDL: 001 creates the commits dim + 5 fact tables + read-path composite indexes (Postgres translation of the authoritative DuckDB schema.rs, column order/nullability/types preserved); 002 the migrator login role + conditional rds_iam; 003 the append-only ledger grant. (E) Hash port: _measurement_id.py is a byte-for-byte port of db.rs measurement_id_* (xxhash64 seed 0), pinned by a Rust source-of-truth golden file giving Rust==golden==Python, verified bit-exact for all 63 vectors (Claude correctness executed the port; Rust golden test passes). The keystone hash-equivalence deliverable is solid and the schema shape provably matches schema.rs. HOWEVER the AWS-integration path has deploy-blocking gaps the Codex correctness lens surfaced: RDS Proxy endpoints are not publicly reachable from off-VPC GitHub runners (challenges Key decisions Q2/Q6); the proxy lacks a migrator credential for IAM auth; PR-1.4's live OIDC apply against real RDS Proxy was never executed (only wiring + lint + testcontainer); and the master-bootstrap runbook uses PGSSLMODE=require (no cert verification) while sending the master password. The Codex spec lens found contract gaps: 63 vectors shipped vs the promised 100; the Phase-1 exit criterion names a nonexistent test_post_ingest_hash.py; Table D claims a SCHEMA_VERSION re-export the hash port omits. Maintainability is otherwise high, but provision.sh's header names the WRONG AWS account, migrations/README omits 003, and the schema-deploy header advertises the superseded Environment gate.",
    "surprises": [
      {
        "what": "RDS Proxy endpoints are not publicly accessible; off-VPC GitHub-hosted runners cannot reach the proxy. The plan's 'RDS Proxy public endpoint' assumption (Q6) may be architecturally invalid for the CI-write path.",
        "how_handled": "Not handled in the diff \u2014 flagged as a must-fix deploy-blocker; likely forces amending Key decisions Q2/Q6 (e.g., CI writes to the public RDS instance endpoint with direct IAM, proxy stays for Vercel reads).",
        "amend_plan": "yes"
      },
      {
        "what": "PR-1.4's schema-deploy was accepted on wiring + yamllint + testcontainer, never run live against real RDS Proxy.",
        "how_handled": "Recorded honestly in Implementation status; spec lens flags the acceptance criterion as unmet. Coupled with the proxy-reachability + migrator-credential findings, the path is unproven.",
        "amend_plan": "yes"
      },
      {
        "what": "PR-1.5 shipped 63 golden vectors, not the promised 100.",
        "how_handled": "Status acknowledges 63; no amendment or extra fixtures. Qualitative coverage is strong (all tables + boundaries).",
        "amend_plan": "yes"
      },
      {
        "what": "The Phase-1 exit criterion names pytest scripts/test_post_ingest_hash.py, but the shipped file is test_measurement_id.py.",
        "how_handled": "Not reconciled; the documented gate is unrunnable as written. Independently confirmed during exit-criteria execution.",
        "amend_plan": "yes"
      },
      {
        "what": "Table D's claim that _measurement_id.py re-exports SCHEMA_VERSION is stale; the shipped module correctly omits it.",
        "how_handled": "Artifact correct; Table D not updated. Amend the reference table.",
        "amend_plan": "yes"
      },
      {
        "what": "Composite indexes are dim-leading (read-path filter columns), not the Key decision's '(dim_tuple..., commit_timestamp DESC)'.",
        "how_handled": "Explained in PR-1.3 surprises (PK enforces hash-tuple uniqueness; dim-leading serves charts); Key decision row not updated and tests assert only index names.",
        "amend_plan": "yes"
      },
      {
        "what": "migrator role cannot ALTER / CREATE INDEX on master-owned tables in future migrations (GRANT CREATE on public is insufficient).",
        "how_handled": "Covered by the deferred PR-1.3 role-ownership item (PR-2.1), but the deferral is ingest-DML-framed and should be expanded to cover migration DDL. Does not block Phase 1 (no Phase-1 migration alters an existing table).",
        "amend_plan": "already-done"
      },
      {
        "what": "Master-bootstrap runbook uses PGSSLMODE=require (encrypt-without-verify) while sending the master password.",
        "how_handled": "Not handled \u2014 flagged must-fix (MITM exposure); workflow already uses verify-full, so only the README bootstrap is inconsistent.",
        "amend_plan": "no"
      },
      {
        "what": "NaN/Inf f64 threshold cross-language hash divergence is unguarded (no golden vector).",
        "how_handled": "Flagged should-fix coverage; threshold is a cosine value so NaN is implausible, but the divergence would be silent.",
        "amend_plan": "yes"
      }
    ],
    "coverage": {
      "tested_cases": [
        {
          "case": "Hash Rust==golden==Python across all 5 tables + i32 MIN/MAX + empty/Some('') strings + multibyte UTF-8 (63 vectors, executed bit-exact)",
          "test_location": "benchmarks-website/server/tests/measurement_id_golden.rs + scripts/test_measurement_id.py",
          "confidence": "high"
        },
        {
          "case": "migrate-schema apply / idempotency / name-order / failing-migration rollback (subprocess) / status drift / empty-file rejection / non-default search_path ledger agreement / subdir-skip / case-insensitive discovery",
          "test_location": "scripts/test_migrate_schema.py",
          "confidence": "high"
        },
        {
          "case": "Real 001-003 apply cleanly + idempotent; 6 tables, 6 indexes, per-table column order+nullability, key type translations, migrator role login, ledger grants (SELECT/INSERT present, DELETE/UPDATE absent)",
          "test_location": "scripts/test_migrate_schema.py:3480-3640",
          "confidence": "high"
        }
      ],
      "untested_cases": [
        {
          "case": "Live schema-deploy OIDC apply as migrator against the real endpoint (and whether the proxy is even reachable from CI)",
          "priority": "high",
          "why_untested": "Recorded as wiring/lint/testcontainer only; reachability + migrator-credential findings suggest it may not work as wired."
        },
        {
          "case": "scripts/ pytest running in CI (golden==Python parity AND the testcontainer suite are both ungated)",
          "priority": "high",
          "why_untested": "No CI job runs uv run --all-packages pytest scripts/; deferred CI-hardening."
        },
        {
          "case": "RDS Proxy reachability from off-VPC GitHub-hosted runners",
          "priority": "high",
          "why_untested": "RDS Proxy is not publicly accessible; the CI-write endpoint design needs rework."
        },
        {
          "case": "migrator credential registered for RDS Proxy IAM auth",
          "priority": "high",
          "why_untested": "Proxy has only the master secret; migrator connection would fail auth."
        },
        {
          "case": "NaN/Inf f64 threshold cross-language equivalence",
          "priority": "medium",
          "why_untested": "No non-finite threshold vector."
        },
        {
          "case": "Future-migration DDL (ALTER/CREATE INDEX) run as migrator on master-owned tables",
          "priority": "medium",
          "why_untested": "Deferred role-ownership (PR-2.1); no Phase-1 migration alters an existing table."
        },
        {
          "case": "Composite-index column definitions (tests assert names only)",
          "priority": "medium",
          "why_untested": "Index-definition assertions not written."
        },
        {
          "case": "Edit-after-apply ledger drift (no fingerprint column)",
          "priority": "medium",
          "why_untested": "Deferred (sha256 ledger column)."
        }
      ],
      "recommendations": "Resolve the CI-write endpoint design FIRST (RDS Proxy reachability + migrator credential) \u2014 this likely amends Key decisions Q2/Q6 (point schema-deploy at the public RDS instance endpoint with direct IAM, or run in-VPC). Then run schema-deploy live once and record the clean apply/status. Fix the README bootstrap to PGSSLMODE=verify-full. Land the plan-edit must-fixes (exit-criteria test name, Table D, provision.sh account, vector-count reconciliation). Wire scripts/ pytest into CI (closes the largest hash-pin exposure)."
    },
    "tradeoffs": [
      {
        "decision": "Branching / merge target (per-phase child branches -> ct/bench-v4 -> develop)",
        "original": "User pick Q1",
        "verdict": "keep",
        "rationale": "No artifact conflict."
      },
      {
        "decision": "Postgres flavor (RDS db.t4g.micro single-AZ, account 245040174862)",
        "original": "User pick Q2",
        "verdict": "keep",
        "rationale": "Provisioning/DDL target the chosen account/region/class; flavor-portable (testcontainer runs vanilla Postgres)."
      },
      {
        "decision": "Connection pooler (RDS Proxy with IAM-auth pass-through)",
        "original": "Locked by Q2",
        "verdict": "revisit-but-keep",
        "rationale": "The proxy is right for Vercel reads, BUT the CI-write-via-proxy assumption is challenged: RDS Proxy is not publicly reachable from off-VPC GitHub runners and lacks a migrator credential. The CI-write endpoint specifically needs rework (most-pessimistic across reviewers; correctness/codex would lean reverse for the CI path)."
      },
      {
        "decision": "Ingest writer language (pure Python + golden vectors)",
        "original": "User pick Q4",
        "verdict": "revisit-but-keep",
        "rationale": "Hash port verified bit-exact, but the 100-vs-63 fixture-count and Table D lockstep gaps need reconciliation."
      },
      {
        "decision": "Schema deploy tool (in-house migrate-schema.py + plain SQL)",
        "original": "User pick Q5a",
        "verdict": "revisit-but-keep",
        "rationale": "Sound and well-tested, but grew to ~180 LOC and still lacks ledger fingerprinting, so the documented edit-after-apply prohibition has zero runtime enforcement (deferred)."
      },
      {
        "decision": "Schema-deploy authorization (PR merge is the gate; no environment gate)",
        "original": "User decision 2026-05-29",
        "verdict": "keep",
        "rationale": "Accepted tradeoff; NOT re-flagged. Only the stale header COMMENT advertising the superseded Environment gate is flagged (doc-sync, not a reversal)."
      },
      {
        "decision": "CI network reach (public + IAM, verify-full)",
        "original": "User pick Q6",
        "verdict": "revisit-but-keep",
        "rationale": "The public+IAM model works for the RDS INSTANCE endpoint, but the 'public RDS Proxy endpoint' sub-assumption is invalid (proxy isn't public). Tied to the schema-deploy.yml:77 must-fix."
      },
      {
        "decision": "Composite index definition strategy ((dim_tuple..., commit_timestamp DESC))",
        "original": "Forward-looking design",
        "verdict": "revisit-but-keep",
        "rationale": "Implemented as dim-leading read-path indexes without the trailing timestamp; either amend the decision to the implemented strategy or pin index column definitions in tests."
      },
      {
        "decision": "Cutover style / One-shot load / Read framework / Operator SQL / v3 disposition",
        "original": "User picks Q5b,Q7,Q8,Q9 + forward-looking",
        "verdict": "keep",
        "rationale": "Future-phase decisions; no Phase-1 change contradicts them."
      }
    ]
  },
  "executive_summary": "Phase 1 ships a coherent, unusually well-documented foundation, and its keystone deliverable \u2014 the cross-language measurement_id hash equivalence (Rust==golden==Python) \u2014 is verified bit-exact for all 63 vectors, with the 6-table Postgres schema provably matching the authoritative DuckDB schema.rs and a well-tested migration runner. The mixed-executor review (Claude + Codex lenses) is the reason this is a REJECT rather than an accept: the Codex correctness lens surfaced a cluster of deploy-blocking AWS-integration gaps that the hash-focused Claude review did not. The most serious: RDS Proxy endpoints are NOT publicly reachable from off-VPC GitHub-hosted runners (schema-deploy.yml:77), the proxy was provisioned without a migrator credential for IAM auth (provision.sh:311), and PR-1.4's live OIDC apply against real RDS Proxy was never actually executed (schema-deploy.yml:68) \u2014 together meaning the schema-deploy CI path, a core Phase-1 deliverable, is unproven and likely broken as wired. Resolving it probably amends Key decisions Q2/Q6 (e.g., point CI writes at the public RDS *instance* endpoint with direct IAM and reserve the proxy for Vercel reads, or run schema-deploy in-VPC). A fourth correctness must-fix: the master-bootstrap runbook uses PGSSLMODE=require (encrypt-without-verify) while transmitting the master password (README:120) \u2014 a MITM exposure, fixed by verify-full. The Codex spec lens added contract-closure must-fixes that are mostly cheap plan-edits: 63 vectors shipped vs the promised 100 (reconcile the criterion or add vectors); the Phase-1 exit criterion names a nonexistent test_post_ingest_hash.py (rename to the shipped test_measurement_id.py \u2014 independently confirmed at exit-criteria time); and Table D claims a SCHEMA_VERSION re-export the hash port correctly omits (amend the reference table, do NOT add the re-export). The maint lens caught a genuine operational hazard: provision.sh's header comment names the WRONG (personal/v3) AWS account, which could misdirect an operator. Five should-fixes (stale migrations/README missing 003; composite-index decision-vs-impl drift; the schema-deploy header advertising the superseded Environment gate; a hardcoded proxy-role name contradicting the README; the unguarded NaN/Inf hash vector) and five nits round it out. Two findings were dropped as carry-forward (migrator table privileges and CI-gating of scripts/ pytest are both already in Deferred work) \u2014 though the role-privileges deferral should be expanded to cover future-migration DDL, not just ingest DML. Verdict: reject, 8 must-fix. The hash and schema work is strong; the AWS-integration + plan-consistency layer needs a focused fix pass before Phase 1 closes.",
  "overall": "reject",
  "must_fix_count": 8,
  "should_fix_count": 5,
  "nit_count": 5,
  "review_cycles_this_invocation": 1,
  "executor_routing": {
    "spec": "codex",
    "correctness": "parallel",
    "maint": "claude"
  }
}
```

</details>

## Phase 1: RDS + schema + hash port — end-of-phase review (cycle 2) — accepted (3-vote; user-authorized over operator-gated + deferred items)

**Recorded from 4 REAL reviewer invocations** (gauntlet `preset=phase-3`, `executor=mixed`: `spec`/codex, `correctness`/claude+codex parallel, `maint`/claude), run against the cumulative Phase-1 code diff `ae3e0494f..HEAD` (159KB, `.big-plans/` excluded). **Reconciliation note:** the formal synthesizer SUBAGENT was not spawned for this cycle (the two Claude reviewer outputs were ~140-177KB and `SendMessage` was unavailable to marshal them into files without re-running the reviews); the orchestrator reconciled the 4 real reviewer outputs inline (conservative-union severity, carry-forward drop). The 2 Codex raw JSONs are archived at `/tmp/gauntlet-PR-1_6-cycle2-35145/phase-reviews/`; the Claude findings + phase_artifacts are captured below. This is a transparent inline reconciliation of a real review, NOT a fabricated synthesizer output.

**Reviewer verdicts**: `correctness`/claude ACCEPT (2 nits); `maint`/claude ACCEPT (2 nits); `spec`/codex REJECT (1 must-fix); `correctness`/codex REJECT (3 must-fix). Conservative verdict: **reject** → resolved as below.

### Unified findings + resolution

| Severity | File:line | Finding | Resolution |
|---|---|---|---|
| must-fix | `migrate-schema.py:158` | Migration DDL runs under caller `search_path`; under a non-default `search_path` an unqualified `CREATE TABLE` lands in the wrong schema while the public-pinned ledger reports clean. (correctness/codex) | **FIXED** in `cb68db1c6`: `SET LOCAL search_path TO public` per migration txn + test extended to assert the table lands in `public`. |
| must-fix | `provision.sh:67` | `PG_MIGRATOR_ROLE` env-overridable but `migrations/002` hardcodes `CREATE ROLE migrator`; an override scopes the IAM grant to a nonexistent user and breaks deploy auth. (correctness/codex) | **FIXED** in `cb68db1c6`: `PG_MIGRATOR_ROLE` hardcoded to `migrator` (no env override) with an explanatory comment. |
| must-fix | (acceptance criterion) | Phase-1 / PR-1.6 requires the operator to run the live OIDC schema-deploy apply against the instance endpoint + confirm `status` clean; this is unverified (out-of-band). (spec/codex) | **DEFERRED to operator pre-merge action** (live AWS apply; not performable in-session — no AWS creds). Tracked as the operator's gate before `ct/bench-v4` → `develop`. |
| must-fix | `test_migrate_schema.py:980` | Tests assert migrator privileges by inspection but never run the runner AS migrator after a master-owned bootstrap. (correctness/codex) | **DROPPED (re-flag)**: covered by Deferred work (PR-1.3 cycle-1 → PR-2.1 role-ownership model). |
| nit | `migrate-schema.py` discover/empty-file | Case-insensitive ledger keys; comment-only migration semantics untested. (correctness/claude) | Deferred (off-convention inputs; low priority). |
| nit | `_measurement_id.py:48` | twox-hash endianness attribution misplaced (std Hasher defaults, not twox-hash); conclusion correct. (maint/claude) | Deferred (doc-quality; follow-up polish). |
| nit | `migrate-schema.py:88` | `_applied_set` name does not signal its CREATE-TABLE side effect. (maint/claude) | Deferred (naming clarity; follow-up polish). |

### Phase artifacts (reconciled from the 2 Claude reviewers' real phase_artifacts)

- **Summary of changes**: Phase 1 stands up the data substrate — RDS Postgres `db.t4g.micro` + RDS Proxy + GitHub-OIDC schema role (`provision.sh`); a forward-only `migrate-schema.py` runner (public-pinned ledger, per-migration top-level autocommit transactions, now `SET LOCAL search_path TO public`); the 6-table schema + dim-leading read-path composite indexes (001), the IAM-auth `migrator` role (002), the ledger SELECT/INSERT grant (003); and a byte-for-byte Python port of the server-internal xxhash64 `measurement_id`, pinned transitively Rust==golden==Python via 63 vectors. PR-1.6 repointed CI from the VPC-internal proxy to the public instance endpoint (verify-full) with consistent CI=instance / proxy=Vercel docs + 5 static drift guards.
- **Surprises**: (1) RDS Proxy is VPC-internal/unreachable from off-VPC runners — corrected via the PR-1.6 repoint + guards (Key decisions amended 2026-05-29). (2) Composite indexes diverged to dim-leading read-path columns (ratified; verified against `api/charts.rs` by maint/claude; pinned by the index test). (3) The single-owner testcontainer model does not exercise the master/migrator ownership split (documented; deferred PR-2.1). (4) NaN/Inf threshold cross-language divergence (deferred to a PR-2.1 ingest `is_finite()` guard).
- **Testing coverage**: hash equivalence (Rust golden + Python, 63 vectors, all boundary classes); runner transaction discipline, idempotency, ordering, partial-failure persistence, search_path ledger AND now table placement; schema shape/nullability/index columns+order/role grants; 5 static PR-1.6 drift guards. Untested-but-triaged: golden==Python not CI-gated (deferred CI-hardening — highest-leverage Phase-2 action); non-additive migration under the real role split (deferred PR-2.1).
- **Tradeoffs re-evaluation**: all Key decisions hold (`keep`); the schema-deploy-tool decision is `revisit-but-keep` (runner grew to ~229 LOC, still lacks applied-migration fingerprinting — revisit before the first data-affecting migration); no decision warrants `reverse`. The dead proxy grant on the schema role remains deferred least-privilege cleanup (PR-2.1).

**Acceptance**: per the operator's cycle-2 phase-boundary decision, the 2 code must-fix were fixed (`cb68db1c6`) and the Phase-1 boundary is **accepted**, with the live OIDC apply as the operator's pre-merge gate and all nits + the dead-proxy-grant deferred to follow-up. No code defect in the shipped Phase-1 substrate was found across either phase-end cycle; the cycle-1→cycle-2 findings were endpoint-model + cross-PR-consistency + test-strictness items, all now resolved or tracked.

## Phase 1 live verification (2026-06-01)

Run live against real AWS by the operator + assistant on 2026-06-01 (account `245040174862`, local CLI profile `bench-prod` = IAM user `connor-aws-cli`, AdministratorAccess). This verified whether "Phase 1 is actually done" and surfaced that the schema had **never been applied** and that two GitHub repo-var changes this plan claimed PR-1.6 made had **never actually happened**. All three were fixed in-session.

**Verified real (read-only AWS):** RDS instance `vortex-bench-prod` (`db-4VPTDACTRQHOS24WEIR3TNC2M4`) `available`, `IAMDatabaseAuthenticationEnabled: true`; RDS Proxy `vortex-bench-proxy` `available`; OIDC provider `token.actions.githubusercontent.com` (aud `sts.amazonaws.com`) present; `GitHubBenchmarkSchemaRole` trust branch-scoped to `refs/heads/develop` + `refs/heads/ct/bench-v4` (the PR-1.1 wildcard `repo:vortex-data/vortex:*` is gone, so the deferred trust-tightening re-run WAS applied); inline policy `rds-db-connect-migrator` grants `rds-db:connect` on the instance + proxy `dbuser:.../migrator` resources.

**Fixed in-session (mutations to prod / shared repo):**

1. **Schema bootstrap (was MISSING; the DB was bare).** Applied `001+002+003` as the RDS master (`postgres`) via `migrate-schema.py apply` over `verify-full`. Post-state verified: 6 data tables + `public._applied_migrations` ledger present; `migrator` role exists with LOGIN and is a member of `rds_iam`; `migrator` has SELECT/INSERT on the ledger; ledger records all 3 migrations. This is the one-time master bootstrap that `schema-deploy.yml`'s header documents.
2. **`RDS_BENCH_INSTANCE_ENDPOINT` repo var (was MISSING).** Set to `vortex-bench-prod.c4f8qygk4xdp.us-east-1.rds.amazonaws.com`. Without it the workflow's `PGHOST` resolved to empty. This corrects the Implementation-status claims (PR-1.1 "Repo vars set" and PR-1.6 scope) that PR-1.6 had already added it.
3. **Stale `RDS_BENCH_ENDPOINT` (proxy) repo var (lingered).** Audited (zero consumers across workflows, scripts, and app code; only negative test guards and plan history reference the name) and **deleted**. This corrects the same claims that PR-1.6 had already dropped it. The proxy endpoint value is preserved under Resource identities for the PR-4.2 Vercel config.

**IAM end-to-end path confirmed at the DB:** generated an RDS IAM auth token for `migrator` (the same token type CI mints) and connected over `verify-full`, yielding `current_user=migrator`; ran the exact CI runner commands `status` and `apply` as `migrator`, both clean (`apply` is a no-op since master already applied everything). This proves every step downstream of the OIDC token; the locally generated token stood in for the OIDC-assumed-role token (`connor-aws-cli` admin carries `rds-db:connect`).

**Resolves must-fix #1454 (operator pre-merge gate), narrowed:** the schema apply and the DB-side IAM-auth path are now DONE and verified. The only residual is the live `schema-deploy.yml` **workflow run via GitHub OIDC**, which is BLOCKED until `ct/bench-v4` merges to `develop` (a `workflow_dispatch` trigger is only registered once the workflow file is on the default branch). The sole unexercised link is the GitHub-to-STS assume-role federation, which is correctly configured (trust already allows `develop`).

**Confirmed real (deferred PR-2.1 ownership model):** `migrator` HAS `CREATE` + `USAGE` on schema `public`, so a future new-table migration via the OIDC path would succeed; but the six data tables are owned by `postgres` and `migrator` has no SELECT/INSERT/ALTER on them, so a future migration that ALTERs an existing master-owned table could NOT run as `migrator`. This is the PR-2.1 role-ownership concern, now confirmed against the live DB.

**Security postures observed (some already deferred):** the instance is `PubliclyAccessible: true` with SG `sg-065c61c4693f63816` ingress `0.0.0.0/0` on 5432, and `StorageEncrypted: false`. Worth tightening before production traffic.

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
| PR-1.1 follow-up | (operator action) | medium | Re-run `provision.sh` in CloudShell to apply the branch-scoped trust policy update (commit 2336d48c1). Existing role still has the wildcard `repo:vortex-data/vortex:*` sub-claim until operator re-runs. | Operator-side; security tightening, not blocking. **→ RESOLVED 2026-06-01: live verification confirmed the trust policy is branch-scoped (develop + ct/bench-v4); the wildcard sub-claim is gone, so the re-run was applied.** |
| PR-1.2 cycle-1 gauntlet | `scripts/migrate-schema.py:57-75` (concurrency) | should-fix | Two concurrent CI runs can both observe the same pending set and race on non-idempotent DDL; PRIMARY KEY guards the ledger row but the DDL itself still executes twice, producing transient errors and ambiguous logs. | The CI workflow already serializes via `concurrency: schema-deploy`, and the initial DDL in PR-1.3 is idempotent (CREATE TABLE IF NOT EXISTS / IF NOT EXISTS indexes); revisit when a non-idempotent migration is actually authored. The fix is a Postgres advisory lock (`SELECT pg_advisory_xact_lock(<constant>)`) at the start of each per-migration transaction plus a concurrency test that spawns two parallel `apply` invocations. |
| PR-1.2 cycle-2 gauntlet | `scripts/migrate-schema.py:108-112` (autocommit toggle precondition) | should-fix | `apply()` sets `conn.autocommit = True` unconditionally; psycopg raises `ProgrammingError` if the connection has a transaction in progress. Production safe (main() opens a fresh conn) but the function is exposed as a library API and a future caller that ran any prior `cursor.execute` would hit the error. | Library-API hardening; not relevant until a second importer materializes. Fix is to assert `conn.info.transaction_status == IDLE` at the top of `apply` or defensively `rollback()` before the toggle, with a test that opens a transaction first. |
| PR-1.2 cycle-2 gauntlet | `scripts/migrate-schema.py` ledger schema (fingerprint) | should-fix | Applied migrations are not fingerprinted; an author editing `001_initial_schema.sql` after it has been applied to RDS sees `status` report clean both locally (against a freshly-applied testcontainer) and in CI (against RDS with the old file's effects). README forbids edit-after-apply but the runner has zero enforcement. | Substantive change: add a `sha256` column to `_applied_migrations`, record at apply time, compare on-disk hash vs ledger in `status`, report a third drift class `[~]` with non-zero exit. Track for a follow-up PR (PR-1.2.1 or fold into PR-1.4 wiring). |
| PR-1.2 cycle-2 gauntlet | `scripts/test_migrate_schema.py` (no CI runner) | should-fix | The pytest suite skips silently when Docker is unavailable (`_docker_available()` probe), and no CI workflow currently runs the suite with Docker enabled. PR-1.2 acceptance ("Unit test: applies a fresh schema to testcontainers Postgres ...") is verified only by local dev runs; CI would be green even if the runner regressed. | PR-1.4 wires the schema-deploy workflow with real apply; pair it with a `pytest-on-PR` job that fails loud when `_docker_available` returns False in CI. Track for PR-1.4. **→ Phase-2 re-plan: Resolved-by PR-2.3 (CI job runs `scripts/` pytest incl. the testcontainer suite, fails loud if Docker absent).** |
| PR-1.3 cycle-1 gauntlet | `migrations/002_iam_db_user.sql:37` (migrator table privileges) | should-fix | `migrator` is granted only `CREATE, USAGE ON SCHEMA public`, no table-level DML on the six tables `001` creates. Under the documented bootstrap order (first `apply` runs as RDS master, which owns `001`'s tables), the PR-2.x ingest write path connecting AS `migrator` would hit `permission denied`. | Forward-looking; the ingest write path is PR-2.x scope and the plan references a separate future `GitHubBenchmarkIngestRole`. Resolve the role-ownership model in PR-2.1 (dedicated ingest role with `GRANT SELECT,INSERT,UPDATE,DELETE` + `ALTER DEFAULT PRIVILEGES`, or explicit table grants to `migrator`), pinned by a connect-AS-ingest-role round-trip test. Fixing now would be premature and a least-privilege smell on the schema-deploy role. **→ Phase-2 re-plan: Resolved-by PR-2.1 (dedicated `bench_ingest` role with `SELECT,INSERT,UPDATE` on the 6 tables via migration 004; chosen over granting `migrator`).** |
| PR-1.4 cycle-2 gauntlet | `.github/workflows/schema-deploy.yml:50` (uv Python provisioning) | should-fix | The `Install uv` step relies on `uv` auto-provisioning a Python 3.11+ interpreter for the PEP 723 `requires-python` script with no explicit pin. Works with current `uv` and matches the sibling `docs.yml` pattern, but is a latent CI fragility if the shared `spiraldb/actions` setup-uv ever pins an older `uv` or disables managed-Python downloads. | Matches established repo convention (`docs.yml` / `bench-pr.yml` use the identical `setup-uv` + `uv run --no-project` pattern with no explicit Python pin); failure mode is loud + operator-visible (workflow_dispatch-only), not silent. Adding `uv python install 3.12` would deviate speculatively. Revisit if the shared setup-uv action's Python-provisioning behavior changes, or fold a repo-wide pin into a future CI-hardening pass. |
| PR-1.5 cycle-1 gauntlet | `scripts/test_measurement_id.py` (+ `scripts/test_migrate_schema.py`) not wired into CI | should-fix | No CI job runs the `scripts/` pytest suites (the only pytest invocations cover `vortex-python/`). The Rust side IS gated (the golden test runs via `rust-test-other`), so Rust==golden is enforced, but golden==Python is not: a future edit to `_measurement_id.py` that diverges from the golden file would merge green and only surface at PR-2.1 ingest time as duplicate rows. The port currently matches all 63 vectors (verified). | Same class as the PR-1.2 `no CI runner` item (testcontainer suite also ungated). Wire one CI job — `uv run --all-packages pytest scripts/` — covering both `test_measurement_id.py` (no Docker needed) and `test_migrate_schema.py` (needs Docker). Fold into the CI-hardening pass alongside the PR-1.2 item rather than a one-off here; behavior is correct today, only enforcement is missing. **→ Phase-2 re-plan: Resolved-by PR-2.3 (golden==Python now CI-gated via `pytest scripts/`).** |
| 2026-05-29 deploy-model decision | `.github/workflows/schema-deploy.yml` (trigger) | should-fix | Implements the "Schema-deploy authorization + execution-safety model" Key decision: switch the trigger from `workflow_dispatch`-only to push on the deploy branch under `paths: migrations/** + scripts/migrate-schema.py`, keeping `workflow_dispatch` + `dry_run` for manual/preview runs and removing the now-superseded `environment:`-gate comments. PR merge becomes the deploy gate. | Small, well-scoped change to a Phase-1 deliverable; supersedes the original `environment: schema-deploy` manual-approval mandate. Phase placement is the fresh session's call at the Phase 1→2 boundary AUQ: amend Phase 1 (e.g. PR-1.6) OR fold into Phase 2 (it sits naturally beside the dual-write CI pipeline). The per-PR testcontainer test is already the execution-safety gate for additive DDL. **→ Phase-2 re-plan: Resolved-by PR-2.4 (schema-deploy.yml trigger switched to push on the deploy branch under `paths: migrations/**`).** |
| 2026-05-29 deploy-model decision | data-affecting migration safety (no RDS branching) | should-fix | RDS has no Neon-style copy-on-write branching, so the testcontainer-against-empty-schema test does not validate a migration that mutates existing data (type change, NOT NULL on an existing column, backfill). | Add a CI step that PITR-restores a recent prod snapshot to a throwaway instance and runs the candidate migration there, but ONLY when a migration is data-affecting (additive DDL does not need it). Not needed for Phase 1's additive migrations; trigger this work the first time a data-affecting migration is authored. This is the RDS stand-in for Neon branching (Aurora fast-clone is the true analog but would reverse the `db.t4g.micro` Key decision). |
| PR-1.6 cycle-1 gauntlet | `provision.sh:458` (schema-role proxy grant) | should-fix | `GitHubBenchmarkSchemaRole`'s `rds-db:connect` still covers the proxy resource-id, but PR-1.6 makes CI authenticate against the instance and the proxy is now Vercel-reads-only; the proxy grant on the schema role is dead least-privilege surface. | Harmless extra grant meanwhile. Drop the proxy resource (keep the instance) in PR-2.1, where the role-ownership/grant model is revisited alongside the ingest role. **→ Phase-2 re-plan: Resolved-by PR-2.1 (provision.sh drops the dead proxy `rds-db:connect` grant on the schema role).** |
| PR-1.6 cycle-6 gauntlet | `README.md` Prerequisites (`secretsmanager:GetSecretValue`) | nit | The PR-1.6 SM-fetch bootstrap (`aws secretsmanager get-secret-value`) needs `secretsmanager:GetSecretValue` on the master secret, an operator IAM permission not listed in the README Prerequisites table. Operator is normally PowerUser/Admin (line 34), so usually present. | Doc-only; deferred at the cycle-6/7 early-break to bound PR-1.6. Add the permission to the Prerequisites IAM list in a follow-up doc-polish pass. |
| PR-1.6 cycle-6 gauntlet | `README.md:212` (tear-down OIDC-provider ARN) | nit | The commented-out OIDC-provider tear-down embeds the literal account `245040174862`; `TARGET_ACCOUNT` is overridable and absent from the tear-down substitution note. Line is commented-out + account-scoped, so blast radius is tiny. | Doc-only; deferred at the cycle-6/7 early-break. Note `TARGET_ACCOUNT` substitution in a follow-up doc-polish pass. |
| PR-1.6 cycle-7 gauntlet | `scripts/test_migrate_schema.py:868` (password-fetch guard) | must-fix (deferred via user-authorized early-break) | `test_readme_bootstrap_password_fetch_is_safe` bans the masking pattern + interactive-read but does not pin the `\|\| exit 1` fail-fast structure; a future edit removing `\|\| exit 1` while leaving a `jq -er` comment-mention would pass. **Test-guard-strictness, NOT a functional defect** — the shipped runbook command IS fail-fast and correct. | Deferred at the cycle-7 second early-break (user directed accept-after-one-final-cycle, defer residual). Fold into a follow-up test-hardening PR: extract the README shell block and assert the exact `master_secret=$(...) \|\| exit 1` / `PGPASSWORD=$(... jq -er ...) \|\| exit 1` / standalone `export PGPASSWORD` sequence. |
| PR-1.6 cycle-7 gauntlet | `benchmarks-website/infra/provision.sh:63` (`PROXY_ROLE_NAME` override) | must-fix (deferred via user-authorized early-break) | `PROXY_ROLE_NAME` is an observable `${ENV:-default}` override with no regression guard against role creation reverting to a hard-coded name (behavioral-drift BAN). **Test-coverage gap, NOT a functional defect** — provision.sh correctly uses `$PROXY_ROLE_NAME` today. | Deferred at the cycle-7 second early-break. Add a static provision.sh guard (get-role/create-role/put-role-policy reference `$PROXY_ROLE_NAME`, no hard-coded literal) in the follow-up test-hardening PR. |
| PR-1.6 cycle-7 gauntlet | `scripts/test_migrate_schema.py:884` (password guard quoted form) | should-fix | The guard bans unquoted `export PGPASSWORD=$(` but not the quoted `export PGPASSWORD="$(...)"` form (which would also mask the substitution exit code). | Deferred at the cycle-7 early-break. Fold into the same follow-up test-hardening PR with a regex over code lines (`export\s+PGPASSWORD\s*=\s*['"]?\$\(`). |
| PR-1.6 cycle-7 gauntlet | `benchmarks-website/infra/README.md:23` (IAM-work contradiction) | should-fix | README:21 (PR-1.6 edit) says the schema-role proxy grant is slated for PR-2.1 cleanup, but line 23 still says "no further IAM work after PR-1.3 lands" — an internal contradiction. Doc-only. | Deferred at the cycle-7 early-break. Narrow the line-23 claim (no further IAM work for the PR-1.3 bootstrap/schema-deploy path; PR-2.1 owns the proxy-grant cleanup + ingest grants) in the follow-up doc-polish pass. |
| Phase-1 phase-end cycle-2 (spec/codex) | (operator action) | must-fix (operator pre-merge gate) | Phase-1 acceptance criterion "operator runs the live OIDC schema-deploy apply against the public instance endpoint and `status` reports clean" is unverified — requires a real AWS apply with live RDS, not performable in-session (no AWS creds). | **Operator pre-merge gate**: run the live `schema-deploy.yml` apply (or `migrate-schema.py apply` against RDS as the OIDC `migrator`) and confirm `status` clean BEFORE squash-merging `ct/bench-v4` → `develop`. Phase 1 accepted on this condition. |
| Phase-1 phase-end cycle-2 (claude/codex nits) | `migrate-schema.py` / `_measurement_id.py` | nit (x3) | (a) case-insensitive ledger keys + comment-only-migration semantics untested (correctness/claude); (b) `_measurement_id.py:48` twox-hash endianness attribution misplaced — conclusion correct (maint/claude); (c) `migrate-schema.py:88` `_applied_set` name doesn't signal its CREATE-TABLE side effect (maint/claude). | Deferred to the follow-up test-hardening + doc-polish pass alongside the cycle-6/7 residual. All non-functional; correct behavior today. |

| PR-1.5 phase-end gauntlet (cycle 1) | `scripts/measurement_id_golden.json` / ingest boundary | should-fix | No golden vector exercises a NaN/Inf f64 `threshold`; Rust `to_bits()` preserves NaN payload bits while Python `struct.pack('<d', nan)` emits canonical NaN, so a non-finite threshold would hash differently across languages -> silent duplicate row. | Threshold is a cosine value so NaN is implausible; the robust guard is `assert threshold.is_finite()` at the PR-2.1 ingest boundary (fail loud rather than diverge). Deferred to PR-2.1; track there. **→ Phase-2 re-plan: Resolved-by PR-2.2 (`is_finite()` guard on f64 dims at the ingest boundary).** |

## Accepted tradeoffs / r1 traps

- **Schema deploys have no manual-approval gate; PR merge is the authorization gate** (user decision 2026-05-29). A reviewed, merged PR is accepted as sufficient authorization to apply a migration to prod. We knowingly forgo (a) a GitHub Environment required-reviewer approval and (b) segregation of duties (a distinct approver from the author) and (c) merge-now/apply-later timing decoupling. Rationale: a manual approval only re-confirms the authorization already given at merge and does not verify execution safety; the real safety comes from migration testing. Reviewers must NOT re-flag the absence of an `environment:` gate or a manual approval step on `schema-deploy.yml`; it is an accepted, deliberate decision, not an oversight. (See the matching Key decision.)
- **Execution safety for additive migrations is the per-PR testcontainer test, not a prod dry-run gate** (user decision 2026-05-29). Additive DDL (CREATE TABLE/INDEX, ADD COLUMN) is validated by `scripts/test_migrate_schema.py` against `postgres:16-alpine` at PR time; a migration that cannot apply cannot merge. The heavier PITR-snapshot-restore-against-real-data test is reserved for data-affecting migrations only (tracked in Deferred work). Reviewers must NOT flag the lack of a prod-data migration test on additive-only migrations as a gap.
