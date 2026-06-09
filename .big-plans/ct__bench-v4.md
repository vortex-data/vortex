# benchmarks-website migration to hosted Postgres + Next.js on Vercel — big-plans plan

## Current State

```yaml
status: executing
branch: ct/bench-v4
planning_sub_flow: null
current_phase: "Phase 4: Next.js read service on Vercel"
phase_index: 4
current_pr: null
pr_index: 7
outstanding_must_fix: 0
deferred_items_total: 11
last_user_touchpoint: 2026-06-09T18:28:08Z
last_user_touchpoint_what: "PHASE-4 HOLISTIC MID-PHASE REVIEW (user-requested) COMPLETE: 3 parallel lenses (coherence/claude + maintainability/claude via Agent; correctness/codex via companion gpt-5.5 xhigh) over the cumulative phase-4 product. 1 must-fix CONFIRMED + FIXED (read-route caching incoherence: /api/groups revalidate=300 forced a build-time prerender so DB-less next build FAILED; slugged routes revalidate inert via request.url; replaced with READ_API_CACHE_CONTROL s-maxage=300 CDN headers via new web/lib/cache.ts; DB-less next build now GREEN + pinned by route-test header assertions). Also fixed: v3 {error,message} 400/404 envelopes; reqNumber finiteness (forged 1e400 slug -> 400, serde f64 parity, raw-payload test); shared web/lib/test-harness.ts (4 suites deduped, net -208 lines); compareCodeUnits dedupe; noscript empty-state v3 parity; .group-info-icon comment corrected; doc polish + em-dash sweep. All in commit b53e07727; tsc/eslint/prettier clean; 128 vitest pass (+2 new). Landing-page caching DEFERRED to PR-4.5 (deferred 10->11; full Implementation-status entry recorded). HISTORY CLEANUP DONE (user-requested 2026-06-09): branch re-squashed to one commit per phase (phase-1/2 squashes kept from the prior cleanup; phase 3 newly squashed; phase-4-so-far newly squashed) + this plan commit; REBASED onto origin/develop 2026-06-09 post-squash (20 upstream commits; only Cargo.lock conflicted, resolved by taking upstream + cargo re-resolve of the migrate crate; squash SHAs after rebase: phase 3 = 8f249165b, phase 4 = f92713a49); pre-squash PRE-REBASE history preserved at LOCAL ref backup/bench-v4-pre-squash-20260609 (75841fae1); per-PR ending-at SHAs in Implementation status entries resolve only via that backup ref. phase_entry_sha repointed to the phase-3 squash so git diff phase_entry_sha..HEAD still spans all phase-4 work. BETWEEN-PRs (status executing + current_pr null): on resume Step 2.1 sets current_pr from pr_index=7 = PR-4.4.b (chart client island + interactivity + permalink; ALSO owns deferred header interactivity incl. mobile-nav GitHub fallback; landing-cache decision input lands at PR-4.5). PR-4.4 ARCH (locked): RSC shell + per-chart client islands; shard endpoint DROPPED. PHASE-4-BOUNDARY FLAG: THREE+ preserved/parity dismissals accumulate (PR-4.3.c statpopgen-descriptions + same-timestamp-tie; PR-4.4.a round-half-even); surface at Step 3.4 for a user decision on a deliberate 'v4 correctness/fidelity improvements over v3' effort vs preserve-v3. ORCH (load-bearing for review): CODEX at /Users/connor/.config/claude/plugins/cache/openai-codex/codex/1.0.4/scripts/codex-companion.mjs (gauntlet's ~/.claude probe WRONG; Codex at ~/.config/claude; gpt-5.5 xhigh). Custom 2-vote = parallel Claude+Codex per lens: compose pr-2 prompts from gauntlet/0.4.0/reference via compose_prompts.py (--lenses fresh,correctness --executor-routing fresh=parallel,correctness=parallel + --plan-section/--bans/--accepted-tradeoffs/--deferred-work files); Claude reviewers via Agent; Codex via 'node <companion> task --background --model gpt-5.5 --effort xhigh --prompt-file <f>' then poll 'result <job-id>'; synthesize inline (conservative-union dedupe). STILL-LIVE: signoff connor@spiraldb.com (commit -F); targeted git add; web/lib/ gitignore-negated; Next 15.5.19; docker up; CAT-K clean as of the 2026-06-09 rebase onto origin/develop; review calibration = trusted-input low-stakes, cap ~3 cycles."
subagent_invocations_this_pr: 0
subagent_invocations_total: 97
review_cycles_this_pr: 0
phase_entry_sha: 8f249165b
phase_end_cycle: 0
phase_end_reject_cycles: 0
last_phase_end_verdict: null
current_pr_is_ci_reopen: null
last_commit: f92713a49
last_cycle_commits: []
```

## Context

The `benchmarks-website/` subsystem is the public face of Vortex's continuous-benchmark numbers. **Corrected current-state (2026-06-04 re-plan, verified against the repo):** the LIVE public site at `benchmarks.vortex.dev` is the **v2** Vite/React SPA served by a Node `server.js` that reads benchmark data from the S3 bucket `vortex-ci-benchmark-results/data.json.gz` (+ `commits.json`), refreshed every ~5 min; v2 is published as a Docker image by `publish-benchmarks-website.yml`. A **v3** system (Rust/Axum + embedded DuckDB on EC2, custom systemd deploy + hourly S3 backup, bearer-token CI ingestion at `POST /api/ingest`, in-process artifact cache `read_model.rs`) is **built and live and CI-fed but has never served public traffic** — the v2→v3 public cutover was never completed (DNS stays on v2; the `7efbcacd2` "remove v2" commit lives on an unmerged branch). CI runs (one writer on `ubuntu-latest`, two on `bench-dedicated`, eleven from the SQL bench matrix) fan in to ~14 parallel `--server` envelope POSTs to the v3 endpoint per push to `develop`. **v3's structured DuckDB is the authoritative, clean source the v4 migration loads history from** (far better-structured than v2's S3 JSON blobs). The migration target is **v4** (Postgres + Next.js), replacing BOTH v2 and v3; cutover is **v2→v4 direct** (v3 is skipped, then decommissioned).

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
- Best-effort (`continue-on-error: true`) on the **v3** CI ingest step — v3 stays hard-required (it feeds the DuckDB migration source + the live read path). **(2026-06-04 re-plan note:** the **v4** ingest step IS intentionally best-effort *during the dual-write soak* — a v4 hiccup must not break the proven v3 pipeline while v4 is unproven — and is promoted to required at cutover. This deliberately scopes the prior blanket "no best-effort" rule to v3-only.)
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
| Cutover style | **Short dual-write window** (CI keeps writing v3 AND adds a best-effort v4 write) for ~3-7 days of soak; then promote v4 to required + drop v3-write; then DNS flip from v2 directly to v4; then decommission v2 + v3 | User pick (Q7), **amended 2026-06-04 (lean re-plan)**. v3 is a stepping stone that never serves public traffic (DNS stays on v2 until the v4 flip), but its DuckDB IS the structured migration source. **Lean de-risking:** the v4 write is BEST-EFFORT during the soak (don't gate the proven v3 pipeline on the unproven v4 path), promoted to required at cutover. The benchmark-data-loss safety net is provided WITHOUT heavy reconciliation machinery: v2 stays live, v3 stays live, the DuckDB snapshot is kept ≥90 days, and Phase-3's one-shot `migrate --verify` (DuckDB↔Postgres row/id comparison) is the primary correctness gate. (Supersedes the prior "both-must-succeed dual-write + reconciliation script + incident.io" model as disproportionate to a trusted-input low-stakes dashboard with four independent safety nets.) |
| Read service framework | **Next.js 15 + App Router + React Server Components + `unstable_cache` + `revalidateTag`** at `benchmarks-website/web/` | User pick (Q8). Server components fetch directly from Postgres; per-chart cache tags invalidated from the Python writer's CLI via a Vercel revalidation endpoint. Latest stable Next.js. Pages Router avoided. |
| Operator SQL replacement | **`scripts/psql-bench.sh`** — tiny helper that runs `aws rds generate-db-auth-token` and pipes into psql with IAM creds | User pick (Q9). Replaces `/api/admin/sql`. No bearer tokens, no Lambda. Documented in benchmarks-website/web/README.md. RDS PITR (35-day) replaces `/api/admin/snapshot`. |
| Composite index definition strategy | Net-new in `migrations/001_initial_schema.sql`. **As-shipped: dim-leading composite indexes following the read-path chart-query filter columns** (per `api/charts.rs`), NOT the hash field order. | **Amended 2026-05-29 (Phase-1 re-plan)** to match what PR-1.3 shipped: the original `(dim_tuple..., commit_timestamp DESC)` framing was superseded — every chart query filters on the dim columns and joins `commits` on `commit_sha`, so a dim-leading index serves the read path; PK uniqueness over the full hash tuple is already enforced by `measurement_id`. (PR-1.3 surprise, ratified here; an index-column-definition test is folded into PR-1.6.) |
| CI-write endpoint (re-plan 2026-05-29) | **Public RDS instance endpoint + direct IAM** for all CI writers (schema-deploy + Phase-2 ingest); RDS Proxy is Vercel-reads-only | Phase-1 phase-end gauntlet found the RDS Proxy is VPC-internal (unreachable from off-VPC GitHub runners). The instance was already provisioned `--publicly-accessible` with IAM auth, so CI writers connect to it directly with OIDC→IAM tokens + verify-full TLS. This **moots** the "register a migrator credential in the proxy auth config" finding for the CI write path (proxy auth config becomes a Phase-4 concern for the Vercel read role). Supersedes the proxy-for-CI assumption in the original pooler/Q6 decisions. |
| v3 EC2 final disposition | **Decommissioned at end of Phase 5** (single deletion PR removes `benchmarks-website/server/`, `benchmarks-website/ops/`, `benchmarks-website/migrate/`, top-level v2 files, `publish-benchmarks-website.yml`, `INGEST_BEARER_TOKEN`/`ADMIN_BEARER_TOKEN` secrets). EC2 instance terminated by hand after PR merges. | v3 never goes live; Q7 cutover model goes v2→v4 directly. |
| Phase-2 ingest DB identity (re-plan 2026-06-01) | **Dedicated `bench_ingest` role** (DB-side) + **`GitHubBenchmarkIngestRole`** (AWS-side), separate from the `migrator` / `GitHubBenchmarkSchemaRole` schema-deploy identity. `bench_ingest` gets DML-only (`SELECT,INSERT,UPDATE`, no DELETE/DDL) on the 6 data tables via migration 004. | Re-plan Q2. The ~14-writer ingest path runs on every push against a `PubliclyAccessible: true` instance with `0.0.0.0/0:5432` ingress (live-verified Phase-1 posture); a separate least-privilege identity means a leaked CI token can do data DML only, never DDL/migrations/role changes. Matches the `GitHubBenchmarkIngestRole` the original plan already anticipated. Cost: one migration + one provision.sh role block. Rejected: reuse `migrator` (conflates schema-deploy authority with the most-exposed code path). |
| Phase-2 dual-write verify scope | **SUPERSEDED 2026-06-04 (lean re-plan) → verify-once, no standalone reconciliation machinery.** Correctness of the v4 data is verified by **Phase 3's one-shot `migrate --verify`** (authoritative DuckDB↔Postgres row/id comparison) plus a short MANUAL spot-check during the soak (e.g. `psql-bench.sh` row-count + a few measurement_id lookups). **DROPPED:** the standalone `reconcile-ingest.py` service, the `dual-write-verify.yml` workflow, and incident.io paging. | The prior 2026-06-01 plan added a per-push Postgres-side reconciliation harness + incident.io alert. The 2026-06-04 course-correction found this disproportionate: benchmark numbers are trusted-input + regenerable, and four independent safety nets already cover Risk #4 (v2 live, v3 live, ≥90-day DuckDB snapshot, Phase-3 `migrate --verify`). A paged production-pipeline reconciliation service for a low-stakes dashboard is over-built. The one-shot verify is the load-bearing gate; the soak just needs eyeballs, not on-call. |
| Review calibration (lean re-plan 2026-06-04) | **Trusted-input, low-stakes calibration for all REMAINING reviews** (see the `## Review calibration` section). Reviewers flag DATA-CORRECTNESS + does-it-work, NOT adversarial-input robustness on trusted `vortex-bench` CI data. Code PRs use **2-vote** (fresh+correctness); only the final cutover phase (Phase 5) uses **3-vote**. Inner-loop **capped at ~3 cycles** (an open finding at cycle 3 is deferred or accepted, never spiraled). Deferred backlog pruned to data-correctness items only. | PR-2.2's 15-cycle / ~75-subagent spiral came from adversarial lenses hunting untrusted-input edge cases (NUL / non-UTF-8 / lone-surrogate / RecursionError + the 4-cycle `git_show_field` non-UTF-8 saga) on trusted CI JSON, with no cycle cap. This recalibrates the skill's default "thoroughness over cost" to the actual artifact (a benchmarks dashboard), preventing recurrence across Phases 3-5. |
| PR-4.4 UI architecture (2026-06-09) | **RSC shell + per-chart client islands; shard endpoint DROPPED** | User pick (PR-4.4 fork AUQ). Chart.js canvas charts are inherently client-rendered, so a chart is a `'use client'` island in every variant; the real fork was the shell + the deferred shard endpoint's fate. Decision: server components render the layout / group-section / summary / filter-bar shell (cached via `unstable_cache` + edge CDN, matching the Phase-4 `Read service framework` decision); each chart is a thin client island that lazily fetches `/api/chart/[slug]` on group-open (groups collapsed by default, faithful to v3's lazy-on-expand at `html/mod.rs`). The v3 `/api/artifacts/{generation}/groups/{slug}/shards/{i}` endpoint is **dropped**: its two jobs — per-group batch fetch + immutable `{generation}`-versioned caching — are covered in v4 by the existing `/api/group/[slug]` + `/api/chart/[slug]` routes + Next `revalidate=300` + edge CDN; `{generation}` (an in-process `read_model.rs` snapshot id, 8 retained generations) has no stateless analog on Vercel. Supersedes the PR-4.4 row's prior "(if kept) shard route" conditional. |
| Prod historical load TIMING (2026-06-05) | **Deferred from Phase 3 to the Phase 5 cutover.** Phase 3 builds + validates the full migration toolkit (loader / value-verify / rehearsal harness / cross-check, all green) AND runs a **real-snapshot LOCAL rehearsal** (the real v3 DuckDB -> a local postgres:16 -> load + verify + cross-check clean). The actual one-shot PROD load into RDS runs at cutover (Phase 5), reusing the SAME validated tools against the prod DSN. | User decision 2026-06-05. Seeding prod RDS in Phase 3 is premature: Phase 4 has not built the v4 reader yet, so any loaded history sits unread until cutover, and an irreversible prod write before cutover-commitment buys nothing. Loading at cutover yields a FRESHER snapshot (the best-effort dual-write soak since Phase 2 fills the gap), no stale prod data, and no premature prod side-effect. The prod load IS "run the existing Phase-3 tools against the prod DSN" (the user's framing) -- a deferred EXECUTION, not unbuilt work. The real-snapshot LOCAL rehearsal still de-risks NOW: it is the only check the synthetic fixtures cannot give (proof the ACTUAL v3 data shape loads + verifies clean). |

## Project-specific BANS

**REVIEW CALIBRATION (load-bearing — read first; 2026-06-04 lean re-plan).** This is a **benchmarks dashboard**: public-read, no user accounts, no payments. Its data comes exclusively from the project's own `vortex-bench --gh-json-v3` CI (TRUSTED input), and it is backed by four independent safety nets (the live v2 site, the live v3 system, a ≥90-day DuckDB snapshot, and Phase-3 `migrate --verify`). Calibrate findings accordingly:
- **DO flag:** data-correctness bugs (measurement_id / hash parity, NaN/Inf divergence, wrong upsert/ON CONFLICT semantics, schema/SCHEMA_VERSION drift, lost/duplicated rows), and "does it actually work" bugs (real runtime errors, broken CI, wrong query results, broken charts).
- **DO NOT flag (over-engineering for this artifact):** adversarial-input robustness on trusted CI/JSON/git data (NUL bytes, non-UTF-8, lone surrogates, `RecursionError` from nested JSON, oversized-integer literals, exotic `git` commit-metadata encodings); availability/pipeline-reliability hardening disproportionate to "a benchmark number is briefly stale" (the data is regenerable and v2/v3 stay live); test-completeness-of-test-completeness spirals.
- **Cycle cap:** inner-loop is capped at ~3 gauntlet cycles. A finding still open at cycle 3 is deferred (to `Deferred work`) or accepted — never spiraled into more cycles. Code PRs are 2-vote (fresh + correctness); only the final cutover phase is 3-vote. (Rationale + the PR-2.2 15-cycle spiral that motivated this: see the `Review calibration` Key-decision row.)

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
| 2 | Postgres writer + best-effort v4 CI | **(lean re-plan 2026-06-04)** `bench_ingest` role + grants (004); `--postgres` mode + IAM-auth + NaN/Inf guard; `scripts/` pytest in CI; **add a BEST-EFFORT v4 write to the 3 ingest workflows (v3 `--server` stays hard-required + untouched) + switch schema-deploy trigger to push-on-deploy-branch.** PR-2.1/2.2/2.3 DONE; only PR-2.4 remains. **DROPPED: PR-2.5 (reconcile-ingest.py + dual-write-verify.yml + incident.io).** | PR-2.1/2.2/2.3 accepted (done); PR-2.4: `yamllint --strict` clean; the 3 workflows add a `continue-on-error` v4 `--postgres` step (v3 step unchanged + required); `schema-deploy.yml` triggers `apply` on push to the deploy branch under `paths: migrations/**`. v4 correctness is gated by Phase-3 verify, not a Phase-2 reconciliation harness. | 4 | 2-vote |
| 3 | Historical data load (DuckDB → Postgres) + value-verify | **(re-plan 2026-06-05, folds in 4 audit gaps; 2026-06-05 prod-load deferred to Phase 5)** Build + validate the migration TOOLKIT: `vortex-bench-migrate --postgres-target` (atomic single-txn COPY, NO aws-sdk-rds) + 004-as-master ordering guard; **value-column verify** (per-`measurement_id` VALUE+`env_triple`+array+commits-metadata compare — the PRIMARY v4-correctness gate); local postgres:16 **rehearsal harness** + **Python-writer-vs-RDS cross-check** harness. Phase 3 closes on a **real-snapshot LOCAL rehearsal** (acquire the real v3 DuckDB off the EC2 host -> load + verify + cross-check against a LOCAL postgres:16). **The one-shot PROD load + prod verify + prod cross-check are DEFERRED to the Phase-5 cutover** (run the same tools against the prod DSN at cutover, freshest snapshot). | `migrate verify --postgres-target` reports 0 presence diffs AND 0 value-column / `env_triple` / `all_runtimes_ns`-array / `commits`-metadata mismatches; the synthetic-fixture rehearsal (PR-3.3) + the cross-check (PR-3.5) are green vs local PG16; **the REAL-snapshot LOCAL rehearsal (PR-3.4) loads the actual v3 DuckDB into a local PG16 and `verify` + cross-check report clean** (the real-data validation gate). The prod-load row-counts + prod cross-check are captured at Phase-5 cutover, NOT here. **(amend 2026-06-08: +PR-3.6 applies the 3 phase-end-cycle-1 should-fix doc/status items; +PR-3.7 clears the 3 phase-end-cycle-2 doc nits.)** | 7 | 3-vote |
| 4 | Next.js read service on Vercel | Scaffold `benchmarks-website/web/`; connection lib; port read endpoints as RSC/route handlers; port chart UI; deploy to Vercel. **(lean re-plan: TIME-BASED revalidation (`export const revalidate = ~300`, matching v2's 5-min S3 refresh) instead of push-based HMAC; DROPPED the standalone revalidate endpoint + writer hook + `REVALIDATE_SECRET` (was PR-4.6).)** **(2026-06-08: PR-4.3 split into PR-4.3.a/b/c; the shard endpoint deferred from PR-4.3 to PR-4.4 — see PR enumeration.)** **(2026-06-09: PR-4.4 split into PR-4.4.a [server shell + CSS] / PR-4.4.b [chart client island + interactivity + permalink] — faithful v2 UI port exceeds single-PR size.)** | `vercel deploy --target=preview` serves all chart slugs; `curl preview-url/api/groups \| jq '.groups[].charts[].slug' \| sort` matches the family registry; **charts match v2 for representative slugs (manual visual check)** | 8 | 2-vote |
| 5 | Cutover + decommission | **(2026-06-05: now opens with the deferred one-shot PROD historical load.)** Run the validated Phase-3 toolkit against PROD RDS (acquire freshest v3 snapshot -> `load` + `verify --postgres-target` + cross-check); then **promote v4 ingest to required + drop v3-write from CI**; DNS flip v2 → v4; delete v2 frontend + server/ + ops/ + migrate/ + publish-benchmarks-website.yml + bearer-token secrets | the prod load reports per-table row-counts + `verify` 0-diffs + cross-check clean against prod RDS (RDS PITR is the rollback); `git grep -n INGEST_BEARER_TOKEN` returns 0; `gh workflow list` does not include `publish-benchmarks-website`; production DNS resolves to Vercel; v3 EC2 terminated | 4 | 3-vote (final) |

Total: **5 phases, 30 PRs** (lean re-plan 2026-06-04: Phase 2 5→4 [dropped PR-2.5], Phase 4 6→5 [dropped PR-4.6]; amend 2026-06-05: Phase 2 +3 [PR-2.6/2.7/2.8]; re-plan 2026-06-05: Phase 3 3→5 [+PR-3.4, +PR-3.5], Phase 3 review-count 2→3-vote; **2026-06-05 prod-load deferral: PR-3.4 re-scoped prod-load → REAL-snapshot LOCAL rehearsal, and the prod load split out as new PR-5.0 [Phase 5 3→4]**; **2026-06-08 PR-4.3 split: PR-4.3 → PR-4.3.a/4.3.b/4.3.c [Phase 4 5→7], shard endpoint deferred PR-4.3→PR-4.4 per user fork decision**; **2026-06-09 PR-4.4 split: PR-4.4 → PR-4.4.a/4.4.b [Phase 4 7→8], shard endpoint DROPPED per the RSC-shell fork decision**). Done: Phase 1 (6) + Phase 2 (7) + **Phase 3 agent code: PR-3.1/3.2/3.3/3.5 complete + reviewed**. Remaining: **PR-3.4 (real-snapshot LOCAL rehearsal — NEXT, agent-doable), Phase 3 phase-end review/close**, Phase 4 (5), Phase 5 (4, incl. PR-5.0 prod load). The 4 Phase-3 audit gaps are folded into the PR enumeration below.

### Phase 3 re-plan decisions (2026-06-05)

Re-plan at the Phase 2→3 boundary (operator-chosen at the Step 3.4 gate) to fold in the 4
load-bearing gaps from the 2026-06-05 complexity & gap audit. Decisions are grounded in 3 parallel
exploration agents over `migrate/`, the DuckDB source + cross-account boundary, and the schema
value/dim/env columns:

- **Execution model: operator-run-locally with the RDS master password (NOT IAM/OIDC).** The OIDC
  `GitHubBenchmark{Schema,Ingest}Role`s are GitHub-Actions-only (trust policy federates only
  `token.actions.githubusercontent.com` for develop + ct/bench-v4 refs) and are NOT assumable from a
  laptop; the only human-usable RDS-write credential is the Secrets-Manager master password
  (`provision.sh:269-271`, `infra/README.md:113-157`). The v3 DuckDB lives in a DIFFERENT account's
  S3/EC2 (`vortex-benchmark-results-database`) with no cross-account bridge. So the one-shot load
  runs locally with two credential sets (v3-account S3 read for the snapshot + `245040174862` master
  password for the RDS write). **PR-3.1 DROPS `aws-sdk-rds`/IAM token-minting** — it connects with the
  master-password DSN over verify-full TLS. (Reverses the original PR-3.1 "sqlx-postgres +
  aws-sdk-rds" note; a minimal async client — e.g. `tokio-postgres` + a TLS connector — suffices.)
- **Source account/region is operator-verified live, not trusted from the audit.** `375504701696` /
  `us-east-2` for the v3 DuckDB appears ONLY in this plan doc — it is pinned NOWHERE in the repo
  (`provision.sh`, `ops/`, workflows all confirm RDS = `245040174862`/`us-east-1` but never the v3
  source). PR-3.3's rehearsal runbook adds a live verification step (`aws sts get-caller-identity` /
  `aws s3api get-bucket-location`) before the prod load. The loader takes `--duckdb <path>`; the
  operator acquires the snapshot (rehydrate the per-table Vortex S3 backup via `duckdb` + `INSTALL
  vortex`, OR scp the live `/var/lib/vortex-bench/bench.duckdb`; the S3 backup has a 7-day lifecycle).
- **Value-column verify shape (gap #1, PR-3.2 — the PRIMARY gate).** `measurement_id` hashes only
  `commit_sha` + the dim tuple, so a PK/count match pins ZERO bytes of any VALUE or ENV column.
  Verify joins DuckDB-source vs Postgres-target on `measurement_id` (1:1 — accumulators dedup by id)
  and compares, per row, every non-hashed column: `query_measurements`{`value_ns`, `all_runtimes_ns`,
  `peak_physical`, `peak_virtual`, `physical_delta`, `virtual_delta`, `env_triple`};
  `compression_times`{`value_ns`, `all_runtimes_ns`, `env_triple`}; `compression_sizes`{`value_bytes`}
  (no env); `random_access_times`{`value_ns`, `all_runtimes_ns`, `env_triple`};
  `vector_search_runs`{`value_ns`, `all_runtimes_ns`, `matches`, `rows_scanned`, `bytes_scanned`,
  `iterations`, `env_triple`}; `commits`{8 metadata columns} per `commit_sha`. `all_runtimes_ns` is
  an ordered `BIGINT[]` — compare element-wise. **Full** comparison (one-shot gate, not sampled).
  `env_triple` is the audit's "env corruption invisible to count-match" case — now explicitly compared.
- **Rehearsal substrate (gap #2, PR-3.3): local postgres:16 testcontainer** (operator-chosen). Same
  engine as RDS 16.4; exercises the actual correctness risk (COPY value fidelity, type coercion,
  `BIGINT[]` handling, verify logic) at zero AWS cost. The endpoint/TLS/master-password path was
  already live-verified in Phases 1-2; the prod load (PR-3.4) is then a DSN swap guarded by
  value-verify + PITR rollback.
- **Python-writer-vs-RDS cross-check (gap #3, PR-3.5): built in Phase 3, run in BOTH Phase 3 + Phase
  5** (operator-chosen). Python==Rust hashing is already CI-gated (golden==Python port); the
  cross-check's novel value is the LIVE property — the Python `post-ingest.py --postgres` writer
  round-trips against REAL RDS and UPDATEs (not duplicate-INSERTs) the Rust-seeded rows. The harness
  writes REAL envelopes matching existing seeded dim tuples (so it UPDATEs — no synthetic prod
  pollution), asserts `measurement_id` matches the seeded row + value/env columns round-trip. Run
  right after the prod seed (PR-3.5, earliest detection) AND re-run as the PR-5.1 pre-promotion gate.
- **004-as-master ordering (gap #4): preflight guard in `migrate-schema.py` (PR-3.1).** 004's `ALTER
  DEFAULT PRIVILEGES` self-grant needs a master-capable role; documented in the SQL, enforced
  nowhere. Add a preflight assertion (the connected role has the needed privilege before applying
  002/004) + a test that a non-master role fails loud with a clear error.
- **Review-count: Phase 3 → 3-vote** (up from 2) — it is now the primary v4-correctness gate
  (establishes the `measurement_id`s the whole upsert-not-duplicate invariant depends on), so it
  earns Spec + Correctness + Maintainability.

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
| PR-2.4 | 2 | **(lean re-plan 2026-06-04)** Add a BEST-EFFORT v4 write to `bench.yml`, `sql-benchmarks.yml`, `v3-commit-metadata.yml`: after the existing (unchanged, hard-required) `post-ingest.py --server` (v3) step, add a SECOND `post-ingest.py --postgres` (v4) step via OIDC → `GitHubBenchmarkIngestRole` marked `continue-on-error: true` (v4 is unproven during the soak; a v4 failure must NOT break the proven v3 pipeline — it is promoted to required in Phase 5). Add `id-token: write` + an assume-role step where missing (e.g. `v3-commit-metadata.yml`). AND switch `schema-deploy.yml` from `workflow_dispatch`-only to also push on the deploy branch under `paths: migrations/** + scripts/migrate-schema.py` (keep `workflow_dispatch` + `dry_run`), removing superseded `environment:`-gate comments | `.github/workflows/{bench.yml,sql-benchmarks.yml,v3-commit-metadata.yml,schema-deploy.yml}` | `yamllint --strict` clean; each ingest workflow has the v3 step (required) + a `continue-on-error: true` v4 `--postgres` step; `schema-deploy.yml` `on:` includes push to the deploy branch under `paths: migrations/**`; CloudWatch shows Postgres writes appearing per develop push (best-effort) |
| ~~PR-2.5~~ | 2 | **DROPPED (lean re-plan 2026-06-04).** Was: Postgres-side dual-write reconciliation (`reconcile-ingest.py` + `dual-write-verify.yml` + incident.io paging). Removed as disproportionate to a trusted-input low-stakes dashboard with four safety nets (v2 live, v3 live, ≥90-day DuckDB snapshot, Phase-3 `migrate --verify`). v4 correctness is gated by Phase-3's one-shot `migrate --verify` + a short manual soak spot-check (`psql-bench.sh` row-count + a few measurement_id lookups) — no standalone service, no paging. | (none) | n/a — dropped |
| PR-2.6 | 2 | **(amend 2026-06-05, audit must-fix)** Fix the LIVE gate-var bug: the 3 ingest workflows gate the v4 dual-write `if:` on `vars.RDS_BENCH_INSTANCE_ENDPOINT != ''` (live: SET) but assume-role uses `vars.GH_BENCH_INGEST_ROLE_ARN` (live: UNSET), so on the next `develop` push the v4 steps FIRE and FAIL at assume-role (swallowed by `continue-on-error`) instead of cleanly no-op'ing. Re-key all 9 v4 sub-step `if:` gates to `vars.GH_BENCH_INGEST_ROLE_ARN != ''` (the var that must exist for assume-role to succeed); keep the `inputs.mode == 'develop'` clause in `sql-benchmarks.yml`. | `.github/workflows/{bench.yml,sql-benchmarks.yml,v3-commit-metadata.yml}` | `yamllint --strict` clean; every v4 sub-step `if:` keys on `GH_BENCH_INGEST_ROLE_ARN`; with the role ARN unset the v4 steps no-op (do not fire), restoring dormant-until-wired behavior |
| PR-2.7 | 2 | **(amend 2026-06-05, audit de-gold-plate)** Remove trusted-input over-hardening from `scripts/post-ingest.py` (pre-lean-re-plan 15-cycle residue; stricter-than-v3-source on TRUSTED CI input per the calibration) + dead code: delete `_is_local_host` (dead; inline the loopback set into the one test fixture), the unreachable `RecursionError` branch in `read_records`, `_reject_unstorable_str` (NUL/lone-surrogate string guards), the `git_show_field` bytes-then-explicit-UTF-8 rewrite (revert to `subprocess(text=True)`), the `read_records` universal-newline/non-UTF-8 hardening, and the `_require_finite` `OverflowError` branch; delete the ~12 matching tests. **KEEP**: measurement_id parity, ON CONFLICT/RETURNING(xmax=0), retry, IAM/TLS auth, `deny_unknown_fields` + typed i32/i64/finite(in-range) validation, memory-quartet/storage-enum. | `scripts/post-ingest.py`, `scripts/test_post_ingest_postgres.py` | `uv run --all-packages pytest scripts/` green; the deleted guards + their tests gone; all measurement_id / ON CONFLICT / IAM-TLS / typed-validation tests still pass |
| PR-2.8 | 2 | **(amend 2026-06-05, retained merge-blocker + phase-end nits)** (1) Reconcile the bench-v4 Python files to the repo `ruff` line-length-120 + style (clear the pre-existing E501 at `test_migrate_schema.py:831` + the established multi-line style) — the RETAINED code-quality merge blocker. (2) Fix the 2 phase-end nits: correct the `benchmarks-website/infra/README.md` var-table PR-lineage mislabel (PR-2.2 → PR-2.4 for the ingest-workflow consumers of `RDS_BENCH_INSTANCE_ENDPOINT`/`GH_BENCH_INGEST_ROLE_ARN`); add a `post-ingest.py` comment noting the v4 step inherits the v3 step's `build_commit` git-history assumption. | `scripts/*.py`, `benchmarks-website/infra/README.md` | `ruff check scripts/` clean (no E501 on the bench-v4 files); README var-table references PR-2.4 for the ingest consumers; `uv run --all-packages pytest scripts/` green |
| PR-3.1 | 3 | **(re-plan)** Extend `vortex-bench-migrate` with `--postgres-target $DSN`: add a minimal async Postgres client (`tokio-postgres` + TLS connector for verify-full; **NO `aws-sdk-rds`** — operator-local master-password DSN) + per-table `COPY FROM STDIN` (text format; `BIGINT[]` via array literals) for `commits` + the 5 fact tables, reusing the existing accumulators + `vortex_bench_server::db::measurement_id_*`. **Atomic: all tables in ONE transaction** (BEGIN → COPY each → COMMIT) so a mid-load failure rolls back to empty, never half-seeded. **+ 004-as-master ordering preflight** in `migrate-schema.py` (assert the connected role can apply 002/004 before doing so; fail loud otherwise) | `benchmarks-website/migrate/Cargo.toml` (+`tokio-postgres` + TLS; NO aws-sdk-rds), `benchmarks-website/migrate/src/postgres.rs`, `benchmarks-website/migrate/src/main.rs` (CLI flag), `scripts/migrate-schema.py`, `scripts/test_migrate_schema.py` | Embedded-DuckDB unit tests (no Docker, green) cover the read + COPY-text pipeline (escaping incl. backslash/tab/newline, `BIGINT[]` array literals, NULLs, negative ints, UTC timestamp via CAST, f64 shortest-round-trip); 004-as-master preflight rejects a non-master role (testcontainer-gated) + marker-detection unit tests (no Docker). **The full Postgres-execution integration (6-table load + atomic mid-load rollback) is delivered by PR-3.3's rehearsal harness** (which owns the local postgres:16 container infra and asserts PR-3.2 verify-clean), to avoid standing up the Postgres-container test infra twice. |
| PR-3.2 | 3 | **(re-plan — PRIMARY v4-correctness gate)** Extend `migrate/src/verify.rs` + a Postgres read path to compare DuckDB-source vs Postgres-target **per `measurement_id`** (1:1): for each fact table compare every VALUE column + `env_triple` (`all_runtimes_ns` element-wise); for `commits` compare the 8 metadata columns per `commit_sha`. `VerifyReport` gains a value-mismatch list `{table, id, column, duckdb_value, pg_value}`. Full comparison (not sampled). `verify --postgres-target` exits non-zero on ANY presence diff OR value mismatch. **(2026-06-05 impl clarifications, recorded before review):** (a) the value gate lands as a NEW `PgVerifyReport`/`ValueMismatch` pair rather than overloading the unrelated v2-vs-v3 structural `VerifyReport` (single-responsibility; the structural report compares group/chart shape, not stored values); (b) `commits.timestamp` is compared as engine-independent **epoch microseconds** (DuckDB `epoch_us` vs Postgres `(extract(epoch ...) * 1e6)::bigint`, exact for whole-second git timestamps) to sidestep cross-engine timestamp text-rendering divergence; (c) `Cargo.toml` is NOT touched (PR-3.1 already added `postgres`/`native-tls`; value verify reuses them). | `benchmarks-website/migrate/src/verify.rs`, `benchmarks-website/migrate/src/postgres.rs` (expose `ColKind`/`column_kind`/`connect_postgres` as `pub(crate)`), `benchmarks-website/migrate/src/main.rs` (`Verify --postgres-target`) | **Comparison-core (no-Docker, PR-3.2):** unit tests prove discrimination — seed source==target -> clean; mutate ONE value column / `env_triple` / one `all_runtimes_ns` element (incl. reorder) / one `commits` field / NULL-vs-value -> exactly one mismatch naming the exact `(table, key, column, duckdb_value, pg_value)`; presence-only diffs caught **both directions**; epoch-microsecond timestamp pinned. **Live PG16 end-to-end (PR-3.3):** the `load -> verify exits 0; mutate -> exits non-zero` against a real postgres:16 is delivered by PR-3.3's rehearsal harness, which **owns the container infra and already asserts PR-3.2 verify-clean** (same PR-3.2/PR-3.3 boundary as PR-3.1's deferred Postgres-execution coverage; avoids standing the container infra up twice; Docker is unavailable in the dev env). |
| PR-3.3 | 3 | **(re-plan — bulk-load rehearsal, gap #2)** Build the rehearsal harness: an end-to-end integration test that loads a representative fixture DuckDB into a **local postgres:16 testcontainer** via PR-3.1's loader and asserts PR-3.2's verify is clean — proving the load+verify CODE works before any prod write. + Document the operator runbook for the REAL-snapshot rehearsal: snapshot acquisition (rehydrate the S3 Vortex backup or scp the live `bench.duckdb`) + **live source account/region verification** (`aws sts get-caller-identity`, `aws s3api get-bucket-location`) since `375504701696/us-east-2` is unverified in-repo | `benchmarks-website/migrate/tests/end_to_end.rs` (or a new `postgres_e2e.rs`), `benchmarks-website/migrate/README.md` (rehearsal + acquisition + verification runbook) | The end-to-end test loads a fixture DuckDB → local PG16 → `verify` clean (CI, fails loud if Docker absent per the established pattern); **also asserts all 6 tables' per-table row counts match the source AND that a forced mid-load failure leaves the target empty (atomic single-txn rollback) — the Postgres-execution coverage for PR-3.1's loader**; the runbook documents snapshot acquisition + the live source-account/region verification step + the local rehearsal procedure. **(2026-06-05 carry-forward from the PR-3.2 review, correctness/codex nit):** the fixture seeds at least one `commits` row with a sub-second (and ideally pre-1970) timestamp, so the live PG16 run confirms the Postgres `(extract(epoch from ts) * 1000000)::bigint` epoch equals DuckDB `epoch_us` to the microsecond — the one verify path the PR-3.2 no-Docker tests cannot exercise (Postgres-side epoch SQL is never executed without a container). |
| PR-3.4 | 3 | **(re-scoped 2026-06-05: REAL-snapshot LOCAL rehearsal — the prod load moved to Phase 5.)** Acquire the REAL v3 DuckDB off the EC2 host, load it into a LOCAL `postgres:16` (the existing PR-3.3 container path) via PR-3.1's loader, then run PR-3.2's `verify --postgres-target` + PR-3.5's cross-check locally. This is the real-data validation gate (the only check the synthetic fixtures cannot give) with ZERO prod risk. Capture per-table row counts + verify-clean + cross-check-clean in `Implementation status`. **Mostly agent-doable now** (see operational facts below). | (no production code; operational PR + logs in Implementation status) | The acquired real v3 DuckDB loads into a local PG16 with per-table row counts captured; `migrate verify --duckdb <snapshot> --postgres-target $LOCAL_DSN` reports 0 presence diffs AND 0 value mismatches; the PR-3.5 cross-check reports clean (writer UPDATEs, not duplicates) against the locally-loaded data. **Operational facts (verified 2026-06-05):** v3 EC2 host `ec2-18-219-54-101.us-east-2.compute.amazonaws.com` (SSH port 22 reachable + in known_hosts; an "AWS EC2" RSA key is in the ssh-agent; SSH user unconfirmed — try `ec2-user`/`ubuntu`; needs explicit approval per the auto-mode classifier); DuckDB at `/var/lib/vortex-bench/bench.duckdb` (live v3 Axum server on `:3000` holds it open -> for a CONSISTENT copy either briefly `systemctl stop` the v3 service then `cp bench.duckdb*` then restart, OR copy the live file + WAL and let DuckDB recover on open); v3 source region CONFIRMED `us-east-2` (was unverified). For the eventual Phase-5 prod load: prod RDS `vortex-bench-prod.c4f8qygk4xdp.us-east-1.rds.amazonaws.com:5432` is PubliclyAccessible; the `bench-prod` AWS CLI profile (connor-aws-cli) reaches account `245040174862`; RDS Proxy = `vortex-bench-proxy.proxy-c4f8qygk4xdp.us-east-1.rds.amazonaws.com`. |
| PR-5.0 | 5 | **(2026-06-05: the deferred one-shot PROD load — first Phase-5 cutover step.)** Run the validated Phase-3 toolkit against PROD RDS: acquire the freshest v3 snapshot, `load` it (atomic single-txn) over verify-full TLS with the master-password (or IAM) DSN, run `verify --postgres-target`, then the PR-3.5 cross-check. Capture per-table row counts + verify-clean + cross-check-clean. Production data seed (hard-to-reverse) -> RDS PITR (35-day) is the rollback. | (no production code; operational PR + logs; reuses PR-3.1/3.2/3.5 tools) | Prod RDS holds the full v3 history (per-table row counts in Implementation status); `migrate verify --duckdb <snapshot> --postgres-target $PROD_DSN` reports 0 presence diffs AND 0 value mismatches; the cross-check confirms the Python writer UPDATEs the seeded rows; the PITR rollback command is documented. |
| PR-3.5 | 3 | **(re-plan — Python-writer-vs-RDS cross-check, gap #3)** Build a cross-check harness: take a few REAL v3 envelopes whose dim tuples exist in the seeded data, run `post-ingest.py --postgres` against RDS, and assert each computed `measurement_id` matches a Rust-seeded row (→ UPDATE, 0-inserted/N-updated, `xmax != 0`) and the VALUE+`env_triple` columns round-trip (re-read + compare). Run it right after PR-3.4's seed (earliest detection). The harness is re-run as the PR-5.1 pre-promotion gate | `scripts/cross_check_python_writer.py` (or extend `post-ingest.py` test tooling), `scripts/test_*` (local-container discrimination test) | Local-container test: the harness reports UPDATE-not-INSERT + value round-trip on correct input, and FAILS when a value is deliberately wrong (discriminating). Operator runs it against prod RDS after PR-3.4; the run is logged clean (0 duplicate INSERTs; value columns match the seeded rows). |
| PR-3.6 | 3 | **(amend 2026-06-08 — phase-end cycle-1 should-fix sweep)** Apply the 3 should-fix doc/status items from the Phase-3 phase-end review: (a) add a `requires-superuser` subsection to `migrations/README.md` documenting the marker + the `rolsuper OR rolcreaterole` preflight + the "apply 002/004 as RDS master before any migrator deploy" ordering (the operator-facing doc the `_assert_master_capable` PermissionError points at); (b) annotate the PR-3.1 Implementation-status entry with the deliberate sync-`postgres`-over-plan-named-`tokio-postgres` choice; (c) annotate the PR-3.4 Implementation-status entry that `vector_search_runs` had 0 real rows + was omitted from the cross-check envelope, so that table's real-data coverage is fixture-only (PR-5.0 closes it). | `migrations/README.md` (the reviewable code/doc change); `.big-plans/` Implementation-status entries (plan edits) | `migrations/README.md` has a requires-superuser subsection matching the `_assert_master_capable` error pointer; the PR-3.1 + PR-3.4 status entries carry the two annotations; no behavior change (doc/status only). |
| PR-3.7 | 3 | **(amend 2026-06-08 — phase-end cycle-2 nit sweep)** Clear the 3 cycle-2 doc nits: (a) align the hypothetical directive example in `migrations/README.md` authoring rules from `-- migrate: no-transaction` to `-- migrate-schema: no-transaction` (match the only-implemented directive's `migrate-schema:` namespace); (b) add one sentence to `benchmarks-website/migrate/src/postgres.rs` `pub fn load` doc comment noting the target schema (migrations/001) must already be applied (it only COPYs into existing tables); (c) add a one-line note to the `migrate/README.md` rehearsal runbook (§2 Acquire the snapshot) that a pre-acquired on-disk snapshot is an acceptable rehearsal source (matching PR-3.4's execution). | `migrations/README.md`, `benchmarks-website/migrate/src/postgres.rs` (doc comment), `benchmarks-website/migrate/README.md` | the 3 nits are resolved; `cargo +nightly fmt --check -p vortex-bench-migrate` clean (doc-comment edit); no behavior change (doc-only). |
| PR-4.1 | 4 | Scaffold `benchmarks-website/web/` Next.js 15 project: `package.json`, `tsconfig.json` (matching `vortex-web/`'s strict flags), `next.config.js`, `.gitignore`, SPDX headers everywhere | `benchmarks-website/web/{package.json,tsconfig.json,next.config.js,.gitignore}`, `app/layout.tsx`, `app/page.tsx` (stub) | `pnpm install && pnpm build` succeeds; lints clean. |
| PR-4.2 | 4 | Connection lib at `web/lib/db.ts`: `pg.Pool` + `@aws-sdk/rds-signer` for IAM token gen + token-refresh-before-expiry; expose `sql` tagged-template helper | `web/lib/db.ts`, `web/lib/db.test.ts` | Integration test (testcontainers Postgres): pool connects via password (test fixture); pool roundtrips a SELECT. IAM-auth path mocked. |
| PR-4.3.a | 4 | **(2026-06-08 split of PR-4.3, foundation + health)** Read-port foundation: `web/lib/schema-version.ts` (`export const SCHEMA_VERSION = 1` — the Table D read-path gate site), `web/lib/slug.ts` (encode/decode the 5 `ChartKey` + 5 `GroupKey` variants as `<prefix>.<base64url-json>`, matching `server/src/slug.rs` prefixes), `web/lib/window.ts` (`?n=` → commit window: default 100, numeric clamp `[1, 1000]`, `all` unbounded — ports `server/src/api/window.rs`), `web/lib/families.ts` (5-fact-table registry + metadata), and the `/health` route (preserve the `HealthResponse` snake_case shape; adapt `db_path`→DB host, `build_sha`→`VERCEL_GIT_COMMIT_SHA`/"unknown", `schema_version`→the const). | `web/lib/{schema-version,slug,window,families}.ts`, `web/app/api/health/route.ts`, `web/lib/*.test.ts` | slug round-trips for all 10 key variants (decode∘encode = id, prefixes match `slug.rs`); `?n=` parsing matches `window.rs` (default 100, clamp 1..1000, `all`); `/health` returns the snake_case shape with correct `row_counts` keys + `schema_version = 1`; vitest green. |
| PR-4.3.b | 4 | **(2026-06-08 split of PR-4.3, chart endpoint)** `web/lib/queries.ts` `chartPayload` — the two-pass seeded-commit-window port for all 5 chart types (`query_measurements`, `compression_times`, `compression_sizes`, `random_access_times`, `vector_search_runs`), Postgres-parameterized (`IS NOT DISTINCT FROM` nullable-dim equality; oldest-first `commits[]`; `series` map + `series_meta` + `unit_kind` + `ChartHistory`), and the `/api/chart/{slug}` route with `export const revalidate = 300`. | `web/lib/queries.ts`, `web/app/api/chart/[slug]/route.ts`, `web/lib/queries.test.ts` | `/api/chart/{slug}` returns the exact `ChartResponse` snake_case shape (`display_name`/`unit_kind`/`history`/`commits`/`series`/`series_meta`) byte-equivalent to the Axum server for representative fixtures (snapshot test vs `server/tests/chart_api.rs` + `server/fixtures/`); `measurement_id` absent from the wire; `?n=all` uncapped, numeric capped at 1000. |
| PR-4.3.c | 4 | **(2026-06-08 split of PR-4.3, groups + group endpoints)** `collectGroups` discovery over the 5 families (slug generation + `GROUP_ORDER` sort), `web/lib/summary.ts` (4 summaries: `randomAccess` rankings, `compression` speedup geomean, `compressionSize` ratio distribution, `queryBenchmark` rankings — ports `server/src/api/summary.rs`), `web/lib/descriptions.ts` (editorial blurbs ported from v2), and the `/api/groups` + `/api/group/{slug}` routes (`revalidate = 300`). | `web/lib/{queries,summary,descriptions}.ts`, `web/app/api/groups/route.ts`, `web/app/api/group/[slug]/route.ts`, tests | `/api/groups` returns `{groups:[Group…]}` with `GROUP_ORDER` sort + correct `Summary` tagged-union shape (camelCase variant fields per `dto.rs`) + descriptions; `/api/group/{slug}` returns `GroupChartsResponse` with flattened `NamedChartResponse` charts byte-equivalent to Axum for representative groups (snapshot vs `server/tests/group_api.rs`). |
| PR-4.4.a | 4 | **(2026-06-09 split of PR-4.4, server shell + CSS)** **(port-source refinement 2026-06-09: the port source is v3's server-rendered HTML layer — `server/src/html/{render,landing,summary}.rs` + `server/static/style.css` — NOT v2's React, because v3 is ALREADY the server-shell+client-island model + reproduces the v4 endpoints' ns data shape + uses v2's CSS class vocabulary; v2 React stays the secondary visual cross-check. v3 folded v2's Sidebar into the header nav, so NO separate Sidebar component — scope unchanged.)** Server-rendered landing shell: `web/app/layout.tsx` (`<html>`/`<head>` fonts + favicons + theme-bootstrap inline script + `globals.css` import; `<body>`), `web/app/page.tsx` (server component calling `collectGroups()` which embeds per-group summary + description, rendering one collapsible `<details>` `<section.group-details>` per group in `GROUP_ORDER` — disclosure header = group name + ⓘ info-icon + chart count; summary card above the chart grid; an empty chart-card mount point per chart carrying `data-chart-slug` + a stable per-page `data-chart-index` + an empty `<canvas>`), the static server-component pieces (`web/components/Header.tsx` = logo/title/GitHub static chrome [interactive nav/theme/filter deferred to 4.4.b], `web/components/GroupSection.tsx`, `web/components/SummaryCard.tsx` = port of `summary.rs` 4 variants incl. `formatTimeNs`, `web/components/Footer.tsx` = build-SHA via `VERCEL_GIT_COMMIT_SHA`), `web/lib/format.ts` (`formatTimeNs`, ns-based, ported from `summary.rs::format_time_ns`), and `web/app/globals.css` ported from v3 `server/static/style.css`. Native `<details>` gives working per-group expand/collapse with NO JS. NO chart canvas / interactivity yet (chart-card shells are empty mount points). Preserve UI BANS that apply to the shell. | `web/app/{layout,page}.tsx`, `web/components/{Header,GroupSection,SummaryCard,Footer}.tsx`, `web/lib/format.ts`, `web/app/globals.css`, tests | landing page renders server-side for all groups in `GROUP_ORDER` with the correct summary card (ns values via `formatTimeNs`) + description per group + one chart-card shell per chart (each carrying `data-chart-slug`); structure matches v3's landing/summary layout (v2-equivalent) on a manual visual check; `next build` + `tsc` + `eslint` + `prettier` + vitest green. |
| PR-4.4.b | 4 | **(2026-06-09 split of PR-4.4, chart client island + interactivity + permalink)** The interactive surface as client islands: `web/components/Chart.tsx` (`'use client'` Chart.js line chart — lazily fetches `/api/chart/[slug]` on group-open/visible, LTTB downsample over the cached payload, range strip + zoom/pan toolbar, custom tooltip positioner; ports v2 `ChartContainer.jsx` + the relevant `chart-init.js` behavior), `web/components/FilterBar.tsx` (`'use client'` chip toggles over the filter universe), `web/components/Modal.tsx` (expanded-chart modal), shared `web/lib/chart-format.ts` (ports v2 `utils.js` — `formatDate`/`stringToColor`/LTTB), and the `web/app/chart/[slug]/page.tsx` permalink page. **Also owns the deferred-from-PR-4.4.a header interactivity**: the mobile nav (hamburger `.nav-controls` + `.nav-controls-github` mobile GitHub fallback — resolves the PR-4.4.a Deferred-work mobile-GitHub-link gap), expand/collapse-all, theme toggle + the theme-bootstrap inline script. Preserve ALL UI BANS (oldest-first `commits[]` predecessor walk `idx-1`; no `pointer-events:auto` on tooltip host; throttled `input` not `change` on sliders; no refetch on pan/zoom/slider beyond the one-shot `?n=all` hop). **NO shard route.** | `web/components/{Chart,FilterBar,Modal}.tsx`, `web/lib/chart-format.ts`, `web/app/chart/[slug]/page.tsx`, the header-interactivity island(s), tests | **(lean re-plan: relaxed acceptance)** charts match the current v2 site for ~5 representative slugs on a manual visual check (dropped the byte-equivalent + lighthouse≥90 + ≤5%-pixel-diff bars as over-specified for a benchmarks dashboard); the mobile GitHub link is present <768px; all UI BANS preserved (verified against the diff); `next build` + `tsc` + `eslint` + `prettier` + vitest green. |
| PR-4.5 | 4 | Vercel deploy config (`vercel.json`); GitHub Action `web-deploy.yml` for preview-per-PR + production deploy on merge | `vercel.json`, `.github/workflows/web-deploy.yml` | PR opens trigger preview deploy; merging to ct/bench-v4 triggers prod deploy (still behind dev-only Vercel domain at this stage). |
| ~~PR-4.6~~ | 4 | **DROPPED (lean re-plan 2026-06-04).** Was: a push-based HMAC revalidate endpoint (`REVALIDATE_SECRET`) + a `post-ingest.py` revalidation hook. Removed in favor of time-based `revalidate` (PR-4.3) — ~5-min staleness matches v2's existing behavior, and a push pipeline + shared secret is unnecessary machinery for a benchmarks dashboard. | (none) | n/a — dropped |
| PR-5.1 | 5 | **Promote the v4 `--postgres` step to required (remove its `continue-on-error`)** + drop the v3 `--server` write from the 3 CI workflows; `post-ingest.py` runs only with `--postgres`. **Pre-promotion gate (Phase-5 half of audit gap #3): re-run the PR-3.5 Python-writer-vs-RDS cross-check against accumulated soak data and confirm clean BEFORE removing `continue-on-error`.** | `.github/workflows/{bench.yml,sql-benchmarks.yml,v3-commit-metadata.yml}`, `scripts/post-ingest.py` (remove --server mode) | The PR-3.5 cross-check re-run is clean (Python writer UPDATEs seeded rows, no duplicates) BEFORE promotion; v4 step no longer `continue-on-error` (now gates CI); v3 step removed; workflows green; CloudWatch shows zero traffic to v3 EC2. |
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
10. **Cross-account one-shot migration**: P=low-med; impact=moderate. v4 RDS lives in `245040174862/us-east-1` (bench account); v3 EC2 + DuckDB live in `375504701696/us-east-2` (personal account). Phase 3 one-shot DuckDB→Postgres load crosses BOTH account and region boundaries. Mitigation options: (a) write a DuckDB snapshot from `375504701696/us-east-2` to a cross-account-accessible S3 bucket (either in `245040174862` with a bucket policy permitting `375504701696` to PutObject, or vice versa), then download from `245040174862` and load locally to RDS; (b) operator runs the migrator with both profiles configured (`AWS_PROFILE` switching for read vs write); (c) temporarily grant `375504701696` IAM identity an `rds-db:connect` role in `245040174862` for direct write. Steady-state CI ingest after Phase 3 is unaffected (GH Actions OIDC into `245040174862` already works). **Chosen (re-plan 2026-06-05):** a blend of (a)+(b) — the operator acquires a DuckDB snapshot from the v3 side (rehydrate the S3 Vortex backup or scp the live `bench.duckdb`) and runs the loader LOCALLY against prod RDS with the `245040174862` Secrets-Manager **master password** over verify-full TLS. NOT (c): the OIDC `rds-db:connect` roles are GitHub-Actions-only and not assumable from a laptop, so direct IAM write is unavailable off-Actions. PR-3.3's runbook verifies the source account/region live (`375504701696/us-east-2` is unverified in-repo). The one-shot load is atomic (single txn) with RDS PITR as the rollback.

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

### PR-2.1: (re-plan) Foundational ingest identity — `bench_ingest` role + 6-table DML grants + `GitHubBenchmarkIngestRole`  (6 code/fix commits across 2 inner-loop gauntlet cycles, ending at 44245c4a8)
- Scope shipped: `migrations/004_ingest_role.sql` creates the least-privilege `bench_ingest` IAM role (USAGE + per-table SELECT/INSERT/UPDATE on the 6 data tables, no DELETE/DDL) plus a default-privilege rule so future `migrator`-created tables auto-grant the ingest role; `provision.sh` adds `GitHubBenchmarkIngestRole` (OIDC, `rds-db:connect` for `bench_ingest`) and drops the dead RDS-Proxy grant on the schema role.
- Tests added: `test_real_migrations_create_bench_ingest_role`, `test_bench_ingest_has_dml_only_on_data_tables`, `test_bench_ingest_can_upsert_and_is_denied_ddl_delete`, `test_bench_ingest_default_privileges_cover_future_migrator_tables`, and (cycle-1 must-fix regression) `test_real_migrations_apply_as_non_superuser_createrole_master` (applies 001..004 as a REAL non-superuser CREATEROLE login).
- Review: 2-vote (pr-2: fresh + correctness, claude-routed) / accepted (cycles: 2). Cycle 1 REJECT (1 must-fix: 004 ADP failed on the non-superuser RDS master). Cycle 2 ACCEPT (0 must-fix; 3 should-fix + 2 nit — cheap ones applied in f3c5771e8/44245c4a8, 2 deferred).
- Confidence: high. The must-fix was a subtle prod-gating bootstrap failure; the fix (guarded INHERIT self-grant via the master's ADMIN option, then revoke) was validated end-to-end against a local PostgreSQL 16.14 cluster — RED pre-fix, GREEN post-fix for BOTH a real non-superuser master AND the superuser path; role graph restored (master retains only its ADMIN auto-grant). The cycle-2 correctness reviewer independently confirmed the REVOKE removes only the self-grant (separate `pg_auth_members` records by grantor).
- Deferred items: 2 (cycle-2: master-lacks-ADMIN-on-pre-existing-`migrator` negative test; `createrole_self_grant='inherit'` no-op-branch test — both test-coverage hardening for an unsupported / conceptually-validated path). Also resolved 1 prior deferred item (PR-1.6 proxy-grant cleanup, via `provision.sh` in commit 50a8d4a08).
- Surprises during implementation: (1) the cycle-1 reviewer's RECOMMENDED fix (`SET ROLE migrator` wrapper) was EMPIRICALLY DISPROVEN — it also fails on a real non-superuser master (`SET ROLE` is checked against the session user; the PG16 CREATEROLE auto-grant has SET FALSE). The shipped fix is the guarded INHERIT self-grant. (2) Docker/testcontainers are unavailable locally, so validation used a local Homebrew PG16.14 cluster + the real runner; CI runs the canonical testcontainer suite. (3) Discovered (OUT OF SCOPE, pre-existing, predates Phase 2): the bench-v4 Python files are ~100-col and fail the repo's `ruff check .` / `ruff format --check .` (line-length 120) — e.g. the pre-existing E501 at `scripts/test_migrate_schema.py:798` (commit 74175ec85d). Not fixed here; reformatting the whole file is unrelated to the ADP must-fix and conflicts with the file's established style. Flag for the operator: the bench Python files need a branch-wide ruff reconciliation before merge to `develop`.

### PR-2.2: `post-ingest.py --postgres` IAM-upsert writer + tests  (impl `0b2ba4b7d` + many fix commits across 15 inner-loop gauntlet cycles, ending at `06087c784`)
- Scope shipped: `scripts/post-ingest.py` gains a `--postgres $RDS_DSN` mode that parses the JSONL, computes `measurement_id` via `_measurement_id.py`, mints an RDS IAM token (boto3) for the least-privilege `bench_ingest` role, connects over verify-full TLS (asserting `conn.pgconn.ssl_in_use` post-connect), upserts `commits` first then the 5 fact tables via `INSERT ... ON CONFLICT (measurement_id) DO UPDATE` in one all-or-nothing transaction, retries on Postgres deadlock/serialization conflict, pins `search_path=public`, and applies an `is_finite()` guard on f64 dims at the ingest boundary. The v3 `--server` path is preserved (stdlib-only under bare `python3`); `SCHEMA_VERSION` kept in lockstep.
- Behavior-preservation verified byte-for-byte against the v3 server boundary: `_RECORD_FIELDS`/`_FIELD_TYPES` mirror `records.rs` (serde `tag="kind"` + `deny_unknown_fields`, per-field i32/i64/f64/str type+range validation); the 5 `ON CONFLICT SET` lists mirror `ingest.rs` (dim columns excluded); the `measurement_id` port matches `db.rs` Table A/B field order + tags; duplicate-JSON-key last-wins matches v3; commit-upsert-first then per-record (commit_sha-match -> value validation -> apply) ordering mirrors `apply_envelope_once`.
- Tests added: 102-test `scripts/test_post_ingest_postgres.py` (pure-unit + testcontainers Postgres) — insert-then-update idempotency, measurement_id==`_measurement_id.py`, NaN/Inf rejection + rollback, all 5 ON CONFLICT SET lists, deny_unknown_fields, type/range boundaries, parse-boundary hardening (universal newlines + ValueError/RecursionError/UnicodeDecodeError), record+commit string storability (NUL/non-UTF-8/lone-surrogate), write-conflict retry, verify-full TLS actually-in-use (incl. a container test pinning the `conn.pgconn` traversal), least-privilege enforcement, search_path pinning, CLI composition, and the `git_show_field` UTF-8 decode boundary with a mutation-pinned subprocess bytes-mode contract guard (`_assert_git_show_bytes_mode` covers text/universal_newlines/encoding/errors).
- Review: 2-vote (pr-2: fresh + correctness, `executor=parallel` Claude+Codex per lens) / **accepted at cycle 15** (15 inner-loop cycles; a 7-cycle early-break at cycle 7 -> operator Continue; then operator-directed fix+review at cycles 12/13/14 and a standing-autonomy land-if-clean at cycle 15). Cycle 15: 0 must-fix; both Claude lenses proved the bytes-mode contract exhaustive against CPython's `Popen.__init__` source; 1 non-blocking should-fix deferred.
- Confidence: high. Genuinely production-breaking bugs were caught and fixed mid-loop (cycle-9 `conn.info.ssl_in_use` AttributeError regression -> `conn.pgconn.ssl_in_use`, caught by fresh/claude verifying the real psycopg API; cycle-12 `git_show_field` non-UTF-8 decode). The full testcontainer suite (102) passes; the production decode/validation boundaries are mutation-verified.
- Deferred items: 3 (PR-2.2 cycle-4 destructive-scrub loopback guard; cycle-7 real-deadlock + `autocommit=False` container coverage; cycle-15 `test_server_mode_requires_benchmark_id` isolation) — all test-hardening, **Resolved-by PR-2.3**. Accepted tradeoffs recorded (dup-key last-wins; PEP 723 `dependencies=[]`; IAM region `--region`>session>host precedence).
- Surprises during implementation: (1) the multi-vote/disjoint-coverage approach earned its keep — Codex lenses probed adversarial inputs + test-discriminating-power while Claude lenses verified invariants against real library APIs and ran the full suite; the cycle-9 regression was caught only because fresh/claude checked the real psycopg accessor against mocks shaped to the wrong API. (2) The last ~3 cycles converged into a diminishing-returns spiral on the `git_show_field` regression test's contract-modeling completeness (text=True -> the universal_newlines alias), terminated once the contract was proven exhaustive against CPython source. (3) The pre-existing bench-v4 ruff line-length-120 reconciliation (flagged in PR-2.1) still pends before the `develop` merge.

### PR-2.3: (re-plan, CI-hardening) wire `scripts/` pytest into CI + fail-loud-on-no-Docker  (3 commits across 1 inner-loop gauntlet cycle, ending at `d21742dc4`)
- Scope shipped: a `scripts-test` job in `.github/workflows/ci.yml` runs `uv run --all-packages pytest scripts/` on the amd64-large runner (Docker present; `ubuntu-latest` fork fallback), mirroring the `python-test` job's `runs-on/action` sccache + `setup-prebuild` pattern, with a `docker info` fast-fail gate. The two testcontainer suites (`test_migrate_schema.py`, `test_post_ingest_postgres.py`) now FAIL LOUD in CI instead of silently skipping when Docker is absent, via a per-file `_require_docker_for_testcontainers()` (CI env set -> `pytest.fail`; local -> `pytest.skip`). Resolves the deferred scripts/-not-CI-gated items (PR-1.2, PR-1.5: golden==Python now gated).
- Tests added: `test_require_docker_fails_loud_in_ci` + `test_require_docker_skips_without_ci` in BOTH test files (monkeypatch `CI` + `_docker_available`); the improved `test_server_mode_requires_benchmark_id` (resolves the PR-2.2 cycle-15 deferral). Full suite 217 passed.
- Review: 2-vote (pr-2: fresh + correctness, `executor=parallel`) / **accepted at cycle 1**. 0 must-fix. 1 should-fix (correctness/claude): the fail-loud guard tests could not catch the always-skip regression they guard -- `pytest.skip()` raises `Skipped` (not a `Failed` subclass), so an always-skipping helper escaped `pytest.raises(Failed)` and registered as a green test-skip; mutation-verified. Applied in `d21742dc4` (catch both outcome types + assert the specific one; re-verified by mutation). 2 nits: pytest-scripts-collects-a-4th-file (applied a clarifying ci.yml comment); per-file helper duplication (dismissed -- accepted pre-existing pattern).
- Confidence: high. The central CI/test risk (a green job that silently skips the testcontainer tests) is closed by a double layer (the `docker info` job gate + the CI-env fixture fail-branch), end-to-end-verified (217 tests run with zero skips when `CI=true` + Docker present). The disjoint-coverage approach again earned its keep: correctness/claude found the ironic guard-test silent-skip gap that the other 3 lenses missed.
- Commits: `8402c1990` (CI job + fail-loud + guard tests), `ed585f451` (--server benchmark-id isolation), `d21742dc4` (guard-test should-fix + ci.yml scope comment).
- Scope decision (explicit): the larger PR-2.2 cycle-7 real-deadlock/`autocommit=False` container tests + cycle-4 destructive-scrub guard are RE-MAPPED to a follow-up test-hardening PR (PR-2.3 delivered the CI prerequisite that makes them run; the real-deadlock test needs threaded 2-connection interleaving warranting focused attention). Recorded in Deferred work.
- Surprises during implementation: (1) the guard-test silent-skip gap is the same failure class PR-2.3 closes, one level up in the test that guards it -- a nice illustration of why the fail-loud invariant needs its own discriminating test. (2) `pytest scripts/` is broader than the 3 named suites (also `scripts/tests/test_benchmark_reporting.py`); harmless + matches the acceptance command. (3) The pre-existing bench-v4 ruff line-length-120 reconciliation (1 E501 at `test_migrate_schema.py:831` + the file's established multi-line style) still pends before the `develop` merge (OPEN ITEM since PR-2.1); new lines were kept ruff-clean.

### PR-2.4: (lean re-plan) best-effort v4 Postgres dual-write in 3 ingest workflows + schema-deploy push trigger  (1 code commit + 1 inner-loop gauntlet cycle, ending at `d8bbeb6ed`)
- Scope shipped: after the unchanged hard-required v3 `post-ingest.py --server` step, added a SECOND best-effort `post-ingest.py --postgres` step (3 sub-steps: OIDC assume `GitHubBenchmarkIngestRole` -> install uv -> ingest) to `bench.yml`, `sql-benchmarks.yml`, and `v3-commit-metadata.yml`, all `continue-on-error: true` and gated on `vars.RDS_BENCH_INSTANCE_ENDPOINT != ''` (plus `inputs.mode == 'develop'` in the reusable sql-benchmarks.yml). Added `id-token: write` to `v3-commit-metadata.yml`. Switched `schema-deploy.yml` from `workflow_dispatch`-only to also `push` on `develop` under `paths: [migrations/**, scripts/migrate-schema.py]` (kept `workflow_dispatch` + `dry_run`); rewrote the now-superseded `environment:`-gate header comment. The v4 ingest mints the RDS IAM token internally via boto3; DSN uses `sslmode=verify-full` + the downloaded RDS global CA bundle; deps via `uv run --no-project --with psycopg[binary]/boto3/xxhash`. Live-v2 S3 path untouched.
- Tests added: none (CI-workflow-only change; acceptance is `yamllint --strict` clean + the structural criteria). v4 ingest correctness is gated by the Phase-3 `migrate --verify` harness, not a Phase-2 reconciliation step (PR-2.5 dropped in the lean re-plan).
- Review: 2-vote (pr-2: fresh + correctness, `executor=claude` per lean calibration) / **accepted at cycle 1**. 0 must-fix, 0 should-fix. 2 nits, both dismissed: (a) redundant AWS region double-sourcing (`AWS_REGION` env + explicit `--region`) across all 3 workflows -- harmless, resolves correctly under the accepted IAM-token region-precedence carry-forward; (b) the v4 step also fires from `nightly-bench.yml` develop-mode cron -- intentional + symmetric with the v3 step's identical gate.
- Confidence: high. Both lenses independently verified end-to-end against the real repo: OIDC `id-token: write` is present at every entry point (bench.yml + v3-commit-metadata.yml at workflow level; the reusable sql-benchmarks.yml inherits it from its develop-mode callers bench.yml/nightly-bench.yml, while PR-mode callers gate the v4 step off via `inputs.mode == 'develop'`); the `post-ingest.py --postgres` CLI signature matches the invocation exactly (positional jsonl + `--postgres` DSN + `--commit-sha` + `--region`; `--benchmark-id` genuinely unused in `--postgres` mode); the v3 step writes `results.v3.jsonl` and the v4 step reads the same path; `v3-commit-metadata.yml` self-writes `empty.jsonl` before reading it (commit-only upsert); the DSN uses the `bench_ingest` role (matching the enforced `_INGEST_ROLE` guard) + `sslmode=verify-full`; the schema-deploy push paths (`migrations/**`, `scripts/migrate-schema.py`) are correct relative to repo root; no token/`PGPASSWORD` echo; `if: failure()` incident.io alert correctly will not fire on a best-effort v4-only failure.
- Commits: `d8bbeb6ed` (the single CI-workflow code commit).
- Deferred items: 0 new. (Retained backlog unchanged: bench-v4 ruff reconcile + Phase-1 live OIDC apply gate.)
- Surprises during implementation: none. OPEN DEPENDENCY for the live v4 write (operator-side, not in-session): operator must set repo vars `GH_BENCH_INGEST_ROLE_ARN` + `RDS_BENCH_INSTANCE_ENDPOINT`/`RDS_BENCH_DB_NAME`/`RDS_BENCH_REGION` (the v4 steps no-op until then), and CloudWatch should confirm Postgres writes appearing per develop push.

### PR-2.6: (amend 2026-06-05) fix the live gate-var bug — gate v4 dual-write on GH_BENCH_INGEST_ROLE_ARN  (1 code commit + 1 inner-loop gauntlet cycle, ending at `077470973`)
- Scope shipped: re-keyed all 9 v4 dual-write sub-step `if:` gates (3 each in `bench.yml`, `sql-benchmarks.yml`, `v3-commit-metadata.yml`) from `vars.RDS_BENCH_INSTANCE_ENDPOINT != ''` (set live) to `vars.GH_BENCH_INGEST_ROLE_ARN != ''` (the assume-role input, unset live), restoring the dormant-until-wired behavior; `sql-benchmarks.yml` keeps its `inputs.mode == 'develop' &&` prefix. Endpoint var stays in `env:`/DSN (the DSN host). Updated the 3 explanatory comments. Surfaced by the 2026-06-05 complexity/gap audit (the bug: gate keyed on the wrong var, so the steps fired+failed at assume-role instead of no-op'ing).
- Tests added: none (CI-workflow gate change; acceptance is `yamllint --strict` clean + grep-verified gate re-key).
- Review: 2-vote (pr-2: fresh + correctness, `executor=claude`) / **accepted at cycle 1**. 0 must-fix, 0 should-fix, 1 nit dismissed (optional both-vars gate hardening — the role-ARN-set-but-endpoint-unset edge case fails loud + swallowed + is not a real wiring path; provision.sh emits both vars together). Both lenses verified end-to-end against the real repo: all 9 gates re-keyed, 0 stale gates, no non-v4 step touched (schema-deploy.yml's endpoint PGHOST ref is the unrelated migrator path), endpoint var retained in env:/DSN, yamllint clean.
- Confidence: high. Both lenses independently judged role-ARN-alone is the correct single gate (it is precisely the assume-role input). Live-var state confirmed the bug (endpoint SET, role ARN UNSET) and confirms the fix (gate now evaluates false → steps no-op until the operator wires the role ARN).
- Commits: `077470973` (the single CI-workflow gate fix).
- Deferred items: 0 new.
- Surprises during implementation: none.

### PR-2.7: (amend 2026-06-05) de-gold-plate post-ingest.py — drop trusted-input over-hardening + dead code  (1 impl commit + 1 fix commit across 2 inner-loop gauntlet cycles, ending at `b6d1f292b`)
- Scope shipped: removed ~35% trusted-input over-hardening (net -194 lines) per the 2026-06-05 audit: `read_records` back to a plain text-mode iterator (dropped bytes/splitlines/explicit-UTF-8/RecursionError/oversized-int machinery; still fails loud `path:line` on bad JSON); `git_show_field` reverted to `subprocess(text=True)`; `_require_finite` OverflowError sub-branch dropped (NaN/Inf guard kept); deleted `_reject_unstorable_str`+`_require_storable_str` (NUL/lone-surrogate guards) + their calls in `_require_str`/`_require_opt_str`/`_upsert_commit`; deleted dead `_is_local_host` (inlined the loopback check into the one test fixture). Deleted the ~8 corresponding tests. KEPT all load-bearing code: measurement_id parity, ON CONFLICT/RETURNING(xmax=0), retry, IAM/TLS auth, deny_unknown_fields + typed i32/i64/finite(in-range) validation, memory-quartet/storage-enum.
- Tests added: 2 minimal replacements (mixed-newline happy path, malformed-JSON loud-fail) + a simplified git_show text-mode decode test; net test delta -127 lines.
- Review: 2-vote (pr-2: fresh + correctness, `executor=claude`) / **cycle 1 REJECT (1 must-fix), cycle 2 ACCEPT (0 findings)**. Cycle-1 correctness caught an INCOMPLETE deletion: the OverflowError-branch removal left an orphaned `pytest.param(10**309, id='out_of_range_int')` in the KEPT testcontainer test `test_nonfinite_threshold_raises_and_rolls_back`, which would error (uncaught OverflowError) in CI (it only skipped locally for lack of Docker). Fresh missed it (only checked deleted tests). Fixed in `b6d1f292b` (dropped the param; nan/inf/-inf retained). Cycle 2: both lenses accept — fix complete, no other dangling refs, behavior-preservation vs v3 holds (the removed guards are stricter than the v3 Rust source; removing them moves the writer CLOSER to v3, and their failure modes stay LOUD — no silent-wrong-write path).
- Confidence: high. Disjoint-lens value earned its keep again: correctness/claude's cumulative-test trace caught the orphaned param that the per-PR diff alone (and fresh) missed. py_compile OK; ruff clean; 38 non-Docker unit tests pass.
- Commits: `6d01990e2` (de-gold-plate impl), `b6d1f292b` (cycle-1 must-fix: orphaned test param).
- Deferred items: 0 new.
- Surprises during implementation: the OverflowError-branch removal's blast radius reached a KEPT testcontainer test's parametrize list — a reminder that deleting a guarded code path requires sweeping ALL tests (not just the standalone ones) for params/cases that exercised it.

### PR-2.8: (amend 2026-06-05) bench-v4 ruff reconcile + 2 phase-end nits  (1 impl commit + 1 nit-fix commit, 1 inner-loop gauntlet cycle, ending at `f0e93b9f4`)
- Scope shipped: (1) cleared the 1 remaining ruff E501 (`test_migrate_schema.py:833`, 122>120) via an `instance_body` local — `ruff check scripts/` now clean (the RETAINED code-quality merge blocker, resolved); (2) fixed the phase-end nit: the `benchmarks-website/infra/README.md` var-table attributed the ingest-workflow consumers of `RDS_BENCH_INSTANCE_ENDPOINT`/`GH_BENCH_INGEST_ROLE_ARN` to PR-2.2, but the dual-write steps landed in PR-2.4 (git-verified: `d8bbeb6ed`); corrected both rows; (3) added a `_main_postgres` comment noting `build_commit` runs `git show <sha>` so the v4 step inherits the v3 step's checkout/git-history assumption (best-effort).
- Tests added: none (doc/quality; the E501 fix is a test-internal local binding covered by the existing passing test).
- Review: 2-vote (pr-2: fresh + correctness, `executor=claude`) / **accepted at cycle 1**. 0 must-fix, 0 should-fix, 1 nit fixed inline: fresh flagged the new comment's last line at 102 cols (over the user CLAUDE.md 100-col rule, under ruff's 120); rewrapped to <=100 in `f0e93b9f4`. Correctness git-verified the README PR-2.4 correction + confirmed the E501 fix is semantically identical + the comment factually accurate.
- Confidence: high. Closes the retained ruff merge-blocker and both phase-end nits. ruff clean; 51 non-Docker unit tests pass.
- Commits: `06f9e8a13` (E501 + README + comment), `f0e93b9f4` (comment-width nit fix).
- Deferred items: 0 new. The RETAINED deferred backlog is now: only the Phase-1 live OIDC `schema-deploy` apply operator pre-merge gate (the ruff reconcile is DONE).
- Surprises during implementation: the ruff reconcile was a single E501 (the "established multi-line style" was already in place), so PR-2.8 was much smaller than the deferred item implied.

### PR-3.1: (re-plan) DuckDB→Postgres bulk loader + 004-as-master guard  (2 impl commits + 1 should-fix commit, 1 inner-loop gauntlet cycle, ending at `a6647765f`)
- Scope shipped: (a) new `benchmarks-website/migrate/src/postgres.rs` + `Load` subcommand in `vortex-bench-migrate` — reads each of the 6 tables from an existing v3 DuckDB snapshot as Arrow batches (`query_arrow`) and streams them into Postgres via `COPY ... FROM STDIN` (text format) inside ONE transaction (atomic; mid-load failure rolls back to empty). `measurement_id` (+ `commit_sha`) copied verbatim — no hash recompute. `commits.timestamp` `CAST` to VARCHAR under `SET TimeZone='UTC'`; `all_runtimes_ns` → `{a,b,c}`; `threshold` f64 shortest-round-trip (non-finite rejected). NO `aws-sdk-rds`: `NoTls` (local rehearsal) or native-tls with `--ca-cert` (prod verify-full). (b) gap-#4: a `-- migrate-schema: requires-superuser` marker on `002`/`004` + a `rolsuper OR rolcreaterole` preflight in `migrate-schema.py` that fails loud + early before a non-master `apply` of a marked migration.
- Tests added: 11 embedded-DuckDB loader unit tests (escaping incl. backslash/tab/newline, `BIGINT[]` arrays incl. empty/negative, NULLs, negative ints, UTC timestamp via CAST incl. non-UTC-offset normalization, f64 round-trip, literal-`\N`-vs-NULL disambiguation, empty tables) + 3 marker-detection unit tests + a real-file marker-placement test + 2 testcontainer-gated guard tests (non-master role rejected before DDL; superuser is master-capable). All non-Docker green (11 loader + 16 migrate-schema non-Docker); clippy + nightly fmt clean.
- Review: 2-vote (pr-2: fresh + correctness, `executor=claude`) / **accepted at cycle 1**. 0 must-fix, 1 should-fix (applied `a6647765f`: pinned the literal-`\N`, non-UTC-offset timestamp, and empty/negative-array COPY cases — verified-correct by both reviewers but unpinned), 3 nits (1 applied: redundant `format_bigint_array` downcast-context; 2 dismissed per the PR-2.7 de-gold-plate calibration: NUL-byte hardening + a fail-loud bail on the unreachable array-element-NULL branch).
- Confidence: medium. The loader's value-fidelity-critical logic (DuckDB read + COPY-text formatting) is high-confidence — fully unit-tested via embedded DuckDB, with column list/order/ColKind programmatically cross-checked against `migrations/001` for all 6 tables. The Postgres-EXECUTION path (connect / COPY streaming / commit / atomic Drop-rollback) and verify-full TLS COMPILE but are NOT runtime-tested here (no Docker); exercised by PR-3.3's rehearsal harness + the PR-3.4 runbook.
- Deferred items: 2 (both explicit, per the PR-3.1/PR-3.3 boundary plan adjustment `7378345da`): full Postgres-execution integration (6-table load + atomic mid-load rollback) → PR-3.3; verify-full TLS path → PR-3.4 runbook.
- Surprises during implementation: (1) PR-3.1 reads an EXISTING v3 DuckDB rather than re-accumulating from v2, so `measurement_id` is copied verbatim (no hash recompute) — simpler + safer than the original plan implied. (2) The migrate crate's `appender-arrow` feature already provides `query_arrow` (the `arrow` feature does not exist on the duckdb crate). (3) The 2 review Agents raced on the working tree (one/both added a transient `zz_probe` test to verify timestamp/literal-`\N` behavior, then reverted); the orchestrator verified the committed tree clean post-review and de-contaminated the fresh-lens findings to their real shared signal (the coverage should-fix). (4) **[phase-end cycle-1 should-fix annotation, added by PR-3.6]** PR-3.1 shipped the synchronous `postgres` 0.19 crate (+ `native-tls`/`postgres-native-tls`), NOT the plan-row-named `tokio-postgres` — a deliberate choice: a one-shot CLI needs no async runtime, the binding hard `NO aws-sdk-rds` constraint IS met, and sync `postgres` is a thin blocking wrapper that pulls `tokio-postgres` in transitively. The named-dependency swap was unremarked at PR-3.1 time; recorded here per the Phase-3 phase-end review (spec lens, should-fix).

### PR-3.2: (re-plan) DuckDB→Postgres per-`measurement_id` value verify — the PRIMARY v4-correctness gate  (1 impl commit + 1 review-follow-up commit, 1 inner-loop gauntlet cycle, ending at `f796cfc24`)
- Scope shipped: `run_postgres_value_verify` in `migrate/src/verify.rs` + the `verify --postgres-target` CLI mode (mutually exclusive with the existing `--against` v2-structural mode). Joins a DuckDB source snapshot vs the Postgres target per `measurement_id` (1:1; `commits` per `commit_sha`) and compares EVERY non-hashed value column per row, FULL (not sampled), exiting non-zero on any presence diff OR value mismatch and naming the exact `(table, key, column, duckdb_value, pg_value)`. Values normalize to an engine-independent `CellValue` (`Null`/`Int(i64)`/`Text`/`Array(Vec<i64>)`): `all_runtimes_ns` element-wise + order-sensitive; `i32` `iterations` widened to `i64`; `commits.timestamp` compared as epoch microseconds (DuckDB `epoch_us` vs PG `(extract(epoch ...) * 1000000)::bigint`) to sidestep cross-engine timestamp text rendering. New `PgVerifyReport`/`ValueMismatch` (kept separate from the unrelated v2-structural `VerifyReport`). Value-column types are single-sourced from the loader via new `pub(crate) postgres::column_kind` (+ `ColKind`/`connect_postgres` exposed); NO `Cargo.toml` change (reuses PR-3.1's `postgres`/`native-tls`).
- Tests added: 14 no-Docker `value_verify_tests` (in-memory DuckDB): epoch-microseconds pinned (1s→1_000_000); clean-when-identical across all 6 tables; mismatch detection for value column / env_triple / array element / array reorder / commits metadata / NULL-vs-value / `vector_search_runs` Int32 `iterations` + side counter; empty-`BIGINT[]` clean + mismatch; presence diffs BOTH directions; SQL-builder epoch wrapping; `read_kinds` drift guard; report Display. Full migrate suite 98/98 green; clippy `--all-targets` + nightly fmt clean.
- Review: 2-vote (pr-2: fresh + correctness, `executor=claude`) / **accepted at cycle 1**. 0 must-fix, 2 should-fix (both applied `f796cfc24`: corrected the epoch-mechanism doc — PG14+ `extract(epoch)` returns `numeric`, exact for any microsecond ts, not just whole seconds; added the `vector_search_runs` Int32/side-counter mismatch test), 4 nits (2 applied: empty-array verify test + SET-TimeZone-not-needed comment; 1 dismissed per de-gold-plate calibration: single-MVCC-snapshot — a one-shot gate against a quiescent post-load target; 1 carried into PR-3.3's acceptance: live-PG16 epoch seed with a sub-second/pre-1970 commit). Both lenses converged on the epoch-doc should-fix. (cycles: 1)
- Confidence: high. The comparison core (the value-fidelity-critical logic) is high-confidence — both lenses verified it neither misses a value/env corruption nor false-positives on a faithful load, and `ReadKind` is resolved ONCE per table from the loader's authoritative `column_kind` and shared across both reads so an asymmetric mis-read is structurally impossible. The Postgres-EXECUTION read path (`read_pg_table`/`pg_cell`/`pg_select_sql` against a live PG16) COMPILES but is NOT runtime-tested here (no Docker); exercised by PR-3.3's rehearsal harness (which owns the container infra and asserts verify-clean).
- Deferred items: 0 new. (Live PG16 end-to-end is the already-planned PR-3.3 boundary, not a new deferral; `deferred_items_total` stays 2.)
- Surprises during implementation: (1) the loader's `commits.timestamp` is COPY'd as a UTC text string then re-parsed by Postgres, so the verify compares it as an absolute epoch on both sides (engine-independent) rather than as text — chosen to avoid DuckDB-vs-Postgres timestamp text-rendering divergence. (2) The acceptance row literally said "Integration test (local PG16)", but Docker is unavailable and PR-3.1's note already designated PR-3.3 as the container-infra owner; clarified the PR-3.2/PR-3.3 test boundary in the plan (commit `bab556bb0`) before review so the no-Docker comparison-core tests are the PR-3.2 deliverable and the live e2e is PR-3.3's — neither reviewer flagged the absent live test as a gap. (3) Codex was available so gauntlet's dynamic default would have routed `mixed`/`parallel`; chose `executor=claude` for reliability in this long orchestration (documented opt-out).

### PR-3.3: (re-plan) bulk-load rehearsal harness + runbook (gap #2)  (1 impl commit + 1 review-follow-up commit, 1 inner-loop gauntlet cycle, ending at `fdb6c386a`)
- Scope shipped: `tests/postgres_e2e.rs` — the first Rust testcontainer in the migrate crate. Builds a representative v3 DuckDB fixture, stands up a `postgres:16-alpine` testcontainer (schema applied from `migrations/001` via the init-SQL `/docker-entrypoint-initdb.d/` entrypoint), and asserts (a) the loader's per-table row counts match the source, (b) `verify --postgres-target` is clean, (c) a forced mid-load PK conflict on the LAST-loaded table (`vector_search_runs`) rolls the whole single-transaction load back to empty. Added `testcontainers` + `testcontainers-modules` (`postgres`, `blocking`) dev-deps (root `Cargo.lock` +342; migrate IS a workspace member, so `cargo nextest --workspace` runs the e2e in CI). `README.md` operator runbook: live source-account/region verification (since `375504701696`/`us-east-2` is unverified in-repo) + snapshot acquisition + the local PG16 rehearsal. Docker-gated (skip locally, fail-loud in CI).
- Tests added: 2 testcontainer e2e tests (`rehearsal_load_then_verify_is_clean`, `rehearsal_mid_load_failure_rolls_back_to_empty`). Both PASS against real `postgres:16-alpine`; full migrate suite 100/100. This is the FIRST RUNTIME validation of PR-3.1's loader Postgres-EXECUTION path (the deferred gap) AND PR-3.2's verify Postgres-READ path — incl. the epoch SQL (sub-second `2024-06-15 ...123456` + pre-1970 `1969-07-20` timestamps both confirmed `epoch_us`==PG-numeric-epoch by the reviewers), literal-`\N`, multibyte text (now in a value-compared column), empty/negative `BIGINT[]`.
- Review: 2-vote (pr-2: fresh + correctness, `executor=claude`) / **accepted at cycle 1**. 0 must-fix, 2 should-fix (both applied `fdb6c386a`: moved multibyte into a value-compared column so the e2e actually discriminates it; runbook readiness-wait + cwd note), 3 nits (1 applied: corrected the stale `intentionally outside the workspace` Cargo.toml comment, flagged by BOTH lenses; 2 dismissed per de-gold-plate calibration: IPv6 dsn edge + raw-string SQL line length). Both lenses independently verified the rollback test genuinely DISCRIMINATES atomicity (a non-atomic loader would fail it). (cycles: 1)
- Confidence: high. The loader + verify Postgres paths are now runtime-validated against a real PG16 (no longer compile-only); both reviewers confirmed the harness genuinely discriminates and the epoch path is exact. Closes the PR-3.1 deferred Postgres-execution gap + the PR-3.2 live-read gap.
- Deferred items: 0 new. (PR-3.1's 2 deferrals are partially discharged here: Postgres-EXECUTION is now runtime-tested via the container; verify-full TLS remains PR-3.4's runbook coverage since the container uses `NoTls`. `deferred_items_total` stays 2.)
- Surprises during implementation: (1) the migrate crate is a workspace MEMBER (root `Cargo.toml:70`), not "outside the workspace" as its stale comment claimed — load-bearing because membership is why CI's `--workspace` run executes the Docker-gated e2e. (2) `testcontainers-modules` `with_init_sql` mounts to `/docker-entrypoint-initdb.d/`, so the schema applies at container init with no separate apply step; its wait-strategy (stderr THEN stdout "ready to accept connections") correctly handles the init-SQL double-message. (3) Operator started Docker Desktop mid-PR (chose local validation over the CI-gated default), so the harness was iterated to green against real PG16 rather than written blind.

### PR-3.5: (re-plan) Python-writer-vs-RDS cross-check harness (gap #3)  (1 impl + 2 review-follow-up commits, 2 inner-loop gauntlet cycles, ending at `6263cc428`)
- Scope shipped: `scripts/cross_check_python_writer.py` — `cross_check(conn, commit, records) -> CrossCheckReport`: runs `post-ingest.ingest_postgres`, asserts `inserted == 0` (every record UPDATEd a seeded row, not duplicate-INSERTed), re-reads each row's value columns + asserts round-trip; reuses `connect_postgres` (verify-full TLS, `bench_ingest` role, IAM token) + a `--postgres`/`--envelopes` CLI for the prod run. Confirms the LIVE property: the Python writer recomputes the SAME `measurement_id` (Python port == Rust hash, golden-gated) and UPDATEs the Rust-loader-seeded rows.
- Tests added: 5 cross-check tests extending `test_post_ingest_postgres.py` (reuse `postgres_dsn`/`schema_conn` fixtures): a directly-seeded golden-mid row UPDATEd + value overwritten; all 5 fact kinds; a NOT-seeded envelope -> INSERT flagged (the duplicate-risk discrimination); `value_mismatches` discrimination on value_ns / array-reorder / env_triple / a side counter; an env + memory-quartet integration round-trip. All pass vs `postgres:16-alpine`; `ruff format --check` + `ruff check` clean; basedpyright bare-`dict` matches the sibling `post-ingest.py` convention (scoped to vortex-python).
- Review: 2-vote (pr-2: fresh + correctness, `executor=claude`) / cycle 1 REJECTED then cycle 2 ACCEPTED. Cycle 1: lenses SPLIT — correctness accepted (mutation-verified the harness catches a duplicate-INSERT + value corruption); fresh rejected with a CI-gating must-fix it caught that correctness missed (`uvx ruff format --check .`, ci.yml:64) + a should-fix coverage gap (env/memory/side-counters in `_VALUE_COLUMNS` but never deliberately mismatched). Both fixed (`cee7e224c` ruff format; `6263cc428` discrimination tests). Cycle 2: both accept, 0 findings — both MUTATION-TESTED the new env/memory test (deleting `env_triple`/`peak_physical` from the writer's ON CONFLICT SET list makes it fail with the exact harness diagnostic). (cycles: 2)
- Confidence: high. The harness's discrimination is mutation-verified across both cycles (it catches a duplicate-INSERT AND a writer SET-list omission AND a value corruption). The LIVE cross-language property against real Rust-seeded RDS rows is the deferred operator gate (PR-3.5 prod run).
- Deferred items: 0 new. (PR-3.5's PROD run is the already-documented operator gate; `deferred_items_total` stays 4.)
- Surprises during implementation: (1) the existing `post-ingest.py` + `test_post_ingest_postgres.py` already provided `ingest_postgres` (with a `RETURNING (xmax=0)` insert/update classifier), `connect_postgres`, and the record/value-column structure, so the harness is thin glue + a discrimination test rather than new writer logic. (2) The cycle-1 split was a textbook multi-lens win: the correctness lens's deep mutation-testing of the discrimination logic and the fresh lens's CI-format-gate catch were disjoint — neither alone would have yielded the right verdict (accept-worthy substance + a real merge-blocker). (3) A `git commit -m` with backticks hit a shell command-substitution trap (twice); fixed by amend + switching to a quoted heredoc.

### PR-3.4: (re-scoped) REAL-snapshot LOCAL rehearsal — the real-data validation gate  (no production code; operational run, 2026-06-08)
- Scope shipped: ran the validated Phase-3 toolkit end-to-end against the REAL v3 history on the operator's laptop, ZERO prod write. Source: the on-disk `./bench.duckdb` (468MB, gitignored, last written 2026-05-04; the real v3 DuckDB — 6 tables, commits 2025-01-02..2026-05-04). Per the 2026-06-08 user decision, used the local on-disk snapshot instead of an SSH/scp pull off the prod EC2 host — the May-4 file IS real v3 data and freshness is irrelevant for the rehearsal gate (it validates the load+verify+cross-check code against the real data shape); freshness is PR-5.0/cutover's concern. Substrate: a throwaway local Homebrew `postgresql@16` cluster (16.14) on `127.0.0.1:55432` — Docker was DOWN, so the PR-3.3 testcontainer path was unavailable; a real local PG16 cluster is the equivalent rehearsal substrate the runbook describes. Schema applied from `migrations/001_initial_schema.sql`; loader = the current `target/debug/vortex-bench-migrate` (Jun-5 build, rebuilt clean off HEAD).
- Results captured (the acceptance evidence):
  - **load** (`vortex-bench-migrate load --duckdb bench.duckdb --postgres-target postgresql://postgres@127.0.0.1:55432/bench`): atomic single-txn, 25s; per-table row counts match the DuckDB source EXACTLY — `commits` 4202, `query_measurements` 4223133, `compression_times` 222150, `compression_sizes` 88161, `random_access_times` 27064, `vector_search_runs` 0.
  - **verify** (`verify --duckdb bench.duckdb --postgres-target ...`): exit 0, `value verify: source and target match (0 presence diffs, 0 value mismatches)` — FULL per-`measurement_id` comparison of every value column + `env_triple` + `all_runtimes_ns` arrays + `commits` metadata across all 4.2M+ rows, 41s. The PRIMARY v4-correctness gate, GREEN against real data.
  - **cross-check** (the shipping `scripts/cross_check_python_writer.py` over verify-full TLS as the `bench_ingest` role): built an 11-record envelope from REAL seeded rows for commit `344cda8e…` (3 query_measurement + 3 compression_time + 3 compression_size + 2 random_access_time); result `11 records, 11 updated, 0 inserted -- CLEAN` (the Python writer recomputed the SAME `measurement_id` as the Rust-seeded rows and UPDATEd them, values round-trip). Negative discrimination confirmed: mutating one record's `engine` dim → `1 inserted` → `FAILED` (the would-be-duplicate is caught); the junk INSERT was deleted so the target again matches the source. To satisfy `connect_postgres`'s hard contract locally, a self-signed `localhost` cert enabled verify-full TLS and a local `bench_ingest` role (LOGIN + SELECT/INSERT/UPDATE grants, no `rds_iam`) was created.
- Tests added: none (no production code; operational rehearsal). The throwaway envelope generator + local PG cluster + cert live outside the repo (`/tmp`); nothing committed to the tree.
- Review: **no inner-loop gauntlet** — PR-3.4 ships no production code (empty repo diff), so there is nothing for a 2-vote review to read (per the plan's PR-3.4 "no code -> no gauntlet review needed"). The Phase-3 cumulative diff (PR-3.1/3.2/3.3/3.5 code) gets the 3-vote phase-end gauntlet next.
- Confidence: high. All three gates are GREEN against the REAL 4.2M-row v3 snapshot, not synthetic fixtures: exact row-count match, full value-verify clean, and the live cross-language UPDATE-not-duplicate + value-round-trip property — with the cross-check's discrimination re-confirmed against this real data.
- **[phase-end cycle-1 should-fix annotation, added by PR-3.6] Real-data coverage caveat:** the real snapshot held `vector_search_runs` = 0 rows AND the PR-3.4 cross-check envelope omitted that kind, so that ONE table's load-COPY (Double/`threshold` dim, four Int side counters, the only Int32 value column) / value-verify / cross-check paths were exercised only by the synthetic PR-3.3 fixture + PR-3.2/3.5 unit tests, NOT against the real data shape. The other 5 tables got full real-data validation. PR-5.0's freshest-snapshot prod load is expected to close this (or, if the live v3 DB still has zero vector rows at cutover, `vector_search_runs` stays fixture-only-validated). Surfaced by the Phase-3 phase-end review (spec lens should-fix; maint lens judged it fixture-covered/acceptable).
- Deferred items: 0 new. The one-shot PROD load + prod verify + prod cross-check remain PR-5.0/PR-5.1 (Phase-5 cutover, freshest snapshot); `deferred_items_total` stays 4.
- Surprises during implementation: (1) the rehearsal needed no prod access at all — a real v3 DuckDB was already on disk, so the whole gate ran locally (the user's premise was correct). (2) Docker being down was a non-issue: a Homebrew PG16 cluster is an equivalent (arguably cleaner) rehearsal substrate vs the testcontainer; the migrate tools only need a `--postgres-target` DSN. (3) `connect_postgres` hard-requires verify-full TLS + the `bench_ingest` role with NO local-affordance exception (by design — it is the prod ingest connector), so the local cross-check needed a self-signed-cert TLS endpoint + a real `bench_ingest` role rather than a plaintext shortcut; this exercised the actual shipping CLI end-to-end rather than a bypass.

### PR-3.6: (amend) apply Phase-3 phase-end cycle-1 should-fix doc/status items  (1 doc commit + 1 review-nit commit + 2 plan annotations, 1 inner-loop 2-vote cycle, 2026-06-08)
- Scope shipped: applied the 3 should-fix items from the Phase-3 phase-end review. (a) Added a `## Bootstrap ordering — requires-superuser migrations (002 / 004)` subsection to `migrations/README.md` documenting the `-- migrate-schema: requires-superuser` marker, the `rolsuper OR rolcreaterole` master-capable preflight, and the "apply 002/004 as RDS master before any migrator deploy" ordering — the operator-facing doc the `_assert_master_capable` PermissionError points at (closes the maint should-fix). (b) Annotated the PR-3.1 status with the deliberate sync-`postgres`-over-plan-named-`tokio-postgres` choice (spec should-fix). (c) Annotated the PR-3.4 status that `vector_search_runs` had 0 real rows + was omitted from the cross-check envelope, so its real-data coverage is fixture-only, with PR-5.0 closing it (spec/maint should-fix).
- Tests added: none (doc/status only; no behavior change).
- Review: 2-vote (pr-2: fresh + correctness, executor=claude) / accepted at cycle 1. 0 must-fix, 0 should-fix, 1 nit (applied `0c33c11e2`). Both reviewers verified every README claim against the live `scripts/migrate-schema.py` source (the `_REQUIRES_SUPERUSER_DIRECTIVE` marker string, the `rolsuper OR rolcreaterole` capability SQL, the `_assert_master_capable` PermissionError text + that the guard fires BEFORE the migration transaction, and that exactly 002/004 carry the marker while 001/003 do not); the correctness lens could not construct a planted-bug case; the fresh lens's nit refined the failure-statement list to include `GRANT` (since `CREATE ROLE` is `IF NOT EXISTS`-guarded in a `DO` block).
- Confidence: high. The README subsection was verified line-by-line against the actual code contract by both lenses; the 2 plan annotations are factual status records.
- Deferred items: 0 new (`deferred_items_total` stays 4).
- Surprises during implementation: none — the should-fix items were exactly as the phase-end review scoped them; the README documents an already-tested contract.

### PR-3.7: (amend) clear Phase-3 phase-end cycle-2 doc nits  (1 doc commit + 1 should-fix repoint commit, 1 inner-loop 2-vote cycle, 2026-06-08)
- Scope shipped: cleared the 3 cycle-2 nits — (a) aligned `migrations/README.md`'s hypothetical directive example `-- migrate: no-transaction` → `-- migrate-schema: no-transaction` (the only implemented directive namespace); (b) added a `pub fn load` doc-comment sentence noting the target schema (`migrations/001`) must already be applied (`load` only `COPY`s, never DDL); (c) added a `migrate/README.md` runbook bullet that a pre-acquired on-disk snapshot is an acceptable LOCAL-rehearsal source. The inner-loop correctness lens caught a bonus should-fix: 3 pre-existing stale `prod load (PR-3.4)` references in `migrate/README.md` (from before the 2026-06-05 re-scope) — repointed to PR-5.0 (`cbbd6f25c`), making the runbook internally consistent with PR-3.4=rehearsal / PR-5.0=prod-load.
- Tests added: none (doc/doc-comment only; no behavior change; `cargo +nightly fmt --check -p vortex-bench-migrate` clean).
- Review: 2-vote (pr-2: fresh + correctness, executor=claude) / accepted at cycle 1. 0 must-fix. fresh found nothing (verified all 3 edits against source: the `migrate-schema:` namespace, `load`-never-runs-DDL via grep, PR-3.4-on-disk-rehearsal via git history); correctness verified the same + flagged the stale-prod-load-references should-fix (applied).
- Confidence: high. All edits verified against the live source by both lenses.
- Deferred items: 0 new (`deferred_items_total` stays 4).
- Surprises during implementation: the correctness lens surfaced a pre-existing doc inconsistency (the runbook still called PR-3.4 the prod load) that the new bullet (c) made visible — a genuine bonus catch, now fixed.

### PR-4.1: Scaffold benchmarks-website/web/ Next.js 15 App Router project  (1 impl commit, 1 inner-loop 2-vote cycle, 2026-06-08)
- Scope shipped: created the Next.js 15.5.19 + React 19 + App Router scaffold under `benchmarks-website/web/`. Files: `package.json` (pnpm; scripts dev/build/start/lint/format/format:check; pinned Next 15.x since 16 is published but the migration targets 15), `tsconfig.json` (Next base + all 5 vortex-web strict flags: `strict`/`noUnusedLocals`/`noUnusedParameters`/`noFallthroughCasesInSwitch`/`noUncheckedSideEffectImports`), `next.config.js` (CommonJS; `outputFileTracingRoot` pinned to the app dir to resolve the repo's multi-lockfile root ambiguity), `eslint.config.mjs` (flat config via `eslint-config-next` + `FlatCompat` over `next/core-web-vitals` + `next/typescript`; added `@eslint/eslintrc` devDep), `.prettierrc.json` (byte-identical to vortex-web shape), `.gitignore`, `pnpm-workspace.yaml` (`allowBuilds: sharp + unrs-resolver`), `app/layout.tsx` + `app/page.tsx` stubs. SPDX header pair on every comment-capable file (JSON exempt, matching vortex-web). `pnpm-lock.yaml` committed.
- Tests added: none (scaffold; no runtime logic). Acceptance verified empirically: `pnpm install && pnpm build` succeed; `pnpm lint` + `pnpm format:check` clean; strict flags confirmed genuinely enforced (injected unused local → tsc TS6133, run by `next build`).
- Review: 2-vote (`pr-2`: fresh + correctness, executor=mixed→**parallel**; 4 reviewer outputs — Claude + Codex per lens) / accepted at cycle 1. 0 must-fix, 0 should-fix, 4 nits. **One disagreement**: Codex/fresh flagged `web/.prettierrc.json` as must-fix ("second formatter config"); the synthesizer + both Claude lenses adjudicated it compliant (the BAN targets formatter-SHAPE divergence, not file count; `web/` is a separately-deployable pnpm/Vercel package so a sibling-tree config is undiscoverable without a fragile `--config` path or out-of-scope repo-root config) → downgraded to nit + recorded in `Accepted tradeoffs` (commit `c6b78598e`) so future Phase-4 PR + phase-end reviews drop the re-flag.
- Confidence: high. Both Claude reviewers re-ran build/lint/format/type-check independently; all BANS verified (SPDX header pair, 5 strict flags, no WASM, no v2 top-level files touched, prettier shape match).
- Deferred items: 0 new (`deferred_items_total` stays 4). 3 nits dismissed (no `engines.node` pin; format-glob misses root config files; unused-var is a lint warning with `next build` type-check as the real gate) per the lean trusted-input calibration.
- Surprises during implementation: (a) pnpm not preinstalled (installed pnpm 11.5.2 globally); (b) pnpm 11 moved `onlyBuiltDependencies` → `pnpm-workspace.yaml` `allowBuilds`; (c) `eslint-config-next` 15.x ships legacy eslintrc configs (used `FlatCompat` + added `@eslint/eslintrc`, the canonical Next-15 flat-config bridge — avoids the deprecated `next lint`); (d) the user's mid-session `develop` rebase rewrote/orphaned `phase_entry_sha` `dcc24f748` → repointed to the rewritten entry commit `5a9d24471` (`git diff 5a9d24471..HEAD` verified clean of develop's byte_length changes); (e) 1Password SSH commit-signing re-locked during the ~10-min review (user unlocked, retry succeeded).

### PR-4.2: Postgres connection lib (web/lib/db.ts, pg.Pool + RDS IAM auth) + testcontainers test  (1 impl + 1 should-fix fix commit, 1 inner-loop 2-vote cycle, 2026-06-08)
- Scope shipped: `web/lib/db.ts` — a `globalThis`-cached singleton `pg.Pool` (warm-invocation + dev-HMR reuse) whose password is an async provider minting a fresh `@aws-sdk/rds-signer` IAM token per connection (token-refresh-before-expiry, ~15-min tokens); a `BENCH_DB_PASSWORD` static-password path bypasses IAM for local/test; `resolveSsl` (verify-full default, disable for tests, **fail-loud on unknown mode**); a pure `buildQuery` helper + the injection-safe `sql` tagged-template ($1..$n binding); a `resetPool` teardown helper. Also: re-included `web/lib/` in `web/.gitignore` (the repo-root Python-artifact `lib/` pattern was silently swallowing it), widened the prettier globs to `{app,lib}/**` + `vitest.config.ts`, and added pg/@aws-sdk/rds-signer/vitest/@testcontainers/postgresql/@types/pg deps (ssh2/cpu-features/protobufjs native builds skipped via `allowBuilds` — not needed for local-Docker testcontainers).
- Tests added: 15 (vitest 4) — 2 LIVE testcontainers Postgres roundtrip (connect-via-password-fixture + SELECT + parameter binding; user chose "run for real" so verified against a real `postgres:16-alpine` container, Docker up), 1 per-connection-IAM-refresh (distinct tokens across two `provider()` calls — pins fresh-per-connection, not single-cache), static-password + missing-region-throws, 3 `buildQuery` (0/1/N values), 4 `resolveSsl` (default/disable/ca-merge/unknown-throws), 3 `requireEnv`. The testcontainers describe is Docker-skip-guarded (mirrors the repo's Python `_docker_available()` precedent).
- Review: 2-vote (`pr-2`: fresh + correctness, executor=claude) / accepted at cycle 1. 0 must-fix. Both lenses verified the `$N` reducer + IAM branching CORRECT on disk (tsc/eslint/prettier/vitest all green; container roundtrip ran live). NO disagreement (synthesized inline). 5 should-fix + 3 nits surfaced; substantive ones APPLIED in fix-commit `5201aff38`: SSL fail-loud on unknown mode (the one finding with real security character — a typo previously disabled cert verification silently), per-connection-refresh test discrimination, `buildQuery` extraction + non-Docker unit coverage, `resolveSsl`/`requireEnv` unit tests, verify-full doc accuracy, format-glob widening, pool reset. 1 nit dismissed (silent NaN port — trusted-config input-hardening, out of the lean calibration). 1 should-fix deferred (wire the vitest suite into CI → PR-4.5). No cycle-2 (should-fix-only changes on an accept verdict; re-verified build/lint/test/format all green).
- Confidence: high. Core logic (IAM per-connection refresh, `sql` parameterization, SSL mode handling) verified live + unit-pinned by both lenses.
- Deferred items: 1 new (vitest-CI-runner gap, Resolved-by PR-4.5; `deferred_items_total` 4 -> 5).
- Surprises during implementation: (a) the repo-root `.gitignore` `lib/` (Python packaging-artifact) pattern silently ignored `web/lib/` — fixed with a scoped `!/lib/` negation in `web/.gitignore`; (b) a `vi.fn` arrow-mock is not `new`-able — used a named `function` expression for the Signer mock; (c) the testcontainers Postgres container started + roundtripped cleanly once Docker was up; (d) `executor=claude` chosen for this inner-loop review (lean 2-vote calibration + faster than the Codex-polling path), unlike PR-4.1's mixed->parallel.

### PR-4.3.a: Read-port foundation (schema-version, slug, window, families) + /health  (2 impl + 3 should-fix/nit fix commits, 1 inner-loop 2-vote cycle accepted, 2026-06-08)
- Scope shipped: the foundation layer for the read-endpoint port. `web/lib/schema-version.ts` (`SCHEMA_VERSION = 1` read-path gate, Table D site, lockstep with `server/src/schema.rs`); `web/lib/families.ts` (5-fact-table registry: tableName + chart/group slug prefixes qm/ct/cs/rat/vsr + qmg/ctg/csg/rag/vsg + kind discriminants, in `family.rs` order; `familyForChartKind`/`familyForGroupKind`; sorted `HEALTH_TABLES`); `web/lib/window.ts` (`?n=` -> CommitWindow port of `window.rs`: default 100, floor-0->1, clamp 1..1000, case-insensitive trimmed `all`, u32-overflow->fallback); `web/lib/slug.ts` (port of `slug.rs`: 5 ChartKey + 5 GroupKey variants, `<prefix>.<base64url-no-pad(canonical-JSON)>` with `k`-first serde-tagged field order + explicit-null Options; decode validates full per-variant shape, NOT just `k`); `web/lib/health.ts` + `web/app/api/health/route.ts` (snake_case `HealthResponse` shape preserved; `db_path`->DB host, `build_sha`->`VERCEL_GIT_COMMIT_SHA`, `schema_version`->const; sorted `row_counts` via per-table `COUNT(*)::int`; `force-dynamic` liveness route). Slugs treated as OPAQUE round-trippable tokens (not Rust-byte-identical) per the shard-endpoint deferral — see Accepted tradeoffs.
- Tests added: 65 vitest total (was 58 in PR-4.2; +7 this PR). slug: round-trip all 10 variants + prefix checks + malformed rejection + 2 canonical-encoding golden vectors (pin `k`-first/explicit-null byte order) + 4 validation-discrimination tests (reject missing/mistyped required fields; absent-Option->null serde parity). window: full `window.rs` parity (default/all/clamp/floor/overflow/malformed). families: order + distinct prefixes + exact-prefix pin + lookups + sorted HEALTH_TABLES. health: pure `assembleHealth`/`buildRowCounts` + 2 LIVE testcontainers `postgres:16-alpine` integration tests (applies real `migrations/001`; empty-schema zeros + real-count + timestamp-format). schema-version: 1-line `=== 1` drift sentinel.
- Review: 2-vote (`pr-2`: fresh + correctness, executor=claude) / accepted at cycle 1. 0 must-fix, 2 should-fix, 5 nits. Both lenses cross-checked every TS module vs its Rust source-of-truth (slug.rs/window.rs/family.rs/dto.rs/mod.rs/migrations-001) + confirmed window parse vs `rustc`; no disagreement. Applied: F1 (slug decode per-variant field validation — the substantive should-fix, both lenses; slugs are user-controllable URL params for 4.3.b) in `99a8b6f96`; F2 canonical golden test; F3 SCHEMA_VERSION sentinel + F7 redundant-slice in `90b6b04e6`; F4 timestamp doc note in `3da6da8d2` (`fix(docs)`). F5 (base64url leniency) + F6 (threshold f64 1.0-vs-1) dismissed -> Accepted tradeoffs (opaque-slug decision moots cross-language byte-identity). No cycle-2 (should-fix/nit on an accept verdict, re-verified tsc/eslint/prettier/65-tests/next-build all green).
- Confidence: high. Faithful behavior-preservation port verified line-by-line against the Rust oracle by both lenses + the full suite green incl. live PG16 container.
- Deferred items: 0 new (`deferred_items_total` stays 5).
- Surprises during implementation: (a) verified wire format is snake_case (the exploration agents had wrongly assumed camelCase) — only the `Summary` enum variant fields are camelCase + `UnitKind` is snake_case, per `dto.rs`; (b) `SCHEMA_VERSION` confirmed `= 1` from `schema.rs` (an exploration agent had hallucinated `2`); (c) the artifacts/shard endpoint was deferred to PR-4.4 via a user fork decision (1 AUQ this phase) — its in-memory generation-store + ETag/brotli machinery is what the lean re-plan moved to the edge CDN.

### PR-4.3.b: Chart endpoint port (queries.ts `chartPayload` + `/api/chart/[slug]` + tests)  (1 impl + 1 must-fix + 1 test-hardening commit, 2 inner-loop 2-vote cycles, 2026-06-08)
- Scope shipped: `web/lib/queries.ts` (`chartPayload` — faithful port of `charts.rs`: `SeriesAccumulator` + `seeded_commits_in_window` two-pass for all 5 chart families; `IS NOT DISTINCT FROM` nullable dims; value cols `::float8`; `count(*)::int`; commit timestamp via `to_char(... AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS"+00"')` reproducing the DuckDB `CAST(... AS VARCHAR)` text byte-for-byte; `BTreeMap`-sorted series/series_meta; `series_meta` omitted when empty per serde skip-if-empty); `web/app/api/chart/[slug]/route.ts` (`revalidate=300`; 400/404/200; `?n` window); `web/vitest.config.ts` (+`@/` alias to import the route handler in tests). Plus the cycle-1 must-fix: `reqI32` in `web/lib/slug.ts` (validate `query_idx` as a 32-bit integer matching serde, so a forged non-i32 `query_idx` returns 400, not an unhandled 500 from the Postgres int4 bind).
- Tests added: `queries.test.ts` (testcontainers PG16): keystone `toEqual` vs the v3 golden snapshot `chart_page_query.snap` (modulo numeric rendering); `unit_kind` per family; `series_meta` tag/no-tag/format-only; window `n`/`all`/default; history placement (125-commit bounded + all); missing->null; route 400/404/200 + `?n=1`/`?n=all`/malformed; no-DB route input-validation (forged non-i32 `query_idx` -> 400); tailored seeded-window describe (null-gap, pre-history exclusion, NULL `scale_factor` `IS NOT DISTINCT FROM` equality, identical-timestamp `commit_sha` tie-break). `slug.test.ts`: `i32`-rejection unit test. 87 vitest total (was 77 at implementation; +10 from review fixes); tsc/eslint/prettier/next-build green.
- Review: 2-vote (`pr-2`: fresh + correctness, executor=parallel Claude+Codex). cycle 1 = REJECT — 1 must-fix found ONLY by the Codex correctness side (the forged non-i32 `query_idx` 400-vs-500 divergence; the disjoint find that justified parallel routing), 3 should-fix coverage + 5 nits. Applied: must-fix `reqI32` + tests (`2d554ecc4`); should-fix coverage (null-gap, NULL-dim equality, tie-break) + nits (route `?n`, series_meta, keystone rename) as test hardening (`e98bf5fcb`). Dismissed: nit #8 (non-contract 400 body — reviewer marked optional) + nit #9 (UTF-16 vs UTF-8 sort comparator — documented, ASCII-only). cycle 2 = ACCEPT — both Claude reviewers verified `reqI32` matches serde `i32` + the tailored tests discriminate, 0 findings; both Codex sides hit a codex-companion infra failure (ENOENT / MODULE_NOT_FOUND in the plugin runtime, mid-session) and degraded to `parallel-to-claude` per the carve-out (Codex is auxiliary, never blocks the verdict).
- Confidence: high. Faithful behavior-preservation port; the keystone `toEqual` pins semantic-equivalence (modulo numeric rendering) vs the v3 Axum golden, and the only behavior divergence found (forged `query_idx`) is fixed + pinned.
- Deferred items: 0 new (`deferred_items_total` stays 5).
- Surprises during implementation: (a) the v3 DuckDB `CAST(... AS VARCHAR)` golden timestamp text is `2026-04-23 12:00:00+00` (space separator, `+00` offset, no `T`/`Z`) — reproduced exactly via `to_char` because the chart `commits[].timestamp` is a wire-compat field (unlike `/health`'s smoke-test timestamp which uses `T...Z`); (b) the cycle-1 must-fix was a PRE-EXISTING latent gap in PR-4.3.a's `slug.ts` (`query_idx` decoded as any number) that PR-4.3.b's int4 bind exposed — fixed at the decode source (`reqI32`); (c) parallel Claude+Codex routing earned its keep — the must-fix was found ONLY by Codex; (d) both Codex sides failed on cycle 2 with a codex-companion infra error, handled via the auxiliary-executor degrade-to-claude carve-out.

### PR-4.3.c: Groups + group endpoints (collectGroups/collectGroupCharts + summary.ts + descriptions.ts + /api/groups + /api/group/[slug])  (6 code commits: 1 impl + 5 review-fix, 3 inner-loop 2-vote cycles, ending at b56c8c516, 2026-06-09)
- Scope shipped: `web/lib/queries.ts` (`collectGroups` discovery over the 5 families with slug generation + `GROUP_ORDER` stable sort by `(pos, name)`; `groupNameQuery` tpch/tpcds/clickbench + legacy fallback; `collectGroupCharts` full re-discovery + flattened `NamedChartResponse`); `web/lib/summary.ts` (4 summaries — `randomAccess` rankings, `compression` speedup geomean, `compressionSize` ratio min/mean/max, `queryBenchmark` missing-series penalty model — ports `summary.rs`; the MF1 fix keeps `MAX(timestamp)` inside SQL via a CTE for both compression-time + size, eliminating the text round-trip that truncated sub-second timestamps); `web/lib/descriptions.ts` (editorial blurbs byte-ported from v2 via `descriptions.rs`); `/api/groups` + `/api/group/[slug]` routes (`revalidate=300`, 400/404/200).
- Tests added: `groups.test.ts` (testcontainers PG16): group ordering, all 4 summary variants, query-penalty exact scores, route 200/404 + `?n`, plus a new `summary math fidelity` describe with isolated truncate-per-test fixtures (sub-second-timestamp regression [discriminating: fails the pre-MF1-fix truncating code], distinct encode/decode timestamps, decode-only fallback, 300k-penalty-floor ranking flip); `descriptions.test.ts` (6 pure-logic). 108 vitest total (was 104 at implementation; +4 from review fixes); tsc/eslint/prettier green.
- Review: 2-vote (`pr-2`: fresh + correctness, executor=parallel Claude+Codex). cycle 1 = REJECT (2 must-fix found only by Codex: MF1 summary-timestamp truncation [fixed], MF2 statpopgen/polarsignals legacy naming [dismissed preserved-v3]; 3 should-fix, 3 nits). cycle 2 = REJECT (1 must-fix, correctness/codex only: same-second-timestamp summary tie [dismissed preserved-v3, verified identical across all 3 Rust summary paths]; 2 should-fix, 3 nits). cycle 3 = ACCEPT (unanimous, all 4 reviewers 0 findings; 365fdaeb2 loop change verified behavior-neutral; carry-forward correctly not re-flagged).
- Confidence: high. Faithful behavior-preservation port; the testcontainer suite pins semantic-equivalence vs the v3 Axum group_api.rs contract values, and the one runtime fidelity gap found (MF1 sub-second timestamp) is fixed + pinned by a discriminating regression test.
- Deferred items: 3 new (`deferred_items_total` 5 -> 8): MF2 statpopgen/polarsignals v2-description restore (v2->v4 display regression, preserved-v3); same-second-timestamp summary tie determinism (preserved-v3 latent bug, all 4 summary paths); groupNameQuery clickbench/variant test-coverage (faithful port, Rust has same gap).
- Surprises during implementation: (a) TWO of the three must-fixes across cycles 1-2 were real latent v3 bugs that the faithful port reproduces (MF2 dead descriptions; same-timestamp tie) — both dismissed as preserved-v3-behavior per PR-4.3.c's v3-semantic-equivalence acceptance criterion and tracked in Deferred work; this accumulating pattern is flagged for a user decision at the Phase-4 boundary (schedule a deliberate "v4 correctness improvements over v3" effort vs preserve v3 exactly); (b) the MF1 timestamp truncation was the only fix-worthy runtime gap, caught only by the Codex correctness lens via deep Rust cross-reading — parallel Claude+Codex routing earned its keep again; (c) Docker/testcontainers ran ~157s once during a cold Docker Desktop start, then recovered to ~5s — transient infra, not a code issue.

### PR-4.4.a: v4 read-service server-rendered landing shell + global CSS  (2 impl + 1 review-fix commit, 1 inner-loop 2-vote cycle accepted, ending at 2b44fc2a4, 2026-06-09)
- Scope shipped: the non-interactive Next.js (App Router, RSC) landing shell at `benchmarks-website/web/`, ported from v3's HTML layer (`render.rs`/`landing.rs`/`summary.rs` + `static/style.css`), NOT v2 React. `app/layout.tsx` (html/body + `globals.css` + external Geist/Funnel fonts hoisted by React 19 + theme-aware favicons via metadata); `app/page.tsx` (`force-dynamic` server component: `collectGroups()` → `<Header>` + a `<GroupSection>` per group in `GROUP_ORDER` + `<Footer>`, page-wide `data-chart-index`, `<noscript>` notice); `components/Header.tsx` (static logo/title/GitHub chrome; interactive nav/theme/filter deferred to PR-4.4.b); `components/GroupSection.tsx` (`<section.group-details>` → collapsed native `<details.group-disclosure>` wrapping ONLY `<summary>`, with `<SummaryCard>` + `.chart-grid` of empty chart-card `<canvas>` shells as SIBLINGS so the ported `.group-disclosure:not([open]) ~ .chart-grid` rule hides charts when collapsed); `components/SummaryCard.tsx` (4 `Summary` variants ported from `summary.rs::summary_markup`, `formatTimeNs` for ns values, `.toFixed(2)x` ratios, empty-content guards, exhaustiveness default); `components/Footer.tsx` (build-SHA via `VERCEL_GIT_COMMIT_SHA`); `lib/format.ts` (`formatTimeNs`, port of `format_time_ns`); `app/globals.css` (v3 `static/style.css` ported ~verbatim, prettier-reformatted); `css.d.ts` (ambient `*.css`); `vitest.config.ts` (oxc automatic-JSX + `components/**` include); logo/favicon assets under `web/public/`.
- Tests added: `lib/format.test.ts` (formatTimeNs tiers/boundaries/sign — 5), `components/SummaryCard.test.tsx` (4 variants + empty/absent guards via `react-dom/server` — 8), `components/GroupSection.test.tsx` (collapsed disclosure, info-icon + count pluralization, summary card, chart-card shells w/ page-wide `data-chart-index` + `data-chart-slug` + permalinks — 5). 126 vitest total (108 prior + 18 new); tsc/eslint/prettier clean; `next build` compiles+lints+typechecks (prerender needs `BENCH_DB_HOST`, a pre-existing PR-4.3 API-route deploy-env condition; the `force-dynamic` page is excluded from prerender).
- Review: 2-vote (`pr-2`: fresh + correctness, executor=parallel Claude+Codex). cycle 1 = ACCEPT (unanimous, all 4 reviewers overall=accept, 0 must-fix). 5 should-fix/nit findings; Codex uniquely caught the mobile-GitHub-link gap. Triaged: 3 fixed in-cycle (noscript, prettier-glob coverage, SummaryCard exhaustiveness guard); 2 deferred (mobile-nav GitHub → PR-4.4.b; round-half-even display parity → Deferred work).
- Confidence: high. Faithful v3-HTML-layer port with a small authored surface (mostly verbatim CSS + declarative server components + a tested pure helper); zero must-fix; SummaryCard/formatTimeNs/GroupSection verified against the v3 Rust source and pinned by the new tests.
- Deferred items: 2 new (`deferred_items_total` 8 → 10): mobile GitHub link <768px (Resolved-by PR-4.4.b); round-half-even display-rounding parity (low-priority, pairs with the Phase-4-boundary v4-vs-v3 decision).
- Surprises during implementation: (a) the better port source is v3's server-rendered HTML layer, NOT v2 React (v3 is already server-shell + client-island, reproduces the v4 endpoints' ns `Summary` shape, uses v2's CSS class vocabulary); recorded as the PR-4.4.a port-source-refinement Key decision. (b) rolldown-vite (vite@8) ignores `esbuild.jsx`; JSX-in-vitest needed `oxc: { jsx: { runtime: 'automatic' } }`. (c) `noUncheckedSideEffectImports` rejects the `import './globals.css'` side-effect import without an ambient `*.css` declaration (added `css.d.ts` rather than disabling the flag, per the BANS). (d) parallel Claude+Codex earned its keep again — the mobile-GitHub finding was Codex-only.

### Phase-4 holistic mid-phase review (2026-06-09, user-requested; not a planned PR; fixes at b53e07727)
- Scope: 3-lens parallel review of the cumulative phase-4 product (`benchmarks-website/web/`): coherence/claude + maintainability/claude via `Agent`, correctness/codex via the companion (gpt-5.5 xhigh); conservative-union triage; all fixes in one consolidated commit `b53e07727`.
- Must-fix (found by coherence/claude, independently by correctness/codex): the read-route caching story was incoherent across PRs. `/api/groups` exported `revalidate = 300`, which forced a BUILD-time prerender so a DB-less `next build` FAILED (verified; contradicting page.tsx's documented `force-dynamic` decision); the same export was inert on `/api/chart/[slug]` + `/api/group/[slug]` (reading `request.url` forces dynamic), so their CDN-caching comments were false. Fixed by replacing route-segment ISR with `Cache-Control: public, s-maxage=300, stale-while-revalidate=300` headers on the three data routes' 200 responses (full-URL-keyed Vercel edge caching, v2's 5-min cadence) via the new shared `web/lib/cache.ts`; pinned by route-test header assertions + a DB-less `next build` (now green; all routes render dynamically).
- Should-fix applied: 400/404 error envelopes aligned to the Axum `{error: 'bad_request'|'not_found', message}` shape; `reqNumber` finiteness in `lib/slug.ts` (serde f64 parity: forged `threshold: 1e400` is now a 400, pinned by a raw-payload test since `JSON.stringify(Infinity)` would mask it); shared `lib/test-harness.ts` extracted (Docker probe + DDL + container boot + the canonical 3-commit fixture that 4 suites had copy-pasted; net -208 lines).
- Nits applied: `compareCodeUnits` dedupe (was `byCodeUnit`/`cmpStr` twins); `<noscript>` no longer renders in the empty-DB state (v3 `landing_body` parity); misleading ported `.group-info-icon` pointer-events comment rewritten; non-discriminating `?n=all` route test renamed + a discriminating all-vs-default route assertion added to the 125-commit suite; doc comments for `requireEnv`, health-route GET, `sql`-vs-`QueryParams` division of labor, `commitWindowUrlValue` forward reference; `families.ts` table-name consumer claim corrected; em-dash sweep.
- Deferred: landing-page caching strategy (page stays `force-dynamic`, every `/` render hits Postgres until PR-4.5 decides vercel.json `Cache-Control` on `/` vs build-time-DB revalidation). `deferred_items_total` 10 -> 11. Dismissed: next-env.d.ts SPDX (generated file, low confidence).
- Review: 3 reviewers, 1 cycle, unanimous on the must-fix. Verified: tsc/eslint/prettier clean; vitest 128 pass (+2 new); DB-less `next build` green.
- History note (2026-06-09): immediately after this review the branch history was restructured (user-requested) to one squashed commit per phase. Pre-squash per-commit history, including every `ending at <SHA>` reference in the entries above, is preserved at the LOCAL ref `backup/bench-v4-pre-squash-20260609`.

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

## Phase 2: Postgres writer + best-effort v4 CI — end-of-phase review (cycle 1) — accepted (2-vote)
**Synthesizer output from /spiral:gauntlet (preset=phase-2: spec + correctness, executor=claude); full Synthesizer Output JSON in the `<details>` block at the end of this section. Verdict: accept (0 must-fix, 0 should-fix, 2 nits).**

### Summary of changes

Phase 2 delivers the Postgres dual-write ingest path and its CI plumbing, shipped exactly to the lean re-planned scope across PR-2.1/2.2/2.3/2.4 with no drift in either direction; correctness review found zero data-correctness bugs. Ingest identity (PR-2.1): migration 004 creates a least-privilege bench_ingest login role (rds_iam member on RDS, guarded for vanilla PG) holding USAGE-not-CREATE on public and SELECT/INSERT/UPDATE on the six tables (no DELETE/TRUNCATE/DDL), plus ALTER DEFAULT PRIVILEGES FOR ROLE migrator so future migrator tables auto-grant. Because the RDS master is rds_superuser but not a true superuser, the ADP runs via a temporary INHERIT self-grant through the creator's ADMIN option, then revoke. provision.sh adds GitHubBenchmarkIngestRole (OIDC trust branch-scoped to develop+ct/bench-v4, rds-db:connect for bench_ingest on the instance only) and drops the dead proxy grant. Writer (PR-2.2): post-ingest.py --postgres reproduces the v3 serde boundary in Python (deny_unknown_fields, per-field type/range validation, memory quartet, storage enum, commit_sha match), computes measurement_id bit-identically via the PR-1.5 port (_measurement_id.py is a byte-for-byte port of db.rs:162-257, pinned transitively Rust==golden==Python), then upserts commits-first and five fact tables in one all-or-nothing transaction via INSERT ... ON CONFLICT DO UPDATE with deadlock retry, over verify-full TLS authenticated by an RDS IAM token as bench_ingest. An is_finite guard rejects NaN/Inf/out-of-f64-range threshold loudly with rollback (Python stricter-or-equal to serde). All six ON CONFLICT SET lists touch only value/env columns, never dim columns; search_path is pinned to public; ssl_in_use is asserted post-connect. The v3 --server path stays stdlib-only (PEP 723 dependencies=[]). SCHEMA_VERSION lockstep held. CI wiring (PR-2.3): a scripts-test job runs pytest scripts/ behind a docker info hard gate plus a CI-env fail-loud fixture so testcontainer suites cannot silently skip. Dual-write CI (PR-2.4): the three ingest workflows each add a best-effort continue-on-error v4 --postgres step after the unchanged hard-required v3 step (v3-commit-metadata.yml gains id-token: write and ingests commit-row-only empty.jsonl); schema-deploy.yml switches to push-on-develop under paths migrations/**, keeping workflow_dispatch+dry_run. PR-2.5 (reconcile-ingest.py + dual-write-verify.yml + incident.io) was intentionally dropped; v4 correctness is deferred to Phase-3 migrate --verify. All five measurement_id functions, six ON CONFLICT SET lists, field/type sets, and column widths were verified against the Rust source and 001_initial_schema.sql. Two cosmetic nits, no must-fix or should-fix.

### Surprises and discoveries

- **Migration 004 ALTER DEFAULT PRIVILEGES FOR ROLE migrator fails on a real non-superuser RDS master (createrole_self_grant default), masked by the superuser testcontainer.** — Fixed PR-2.1 cycle-1: guarded INHERIT self-grant via the master's ADMIN option then revoke, plus a test applying 001..004 as a real NOSUPERUSER CREATEROLE login. _(amend_plan: already-done)_
- **conn.info.ssl_in_use does not exist in psycopg (ssl_in_use lives on conn.pgconn); the cycle-9 fix shipped the wrong accessor.** — Fixed PR-2.2 (conn.pgconn.ssl_in_use) with a unit test pinning the accessor location and a container test pinning the traversal. _(amend_plan: already-done)_
- **fail-loud-on-no-Docker guard tests could not catch an always-skip regression because pytest.skip raises Skipped (not a Failed subclass).** — Fixed PR-2.3 (catch both outcome types and assert the specific one; mutation-verified). _(amend_plan: already-done)_
- **Python json.loads accepts NaN/Infinity literals that serde_json rejects at parse time.** — The _require_finite guard rejects them loudly with rollback, making the Python writer stricter-or-equal to v3, so no divergent row is written. Already handled. _(amend_plan: no)_
- **The v3 producer emits several fields as u32/u64 (query_idx u32; value_ns/all_runtimes_ns u64) where server and Python validator use i32/i64.** — Both v3 serde and Python _require_int reject values exceeding the signed range, so the substrates agree; values are far inside range. No change needed. _(amend_plan: no)_
- **insert-vs-update classification differs across substrates: Rust does an exists() preflight SELECT; Python derives the flag from RETURNING (xmax = 0).** — Equivalence (incl. same-dim-tuple-twice-in-one-envelope yielding (1,1) not (2,0)) is pinned by test_same_dim_tuple_twice_in_one_envelope_counts_second_as_update. Already handled. _(amend_plan: already-done)_
- **bench-v4 Python files predate Phase 2 at ~100-col and fail the repo ruff line-length-120 (a pre-existing E501).** — Left unfixed and flagged for the operator as a branch-wide ruff reconciliation required before merging ct/bench-v4 to develop; RETAINED deferred item. _(amend_plan: no)_
- **pytest scripts/ collects a fourth file (scripts/tests/test_benchmark_reporting.py) beyond the three named suites.** — Accepted as harmless and matching the acceptance command; a clarifying ci.yml comment added. _(amend_plan: no)_

### Testing coverage assessment

| Case | Location | Confidence |
|---|---|---|
| measurement_id Rust==golden==Python parity across all 5 tables incl. negative i32, null/Some, multibyte UTF-8, whole/negative/tiny/large f64 thresholds; SCHEMA_VERSION lockstep with schema.rs | `scripts/test_measurement_id.py + scripts/measurement_id_golden.json; scripts/test_post_ingest_postgres.py:3105` | high |
| bench_ingest DML-only (SELECT/INSERT/UPDATE) on all 6 tables, denied DELETE/CREATE/DDL; default-privileges cover future migrator tables; 001..004 apply cleanly + idempotently under a real NOSUPERUSER CREATEROLE master | `scripts/test_migrate_schema.py:1960,1928,1993,2069` | high |
| provision.sh provisions ingest role on instance dbuser, emits GH_BENCH_INGEST_ROLE_ARN, drops dead proxy grant | `scripts/test_migrate_schema.py:1805,1835` | high |
| insert-then-update accounting; re-ingest upserts (0 inserted, N updated) with stable counts; measurement_id matches port | `scripts/test_post_ingest_postgres.py:2477,2572` | high |
| ON CONFLICT DO UPDATE overwrites every value/env column while leaving dim tuple / measurement_id stable, per table; dim columns NEVER in any DO UPDATE SET list for all 5 tables | `scripts/test_post_ingest_postgres.py:2867` | high |
| NaN/Inf/out-of-f64-range threshold raises loudly and rolls back; deny_unknown_fields; missing-required; unknown/non-scalar kind; storage enum; memory quartet; commit_sha mismatch; type/range boundaries reproduced | `scripts/test_post_ingest_postgres.py:2633,2887,3066,3074,3094` | high |
| connect_postgres: IAM-token-when-passwordless, always-bench_ingest, rejects weak sslmode (verify-full), forces search_path=public, rejects non-TLS, post-connect ssl_in_use | `scripts/test_post_ingest_postgres.py:3142,3221,3233,3264,3285` | high |
| --server requires --benchmark-id while --postgres does not; mutual exclusivity + dispatch | `scripts/test_post_ingest_postgres.py:3401,3363,3375` | high |
| fail-loud-on-no-Docker in CI vs skip locally (both test files) | `scripts/test_migrate_schema.py:1745; scripts/test_post_ingest_postgres.py:2753` | high |
| write-conflict retry (deadlock/serialization) retries then succeeds, gives up at cap, propagates validation errors immediately | `scripts/test_post_ingest_postgres.py (test_retry_write_conflicts_*)` | medium |

**Untested / gaps:**

| Case | Priority | Why untested |
|---|---|---|
| End-to-end live v4 dual-write per develop push (CloudWatch shows Postgres writes) — the PR-2.4 acceptance criterion needing real RDS + repo vars | high | Operator-side, not performable in-session; the v4 steps no-op until GH_BENCH_INGEST_ROLE_ARN + RDS_BENCH_INSTANCE_ENDPOINT/DB_NAME/REGION are set. Tracked as an open operator dependency. |
| Real psycopg DeadlockDetected inside conn.transaction() across two reversed-order connections on autocommit=False (retry+commit/rollback) | medium | Retry path covered only by mocked-op unit tests; a real-conflict container test needs deterministic threaded interleaving. Explicitly deferred (PR-2.2 cycle-7) to the follow-up test-hardening PR; behavior manually verified against live PG16. |
| migration 004 self-grant when the executing role lacks ADMIN on a pre-existing migrator (unsupported misconfiguration) | low | Deferred; the shipped single-bootstrap-master path is correct and tested. |
| schema_conn BENCH_TEST_PG_DSN destructive-scrub guard against a localhost-tunnel-to-prod | low | Dev-only override never used in CI (testcontainers only); explicitly accepted/deferred as an exotic foot-gun. |

_Recommendations:_ Coverage of the load-bearing data-correctness invariants is comprehensive and pins each against an automated test, satisfying behavior-preservation against the v3 serde/ingest boundary per-table. The only material gap is the live end-to-end dual-write, gated on operator var-setting and best-effort by design. The deferred real-deadlock/autocommit=False container test is the one worth landing in the follow-up test-hardening PR; the remaining untested cases are availability/test-infra hardening already triaged and out of scope per the lean re-plan.

### Tradeoffs re-evaluation

- **Schema-deploy authorization (PR merge is the gate; push-on-develop trigger; no environment/manual-approval gate)** — verdict: **keep**. PR-2.4 implements exactly this: push-on-develop under paths migrations/**, dry_run kept, environment-gate comments removed. Execution safety comes from the per-PR testcontainer migration test. Accepted tradeoff; not re-flagged. _(found_by: spec, correctness)_
- **Ingest writer language (pure Python extending post-ingest.py; xxhash port)** — verdict: **keep**. PR-2.2 delivered exactly this; the port matches Table A/B and SCHEMA_VERSION lockstep holds. _(found_by: spec)_
- **Cutover style (short best-effort dual-write soak, then promote v4); v4 dual-write is continue-on-error during soak, v3 stays hard-required** — verdict: **keep**. PR-2.4's continue-on-error v4 steps are the direct realization; a v4 OIDC/connect hiccup must never break the proven v3 pipeline; v4 promoted to required at cutover (PR-5.1). Both steps consume the same results.v3.jsonl, so neither substrate is written in isolation. Calibrated, accepted exception to the no-best-effort rule. _(found_by: spec, correctness)_
- **Phase-2 ingest DB identity (dedicated bench_ingest + GitHubBenchmarkIngestRole, separate from migrator)** — verdict: **keep**. PR-2.1 implemented precisely this with DML-only grants and a separate OIDC role; separation of duties; the writer enforces bench_ingest unconditionally. Pinned by tests. _(found_by: spec, correctness)_
- **Phase-2 dual-write verify scope (verify-once via Phase-3 migrate --verify; drop reconcile-ingest.py + dual-write-verify.yml + incident.io)** — verdict: **keep**. PR-2.5 correctly dropped; PR-2.4 adds no reconciliation machinery; v4-only failure does not trigger incident.io. Four independent safety nets remain. Not drift. _(found_by: spec)_
- **Duplicate JSON keys collapse last-wins (no rejection), matching v3** — verdict: **keep**. serde_json::Value::Object and Python json.loads are both last-wins; rejecting would break behavior-preservation. Not re-flagged. _(found_by: correctness)_
- **post-ingest.py PEP 723 dependencies = []; --postgres runs from the uv env** — verdict: **keep**. Keeps the v3 --server path stdlib-only under bare python3; --postgres deps come from the uv workspace. Lazy imports verified. Not re-flagged. _(found_by: correctness)_
- **IAM-token region precedence: --region > boto3 session region > RDS-hostname-parsed** — verdict: **keep**. A wrong region fails loud at IAM connect, not silently; ordering matches AWS convention. Not re-flagged. _(found_by: correctness)_

### Disagreements

None. Both lenses (spec + correctness) accepted at high confidence with no contradictory findings or tradeoff verdicts.

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1)</summary>

```json
{
  "schema_version": 1,
  "preset": "phase-2",
  "lenses_used": ["spec", "correctness"],
  "review_count": 2,
  "unified_findings": [
    {
      "severity": "nit",
      "kind": "doc-quality",
      "file_line": "benchmarks-website/infra/README.md:81",
      "description": "README var table still attributes RDS_BENCH_INSTANCE_ENDPOINT/GH_BENCH_INGEST_ROLE_ARN consumers to 'PR-2.2' ingest workflows, but the dual-write steps actually landed in PR-2.4; minor lineage mislabel in shipped docs.",
      "recommended_fix": "Update the 'used by' column references from PR-2.2 to PR-2.4 for the ingest-workflow consumers of RDS_BENCH_INSTANCE_ENDPOINT and GH_BENCH_INGEST_ROLE_ARN, to match where the dual-write steps were actually added.",
      "found_by": ["spec"]
    },
    {
      "severity": "nit",
      "kind": "other",
      "file_line": "scripts/post-ingest.py:1168",
      "description": "_main_postgres calls build_commit() (git show -s <sha>) for every run; bench.yml/sql-benchmarks.yml checkouts are not pinned to fetch-depth the way v3-commit-metadata.yml sets fetch-depth:2. Behavior matches v3 (--server also builds the commit), so it is parity, not drift.",
      "recommended_fix": "No change required for spec adherence (best-effort + continue-on-error absorbs any git-history miss). Optionally note in a comment that the v4 step inherits the v3 step's git-history assumption.",
      "found_by": ["spec"]
    }
  ],
  "disagreements": [],
  "dropped_re_flags": [],
  "phase_artifacts": {
    "summary": "Phase 2 delivers the Postgres dual-write ingest path and its CI plumbing, shipped exactly to the lean re-planned scope across PR-2.1/2.2/2.3/2.4 with no drift in either direction; correctness review found zero data-correctness bugs. Ingest identity (PR-2.1): migration 004 creates a least-privilege bench_ingest login role (rds_iam member on RDS, guarded for vanilla PG) holding USAGE-not-CREATE on public and SELECT/INSERT/UPDATE on the six tables (no DELETE/TRUNCATE/DDL), plus ALTER DEFAULT PRIVILEGES FOR ROLE migrator so future migrator tables auto-grant. Because the RDS master is rds_superuser but not a true superuser, the ADP runs via a temporary INHERIT self-grant through the creator's ADMIN option, then revoke. provision.sh adds GitHubBenchmarkIngestRole (OIDC trust branch-scoped to develop+ct/bench-v4, rds-db:connect for bench_ingest on the instance only) and drops the dead proxy grant. Writer (PR-2.2): post-ingest.py --postgres reproduces the v3 serde boundary in Python (deny_unknown_fields, per-field type/range validation, memory quartet, storage enum, commit_sha match), computes measurement_id bit-identically via the PR-1.5 port (_measurement_id.py is a byte-for-byte port of db.rs:162-257, pinned transitively Rust==golden==Python), then upserts commits-first and five fact tables in one all-or-nothing transaction via INSERT ... ON CONFLICT DO UPDATE with deadlock retry, over verify-full TLS authenticated by an RDS IAM token as bench_ingest. An is_finite guard rejects NaN/Inf/out-of-f64-range threshold loudly with rollback (Python stricter-or-equal to serde). All six ON CONFLICT SET lists touch only value/env columns, never dim columns; search_path is pinned to public; ssl_in_use is asserted post-connect. The v3 --server path stays stdlib-only (PEP 723 dependencies=[]). SCHEMA_VERSION lockstep held. CI wiring (PR-2.3): a scripts-test job runs pytest scripts/ behind a docker info hard gate plus a CI-env fail-loud fixture so testcontainer suites cannot silently skip. Dual-write CI (PR-2.4): the three ingest workflows each add a best-effort continue-on-error v4 --postgres step after the unchanged hard-required v3 step (v3-commit-metadata.yml gains id-token: write and ingests commit-row-only empty.jsonl); schema-deploy.yml switches to push-on-develop under paths migrations/**, keeping workflow_dispatch+dry_run. PR-2.5 (reconcile-ingest.py + dual-write-verify.yml + incident.io) was intentionally dropped; v4 correctness is deferred to Phase-3 migrate --verify. All five measurement_id functions, six ON CONFLICT SET lists, field/type sets, and column widths were verified against the Rust source and 001_initial_schema.sql. Two cosmetic nits, no must-fix or should-fix.",
    "surprises": [
      { "what": "Migration 004 ALTER DEFAULT PRIVILEGES FOR ROLE migrator fails on a real non-superuser RDS master (createrole_self_grant default), masked by the superuser testcontainer.", "how_handled": "Fixed PR-2.1 cycle-1: guarded INHERIT self-grant via the master's ADMIN option then revoke, plus a test applying 001..004 as a real NOSUPERUSER CREATEROLE login.", "amend_plan": "already-done" },
      { "what": "conn.info.ssl_in_use does not exist in psycopg (ssl_in_use lives on conn.pgconn); the cycle-9 fix shipped the wrong accessor.", "how_handled": "Fixed PR-2.2 (conn.pgconn.ssl_in_use) with a unit test pinning the accessor location and a container test pinning the traversal.", "amend_plan": "already-done" },
      { "what": "fail-loud-on-no-Docker guard tests could not catch an always-skip regression because pytest.skip raises Skipped (not a Failed subclass).", "how_handled": "Fixed PR-2.3 (catch both outcome types and assert the specific one; mutation-verified).", "amend_plan": "already-done" },
      { "what": "Python json.loads accepts NaN/Infinity literals that serde_json rejects at parse time.", "how_handled": "The _require_finite guard rejects them loudly with rollback, making the Python writer stricter-or-equal to v3, so no divergent row is written. Already handled.", "amend_plan": "no" },
      { "what": "The v3 producer emits several fields as u32/u64 (query_idx u32; value_ns/all_runtimes_ns u64) where server and Python validator use i32/i64.", "how_handled": "Both v3 serde and Python _require_int reject values exceeding the signed range, so the substrates agree; values are far inside range. No change needed.", "amend_plan": "no" },
      { "what": "insert-vs-update classification differs across substrates: Rust does an exists() preflight SELECT; Python derives the flag from RETURNING (xmax = 0).", "how_handled": "Equivalence (incl. same-dim-tuple-twice-in-one-envelope yielding (1,1) not (2,0)) is pinned by test_same_dim_tuple_twice_in_one_envelope_counts_second_as_update. Already handled.", "amend_plan": "already-done" },
      { "what": "bench-v4 Python files predate Phase 2 at ~100-col and fail the repo ruff line-length-120 (a pre-existing E501).", "how_handled": "Left unfixed and flagged for the operator as a branch-wide ruff reconciliation required before merging ct/bench-v4 to develop; RETAINED deferred item.", "amend_plan": "no" },
      { "what": "pytest scripts/ collects a fourth file (scripts/tests/test_benchmark_reporting.py) beyond the three named suites.", "how_handled": "Accepted as harmless and matching the acceptance command; a clarifying ci.yml comment added.", "amend_plan": "no" }
    ],
    "coverage": {
      "tested_cases": [
        { "case": "measurement_id Rust==golden==Python parity across all 5 tables incl. negative i32, null/Some, multibyte UTF-8, whole/negative/tiny/large f64 thresholds; SCHEMA_VERSION lockstep with schema.rs", "test_location": "scripts/test_measurement_id.py + scripts/measurement_id_golden.json; scripts/test_post_ingest_postgres.py:3105", "confidence": "high" },
        { "case": "bench_ingest DML-only (SELECT/INSERT/UPDATE) on all 6 tables, denied DELETE/CREATE/DDL; default-privileges cover future migrator tables; 001..004 apply cleanly + idempotently under a real NOSUPERUSER CREATEROLE master", "test_location": "scripts/test_migrate_schema.py:1960,1928,1993,2069", "confidence": "high" },
        { "case": "provision.sh provisions ingest role on instance dbuser, emits GH_BENCH_INGEST_ROLE_ARN, drops dead proxy grant", "test_location": "scripts/test_migrate_schema.py:1805,1835", "confidence": "high" },
        { "case": "insert-then-update accounting; re-ingest upserts (0 inserted, N updated) with stable counts; measurement_id matches port", "test_location": "scripts/test_post_ingest_postgres.py:2477,2572", "confidence": "high" },
        { "case": "ON CONFLICT DO UPDATE overwrites every value/env column while leaving dim tuple / measurement_id stable, per table; dim columns NEVER in any DO UPDATE SET list for all 5 tables", "test_location": "scripts/test_post_ingest_postgres.py:2867", "confidence": "high" },
        { "case": "NaN/Inf/out-of-f64-range threshold raises loudly and rolls back; deny_unknown_fields; missing-required; unknown/non-scalar kind; storage enum; memory quartet; commit_sha mismatch; type/range boundaries reproduced", "test_location": "scripts/test_post_ingest_postgres.py:2633,2887,3066,3074,3094", "confidence": "high" },
        { "case": "connect_postgres: IAM-token-when-passwordless, always-bench_ingest, rejects weak sslmode (verify-full), forces search_path=public, rejects non-TLS, post-connect ssl_in_use", "test_location": "scripts/test_post_ingest_postgres.py:3142,3221,3233,3264,3285", "confidence": "high" },
        { "case": "--server requires --benchmark-id while --postgres does not; mutual exclusivity + dispatch", "test_location": "scripts/test_post_ingest_postgres.py:3401,3363,3375", "confidence": "high" },
        { "case": "fail-loud-on-no-Docker in CI vs skip locally (both test files)", "test_location": "scripts/test_migrate_schema.py:1745; scripts/test_post_ingest_postgres.py:2753", "confidence": "high" },
        { "case": "write-conflict retry (deadlock/serialization) retries then succeeds, gives up at cap, propagates validation errors immediately", "test_location": "scripts/test_post_ingest_postgres.py (test_retry_write_conflicts_*)", "confidence": "medium" }
      ],
      "untested_cases": [
        { "case": "End-to-end live v4 dual-write per develop push (CloudWatch shows Postgres writes) — the PR-2.4 acceptance criterion needing real RDS + repo vars", "priority": "high", "why_untested": "Operator-side, not performable in-session; the v4 steps no-op until GH_BENCH_INGEST_ROLE_ARN + RDS_BENCH_INSTANCE_ENDPOINT/DB_NAME/REGION are set. Tracked as an open operator dependency." },
        { "case": "Real psycopg DeadlockDetected inside conn.transaction() across two reversed-order connections on autocommit=False (retry+commit/rollback)", "priority": "medium", "why_untested": "Retry path covered only by mocked-op unit tests; a real-conflict container test needs deterministic threaded interleaving. Explicitly deferred (PR-2.2 cycle-7) to the follow-up test-hardening PR; behavior manually verified against live PG16." },
        { "case": "migration 004 self-grant when the executing role lacks ADMIN on a pre-existing migrator (unsupported misconfiguration)", "priority": "low", "why_untested": "Deferred; the shipped single-bootstrap-master path is correct and tested." },
        { "case": "schema_conn BENCH_TEST_PG_DSN destructive-scrub guard against a localhost-tunnel-to-prod", "priority": "low", "why_untested": "Dev-only override never used in CI (testcontainers only); explicitly accepted/deferred as an exotic foot-gun." }
      ],
      "recommendations": "Coverage of the load-bearing data-correctness invariants is comprehensive and pins each against an automated test, satisfying behavior-preservation against the v3 serde/ingest boundary per-table. The only material gap is the live end-to-end dual-write, gated on operator var-setting and best-effort by design. The deferred real-deadlock/autocommit=False container test is the one worth landing in the follow-up test-hardening PR; the remaining untested cases are availability/test-infra hardening already triaged and out of scope per the lean re-plan."
    },
    "tradeoffs": [
      { "decision": "Schema-deploy authorization (PR merge is the gate; push-on-develop trigger; no environment/manual-approval gate)", "original": "Supersedes the original manual-approval mandate (2026-05-29 deploy-model decision)", "verdict": "keep", "rationale": "PR-2.4 implements exactly this: push-on-develop under paths migrations/**, dry_run kept, environment-gate comments removed. Execution safety comes from the per-PR testcontainer migration test. Accepted tradeoff; not re-flagged.", "found_by": ["spec", "correctness"] },
      { "decision": "Ingest writer language (pure Python extending post-ingest.py; xxhash port)", "original": "Avoid sqlx/aws-sdk Rust deps", "verdict": "keep", "rationale": "PR-2.2 delivered exactly this; the port matches Table A/B and SCHEMA_VERSION lockstep holds.", "found_by": ["spec"] },
      { "decision": "Cutover style (short best-effort dual-write soak, then promote v4); v4 dual-write is continue-on-error during soak, v3 stays hard-required", "original": "Amended 2026-06-04 to make v4 best-effort during soak (PR-2.4 lean re-plan)", "verdict": "keep", "rationale": "PR-2.4's continue-on-error v4 steps are the direct realization; a v4 OIDC/connect hiccup must never break the proven v3 pipeline; v4 promoted to required at cutover (PR-5.1). Both steps consume the same results.v3.jsonl, so neither substrate is written in isolation. Calibrated, accepted exception to the no-best-effort rule.", "found_by": ["spec", "correctness"] },
      { "decision": "Phase-2 ingest DB identity (dedicated bench_ingest + GitHubBenchmarkIngestRole, separate from migrator)", "original": "Re-plan Q2 least-privilege split (PR-2.1)", "verdict": "keep", "rationale": "PR-2.1 implemented precisely this with DML-only grants and a separate OIDC role; separation of duties; the writer enforces bench_ingest unconditionally. Pinned by tests.", "found_by": ["spec", "correctness"] },
      { "decision": "Phase-2 dual-write verify scope (verify-once via Phase-3 migrate --verify; drop reconcile-ingest.py + dual-write-verify.yml + incident.io)", "original": "Superseded the 2026-06-01 per-push reconciliation harness", "verdict": "keep", "rationale": "PR-2.5 correctly dropped; PR-2.4 adds no reconciliation machinery; v4-only failure does not trigger incident.io. Four independent safety nets remain. Not drift.", "found_by": ["spec"] },
      { "decision": "Duplicate JSON keys collapse last-wins (no rejection), matching v3", "original": "PR-2.2 cycle-6", "verdict": "keep", "rationale": "serde_json::Value::Object and Python json.loads are both last-wins; rejecting would break behavior-preservation. Not re-flagged.", "found_by": ["correctness"] },
      { "decision": "post-ingest.py PEP 723 dependencies = []; --postgres runs from the uv env", "original": "PR-2.2", "verdict": "keep", "rationale": "Keeps the v3 --server path stdlib-only under bare python3; --postgres deps come from the uv workspace. Lazy imports verified. Not re-flagged.", "found_by": ["correctness"] },
      { "decision": "IAM-token region precedence: --region > boto3 session region > RDS-hostname-parsed", "original": "PR-2.2", "verdict": "keep", "rationale": "A wrong region fails loud at IAM connect, not silently; ordering matches AWS convention. Not re-flagged.", "found_by": ["correctness"] }
    ]
  },
  "executive_summary": "Phase 2 ships the Postgres dual-write ingest path and its CI plumbing, exactly to the lean re-planned scope across PR-2.1/2.2/2.3/2.4 with no drift in either direction. Both lenses accept at high confidence; the correctness lens found zero data-correctness bugs and the spec lens found only two cosmetic nits, so there are no must-fix or should-fix items. PR-2.1 adds a least-privilege bench_ingest login role (migration 004) with SELECT/INSERT/UPDATE on the six tables (no DELETE/TRUNCATE/DDL), USAGE-not-CREATE, and ALTER DEFAULT PRIVILEGES so future migrator tables auto-grant; provision.sh adds a branch-scoped OIDC GitHubBenchmarkIngestRole (rds-db:connect on the instance only) and drops the dead proxy grant. PR-2.2's post-ingest.py --postgres reproduces the v3 serde boundary in Python, computes measurement_id bit-identically via the PR-1.5 port, and upserts commits-first then five fact tables in one all-or-nothing transaction with deadlock retry over verify-full TLS and IAM auth. PR-2.3 runs pytest scripts/ behind a docker hard gate plus a CI fail-loud fixture. PR-2.4 makes each of the three ingest workflows add a best-effort continue-on-error v4 step after the unchanged hard-required v3 step, and switches schema-deploy to push-on-develop under paths migrations/**. PR-2.5 was intentionally dropped; v4 correctness is deferred to Phase-3 migrate --verify. The key engineering surprises were all already resolved in-cycle: the non-superuser RDS master breaking ALTER DEFAULT PRIVILEGES (fixed with a guarded INHERIT self-grant + a real NOSUPERUSER test), the ssl_in_use accessor living on conn.pgconn rather than conn.info (fixed and pinned), and the fail-loud-on-no-Docker guard being unable to catch always-skip because pytest.skip is not a Failed subclass (fixed and mutation-verified). Correctness independently confirmed Python-is-stricter-or-equal handling of NaN/Inf literals, agreement on u32/u64-vs-i32/i64 ranges, and insert-vs-update equivalence between Rust's exists() preflight and Python's xmax=0 classifier. Every Key decision was verdicted keep by both reviewers: schema-deploy PR-merge gating, pure-Python writer + xxhash port, best-effort dual-write during soak with v3 hard-required, the dedicated bench_ingest identity, dropping the reconcile harness, last-wins duplicate-key handling, PEP 723 empty-deps, and IAM region precedence. Coverage of load-bearing data-correctness invariants is comprehensive and per-table-pinned; the one material gap is the live end-to-end dual-write, which is operator-gated on repo vars and best-effort by design. Verdict: ACCEPT.",
  "overall": "accept",
  "must_fix_count": 0,
  "should_fix_count": 0,
  "nit_count": 2,
  "review_cycles_this_invocation": 1
}
```

</details>

## Phase 2: Postgres writer + best-effort v4 CI — end-of-phase review (cycle 2) — accepted (2-vote)
**Synthesizer output from /spiral:gauntlet (preset=phase-2: spec + correctness, executor=claude); amend-triggered re-review of the cumulative amended phase. Full JSON in the `<details>` block + the cycle-2 archive entry. Verdict: accept (0 must-fix, 0 should-fix, 2 dismissable nits).**

### Summary of changes

Amended cumulative Phase 2 delivers the Postgres dual-write ingest path + best-effort v4 CI. Ingest identity (PR-2.1): migrations/004 creates least-privilege bench_ingest (SELECT/INSERT/UPDATE on the 6 tables, USAGE-not-CREATE, ALTER DEFAULT PRIVILEGES for future migrator tables, no DELETE/DDL); provision.sh adds the branch-scoped OIDC GitHubBenchmarkIngestRole (rds-db:connect on the instance dbuser) and drops the dead proxy grant; 004 self-grants migrator INHERIT via the master's ADMIN option then revokes (non-superuser-RDS-master bootstrap fix). Writer (PR-2.2, de-gold-plated by PR-2.7): post-ingest.py --postgres computes measurement_id bit-identically via the byte-exact _measurement_id.py port, validates every field/type/range to reproduce the v3 serde + deny_unknown_fields boundary, upserts commits-first then 5 fact tables via INSERT...ON CONFLICT(measurement_id) DO UPDATE...RETURNING(xmax=0) in one all-or-nothing transaction with deadlock/serialization retry; connect_postgres enforces verify-full TLS + post-connect ssl_in_use + bench_ingest-only + IAM-token mint + search_path=public. CI wiring (PR-2.3/2.4/2.6): a scripts-test job runs pytest scripts/ (Docker-required, fail-loud-not-skip in CI); the 3 ingest workflows add a continue-on-error best-effort v4 --postgres step after the unchanged hard-required v3 step; schema-deploy.yml triggers on push to develop under migrations/**. Amendments: PR-2.6 re-keys the 9 v4 gates from the endpoint var to GH_BENCH_INGEST_ROLE_ARN so they no-op (not fire+fail) until infra is wired; PR-2.7 de-gold-plates ~194 lines of trusted-input over-hardening (NUL/surrogate/non-UTF-8/RecursionError/oversized-int guards + dead _is_local_host) + ~8 tests, moving the writer CLOSER to the v3 Rust source; PR-2.8 clears the ruff E501 merge-blocker + README PR-lineage + a git-history comment. Every preserved invariant (measurement_id parity, ON CONFLICT SET membership, NaN/Inf guard, typed validation, memory-quartet, IAM/TLS) remains pinned by tests; the de-gold-plate removed only stricter-than-v3 hardening on trusted CI input.

### Surprises and discoveries

- **PR-2.7's OverflowError-branch removal left an orphaned pytest.param(10**309) in the KEPT test_nonfinite_threshold_raises_and_rolls_back, which would have gone CI-red (uncaught OverflowError); only skipped locally for lack of Docker.** — Caught by cycle-1 correctness (cumulative-test trace), fixed in b6d1f292b (param dropped; nan/inf/-inf retained). Verified gone. _(amend_plan: already-done)_
- **The de-gold-plate moves the writer CLOSER to v3: the removed NUL/surrogate/non-UTF-8 guards were STRICTER than the v3 Rust source, so removal improves behavior-preservation rather than regressing it; failure modes stayed LOUD (UnicodeEncodeError in _write_str / Postgres DataError at bind, inside the transaction -> rollback). No silent-wrong-write path.** — Authorized scope-reduction per the 2026-06-05 complexity audit; KEEP-list verified intact field-by-field. _(amend_plan: no)_
- **The live gate-var bug (PR-2.6) was a real config mismatch: gates keyed on RDS_BENCH_INSTANCE_ENDPOINT (set) while assume-role uses GH_BENCH_INGEST_ROLE_ARN (unset), so the v4 steps would fire+fail at assume-role on the next develop push.** — Re-keyed all 9 gates to the role-ARN var; live-var state now makes the gate evaluate false (clean no-op until wired). _(amend_plan: already-done)_

### Testing coverage assessment

| Case | Location | Confidence |
|---|---|---|
| measurement_id parity (5 kinds) vs the Python port + stored rows; SCHEMA_VERSION lockstep | `scripts/test_measurement_id.py (65 vectors) + test_post_ingest_postgres.py (measurement_ids_match_python_port, schema_version)` | high |
| ON CONFLICT SET excludes dim columns (BAN) across all 5 _insert_ fns; per-table value-column update | `test_on_conflict_set_excludes_dim_columns + test_update_overwrites_all_value_columns_per_table` | high |
| insert-vs-update accounting incl. same-dim-twice (xmax=0 classifier); commit-before-facts; all-or-nothing rollback | `test_ingest_inserts_then_updates, test_same_dim_tuple_twice..., test_late_validation_failure_rolls_back_earlier_fact_row` | high |
| NaN/+Inf/-Inf threshold rejected + rollback (post de-gold-plate, OverflowError param removed) | `test_nonfinite_threshold_raises_and_rolls_back` | high |
| typed i32/i64/finite + deny_unknown_fields + memory-quartet + storage-enum + commit_sha-mismatch validation | `test_post_ingest_postgres.py value-validation tests` | high |
| connect_postgres: verify-full + post-connect ssl_in_use + bench_ingest-only + IAM-token-vs-password + search_path pin | `test_connect_postgres_* (9 pure-unit tests, pass non-Docker)` | high |
| de-gold-plate replacements: mixed-newline read, malformed-JSON loud-fail, git_show text-mode decode | `test_read_records_happy_path_mixed_newlines / _rejects_malformed_json / test_git_show_field_decodes_and_strips` | high |
| bench_ingest least-privilege + 004 idempotency + non-superuser-master bootstrap; provision.sh ingest role + dropped proxy grant | `scripts/test_migrate_schema.py (testcontainer + static provision tests)` | high |
| v4 gate keys on GH_BENCH_INGEST_ROLE_ARN; CI fail-loud-not-skip on missing Docker | `workflow inspection + test_require_docker_fails_loud_in_ci/_skips_without_ci` | high |

**Untested / gaps:**

| Case | Priority | Why |
|---|---|---|
| Live v4 dual-write end-to-end against real RDS per develop push (OIDC->IAM->verify-full->upsert; CloudWatch confirmation) | medium | Requires live AWS + operator-set repo vars; operator-side soak. Acceptance for PR-2.4 is the structural/yamllint criteria; v4 correctness is gated by Phase-3 migrate --verify. |
| Real two-connection DeadlockDetected on an autocommit=False connection asserting retry+commit | low | Container-only threaded-interleaving test; deferred (PR-2.2 cycle-7) to a follow-up test-hardening PR. Retry pinned by mocked-op unit tests + live-container verification. |
| Live OIDC schema-deploy apply firing on a migrations/** push | low | The retained operator pre-merge gate; not in-session testable. |

_Recommendations:_ Coverage is strong for a trusted-input low-stakes dashboard and matches the lean calibration. The only meaningful gaps are the two retained operator-side gates (live OIDC schema-deploy apply; live v4 dual-write soak + CloudWatch). No new test work is required to accept the amended phase. The deferred real-deadlock/autocommit=False integration test remains a reasonable follow-up but is not a blocker.

### Tradeoffs re-evaluation

- **v4 ingest steps continue-on-error (best-effort) during the soak** — **keep**. v3 stays hard-required + untouched; the 'no continue-on-error' BAN applies at the PR-5.1 cutover, not the soak. PR-2.6 reinforces it (clean no-op until wired).
- **Gate v4 steps on GH_BENCH_INGEST_ROLE_ARN (PR-2.6)** — **keep**. Aligns the gate with the assume-role input so the steps no-op (not fire+fail) until infra is wired; verified across all 9 gates.
- **De-gold-plate trusted-input hardening (PR-2.7)** — **keep**. Removed only stricter-than-v3 guards on trusted CI input; every removed input still fails loud inside the transaction (rollback). No data-correctness guard or its test removed.
- **Schema-deploy authorization (PR-merge gate, push trigger, no environment: gate)** — **keep**. PR-2.4 shipped the trigger; the amendment did not touch it. Accepted tradeoff.
- **Phase-2 dual-write verify scope (verify-once via Phase-3 migrate --verify; PR-2.5 dropped)** — **keep**. The amendment added no reconciliation machinery; consistent with the superseding decision.
- **Pure-Python writer + PEP 723 dependencies=[]; IAM region precedence** — **keep**. Accepted tradeoffs; keeps v3 --server stdlib-only; region mismatch fails loud. Unchanged by the amendment.
- **Dedicated bench_ingest identity separate from migrator** — **keep**. Unchanged; PR-2.6 gates on the ingest role ARN, reinforcing the separate-identity model; connect_postgres still enforces bench_ingest-only.

### Disagreements

None. Both lenses accept at high confidence; the 2 nits are dismissable (comment at 100-col limit; huge-int→OverflowError still fails loud, an intentional de-gold-plate tradeoff).

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1)</summary>

```json
{
  "schema_version": 1, "preset": "phase-2", "lenses_used": ["spec", "correctness"], "review_count": 2,
  "unified_findings": [
    {"severity":"nit","kind":"convention","file_line":"scripts/post-ingest.py:1085","description":"PR-2.8's build_commit git-history comment lines sit at exactly 100 cols (at the user CLAUDE.md limit, not over; well under ruff's 120). Already rewrapped from 102 in f0e93b9f4. No action.","recommended_fix":"None; within the 100-col rule.","found_by":["spec"]},
    {"severity":"nit","kind":"error-path","file_line":"scripts/post-ingest.py:448","description":"After the de-gold-plate dropped the _require_finite OverflowError sub-branch, a huge-int threshold (e.g. 10**309) now raises an uncaught OverflowError instead of a controlled SystemExit. Still fails LOUD + rolls back the transaction (no silent-wrong-write); the input class (oversized-int on trusted CI) is explicitly DO-NOT-flag per the calibration. Intentional de-gold-plate tradeoff.","recommended_fix":"None required (accepted). A one-line `except OverflowError: is_finite_number = False` would restore the controlled error, but is not worth carrying per the trusted-input calibration.","found_by":["correctness"]}
  ],
  "disagreements": [],
  "dropped_re_flags": [],
  "phase_artifacts": {
    "summary": "Amended cumulative Phase 2 delivers the Postgres dual-write ingest path + best-effort v4 CI. Ingest identity (PR-2.1): migrations/004 creates least-privilege bench_ingest (SELECT/INSERT/UPDATE on the 6 tables, USAGE-not-CREATE, ALTER DEFAULT PRIVILEGES for future migrator tables, no DELETE/DDL); provision.sh adds the branch-scoped OIDC GitHubBenchmarkIngestRole (rds-db:connect on the instance dbuser) and drops the dead proxy grant; 004 self-grants migrator INHERIT via the master's ADMIN option then revokes (non-superuser-RDS-master bootstrap fix). Writer (PR-2.2, de-gold-plated by PR-2.7): post-ingest.py --postgres computes measurement_id bit-identically via the byte-exact _measurement_id.py port, validates every field/type/range to reproduce the v3 serde + deny_unknown_fields boundary, upserts commits-first then 5 fact tables via INSERT...ON CONFLICT(measurement_id) DO UPDATE...RETURNING(xmax=0) in one all-or-nothing transaction with deadlock/serialization retry; connect_postgres enforces verify-full TLS + post-connect ssl_in_use + bench_ingest-only + IAM-token mint + search_path=public. CI wiring (PR-2.3/2.4/2.6): a scripts-test job runs pytest scripts/ (Docker-required, fail-loud-not-skip in CI); the 3 ingest workflows add a continue-on-error best-effort v4 --postgres step after the unchanged hard-required v3 step; schema-deploy.yml triggers on push to develop under migrations/**. Amendments: PR-2.6 re-keys the 9 v4 gates from the endpoint var to GH_BENCH_INGEST_ROLE_ARN so they no-op (not fire+fail) until infra is wired; PR-2.7 de-gold-plates ~194 lines of trusted-input over-hardening (NUL/surrogate/non-UTF-8/RecursionError/oversized-int guards + dead _is_local_host) + ~8 tests, moving the writer CLOSER to the v3 Rust source; PR-2.8 clears the ruff E501 merge-blocker + README PR-lineage + a git-history comment. Every preserved invariant (measurement_id parity, ON CONFLICT SET membership, NaN/Inf guard, typed validation, memory-quartet, IAM/TLS) remains pinned by tests; the de-gold-plate removed only stricter-than-v3 hardening on trusted CI input.",
    "surprises": [
      {"what":"PR-2.7's OverflowError-branch removal left an orphaned pytest.param(10**309) in the KEPT test_nonfinite_threshold_raises_and_rolls_back, which would have gone CI-red (uncaught OverflowError); only skipped locally for lack of Docker.","how_handled":"Caught by cycle-1 correctness (cumulative-test trace), fixed in b6d1f292b (param dropped; nan/inf/-inf retained). Verified gone.","amend_plan":"already-done"},
      {"what":"The de-gold-plate moves the writer CLOSER to v3: the removed NUL/surrogate/non-UTF-8 guards were STRICTER than the v3 Rust source, so removal improves behavior-preservation rather than regressing it; failure modes stayed LOUD (UnicodeEncodeError in _write_str / Postgres DataError at bind, inside the transaction -> rollback). No silent-wrong-write path.","how_handled":"Authorized scope-reduction per the 2026-06-05 complexity audit; KEEP-list verified intact field-by-field.","amend_plan":"no"},
      {"what":"The live gate-var bug (PR-2.6) was a real config mismatch: gates keyed on RDS_BENCH_INSTANCE_ENDPOINT (set) while assume-role uses GH_BENCH_INGEST_ROLE_ARN (unset), so the v4 steps would fire+fail at assume-role on the next develop push.","how_handled":"Re-keyed all 9 gates to the role-ARN var; live-var state now makes the gate evaluate false (clean no-op until wired).","amend_plan":"already-done"}
    ],
    "coverage": {
      "tested_cases": [
        {"case":"measurement_id parity (5 kinds) vs the Python port + stored rows; SCHEMA_VERSION lockstep","test_location":"scripts/test_measurement_id.py (65 vectors) + test_post_ingest_postgres.py (measurement_ids_match_python_port, schema_version)","confidence":"high"},
        {"case":"ON CONFLICT SET excludes dim columns (BAN) across all 5 _insert_ fns; per-table value-column update","test_location":"test_on_conflict_set_excludes_dim_columns + test_update_overwrites_all_value_columns_per_table","confidence":"high"},
        {"case":"insert-vs-update accounting incl. same-dim-twice (xmax=0 classifier); commit-before-facts; all-or-nothing rollback","test_location":"test_ingest_inserts_then_updates, test_same_dim_tuple_twice..., test_late_validation_failure_rolls_back_earlier_fact_row","confidence":"high"},
        {"case":"NaN/+Inf/-Inf threshold rejected + rollback (post de-gold-plate, OverflowError param removed)","test_location":"test_nonfinite_threshold_raises_and_rolls_back","confidence":"high"},
        {"case":"typed i32/i64/finite + deny_unknown_fields + memory-quartet + storage-enum + commit_sha-mismatch validation","test_location":"test_post_ingest_postgres.py value-validation tests","confidence":"high"},
        {"case":"connect_postgres: verify-full + post-connect ssl_in_use + bench_ingest-only + IAM-token-vs-password + search_path pin","test_location":"test_connect_postgres_* (9 pure-unit tests, pass non-Docker)","confidence":"high"},
        {"case":"de-gold-plate replacements: mixed-newline read, malformed-JSON loud-fail, git_show text-mode decode","test_location":"test_read_records_happy_path_mixed_newlines / _rejects_malformed_json / test_git_show_field_decodes_and_strips","confidence":"high"},
        {"case":"bench_ingest least-privilege + 004 idempotency + non-superuser-master bootstrap; provision.sh ingest role + dropped proxy grant","test_location":"scripts/test_migrate_schema.py (testcontainer + static provision tests)","confidence":"high"},
        {"case":"v4 gate keys on GH_BENCH_INGEST_ROLE_ARN; CI fail-loud-not-skip on missing Docker","test_location":"workflow inspection + test_require_docker_fails_loud_in_ci/_skips_without_ci","confidence":"high"}
      ],
      "untested_cases": [
        {"case":"Live v4 dual-write end-to-end against real RDS per develop push (OIDC->IAM->verify-full->upsert; CloudWatch confirmation)","priority":"medium","why_untested":"Requires live AWS + operator-set repo vars; operator-side soak. Acceptance for PR-2.4 is the structural/yamllint criteria; v4 correctness is gated by Phase-3 migrate --verify."},
        {"case":"Real two-connection DeadlockDetected on an autocommit=False connection asserting retry+commit","priority":"low","why_untested":"Container-only threaded-interleaving test; deferred (PR-2.2 cycle-7) to a follow-up test-hardening PR. Retry pinned by mocked-op unit tests + live-container verification."},
        {"case":"Live OIDC schema-deploy apply firing on a migrations/** push","priority":"low","why_untested":"The retained operator pre-merge gate; not in-session testable."}
      ],
      "recommendations": "Coverage is strong for a trusted-input low-stakes dashboard and matches the lean calibration. The only meaningful gaps are the two retained operator-side gates (live OIDC schema-deploy apply; live v4 dual-write soak + CloudWatch). No new test work is required to accept the amended phase. The deferred real-deadlock/autocommit=False integration test remains a reasonable follow-up but is not a blocker."
    },
    "tradeoffs": [
      {"decision":"v4 ingest steps continue-on-error (best-effort) during the soak","original":"PR-2.4 lean re-plan amendment","verdict":"keep","rationale":"v3 stays hard-required + untouched; the 'no continue-on-error' BAN applies at the PR-5.1 cutover, not the soak. PR-2.6 reinforces it (clean no-op until wired)."},
      {"decision":"Gate v4 steps on GH_BENCH_INGEST_ROLE_ARN (PR-2.6)","original":"audit live-bug fix","verdict":"keep","rationale":"Aligns the gate with the assume-role input so the steps no-op (not fire+fail) until infra is wired; verified across all 9 gates."},
      {"decision":"De-gold-plate trusted-input hardening (PR-2.7)","original":"2026-06-05 complexity audit","verdict":"keep","rationale":"Removed only stricter-than-v3 guards on trusted CI input; every removed input still fails loud inside the transaction (rollback). No data-correctness guard or its test removed."},
      {"decision":"Schema-deploy authorization (PR-merge gate, push trigger, no environment: gate)","original":"2026-05-29 deploy-model decision","verdict":"keep","rationale":"PR-2.4 shipped the trigger; the amendment did not touch it. Accepted tradeoff."},
      {"decision":"Phase-2 dual-write verify scope (verify-once via Phase-3 migrate --verify; PR-2.5 dropped)","original":"lean re-plan","verdict":"keep","rationale":"The amendment added no reconciliation machinery; consistent with the superseding decision."},
      {"decision":"Pure-Python writer + PEP 723 dependencies=[]; IAM region precedence","original":"PR-2.2","verdict":"keep","rationale":"Accepted tradeoffs; keeps v3 --server stdlib-only; region mismatch fails loud. Unchanged by the amendment."},
      {"decision":"Dedicated bench_ingest identity separate from migrator","original":"re-plan Q2","verdict":"keep","rationale":"Unchanged; PR-2.6 gates on the ingest role ARN, reinforcing the separate-identity model; connect_postgres still enforces bench_ingest-only."}
    ]
  },
  "executive_summary": "Phase 2 cycle-2 phase-end review of the AMENDED cumulative phase (PR-2.1..2.4 accepted at cycle 1, plus the 2026-06-05 amendment PR-2.6/2.7/2.8). Both lenses ACCEPT with high confidence; 0 must-fix, 2 dismissable nits, no disagreements. The amendment did exactly three things and the reviewers verified each: PR-2.6 fixed a real LIVE gate-var bug (the v4 dual-write gates keyed on RDS_BENCH_INSTANCE_ENDPOINT [set] while assume-role uses GH_BENCH_INGEST_ROLE_ARN [unset], so the steps would fire+fail on the next develop push) by re-keying all 9 gates to the role-ARN var, restoring clean no-op-until-wired behavior. PR-2.7 de-gold-plated ~194 lines of trusted-input over-hardening (NUL/lone-surrogate/non-UTF-8/RecursionError/oversized-int guards + dead _is_local_host) that was STRICTER than the v3 Rust source it mirrors; the spec lens confirmed this is authorized scope-reduction (not silent de-scoping) and the correctness lens verified field-by-field that NO load-bearing data-correctness validation was removed (measurement_id parity, ON CONFLICT SET membership, NaN/Inf guard, typed i32/i64/finite, deny_unknown_fields, memory-quartet, IAM/TLS all intact) and that the removed string guards open NO silent-wrong-write path (a NUL or lone surrogate still fails LOUD inside the transaction -> rollback). PR-2.8 cleared the retained ruff E501 merge-blocker + 2 doc nits. The cycle-1 amendment must-fix (an orphaned 10**309 test param after the OverflowError-branch removal) was caught and fixed during PR-2.7's own inner loop. All Phase-2 exit criteria re-verified on the amended tree: yamllint clean, ruff clean, the 3 ingest workflows have the required v3 step + continue-on-error v4 steps, the schema-deploy push trigger, SCHEMA_VERSION lockstep, measurement_id reproduction; 120 non-Docker unit tests pass. Every Key decision remains keep. The two nits are dismissable: the build_commit comment sits at (not over) the 100-col limit, and the huge-int-threshold path now raises an uncaught OverflowError instead of SystemExit but still fails loud + rolls back (an explicitly DO-NOT-flag trusted-input case). Two operator-side gates remain (live OIDC schema-deploy apply; live v4 soak + CloudWatch). Verdict: ACCEPT.",
  "overall": "accept", "must_fix_count": 0, "should_fix_count": 0, "nit_count": 2, "review_cycles_this_invocation": 2
}
```

</details>

## Phase 2 raw gauntlet responses (archive)

### Cycle 1 — preset=phase-2 — accept

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1)</summary>

```json
{
  "schema_version": 1,
  "preset": "phase-2",
  "lenses_used": ["spec", "correctness"],
  "review_count": 2,
  "unified_findings": [
    {
      "severity": "nit",
      "kind": "doc-quality",
      "file_line": "benchmarks-website/infra/README.md:81",
      "description": "README var table still attributes RDS_BENCH_INSTANCE_ENDPOINT/GH_BENCH_INGEST_ROLE_ARN consumers to 'PR-2.2' ingest workflows, but the dual-write steps actually landed in PR-2.4; minor lineage mislabel in shipped docs.",
      "recommended_fix": "Update the 'used by' column references from PR-2.2 to PR-2.4 for the ingest-workflow consumers of RDS_BENCH_INSTANCE_ENDPOINT and GH_BENCH_INGEST_ROLE_ARN, to match where the dual-write steps were actually added.",
      "found_by": ["spec"]
    },
    {
      "severity": "nit",
      "kind": "other",
      "file_line": "scripts/post-ingest.py:1168",
      "description": "_main_postgres calls build_commit() (git show -s <sha>) for every run; bench.yml/sql-benchmarks.yml checkouts are not pinned to fetch-depth the way v3-commit-metadata.yml sets fetch-depth:2. Behavior matches v3 (--server also builds the commit), so it is parity, not drift.",
      "recommended_fix": "No change required for spec adherence (best-effort + continue-on-error absorbs any git-history miss). Optionally note in a comment that the v4 step inherits the v3 step's git-history assumption.",
      "found_by": ["spec"]
    }
  ],
  "disagreements": [],
  "dropped_re_flags": [],
  "phase_artifacts": {
    "summary": "Phase 2 delivers the Postgres dual-write ingest path and its CI plumbing, shipped exactly to the lean re-planned scope across PR-2.1/2.2/2.3/2.4 with no drift in either direction; correctness review found zero data-correctness bugs. Ingest identity (PR-2.1): migration 004 creates a least-privilege bench_ingest login role (rds_iam member on RDS, guarded for vanilla PG) holding USAGE-not-CREATE on public and SELECT/INSERT/UPDATE on the six tables (no DELETE/TRUNCATE/DDL), plus ALTER DEFAULT PRIVILEGES FOR ROLE migrator so future migrator tables auto-grant. Because the RDS master is rds_superuser but not a true superuser, the ADP runs via a temporary INHERIT self-grant through the creator's ADMIN option, then revoke. provision.sh adds GitHubBenchmarkIngestRole (OIDC trust branch-scoped to develop+ct/bench-v4, rds-db:connect for bench_ingest on the instance only) and drops the dead proxy grant. Writer (PR-2.2): post-ingest.py --postgres reproduces the v3 serde boundary in Python (deny_unknown_fields, per-field type/range validation, memory quartet, storage enum, commit_sha match), computes measurement_id bit-identically via the PR-1.5 port (_measurement_id.py is a byte-for-byte port of db.rs:162-257, pinned transitively Rust==golden==Python), then upserts commits-first and five fact tables in one all-or-nothing transaction via INSERT ... ON CONFLICT DO UPDATE with deadlock retry, over verify-full TLS authenticated by an RDS IAM token as bench_ingest. An is_finite guard rejects NaN/Inf/out-of-f64-range threshold loudly with rollback (Python stricter-or-equal to serde). All six ON CONFLICT SET lists touch only value/env columns, never dim columns; search_path is pinned to public; ssl_in_use is asserted post-connect. The v3 --server path stays stdlib-only (PEP 723 dependencies=[]). SCHEMA_VERSION lockstep held. CI wiring (PR-2.3): a scripts-test job runs pytest scripts/ behind a docker info hard gate plus a CI-env fail-loud fixture so testcontainer suites cannot silently skip. Dual-write CI (PR-2.4): the three ingest workflows each add a best-effort continue-on-error v4 --postgres step after the unchanged hard-required v3 step (v3-commit-metadata.yml gains id-token: write and ingests commit-row-only empty.jsonl); schema-deploy.yml switches to push-on-develop under paths migrations/**, keeping workflow_dispatch+dry_run. PR-2.5 (reconcile-ingest.py + dual-write-verify.yml + incident.io) was intentionally dropped; v4 correctness is deferred to Phase-3 migrate --verify. All five measurement_id functions, six ON CONFLICT SET lists, field/type sets, and column widths were verified against the Rust source and 001_initial_schema.sql. Two cosmetic nits, no must-fix or should-fix.",
    "surprises": [
      { "what": "Migration 004 ALTER DEFAULT PRIVILEGES FOR ROLE migrator fails on a real non-superuser RDS master (createrole_self_grant default), masked by the superuser testcontainer.", "how_handled": "Fixed PR-2.1 cycle-1: guarded INHERIT self-grant via the master's ADMIN option then revoke, plus a test applying 001..004 as a real NOSUPERUSER CREATEROLE login.", "amend_plan": "already-done" },
      { "what": "conn.info.ssl_in_use does not exist in psycopg (ssl_in_use lives on conn.pgconn); the cycle-9 fix shipped the wrong accessor.", "how_handled": "Fixed PR-2.2 (conn.pgconn.ssl_in_use) with a unit test pinning the accessor location and a container test pinning the traversal.", "amend_plan": "already-done" },
      { "what": "fail-loud-on-no-Docker guard tests could not catch an always-skip regression because pytest.skip raises Skipped (not a Failed subclass).", "how_handled": "Fixed PR-2.3 (catch both outcome types and assert the specific one; mutation-verified).", "amend_plan": "already-done" },
      { "what": "Python json.loads accepts NaN/Infinity literals that serde_json rejects at parse time.", "how_handled": "The _require_finite guard rejects them loudly with rollback, making the Python writer stricter-or-equal to v3, so no divergent row is written. Already handled.", "amend_plan": "no" },
      { "what": "The v3 producer emits several fields as u32/u64 (query_idx u32; value_ns/all_runtimes_ns u64) where server and Python validator use i32/i64.", "how_handled": "Both v3 serde and Python _require_int reject values exceeding the signed range, so the substrates agree; values are far inside range. No change needed.", "amend_plan": "no" },
      { "what": "insert-vs-update classification differs across substrates: Rust does an exists() preflight SELECT; Python derives the flag from RETURNING (xmax = 0).", "how_handled": "Equivalence (incl. same-dim-tuple-twice-in-one-envelope yielding (1,1) not (2,0)) is pinned by test_same_dim_tuple_twice_in_one_envelope_counts_second_as_update. Already handled.", "amend_plan": "already-done" },
      { "what": "bench-v4 Python files predate Phase 2 at ~100-col and fail the repo ruff line-length-120 (a pre-existing E501).", "how_handled": "Left unfixed and flagged for the operator as a branch-wide ruff reconciliation required before merging ct/bench-v4 to develop; RETAINED deferred item.", "amend_plan": "no" },
      { "what": "pytest scripts/ collects a fourth file (scripts/tests/test_benchmark_reporting.py) beyond the three named suites.", "how_handled": "Accepted as harmless and matching the acceptance command; a clarifying ci.yml comment added.", "amend_plan": "no" }
    ],
    "coverage": {
      "tested_cases": [
        { "case": "measurement_id Rust==golden==Python parity across all 5 tables incl. negative i32, null/Some, multibyte UTF-8, whole/negative/tiny/large f64 thresholds; SCHEMA_VERSION lockstep with schema.rs", "test_location": "scripts/test_measurement_id.py + scripts/measurement_id_golden.json; scripts/test_post_ingest_postgres.py:3105", "confidence": "high" },
        { "case": "bench_ingest DML-only (SELECT/INSERT/UPDATE) on all 6 tables, denied DELETE/CREATE/DDL; default-privileges cover future migrator tables; 001..004 apply cleanly + idempotently under a real NOSUPERUSER CREATEROLE master", "test_location": "scripts/test_migrate_schema.py:1960,1928,1993,2069", "confidence": "high" },
        { "case": "provision.sh provisions ingest role on instance dbuser, emits GH_BENCH_INGEST_ROLE_ARN, drops dead proxy grant", "test_location": "scripts/test_migrate_schema.py:1805,1835", "confidence": "high" },
        { "case": "insert-then-update accounting; re-ingest upserts (0 inserted, N updated) with stable counts; measurement_id matches port", "test_location": "scripts/test_post_ingest_postgres.py:2477,2572", "confidence": "high" },
        { "case": "ON CONFLICT DO UPDATE overwrites every value/env column while leaving dim tuple / measurement_id stable, per table; dim columns NEVER in any DO UPDATE SET list for all 5 tables", "test_location": "scripts/test_post_ingest_postgres.py:2867", "confidence": "high" },
        { "case": "NaN/Inf/out-of-f64-range threshold raises loudly and rolls back; deny_unknown_fields; missing-required; unknown/non-scalar kind; storage enum; memory quartet; commit_sha mismatch; type/range boundaries reproduced", "test_location": "scripts/test_post_ingest_postgres.py:2633,2887,3066,3074,3094", "confidence": "high" },
        { "case": "connect_postgres: IAM-token-when-passwordless, always-bench_ingest, rejects weak sslmode (verify-full), forces search_path=public, rejects non-TLS, post-connect ssl_in_use", "test_location": "scripts/test_post_ingest_postgres.py:3142,3221,3233,3264,3285", "confidence": "high" },
        { "case": "--server requires --benchmark-id while --postgres does not; mutual exclusivity + dispatch", "test_location": "scripts/test_post_ingest_postgres.py:3401,3363,3375", "confidence": "high" },
        { "case": "fail-loud-on-no-Docker in CI vs skip locally (both test files)", "test_location": "scripts/test_migrate_schema.py:1745; scripts/test_post_ingest_postgres.py:2753", "confidence": "high" },
        { "case": "write-conflict retry (deadlock/serialization) retries then succeeds, gives up at cap, propagates validation errors immediately", "test_location": "scripts/test_post_ingest_postgres.py (test_retry_write_conflicts_*)", "confidence": "medium" }
      ],
      "untested_cases": [
        { "case": "End-to-end live v4 dual-write per develop push (CloudWatch shows Postgres writes) — the PR-2.4 acceptance criterion needing real RDS + repo vars", "priority": "high", "why_untested": "Operator-side, not performable in-session; the v4 steps no-op until GH_BENCH_INGEST_ROLE_ARN + RDS_BENCH_INSTANCE_ENDPOINT/DB_NAME/REGION are set. Tracked as an open operator dependency." },
        { "case": "Real psycopg DeadlockDetected inside conn.transaction() across two reversed-order connections on autocommit=False (retry+commit/rollback)", "priority": "medium", "why_untested": "Retry path covered only by mocked-op unit tests; a real-conflict container test needs deterministic threaded interleaving. Explicitly deferred (PR-2.2 cycle-7) to the follow-up test-hardening PR; behavior manually verified against live PG16." },
        { "case": "migration 004 self-grant when the executing role lacks ADMIN on a pre-existing migrator (unsupported misconfiguration)", "priority": "low", "why_untested": "Deferred; the shipped single-bootstrap-master path is correct and tested." },
        { "case": "schema_conn BENCH_TEST_PG_DSN destructive-scrub guard against a localhost-tunnel-to-prod", "priority": "low", "why_untested": "Dev-only override never used in CI (testcontainers only); explicitly accepted/deferred as an exotic foot-gun." }
      ],
      "recommendations": "Coverage of the load-bearing data-correctness invariants is comprehensive and pins each against an automated test, satisfying behavior-preservation against the v3 serde/ingest boundary per-table. The only material gap is the live end-to-end dual-write, gated on operator var-setting and best-effort by design. The deferred real-deadlock/autocommit=False container test is the one worth landing in the follow-up test-hardening PR; the remaining untested cases are availability/test-infra hardening already triaged and out of scope per the lean re-plan."
    },
    "tradeoffs": [
      { "decision": "Schema-deploy authorization (PR merge is the gate; push-on-develop trigger; no environment/manual-approval gate)", "original": "Supersedes the original manual-approval mandate (2026-05-29 deploy-model decision)", "verdict": "keep", "rationale": "PR-2.4 implements exactly this: push-on-develop under paths migrations/**, dry_run kept, environment-gate comments removed. Execution safety comes from the per-PR testcontainer migration test. Accepted tradeoff; not re-flagged.", "found_by": ["spec", "correctness"] },
      { "decision": "Ingest writer language (pure Python extending post-ingest.py; xxhash port)", "original": "Avoid sqlx/aws-sdk Rust deps", "verdict": "keep", "rationale": "PR-2.2 delivered exactly this; the port matches Table A/B and SCHEMA_VERSION lockstep holds.", "found_by": ["spec"] },
      { "decision": "Cutover style (short best-effort dual-write soak, then promote v4); v4 dual-write is continue-on-error during soak, v3 stays hard-required", "original": "Amended 2026-06-04 to make v4 best-effort during soak (PR-2.4 lean re-plan)", "verdict": "keep", "rationale": "PR-2.4's continue-on-error v4 steps are the direct realization; a v4 OIDC/connect hiccup must never break the proven v3 pipeline; v4 promoted to required at cutover (PR-5.1). Both steps consume the same results.v3.jsonl, so neither substrate is written in isolation. Calibrated, accepted exception to the no-best-effort rule.", "found_by": ["spec", "correctness"] },
      { "decision": "Phase-2 ingest DB identity (dedicated bench_ingest + GitHubBenchmarkIngestRole, separate from migrator)", "original": "Re-plan Q2 least-privilege split (PR-2.1)", "verdict": "keep", "rationale": "PR-2.1 implemented precisely this with DML-only grants and a separate OIDC role; separation of duties; the writer enforces bench_ingest unconditionally. Pinned by tests.", "found_by": ["spec", "correctness"] },
      { "decision": "Phase-2 dual-write verify scope (verify-once via Phase-3 migrate --verify; drop reconcile-ingest.py + dual-write-verify.yml + incident.io)", "original": "Superseded the 2026-06-01 per-push reconciliation harness", "verdict": "keep", "rationale": "PR-2.5 correctly dropped; PR-2.4 adds no reconciliation machinery; v4-only failure does not trigger incident.io. Four independent safety nets remain. Not drift.", "found_by": ["spec"] },
      { "decision": "Duplicate JSON keys collapse last-wins (no rejection), matching v3", "original": "PR-2.2 cycle-6", "verdict": "keep", "rationale": "serde_json::Value::Object and Python json.loads are both last-wins; rejecting would break behavior-preservation. Not re-flagged.", "found_by": ["correctness"] },
      { "decision": "post-ingest.py PEP 723 dependencies = []; --postgres runs from the uv env", "original": "PR-2.2", "verdict": "keep", "rationale": "Keeps the v3 --server path stdlib-only under bare python3; --postgres deps come from the uv workspace. Lazy imports verified. Not re-flagged.", "found_by": ["correctness"] },
      { "decision": "IAM-token region precedence: --region > boto3 session region > RDS-hostname-parsed", "original": "PR-2.2", "verdict": "keep", "rationale": "A wrong region fails loud at IAM connect, not silently; ordering matches AWS convention. Not re-flagged.", "found_by": ["correctness"] }
    ]
  },
  "executive_summary": "Phase 2 ships the Postgres dual-write ingest path and its CI plumbing, exactly to the lean re-planned scope across PR-2.1/2.2/2.3/2.4 with no drift in either direction. Both lenses accept at high confidence; the correctness lens found zero data-correctness bugs and the spec lens found only two cosmetic nits, so there are no must-fix or should-fix items. PR-2.1 adds a least-privilege bench_ingest login role (migration 004) with SELECT/INSERT/UPDATE on the six tables (no DELETE/TRUNCATE/DDL), USAGE-not-CREATE, and ALTER DEFAULT PRIVILEGES so future migrator tables auto-grant; provision.sh adds a branch-scoped OIDC GitHubBenchmarkIngestRole (rds-db:connect on the instance only) and drops the dead proxy grant. PR-2.2's post-ingest.py --postgres reproduces the v3 serde boundary in Python, computes measurement_id bit-identically via the PR-1.5 port, and upserts commits-first then five fact tables in one all-or-nothing transaction with deadlock retry over verify-full TLS and IAM auth. PR-2.3 runs pytest scripts/ behind a docker hard gate plus a CI fail-loud fixture. PR-2.4 makes each of the three ingest workflows add a best-effort continue-on-error v4 step after the unchanged hard-required v3 step, and switches schema-deploy to push-on-develop under paths migrations/**. PR-2.5 was intentionally dropped; v4 correctness is deferred to Phase-3 migrate --verify. The key engineering surprises were all already resolved in-cycle: the non-superuser RDS master breaking ALTER DEFAULT PRIVILEGES (fixed with a guarded INHERIT self-grant + a real NOSUPERUSER test), the ssl_in_use accessor living on conn.pgconn rather than conn.info (fixed and pinned), and the fail-loud-on-no-Docker guard being unable to catch always-skip because pytest.skip is not a Failed subclass (fixed and mutation-verified). Correctness independently confirmed Python-is-stricter-or-equal handling of NaN/Inf literals, agreement on u32/u64-vs-i32/i64 ranges, and insert-vs-update equivalence between Rust's exists() preflight and Python's xmax=0 classifier. Every Key decision was verdicted keep by both reviewers: schema-deploy PR-merge gating, pure-Python writer + xxhash port, best-effort dual-write during soak with v3 hard-required, the dedicated bench_ingest identity, dropping the reconcile harness, last-wins duplicate-key handling, PEP 723 empty-deps, and IAM region precedence. Coverage of load-bearing data-correctness invariants is comprehensive and per-table-pinned; the one material gap is the live end-to-end dual-write, which is operator-gated on repo vars and best-effort by design. Verdict: ACCEPT.",
  "overall": "accept",
  "must_fix_count": 0,
  "should_fix_count": 0,
  "nit_count": 2,
  "review_cycles_this_invocation": 1
}
```

</details>


### Cycle 2 — preset=phase-2 — accept

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1)</summary>

```json
{
  "schema_version": 1, "preset": "phase-2", "lenses_used": ["spec", "correctness"], "review_count": 2,
  "unified_findings": [
    {"severity":"nit","kind":"convention","file_line":"scripts/post-ingest.py:1085","description":"PR-2.8's build_commit git-history comment lines sit at exactly 100 cols (at the user CLAUDE.md limit, not over; well under ruff's 120). Already rewrapped from 102 in f0e93b9f4. No action.","recommended_fix":"None; within the 100-col rule.","found_by":["spec"]},
    {"severity":"nit","kind":"error-path","file_line":"scripts/post-ingest.py:448","description":"After the de-gold-plate dropped the _require_finite OverflowError sub-branch, a huge-int threshold (e.g. 10**309) now raises an uncaught OverflowError instead of a controlled SystemExit. Still fails LOUD + rolls back the transaction (no silent-wrong-write); the input class (oversized-int on trusted CI) is explicitly DO-NOT-flag per the calibration. Intentional de-gold-plate tradeoff.","recommended_fix":"None required (accepted). A one-line `except OverflowError: is_finite_number = False` would restore the controlled error, but is not worth carrying per the trusted-input calibration.","found_by":["correctness"]}
  ],
  "disagreements": [],
  "dropped_re_flags": [],
  "phase_artifacts": {
    "summary": "Amended cumulative Phase 2 delivers the Postgres dual-write ingest path + best-effort v4 CI. Ingest identity (PR-2.1): migrations/004 creates least-privilege bench_ingest (SELECT/INSERT/UPDATE on the 6 tables, USAGE-not-CREATE, ALTER DEFAULT PRIVILEGES for future migrator tables, no DELETE/DDL); provision.sh adds the branch-scoped OIDC GitHubBenchmarkIngestRole (rds-db:connect on the instance dbuser) and drops the dead proxy grant; 004 self-grants migrator INHERIT via the master's ADMIN option then revokes (non-superuser-RDS-master bootstrap fix). Writer (PR-2.2, de-gold-plated by PR-2.7): post-ingest.py --postgres computes measurement_id bit-identically via the byte-exact _measurement_id.py port, validates every field/type/range to reproduce the v3 serde + deny_unknown_fields boundary, upserts commits-first then 5 fact tables via INSERT...ON CONFLICT(measurement_id) DO UPDATE...RETURNING(xmax=0) in one all-or-nothing transaction with deadlock/serialization retry; connect_postgres enforces verify-full TLS + post-connect ssl_in_use + bench_ingest-only + IAM-token mint + search_path=public. CI wiring (PR-2.3/2.4/2.6): a scripts-test job runs pytest scripts/ (Docker-required, fail-loud-not-skip in CI); the 3 ingest workflows add a continue-on-error best-effort v4 --postgres step after the unchanged hard-required v3 step; schema-deploy.yml triggers on push to develop under migrations/**. Amendments: PR-2.6 re-keys the 9 v4 gates from the endpoint var to GH_BENCH_INGEST_ROLE_ARN so they no-op (not fire+fail) until infra is wired; PR-2.7 de-gold-plates ~194 lines of trusted-input over-hardening (NUL/surrogate/non-UTF-8/RecursionError/oversized-int guards + dead _is_local_host) + ~8 tests, moving the writer CLOSER to the v3 Rust source; PR-2.8 clears the ruff E501 merge-blocker + README PR-lineage + a git-history comment. Every preserved invariant (measurement_id parity, ON CONFLICT SET membership, NaN/Inf guard, typed validation, memory-quartet, IAM/TLS) remains pinned by tests; the de-gold-plate removed only stricter-than-v3 hardening on trusted CI input.",
    "surprises": [
      {"what":"PR-2.7's OverflowError-branch removal left an orphaned pytest.param(10**309) in the KEPT test_nonfinite_threshold_raises_and_rolls_back, which would have gone CI-red (uncaught OverflowError); only skipped locally for lack of Docker.","how_handled":"Caught by cycle-1 correctness (cumulative-test trace), fixed in b6d1f292b (param dropped; nan/inf/-inf retained). Verified gone.","amend_plan":"already-done"},
      {"what":"The de-gold-plate moves the writer CLOSER to v3: the removed NUL/surrogate/non-UTF-8 guards were STRICTER than the v3 Rust source, so removal improves behavior-preservation rather than regressing it; failure modes stayed LOUD (UnicodeEncodeError in _write_str / Postgres DataError at bind, inside the transaction -> rollback). No silent-wrong-write path.","how_handled":"Authorized scope-reduction per the 2026-06-05 complexity audit; KEEP-list verified intact field-by-field.","amend_plan":"no"},
      {"what":"The live gate-var bug (PR-2.6) was a real config mismatch: gates keyed on RDS_BENCH_INSTANCE_ENDPOINT (set) while assume-role uses GH_BENCH_INGEST_ROLE_ARN (unset), so the v4 steps would fire+fail at assume-role on the next develop push.","how_handled":"Re-keyed all 9 gates to the role-ARN var; live-var state now makes the gate evaluate false (clean no-op until wired).","amend_plan":"already-done"}
    ],
    "coverage": {
      "tested_cases": [
        {"case":"measurement_id parity (5 kinds) vs the Python port + stored rows; SCHEMA_VERSION lockstep","test_location":"scripts/test_measurement_id.py (65 vectors) + test_post_ingest_postgres.py (measurement_ids_match_python_port, schema_version)","confidence":"high"},
        {"case":"ON CONFLICT SET excludes dim columns (BAN) across all 5 _insert_ fns; per-table value-column update","test_location":"test_on_conflict_set_excludes_dim_columns + test_update_overwrites_all_value_columns_per_table","confidence":"high"},
        {"case":"insert-vs-update accounting incl. same-dim-twice (xmax=0 classifier); commit-before-facts; all-or-nothing rollback","test_location":"test_ingest_inserts_then_updates, test_same_dim_tuple_twice..., test_late_validation_failure_rolls_back_earlier_fact_row","confidence":"high"},
        {"case":"NaN/+Inf/-Inf threshold rejected + rollback (post de-gold-plate, OverflowError param removed)","test_location":"test_nonfinite_threshold_raises_and_rolls_back","confidence":"high"},
        {"case":"typed i32/i64/finite + deny_unknown_fields + memory-quartet + storage-enum + commit_sha-mismatch validation","test_location":"test_post_ingest_postgres.py value-validation tests","confidence":"high"},
        {"case":"connect_postgres: verify-full + post-connect ssl_in_use + bench_ingest-only + IAM-token-vs-password + search_path pin","test_location":"test_connect_postgres_* (9 pure-unit tests, pass non-Docker)","confidence":"high"},
        {"case":"de-gold-plate replacements: mixed-newline read, malformed-JSON loud-fail, git_show text-mode decode","test_location":"test_read_records_happy_path_mixed_newlines / _rejects_malformed_json / test_git_show_field_decodes_and_strips","confidence":"high"},
        {"case":"bench_ingest least-privilege + 004 idempotency + non-superuser-master bootstrap; provision.sh ingest role + dropped proxy grant","test_location":"scripts/test_migrate_schema.py (testcontainer + static provision tests)","confidence":"high"},
        {"case":"v4 gate keys on GH_BENCH_INGEST_ROLE_ARN; CI fail-loud-not-skip on missing Docker","test_location":"workflow inspection + test_require_docker_fails_loud_in_ci/_skips_without_ci","confidence":"high"}
      ],
      "untested_cases": [
        {"case":"Live v4 dual-write end-to-end against real RDS per develop push (OIDC->IAM->verify-full->upsert; CloudWatch confirmation)","priority":"medium","why_untested":"Requires live AWS + operator-set repo vars; operator-side soak. Acceptance for PR-2.4 is the structural/yamllint criteria; v4 correctness is gated by Phase-3 migrate --verify."},
        {"case":"Real two-connection DeadlockDetected on an autocommit=False connection asserting retry+commit","priority":"low","why_untested":"Container-only threaded-interleaving test; deferred (PR-2.2 cycle-7) to a follow-up test-hardening PR. Retry pinned by mocked-op unit tests + live-container verification."},
        {"case":"Live OIDC schema-deploy apply firing on a migrations/** push","priority":"low","why_untested":"The retained operator pre-merge gate; not in-session testable."}
      ],
      "recommendations": "Coverage is strong for a trusted-input low-stakes dashboard and matches the lean calibration. The only meaningful gaps are the two retained operator-side gates (live OIDC schema-deploy apply; live v4 dual-write soak + CloudWatch). No new test work is required to accept the amended phase. The deferred real-deadlock/autocommit=False integration test remains a reasonable follow-up but is not a blocker."
    },
    "tradeoffs": [
      {"decision":"v4 ingest steps continue-on-error (best-effort) during the soak","original":"PR-2.4 lean re-plan amendment","verdict":"keep","rationale":"v3 stays hard-required + untouched; the 'no continue-on-error' BAN applies at the PR-5.1 cutover, not the soak. PR-2.6 reinforces it (clean no-op until wired)."},
      {"decision":"Gate v4 steps on GH_BENCH_INGEST_ROLE_ARN (PR-2.6)","original":"audit live-bug fix","verdict":"keep","rationale":"Aligns the gate with the assume-role input so the steps no-op (not fire+fail) until infra is wired; verified across all 9 gates."},
      {"decision":"De-gold-plate trusted-input hardening (PR-2.7)","original":"2026-06-05 complexity audit","verdict":"keep","rationale":"Removed only stricter-than-v3 guards on trusted CI input; every removed input still fails loud inside the transaction (rollback). No data-correctness guard or its test removed."},
      {"decision":"Schema-deploy authorization (PR-merge gate, push trigger, no environment: gate)","original":"2026-05-29 deploy-model decision","verdict":"keep","rationale":"PR-2.4 shipped the trigger; the amendment did not touch it. Accepted tradeoff."},
      {"decision":"Phase-2 dual-write verify scope (verify-once via Phase-3 migrate --verify; PR-2.5 dropped)","original":"lean re-plan","verdict":"keep","rationale":"The amendment added no reconciliation machinery; consistent with the superseding decision."},
      {"decision":"Pure-Python writer + PEP 723 dependencies=[]; IAM region precedence","original":"PR-2.2","verdict":"keep","rationale":"Accepted tradeoffs; keeps v3 --server stdlib-only; region mismatch fails loud. Unchanged by the amendment."},
      {"decision":"Dedicated bench_ingest identity separate from migrator","original":"re-plan Q2","verdict":"keep","rationale":"Unchanged; PR-2.6 gates on the ingest role ARN, reinforcing the separate-identity model; connect_postgres still enforces bench_ingest-only."}
    ]
  },
  "executive_summary": "Phase 2 cycle-2 phase-end review of the AMENDED cumulative phase (PR-2.1..2.4 accepted at cycle 1, plus the 2026-06-05 amendment PR-2.6/2.7/2.8). Both lenses ACCEPT with high confidence; 0 must-fix, 2 dismissable nits, no disagreements. The amendment did exactly three things and the reviewers verified each: PR-2.6 fixed a real LIVE gate-var bug (the v4 dual-write gates keyed on RDS_BENCH_INSTANCE_ENDPOINT [set] while assume-role uses GH_BENCH_INGEST_ROLE_ARN [unset], so the steps would fire+fail on the next develop push) by re-keying all 9 gates to the role-ARN var, restoring clean no-op-until-wired behavior. PR-2.7 de-gold-plated ~194 lines of trusted-input over-hardening (NUL/lone-surrogate/non-UTF-8/RecursionError/oversized-int guards + dead _is_local_host) that was STRICTER than the v3 Rust source it mirrors; the spec lens confirmed this is authorized scope-reduction (not silent de-scoping) and the correctness lens verified field-by-field that NO load-bearing data-correctness validation was removed (measurement_id parity, ON CONFLICT SET membership, NaN/Inf guard, typed i32/i64/finite, deny_unknown_fields, memory-quartet, IAM/TLS all intact) and that the removed string guards open NO silent-wrong-write path (a NUL or lone surrogate still fails LOUD inside the transaction -> rollback). PR-2.8 cleared the retained ruff E501 merge-blocker + 2 doc nits. The cycle-1 amendment must-fix (an orphaned 10**309 test param after the OverflowError-branch removal) was caught and fixed during PR-2.7's own inner loop. All Phase-2 exit criteria re-verified on the amended tree: yamllint clean, ruff clean, the 3 ingest workflows have the required v3 step + continue-on-error v4 steps, the schema-deploy push trigger, SCHEMA_VERSION lockstep, measurement_id reproduction; 120 non-Docker unit tests pass. Every Key decision remains keep. The two nits are dismissable: the build_commit comment sits at (not over) the 100-col limit, and the huge-int-threshold path now raises an uncaught OverflowError instead of SystemExit but still fails loud + rolls back (an explicitly DO-NOT-flag trusted-input case). Two operator-side gates remain (live OIDC schema-deploy apply; live v4 soak + CloudWatch). Verdict: ACCEPT.",
  "overall": "accept", "must_fix_count": 0, "should_fix_count": 0, "nit_count": 2, "review_cycles_this_invocation": 2
}
```

</details>

## Phase 1+2 complexity & gap audit (2026-06-05)

Operator-requested step-back review ("did we go overkill / too much complexity / is stuff missing"), run as 3 parallel auditors (over-engineering, gaps, architecture-coherence) with full repo access, then synthesized. Verdict: **the destination architecture is sound and coherent; the writer/tooling is meaningfully over-built; one real LIVE bug + several Phase-3 gaps.**

**Sound (do not touch):** RDS + IAM-auth + hash-preserving `ON CONFLICT` upsert + Next.js/Vercel is the right shape. `measurement_id` parity verified field-by-field across all 6 tables -> complete + correct, no silent-duplicate risk in the port. `RETURNING (xmax=0)` classification is more correct than the v3 Rust under concurrency.

**LIVE BUG (must-fix, PR-2.4 scope) -- v4 dual-write gates on the WRONG variable.** The 3 ingest workflows gate the v4 step on `if: vars.RDS_BENCH_INSTANCE_ENDPOINT != ''` (live: SET since 2026-06-01) but assume-role uses `vars.GH_BENCH_INGEST_ROLE_ARN` (live: NOT SET). So on the next `develop` push the v4 steps FIRE and FAIL at assume-role (swallowed by `continue-on-error`) rather than cleanly no-op'ing -- the plan's "no-ops until vars set" premise is false. Fix: gate on `GH_BENCH_INGEST_ROLE_ARN != ''` (the var that must exist for assume-role). One-line per workflow (`bench.yml:128/135/139`, `sql-benchmarks.yml:508/515/519`, `v3-commit-metadata.yml:41/48/52`). **Selected for the Phase-2 amendment.**

**Overkill (confirmed by 2 of 3 auditors):** ~35% of `post-ingest.py`'s 987 new `--postgres` lines + ~25-30% of `test_post_ingest_postgres.py`'s 1221 lines are adversarial-input hardening (NUL / lone-surrogate / non-UTF-8 / RecursionError / oversized-int) on TRUSTED CI input -- which the lean calibration itself says is out of scope, and which is STRICTER than the v3 Rust source it mirrors. Frozen residue of the pre-lean-re-plan 15-cycle PR-2.2 spiral.
- Cleanly deletable, ~no risk: `_is_local_host` (dead -- its own docstring says no production role), the unreachable `RecursionError` branch in `read_records`.
- Low risk: `_reject_unstorable_str` (surrogate/NUL guards), the `git_show_field` bytes-decode rewrite, the `read_records` universal-newline hardening, the `_require_finite` `OverflowError` branch, + ~12 tests.
- Keep untouched: hash parity, ON CONFLICT/RETURNING, retry, IAM/TLS auth + `004_ingest_role.sql`, deny_unknown_fields + typed i32/i64/finite validation, memory-quartet/storage-enum.

**Bigger structural simplifications (higher risk; reverse accepted + partly live-verified Phase-1 decisions -- arguably re-plan, not amend):**
- Reuse the existing Rust `measurement_id_*` hasher (already linked by `migrate/`) instead of the Python port -> deletes the golden-vector apparatus + the NaN-payload `is_finite` hazard + the endianness caveat. Medium risk; reverses Key-decision Q4 (pure-Python writer to keep CI writers dep-light).
- Replace the bespoke `migrate-schema.py` (240 LOC + 1431 test lines) with a ~15-line `psql -f` apply loop for 4 idempotent additive migrations. Low-med risk; reverses PR-1.2 (tested + live-verified).
- RDS Proxy is fully provisioned (`provision.sh`) but consumed by NOTHING until PR-4.2 (whose Vercel-to-VPC reachability is unsolved) -> dead infra today. Could defer provisioning to Phase 4.

**Phase-3 gaps (load-bearing; deferred to a Phase-3 re-plan at the Phase 2->3 boundary per operator 2026-06-05):**
1. `migrate --verify` (the PRIMARY v4-correctness gate) does NOT exist yet and is net-new, not an "extension": `migrate/src/verify.rs` today is a v2-vs-v3 structural diff with NO Postgres dependency (no sqlx/tokio-postgres in `migrate/Cargo.toml`). Its stated baseline `matched_rows == duckdb_rows AND only_in_postgres == []` is necessary-but-NOT-sufficient -- a PK-count match cannot detect value-column corruption in the bulk COPY, and `env_triple` is not in the hash so count-match can't see env corruption. Phase-3 verify must compare value-column payloads per `measurement_id`, not just PK presence.
2. The DuckDB->Postgres bulk-load seed (PR-3.1) establishes the existing `measurement_id`s the whole upsert-not-duplicate invariant depends on -- the single most correctness-critical unwritten code. One-shot, crosses an AWS account boundary (`375504701696/us-east-2` DuckDB <-> `245040174862/us-east-1` RDS) + region; NO rehearsal/dry-run specified before the one-shot prod load.
3. The soak exercises the PYTHON writer (`post-ingest.py --postgres`); the cutover gate (`migrate --verify`) exercises the RUST bulk-loader -- the two NEVER cross-check. So a green verify does not imply the steady-state Python writer round-trips against real RDS. Add a Python-writer-vs-RDS cross-check (a few rows' `measurement_id` + value columns vs the Rust-seeded rows) before PR-5.1 makes the Python path required.
4. Migration `004_ingest_role.sql` must be applied AS THE RDS MASTER (or the `ALTER DEFAULT PRIVILEGES` self-grant hits InsufficientPrivilege + rolls back); this ordering ("002+004 by the master before any `migrator` run") is documented in the SQL but enforced nowhere.

## Phase 3: Historical data load (DuckDB → Postgres) + value-verify — end-of-phase review (cycle 1) — accepted (3-vote)
**Synthesizer output from /spiral:gauntlet (preset=phase-3, lenses=spec+correctness+maint, executor=claude). Verdict: ACCEPT — 0 must-fix, 3 should-fix, 4 nits. Full Synthesizer Output JSON in the `<details>` block at the end of this section (also the cycle-1 raw-response archive).**

### Summary of changes
Phase 3 delivers the v3-to-v4 DuckDB-to-Postgres migration TOOLKIT (validated locally; the irreversible prod load is deferred to the Phase-5 cutover as PR-5.0). (1) BULK LOADER (postgres.rs + Load subcommand): reads 6 tables from an existing v3 DuckDB as Arrow batches and streams them via COPY FROM STDIN text format inside ONE transaction (atomic; mid-load failure rolls back). measurement_id/commit_sha verbatim; commits.timestamp CAST to VARCHAR under SET TimeZone=UTC; all_runtimes_ns explodes to {a,b,c}; f64 shortest-round-trip with non-finite rejected; NoTls locally or native-tls+--ca-cert for prod; NO aws-sdk-rds. (2) VALUE VERIFIER (verify.rs + verify --postgres-target): PRIMARY gate, a new PgVerifyReport/ValueMismatch pair (separate from the v2 structural VerifyReport), per-measurement_id full compare of every non-hashed value column, all_runtimes_ns element-wise + order-sensitive, timestamps as engine-independent epoch microseconds, read kinds resolved once from the loader column_kind so the two sides cannot drift. (3) REHEARSAL (postgres_e2e.rs + README): first migrate testcontainer (postgres:16-alpine), asserts count match + verify-clean + atomic mid-load rollback, Docker-gated skip-local/fail-loud-CI. (4) CROSS-CHECK (cross_check_python_writer.py): writer UPDATEs seeded rows (not duplicate-INSERTs), values round-trip, reuses connect_postgres verify-full TLS as bench_ingest. (5) BOOTSTRAP GUARD (migrate-schema.py): requires-superuser marker + rolsuper-OR-rolcreaterole preflight rejecting a non-master apply before any DDL. PR-3.4 ran all three gates GREEN against the real 4.2M-row bench.duckdb into local PG16, zero prod writes. Toolkit honestly labeled throwaway (removal trigger Table F / PR-5.3).

### Surprises and discoveries
- **The loader reads an EXISTING v3 DuckDB (copying measurement_id/commit_sha verbatim) rather than re-accumulating from v2.** — Documented; simpler and safer since ids are byte-preserved. (amend_plan: no)
- **PR-3.1 shipped the synchronous `postgres` 0.19 crate, not the plan-named `tokio-postgres`, with no deviation note.** — Defensible (hard-NO on aws-sdk-rds honored) but unremarked. Add a one-line status note. (amend_plan: yes)
- **PR-3.4's real snapshot held vector_search_runs=0 rows AND the cross-check envelope omitted that kind, so vector_search_runs got zero REAL-data coverage through the gate.** — Code paths fixture-covered; status overstates GREEN. Annotate; PR-5.0 closes it. (Lens disagreement: spec should-fix, maint acceptable.) (amend_plan: yes)
- **The bootstrap PermissionError cites migrations/README.md content (requires-superuser marker / preflight / 002-004 ordering) that does not exist there.** — Not handled; add a README subsection. (amend_plan: yes)
- **The local module `postgres` collides by name with the external `postgres` crate.** — Compiles; cosmetic disambiguation nit. (amend_plan: no)
- **PR-3.1 status claims column order is 'programmatically cross-checked against migrations/001', but the check is indirect (e2e COPY fails on disagreement), not a DDL-parsing test.** — Status log overstates mechanism; behavior covered by the e2e. (amend_plan: no)

### Testing coverage assessment
**Tested (high confidence):**
- loader COPY-text rendering incl. escaping/BIGINT[]/literal-\N-vs-NULL/non-UTC-tz/f64-round-trip+non-finite-reject (postgres.rs:514-712, high)
- value-verify discrimination across 6 tables incl. array reorder/Int32+side-counters/presence-both-directions/epoch-us-pinned (verify.rs:965-1356, high)
- Postgres e2e: counts/verify-clean/atomic mid-load rollback (postgres_e2e.rs:136-226, high)
- cross-check UPDATE-not-INSERT/all-5-kinds/NOT-seeded->INSERT/value_mismatches mutation-verified (test_post_ingest_postgres.py:3795-3947, high)
- requires-superuser guard: marker detection/non-master rejected before DDL (test_migrate_schema.py:3629-3768, high)
- real-data PR-3.4: 4.2M-row load exact match + verify clean + cross-check clean + negative (manual run, medium)

**Untested / under-covered (all non-blocking; plan-acknowledged or status-annotation):**
- vector_search_runs Double/threshold COPY + vector value-verify against REAL data (0 real rows + omitted from cross-check) — priority medium, annotate status + PR-5.0 closes
- verify-full TLS --ca-cert against a live RDS endpoint — priority low, compile+runbook only, plan-acknowledged
- TABLE_SPECS vs migrations/001 column-order as a direct DDL-parsing unit test — priority low, covered behaviorally by the e2e COPY

### Tradeoffs re-evaluation
- **Prod historical load TIMING deferred Phase 3 -> Phase 5 (PR-5.0); Phase 3 closes on the real-snapshot LOCAL rehearsal** — _keep_: PR-3.4 executed exactly this with zero irreversible prod side-effect; RDS PITR is the prod rollback.
- **Value-column verify as a NEW PgVerifyReport pair (not overloading v2-structural VerifyReport)** — _keep_: Single-responsibility; structural vs stored-value comparison are different concerns.
- **commits.timestamp compared as epoch microseconds** — _keep_: Engine-independent and exact; e2e confirmed sub-second + pre-1970.
- **local postgres:16 testcontainer rehearsal, Docker-gated** — _keep_: Same engine as RDS at zero AWS cost; e2e closes the live Postgres-execution gap.
- **operator-run-locally master-password DSN; DROP aws-sdk-rds** — _revisit-but-keep_: Keep (avoids heavy AWS SDK); record the resulting sync-`postgres`-over-`tokio-postgres` swap in the PR-3.1 status.
- **004-as-master requires-superuser preflight guard** — _revisit-but-keep_: Guard is well-shaped and load-bearing; close the migrations/README.md doc gap the PermissionError points at.
- **trusted-input low-stakes review calibration; 3-vote Phase 3** — _keep_: Appropriate; adversarial-input hardening correctly out of scope; retained guards are load-bearing.

### Disagreements
- **vector_search_runs real-data coverage gap through the PR-3.4 validation gate**
  - spec: should-fix / weak-acceptance: one of six tables (the one with the most distinct column shape) got no real-data validation through the gate while the status asserts GREEN without qualifying the fall-through. Annotate the status entry.
  - maint: acceptable / amend_plan: no: the Double/threshold COPY + vector value-verify paths are covered by the synthetic e2e fixture (one seeded vector row), so the gap is covered; no plan amendment needed.
  - **Synthesizer call:** Act on the spec position at should-fix, scoped to a status-log annotation (NOT a code/test change). maint is correct that the paths ARE fixture-exercised (no untested-code defect, no new test warranted for a throwaway crate); spec is correct that PR-3.4's distinguishing value is REAL-data validation that vector_search_runs did not receive. Both reduce to: annotate the PR-3.4 status that vector_search_runs had 0 real rows and point at PR-5.0's freshest-snapshot prod load. Non-blocking.

**Dropped re-flags (carry-forward respected):**
- Dim-column COPY corruption is invisible to the primary value-verify gate (verify.rs omits dim columns; load_table COPYs them verbatim) — covered by Accepted tradeoffs (Accepted tradeoffs / r1 traps)

### Findings (0 must-fix, 3 should-fix, 4 nits)
| # | Severity | Kind | File:line | Description | Found by |
|---|----------|------|-----------|-------------|----------|
| 1 | should-fix | scope-drift | `benchmarks-website/migrate/Cargo.toml:35-37` | PR-3.1 shipped the synchronous `postgres` 0.19 crate (+ native-tls + postgres-native-tls), not the plan-named `tokio-postgres`, with no explicit deviation note in the PR-3.1 implementation-status entry. The binding hard-NO (no aws-sdk-rd... | spec/claude |
| 2 | should-fix | weak-acceptance | `benchmarks-website/migrate/tests/postgres_e2e.rs` | PR-3.4's real-snapshot LOCAL rehearsal gave `vector_search_runs` zero real-data coverage: the real snapshot held 0 such rows AND the cross-check envelope omitted that kind, so its load COPY (Double/threshold, four Int side counters, the ... | spec/claude,maint/claude |
| 3 | should-fix | doc-quality | `scripts/migrate-schema.py:3590` | The new `_assert_master_capable` PermissionError tells the operator to 'see migrations/README.md and the migration header', but migrations/README.md has NO mention of the requires-superuser marker, the master-capable bootstrap, the rolsu... | maint/claude |
| 4 | nit | convention | `benchmarks-website/migrate/src/postgres.rs:1` | Local module is named `postgres` (`crate::postgres`) while the crate also depends on the external `postgres` crate. In verify.rs `crate::postgres::ColKind` and extern `postgres::Client` coexist; a fresh reader must disambiguate. | maint/claude |
| 5 | nit | convention | `benchmarks-website/migrate/src/verify.rs:684` | Two private `fn select_sql` with different signatures coexist: postgres.rs:333 `select_sql(&TableSpec)` and verify.rs:684 `select_sql(&ValueSpec, epoch_expr)`. A grep for `select_sql` lands on two unrelated builders. | maint/claude |
| 6 | nit | coverage | `benchmarks-website/migrate/src/verify.rs:972` | `in_memory_v3()` is duplicated verbatim in postgres.rs:506 and verify.rs:972, and build_fixture_duckdb (postgres_e2e.rs:97) repeats the same DDL-apply loop a third time. Three copies of the v3-schema builder; all iterate FAMILIES/COMMITS... | maint/claude |
| 7 | nit | boundary | `scripts/migrate-schema.py (_migration_requires_superuser)` | requires-superuser marker detection scans each physical line and matches when `line.startswith('--')` and `stripped == directive`. A migration INSERTing multi-line text whose own line begins with `-- migrate-schema: requires-superuser` (... | correctness/claude |

These 3 should-fix items are all NON-code documentation/status annotations; tracked for a follow-up doc/status pass (none block Phase 3). The 4 nits are dismissible per the trusted-input throwaway-crate calibration.

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1) — cycle 1 raw archive</summary>

```json
{
  "schema_version": 1,
  "preset": "phase-3",
  "lenses_used": [
    "spec",
    "correctness",
    "maint"
  ],
  "overall": "accept",
  "must_fix_count": 0,
  "should_fix_count": 3,
  "nit_count": 4,
  "review_cycles_this_invocation": 1,
  "unified_findings": [
    {
      "severity": "should-fix",
      "kind": "scope-drift",
      "file_line": "benchmarks-website/migrate/Cargo.toml:35-37",
      "description": "PR-3.1 shipped the synchronous `postgres` 0.19 crate (+ native-tls + postgres-native-tls), not the plan-named `tokio-postgres`, with no explicit deviation note in the PR-3.1 implementation-status entry. The binding hard-NO (no aws-sdk-rds) IS honored and the swap is defensible (sync `postgres` is a thin blocking wrapper over tokio-postgres), but a silent named-dependency substitution is the quiet plan-narrowing the spec lens surfaces.",
      "recommended_fix": "Add one line to the PR-3.1 implementation-status Surprises bullet noting the deliberate choice of synchronous `postgres` over the plan-named `tokio-postgres`. No code change required.",
      "found_by": [
        "spec/claude"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "weak-acceptance",
      "file_line": "benchmarks-website/migrate/tests/postgres_e2e.rs",
      "description": "PR-3.4's real-snapshot LOCAL rehearsal gave `vector_search_runs` zero real-data coverage: the real snapshot held 0 such rows AND the cross-check envelope omitted that kind, so its load COPY (Double/threshold, four Int side counters, the only Int32 value column), value-verify join, and cross-check were never exercised against real data. The PR-3.4 status asserts GREEN without qualifying which table fell through; coverage rests on the synthetic PR-3.3 fixture + PR-3.2/3.5 unit tests.",
      "recommended_fix": "Annotate the PR-3.4 status entry that `vector_search_runs` had 0 real rows so its paths were validated only by synthetic fixtures, and note PR-5.0's freshest-snapshot prod load is expected to cover it. No code change.",
      "found_by": [
        "spec/claude",
        "maint/claude"
      ]
    },
    {
      "severity": "should-fix",
      "kind": "doc-quality",
      "file_line": "scripts/migrate-schema.py:3590",
      "description": "The new `_assert_master_capable` PermissionError tells the operator to 'see migrations/README.md and the migration header', but migrations/README.md has NO mention of the requires-superuser marker, the master-capable bootstrap, the rolsuper-OR-rolcreaterole preflight, or 002/004 ordering. The cited doc does not back the contract the error advertises.",
      "recommended_fix": "Add a short subsection to migrations/README.md covering the `migrate-schema: requires-superuser` marker, the rolsuper OR rolcreaterole preflight, and 'apply 002/004 as the RDS master before any migrator deploy'.",
      "found_by": [
        "maint/claude"
      ]
    },
    {
      "severity": "nit",
      "kind": "convention",
      "file_line": "benchmarks-website/migrate/src/postgres.rs:1",
      "description": "Local module is named `postgres` (`crate::postgres`) while the crate also depends on the external `postgres` crate. In verify.rs `crate::postgres::ColKind` and extern `postgres::Client` coexist; a fresh reader must disambiguate.",
      "recommended_fix": "Rename the local module (e.g. `loader`/`pg_load`) or import the extern crate as `pg`.",
      "found_by": [
        "maint/claude"
      ]
    },
    {
      "severity": "nit",
      "kind": "convention",
      "file_line": "benchmarks-website/migrate/src/verify.rs:684",
      "description": "Two private `fn select_sql` with different signatures coexist: postgres.rs:333 `select_sql(&TableSpec)` and verify.rs:684 `select_sql(&ValueSpec, epoch_expr)`. A grep for `select_sql` lands on two unrelated builders.",
      "recommended_fix": "Disambiguate the verify one (e.g. `value_select_sql`).",
      "found_by": [
        "maint/claude"
      ]
    },
    {
      "severity": "nit",
      "kind": "coverage",
      "file_line": "benchmarks-website/migrate/src/verify.rs:972",
      "description": "`in_memory_v3()` is duplicated verbatim in postgres.rs:506 and verify.rs:972, and build_fixture_duckdb (postgres_e2e.rs:97) repeats the same DDL-apply loop a third time. Three copies of the v3-schema builder; all iterate FAMILIES/COMMITS_DDL so the table set cannot drift, but the duplication is real.",
      "recommended_fix": "Extract one `pub(crate) fn open_in_memory_v3()` shared by all three call sites.",
      "found_by": [
        "maint/claude"
      ]
    },
    {
      "severity": "nit",
      "kind": "boundary",
      "file_line": "scripts/migrate-schema.py (_migration_requires_superuser)",
      "description": "requires-superuser marker detection scans each physical line and matches when `line.startswith('--')` and `stripped == directive`. A migration INSERTing multi-line text whose own line begins with `-- migrate-schema: requires-superuser` (inside a string literal) would be falsely classified. Author-controlled trusted content; the review calibration argues against fixing.",
      "recommended_fix": "If touched later, strip SQL string literals / inspect only the leading comment block. Not required given trusted-input calibration.",
      "found_by": [
        "correctness/claude"
      ]
    }
  ],
  "disagreements": [
    {
      "topic": "vector_search_runs real-data coverage gap through the PR-3.4 validation gate",
      "positions": [
        {
          "lens": "spec",
          "position": "should-fix / weak-acceptance: one of six tables (the one with the most distinct column shape) got no real-data validation through the gate while the status asserts GREEN without qualifying the fall-through. Annotate the status entry."
        },
        {
          "lens": "maint",
          "position": "acceptable / amend_plan: no: the Double/threshold COPY + vector value-verify paths are covered by the synthetic e2e fixture (one seeded vector row), so the gap is covered; no plan amendment needed."
        }
      ],
      "synthesizer_call": "Act on the spec position at should-fix, scoped to a status-log annotation (NOT a code/test change). maint is correct that the paths ARE fixture-exercised (no untested-code defect, no new test warranted for a throwaway crate); spec is correct that PR-3.4's distinguishing value is REAL-data validation that vector_search_runs did not receive. Both reduce to: annotate the PR-3.4 status that vector_search_runs had 0 real rows and point at PR-5.0's freshest-snapshot prod load. Non-blocking."
    }
  ],
  "dropped_re_flags": [
    {
      "topic": "Dim-column COPY corruption is invisible to the primary value-verify gate (verify.rs omits dim columns; load_table COPYs them verbatim)",
      "reason": "covered by Accepted tradeoffs",
      "reference": "Accepted tradeoffs / r1 traps"
    }
  ],
  "executive_summary": "Phase 3 ships the v3-to-v4 DuckDB-to-Postgres migration TOOLKIT and is accepted by all three lenses (spec, correctness, maint) with ZERO must-fix findings. The phase delivers an atomic single-transaction COPY loader (measurement_id/commit_sha verbatim, UTC-cast timestamps, non-finite f64 rejected, no aws-sdk-rds), a per-measurement_id value verifier as the primary correctness gate (engine-independent epoch-microsecond timestamps, element-wise order-sensitive array compare, read-kinds single-sourced from the loader so the two sides cannot drift), a first local postgres:16 testcontainer rehearsal (counts + verify-clean + atomic mid-load rollback), a Python-writer cross-check, and a requires-superuser bootstrap guard. PR-3.4 ran load+verify+cross-check GREEN against the real 4.2M-row snapshot into local PG16 with zero prod writes, clearing the phase exit criteria. The correctness lens ran a thorough adversarial hunt and found no surviving correctness bug; correctness-critical paths are unusually well-covered. Findings: three should-fix, all NON-code documentation/status annotations (sync-postgres-vs-tokio-postgres deviation note; vector_search_runs real-data coverage annotation; migrations/README.md missing the requires-superuser contract the PermissionError advertises) + four nits. One disagreement surfaced (spec rated the vector_search_runs gap should-fix; maint rated it acceptable) — synthesizer scoped it to a status annotation. Two tradeoffs moved to revisit-but-keep (DROP-aws-sdk-rds; the 004 preflight guard), both keep-with-a-doc-followup. Verdict: ACCEPT, 0 must-fix, 3 should-fix, 4 nits.",
  "phase_artifacts": {
    "summary": "Phase 3 delivers the v3-to-v4 DuckDB-to-Postgres migration TOOLKIT (validated locally; the irreversible prod load is deferred to the Phase-5 cutover as PR-5.0). (1) BULK LOADER (postgres.rs + Load subcommand): reads 6 tables from an existing v3 DuckDB as Arrow batches and streams them via COPY FROM STDIN text format inside ONE transaction (atomic; mid-load failure rolls back). measurement_id/commit_sha verbatim; commits.timestamp CAST to VARCHAR under SET TimeZone=UTC; all_runtimes_ns explodes to {a,b,c}; f64 shortest-round-trip with non-finite rejected; NoTls locally or native-tls+--ca-cert for prod; NO aws-sdk-rds. (2) VALUE VERIFIER (verify.rs + verify --postgres-target): PRIMARY gate, a new PgVerifyReport/ValueMismatch pair (separate from the v2 structural VerifyReport), per-measurement_id full compare of every non-hashed value column, all_runtimes_ns element-wise + order-sensitive, timestamps as engine-independent epoch microseconds, read kinds resolved once from the loader column_kind so the two sides cannot drift. (3) REHEARSAL (postgres_e2e.rs + README): first migrate testcontainer (postgres:16-alpine), asserts count match + verify-clean + atomic mid-load rollback, Docker-gated skip-local/fail-loud-CI. (4) CROSS-CHECK (cross_check_python_writer.py): writer UPDATEs seeded rows (not duplicate-INSERTs), values round-trip, reuses connect_postgres verify-full TLS as bench_ingest. (5) BOOTSTRAP GUARD (migrate-schema.py): requires-superuser marker + rolsuper-OR-rolcreaterole preflight rejecting a non-master apply before any DDL. PR-3.4 ran all three gates GREEN against the real 4.2M-row bench.duckdb into local PG16, zero prod writes. Toolkit honestly labeled throwaway (removal trigger Table F / PR-5.3).",
    "surprises": [
      {
        "what": "The loader reads an EXISTING v3 DuckDB (copying measurement_id/commit_sha verbatim) rather than re-accumulating from v2.",
        "how_handled": "Documented; simpler and safer since ids are byte-preserved.",
        "amend_plan": "no"
      },
      {
        "what": "PR-3.1 shipped the synchronous `postgres` 0.19 crate, not the plan-named `tokio-postgres`, with no deviation note.",
        "how_handled": "Defensible (hard-NO on aws-sdk-rds honored) but unremarked. Add a one-line status note.",
        "amend_plan": "yes"
      },
      {
        "what": "PR-3.4's real snapshot held vector_search_runs=0 rows AND the cross-check envelope omitted that kind, so vector_search_runs got zero REAL-data coverage through the gate.",
        "how_handled": "Code paths fixture-covered; status overstates GREEN. Annotate; PR-5.0 closes it. (Lens disagreement: spec should-fix, maint acceptable.)",
        "amend_plan": "yes"
      },
      {
        "what": "The bootstrap PermissionError cites migrations/README.md content (requires-superuser marker / preflight / 002-004 ordering) that does not exist there.",
        "how_handled": "Not handled; add a README subsection.",
        "amend_plan": "yes"
      },
      {
        "what": "The local module `postgres` collides by name with the external `postgres` crate.",
        "how_handled": "Compiles; cosmetic disambiguation nit.",
        "amend_plan": "no"
      },
      {
        "what": "PR-3.1 status claims column order is 'programmatically cross-checked against migrations/001', but the check is indirect (e2e COPY fails on disagreement), not a DDL-parsing test.",
        "how_handled": "Status log overstates mechanism; behavior covered by the e2e.",
        "amend_plan": "no"
      }
    ],
    "coverage": {
      "tested_cases": [
        "loader COPY-text rendering incl. escaping/BIGINT[]/literal-\\N-vs-NULL/non-UTC-tz/f64-round-trip+non-finite-reject (postgres.rs:514-712, high)",
        "value-verify discrimination across 6 tables incl. array reorder/Int32+side-counters/presence-both-directions/epoch-us-pinned (verify.rs:965-1356, high)",
        "Postgres e2e: counts/verify-clean/atomic mid-load rollback (postgres_e2e.rs:136-226, high)",
        "cross-check UPDATE-not-INSERT/all-5-kinds/NOT-seeded->INSERT/value_mismatches mutation-verified (test_post_ingest_postgres.py:3795-3947, high)",
        "requires-superuser guard: marker detection/non-master rejected before DDL (test_migrate_schema.py:3629-3768, high)",
        "real-data PR-3.4: 4.2M-row load exact match + verify clean + cross-check clean + negative (manual run, medium)"
      ],
      "untested_cases": [
        "vector_search_runs Double/threshold COPY + vector value-verify against REAL data (0 real rows + omitted from cross-check) — priority medium, annotate status + PR-5.0 closes",
        "verify-full TLS --ca-cert against a live RDS endpoint — priority low, compile+runbook only, plan-acknowledged",
        "TABLE_SPECS vs migrations/001 column-order as a direct DDL-parsing unit test — priority low, covered behaviorally by the e2e COPY"
      ]
    },
    "tradeoffs": [
      {
        "decision": "Prod historical load TIMING deferred Phase 3 -> Phase 5 (PR-5.0); Phase 3 closes on the real-snapshot LOCAL rehearsal",
        "verdict": "keep",
        "rationale": "PR-3.4 executed exactly this with zero irreversible prod side-effect; RDS PITR is the prod rollback."
      },
      {
        "decision": "Value-column verify as a NEW PgVerifyReport pair (not overloading v2-structural VerifyReport)",
        "verdict": "keep",
        "rationale": "Single-responsibility; structural vs stored-value comparison are different concerns."
      },
      {
        "decision": "commits.timestamp compared as epoch microseconds",
        "verdict": "keep",
        "rationale": "Engine-independent and exact; e2e confirmed sub-second + pre-1970."
      },
      {
        "decision": "local postgres:16 testcontainer rehearsal, Docker-gated",
        "verdict": "keep",
        "rationale": "Same engine as RDS at zero AWS cost; e2e closes the live Postgres-execution gap."
      },
      {
        "decision": "operator-run-locally master-password DSN; DROP aws-sdk-rds",
        "verdict": "revisit-but-keep",
        "rationale": "Keep (avoids heavy AWS SDK); record the resulting sync-`postgres`-over-`tokio-postgres` swap in the PR-3.1 status."
      },
      {
        "decision": "004-as-master requires-superuser preflight guard",
        "verdict": "revisit-but-keep",
        "rationale": "Guard is well-shaped and load-bearing; close the migrations/README.md doc gap the PermissionError points at."
      },
      {
        "decision": "trusted-input low-stakes review calibration; 3-vote Phase 3",
        "verdict": "keep",
        "rationale": "Appropriate; adversarial-input hardening correctly out of scope; retained guards are load-bearing."
      }
    ]
  }
}
```

</details>

## Phase 3: Historical data load (DuckDB → Postgres) + value-verify — end-of-phase review (cycle 2) — accepted (3-vote)
**Cycle 2 (post PR-3.6 amend). Synthesizer output from /spiral:gauntlet (preset=phase-3, lenses=spec+correctness+maint, executor=claude). Verdict: ACCEPT — 0 must-fix, 1 should-fix (disclosed/deferred to PR-5.0), 3 nits. All 3 cycle-1 should-fix items confirmed ADDRESSED by PR-3.6 (not re-flagged). Full JSON in the `<details>` block at the end.**

### Summary of changes
Phase 3 builds + validates the v3->v4 DuckDB->Postgres migration toolkit and closes on a real-snapshot LOCAL rehearsal, with the one-shot prod load deferred to Phase 5 (PR-5.0). (1) BULK LOADER (PR-3.1, postgres.rs): atomic single-txn per-table COPY FROM STDIN; measurement_id/commit_sha verbatim; UTC-cast timestamp; BIGINT[] as {a,b,c}; f64 shortest-round-trip, non-finite rejected; NO aws-sdk-rds (sync postgres 0.19 + native-tls, NoTls local / --ca-cert verify-full prod). (2) VALUE-VERIFY GATE (PR-3.2, verify.rs): PRIMARY v4-correctness gate; per-measurement_id full compare of every non-hashed value column, env_triple + arrays element-wise + commits metadata, engine-independent epoch-microsecond timestamps; column types single-sourced from the loader's column_kind. (3) BOOTSTRAP GUARD (PR-3.1, migrate-schema.py): requires-superuser marker on 002/004 + rolsuper-OR-rolcreaterole preflight failing loud BEFORE the transaction. (4) REHEARSAL HARNESS (PR-3.3, postgres_e2e.rs): first migrate testcontainer; counts + verify-clean + atomic mid-load rollback; Docker-gated. (5) CROSS-CHECK (PR-3.5): Python writer UPDATEs seeded rows (not duplicate-INSERTs), values round-trip; mutation-verified discrimination. (6) REAL-DATA GATE (PR-3.4): ran the toolkit against a real 4.2M-row snapshot into local PG16 — exact row counts, full value-verify clean, cross-check clean, zero prod write. (7) PR-3.6 amend: README requires-superuser subsection + 2 status annotations (doc/status only). The cumulative diff faithfully implements the Phase-3 contract with no out-of-scope drift.

### Surprises and discoveries
- **PR-3.1 shipped sync `postgres` 0.19, not plan-named `tokio-postgres` (deliberate; no async runtime for a one-shot CLI; NO-aws-sdk-rds honored; tokio-postgres pulled in transitively).** — Annotated in the PR-3.1 status by PR-3.6. (amend_plan: already-done)
- **The real PR-3.4 snapshot held vector_search_runs=0 rows AND the cross-check omitted that kind, so that table got fixture+unit coverage only, not real-data.** — Annotated in the PR-3.4 status by PR-3.6; PR-5.0 (freshest snapshot) expected to close it or record fixture-only as an accepted residual. (amend_plan: already-done)
- **PR-3.4 ran against a pre-existing on-disk ./bench.duckdb on a Homebrew PG16 cluster (Docker was down) rather than the SSH/scp + testcontainer path the runbook describes.** — Captured in the PR-3.4 status as an equivalent rehearsal substrate; the prod runbook still documents the acquisition+verification path PR-5.0 uses (minor runbook-vs-execution divergence, nit). (amend_plan: no)
- **README authoring rules reference a hypothetical `-- migrate: no-transaction` directive whose prefix differs from the only implemented `-- migrate-schema: requires-superuser`.** — Not reconciled; flagged as a doc-consistency nit. (amend_plan: no)

### Testing coverage assessment
**Tested (high confidence):**
- loader COPY-text + DuckDB-read pipeline (escaping, BIGINT[] empty/negative, NULLs, literal-\N-vs-NULL, UTC + non-UTC-offset timestamp, f64 round-trip, empty tables) — 11 unit tests (postgres.rs, high)
- value-verify discrimination across 6 tables incl. array reorder / Int32 + side counters / presence both directions / epoch-us pinned / read_kinds drift guard — 14 no-Docker tests (verify.rs, high)
- live PG16 e2e: counts + verify-clean + atomic mid-load rollback; sub-second + pre-1970 epoch exact — 2 testcontainer tests (postgres_e2e.rs, high)
- requires-superuser marker detection + non-master rejection BEFORE DDL (test_migrate_schema.py, high)
- Python cross-check UPDATE-not-INSERT + all 5 kinds + value/env/memory/side-counter discrimination, mutation-verified (test_post_ingest_postgres.py, high)
- real-data gate: 4.2M-row load exact-match + full value-verify clean + cross-check clean (PR-3.4 operational, high)
- PR-3.6 README subsection cross-verified against migrate-schema.py (marker, proxy SQL, PermissionError pointer, GRANT-in-failure-list) — this review, high

**Untested / under-covered (non-blocking):**
- vector_search_runs against the REAL data shape (Double threshold dim, four Int side counters, only Int32 value column) — priority medium, real snapshot had 0 rows + omitted from cross-check; fixture+unit only; DISCLOSED + deferred to PR-5.0
- verify-full TLS --ca-cert against a live RDS host — priority low, compile + local self-signed-cert run + runbook only; needs live RDS; plan-acknowledged

### Tradeoffs re-evaluation
- **One-shot historical load: retarget migrate/ for DuckDB->Postgres (throwaway)** — _keep_: Loader reads an existing v3 DuckDB and copies measurement_id verbatim; validated clean against the real snapshot; removal trigger documented.
- **Execution model: operator-run-locally master password, NO aws-sdk-rds** — _keep_: aws-sdk-rds absent; sync postgres 0.19 suffices; the tokio-postgres->sync-postgres swap is now annotated as deliberate.
- **Value-column verify as the PRIMARY gate (NEW PgVerifyReport pair)** — _keep_: Per-measurement_id full compare implemented + runtime-validated; green against the real snapshot.
- **commits.timestamp as epoch microseconds** — _keep_: Engine-independent + exact; e2e confirmed sub-second + pre-1970.
- **local postgres:16 testcontainer rehearsal (Docker-gated)** — _keep_: Same engine as RDS; PR-3.4 used an equivalent Homebrew PG16 cluster when Docker was down.
- **004-as-master requires-superuser preflight guard** — _keep_: Guard + fail-loud test shipped; PR-3.6 added the operator-facing README the PermissionError points to (cycle-1 should-fix closed).
- **Prod historical load TIMING deferred Phase 3 -> Phase 5 (PR-5.0)** — _keep_: PR-3.4 ran the real-snapshot LOCAL rehearsal with zero prod write; prod load remains PR-5.0/5.1.
- **trusted-input low-stakes review calibration; 3-vote Phase 3** — _keep_: Appropriate; this cycle-2 review ran the full 3-vote (spec + correctness + maint).

### Disagreements
None — all three lenses accepted.

**Dropped re-flags (carry-forward respected):**
- Cycle-1 should-fix items (sync-postgres deviation note; vector_search_runs annotation; migrations/README requires-superuser doc gap) — addressed by PR-3.6; reviewers confirmed resolved, not re-flagged (PR-3.6 (README subsection + PR-3.1/PR-3.4 status annotations))
- Dim-column COPY corruption invisible to the value-verify gate — covered by Accepted tradeoffs (Accepted tradeoffs / r1 traps)

### Findings (0 must-fix, 1 should-fix, 3 nits)
| # | Severity | Kind | File:line | Description | Found by |
|---|----------|------|-----------|-------------|----------|
| 1 | should-fix | coverage | `.big-plans/ct__bench-v4.md (PR-3.4 status) / PR-5.0` | vector_search_runs got zero real-data validation (0 rows in the snapshot + omitted from the PR-3.4 cross-check envelope); its Double/threshold dim, four Int side counters, and only Int32 value column are exercised only by synthetic fixtu... | spec/claude |
| 2 | nit | cross-bundle-consistency | `migrations/README.md:27 (authoring rules)` | The README authoring-rules section names a hypothetical future directive `-- migrate: no-transaction` (prefix `migrate:`) while the only implemented directive uses prefix `migrate-schema:` (`-- migrate-schema: requires-superuser`). A fut... | maint/claude |
| 3 | nit | doc-comment | `benchmarks-website/migrate/src/postgres.rs (pub fn load)` | `load`'s doc comment does not state that the target schema (migrations/001) must already be applied — `load` only COPYs into existing tables, never DDL. The precondition is in the README + e2e test but not the function doc. | maint/claude |
| 4 | nit | doc-quality | `benchmarks-website/migrate/README.md (REAL-snapshot rehearsal runbook)` | The PR-3.3 runbook presents SSH/scp + S3-rehydrate acquisition + live source-account/region verification as the rehearsal entry path, but PR-3.4 was executed against a pre-existing on-disk ./bench.duckdb. The prod-facing runbook (which P... | spec/claude |

The 1 should-fix is the already-disclosed vector_search_runs residual (tracked to PR-5.0; no Phase-3 action owed). The 3 nits are cheap doc polish, dismissible per the trusted-input calibration.

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1) — cycle 2 raw archive</summary>

```json
{
  "schema_version": 1,
  "preset": "phase-3",
  "lenses_used": [
    "spec",
    "correctness",
    "maint"
  ],
  "cycle": 2,
  "overall": "accept",
  "must_fix_count": 0,
  "should_fix_count": 1,
  "nit_count": 3,
  "unified_findings": [
    {
      "severity": "should-fix",
      "kind": "coverage",
      "file_line": ".big-plans/ct__bench-v4.md (PR-3.4 status) / PR-5.0",
      "description": "vector_search_runs got zero real-data validation (0 rows in the snapshot + omitted from the PR-3.4 cross-check envelope); its Double/threshold dim, four Int side counters, and only Int32 value column are exercised only by synthetic fixtures + unit tests. NOW DISCLOSED by PR-3.6's annotation and tracked to PR-5.0 — no new code/test action is owed within Phase 3; flagged for completeness so PR-5.0 explicitly re-assesses it at cutover.",
      "recommended_fix": "Keep the PR-3.6 annotation; make PR-5.0's freshest-snapshot prod load explicitly re-check vector_search_runs row count and either close it with real rows or record fixture-only as an accepted residual.",
      "found_by": [
        "spec/claude"
      ]
    },
    {
      "severity": "nit",
      "kind": "cross-bundle-consistency",
      "file_line": "migrations/README.md:27 (authoring rules)",
      "description": "The README authoring-rules section names a hypothetical future directive `-- migrate: no-transaction` (prefix `migrate:`) while the only implemented directive uses prefix `migrate-schema:` (`-- migrate-schema: requires-superuser`). A future engineer adding a runner directive can't tell which prefix is canonical. Pre-existing (predates PR-3.6); surfaced now alongside the new requires-superuser subsection.",
      "recommended_fix": "Align the hypothetical example to `-- migrate-schema: no-transaction`, or note that all runner directives share the `migrate-schema:` prefix.",
      "found_by": [
        "maint/claude"
      ]
    },
    {
      "severity": "nit",
      "kind": "doc-comment",
      "file_line": "benchmarks-website/migrate/src/postgres.rs (pub fn load)",
      "description": "`load`'s doc comment does not state that the target schema (migrations/001) must already be applied — `load` only COPYs into existing tables, never DDL. The precondition is in the README + e2e test but not the function doc.",
      "recommended_fix": "Add one sentence to the `load` doc comment noting the target tables must pre-exist.",
      "found_by": [
        "maint/claude"
      ]
    },
    {
      "severity": "nit",
      "kind": "doc-quality",
      "file_line": "benchmarks-website/migrate/README.md (REAL-snapshot rehearsal runbook)",
      "description": "The PR-3.3 runbook presents SSH/scp + S3-rehydrate acquisition + live source-account/region verification as the rehearsal entry path, but PR-3.4 was executed against a pre-existing on-disk ./bench.duckdb. The prod-facing runbook (which PR-5.0 consumes) and the as-executed PR-3.4 rehearsal diverge in acquisition method.",
      "recommended_fix": "Optionally add a one-line note that a pre-acquired on-disk snapshot is an acceptable rehearsal source. No code change; the runbook is the PR-5.0 prod-path doc.",
      "found_by": [
        "spec/claude"
      ]
    }
  ],
  "disagreements": [],
  "dropped_re_flags": [
    {
      "topic": "Cycle-1 should-fix items (sync-postgres deviation note; vector_search_runs annotation; migrations/README requires-superuser doc gap)",
      "reason": "addressed by PR-3.6; reviewers confirmed resolved, not re-flagged",
      "reference": "PR-3.6 (README subsection + PR-3.1/PR-3.4 status annotations)"
    },
    {
      "topic": "Dim-column COPY corruption invisible to the value-verify gate",
      "reason": "covered by Accepted tradeoffs",
      "reference": "Accepted tradeoffs / r1 traps"
    }
  ],
  "executive_summary": "Phase 3 CYCLE 2 (after the PR-3.6 amend) is accepted by all three lenses (spec, correctness, maint) with ZERO must-fix. The amend applied the three cycle-1 should-fix items — a migrations/README requires-superuser bootstrap-ordering subsection plus PR-3.1 (sync-postgres) and PR-3.4 (vector_search_runs) status annotations — and all three reviewers confirmed them ADDRESSED (the README was verified line-for-line against scripts/migrate-schema.py: the marker string, the rolsuper-OR-rolcreaterole proxy, the PermissionError pointer, the guard-fires-before-transaction ordering, and which migrations carry the marker). The correctness lens re-hunted the loader/verifier/guard/cross-check adversarially and found no surviving bug and no false behavioral claim in the new doc. Residual findings are minor and non-blocking: one should-fix (the vector_search_runs real-data coverage gap, which is now DISCLOSED and tracked to PR-5.0 — no Phase-3 action owed) and three nits (a README directive-namespace inconsistency `migrate:` vs `migrate-schema:`; the `load` doc-comment not stating the schema-must-pre-exist precondition; the prod runbook's SSH/scp acquisition path diverging from PR-3.4's on-disk execution). All key tradeoffs hold at keep. Verdict: ACCEPT, 0 must-fix, 1 should-fix (disclosed/deferred), 3 nits. Phase 3 is complete and correct; the only tracked real-data residual (vector_search_runs) routes to the Phase-5 prod load.",
  "phase_artifacts": {
    "summary": "Phase 3 builds + validates the v3->v4 DuckDB->Postgres migration toolkit and closes on a real-snapshot LOCAL rehearsal, with the one-shot prod load deferred to Phase 5 (PR-5.0). (1) BULK LOADER (PR-3.1, postgres.rs): atomic single-txn per-table COPY FROM STDIN; measurement_id/commit_sha verbatim; UTC-cast timestamp; BIGINT[] as {a,b,c}; f64 shortest-round-trip, non-finite rejected; NO aws-sdk-rds (sync postgres 0.19 + native-tls, NoTls local / --ca-cert verify-full prod). (2) VALUE-VERIFY GATE (PR-3.2, verify.rs): PRIMARY v4-correctness gate; per-measurement_id full compare of every non-hashed value column, env_triple + arrays element-wise + commits metadata, engine-independent epoch-microsecond timestamps; column types single-sourced from the loader's column_kind. (3) BOOTSTRAP GUARD (PR-3.1, migrate-schema.py): requires-superuser marker on 002/004 + rolsuper-OR-rolcreaterole preflight failing loud BEFORE the transaction. (4) REHEARSAL HARNESS (PR-3.3, postgres_e2e.rs): first migrate testcontainer; counts + verify-clean + atomic mid-load rollback; Docker-gated. (5) CROSS-CHECK (PR-3.5): Python writer UPDATEs seeded rows (not duplicate-INSERTs), values round-trip; mutation-verified discrimination. (6) REAL-DATA GATE (PR-3.4): ran the toolkit against a real 4.2M-row snapshot into local PG16 — exact row counts, full value-verify clean, cross-check clean, zero prod write. (7) PR-3.6 amend: README requires-superuser subsection + 2 status annotations (doc/status only). The cumulative diff faithfully implements the Phase-3 contract with no out-of-scope drift.",
    "surprises": [
      {
        "what": "PR-3.1 shipped sync `postgres` 0.19, not plan-named `tokio-postgres` (deliberate; no async runtime for a one-shot CLI; NO-aws-sdk-rds honored; tokio-postgres pulled in transitively).",
        "how_handled": "Annotated in the PR-3.1 status by PR-3.6.",
        "amend_plan": "already-done"
      },
      {
        "what": "The real PR-3.4 snapshot held vector_search_runs=0 rows AND the cross-check omitted that kind, so that table got fixture+unit coverage only, not real-data.",
        "how_handled": "Annotated in the PR-3.4 status by PR-3.6; PR-5.0 (freshest snapshot) expected to close it or record fixture-only as an accepted residual.",
        "amend_plan": "already-done"
      },
      {
        "what": "PR-3.4 ran against a pre-existing on-disk ./bench.duckdb on a Homebrew PG16 cluster (Docker was down) rather than the SSH/scp + testcontainer path the runbook describes.",
        "how_handled": "Captured in the PR-3.4 status as an equivalent rehearsal substrate; the prod runbook still documents the acquisition+verification path PR-5.0 uses (minor runbook-vs-execution divergence, nit).",
        "amend_plan": "no"
      },
      {
        "what": "README authoring rules reference a hypothetical `-- migrate: no-transaction` directive whose prefix differs from the only implemented `-- migrate-schema: requires-superuser`.",
        "how_handled": "Not reconciled; flagged as a doc-consistency nit.",
        "amend_plan": "no"
      }
    ],
    "coverage": {
      "tested_cases": [
        "loader COPY-text + DuckDB-read pipeline (escaping, BIGINT[] empty/negative, NULLs, literal-\\N-vs-NULL, UTC + non-UTC-offset timestamp, f64 round-trip, empty tables) — 11 unit tests (postgres.rs, high)",
        "value-verify discrimination across 6 tables incl. array reorder / Int32 + side counters / presence both directions / epoch-us pinned / read_kinds drift guard — 14 no-Docker tests (verify.rs, high)",
        "live PG16 e2e: counts + verify-clean + atomic mid-load rollback; sub-second + pre-1970 epoch exact — 2 testcontainer tests (postgres_e2e.rs, high)",
        "requires-superuser marker detection + non-master rejection BEFORE DDL (test_migrate_schema.py, high)",
        "Python cross-check UPDATE-not-INSERT + all 5 kinds + value/env/memory/side-counter discrimination, mutation-verified (test_post_ingest_postgres.py, high)",
        "real-data gate: 4.2M-row load exact-match + full value-verify clean + cross-check clean (PR-3.4 operational, high)",
        "PR-3.6 README subsection cross-verified against migrate-schema.py (marker, proxy SQL, PermissionError pointer, GRANT-in-failure-list) — this review, high"
      ],
      "untested_cases": [
        "vector_search_runs against the REAL data shape (Double threshold dim, four Int side counters, only Int32 value column) — priority medium, real snapshot had 0 rows + omitted from cross-check; fixture+unit only; DISCLOSED + deferred to PR-5.0",
        "verify-full TLS --ca-cert against a live RDS host — priority low, compile + local self-signed-cert run + runbook only; needs live RDS; plan-acknowledged"
      ],
      "recommendations": "Coverage for the phase's scope is strong and the value-fidelity-critical paths are runtime-validated against both a real PG16 container and a real 4.2M-row snapshot. The single real-data residual (vector_search_runs) is disclosed and tracked to PR-5.0. No new tests are owed within Phase 3."
    },
    "tradeoffs": [
      {
        "decision": "One-shot historical load: retarget migrate/ for DuckDB->Postgres (throwaway)",
        "verdict": "keep",
        "rationale": "Loader reads an existing v3 DuckDB and copies measurement_id verbatim; validated clean against the real snapshot; removal trigger documented."
      },
      {
        "decision": "Execution model: operator-run-locally master password, NO aws-sdk-rds",
        "verdict": "keep",
        "rationale": "aws-sdk-rds absent; sync postgres 0.19 suffices; the tokio-postgres->sync-postgres swap is now annotated as deliberate."
      },
      {
        "decision": "Value-column verify as the PRIMARY gate (NEW PgVerifyReport pair)",
        "verdict": "keep",
        "rationale": "Per-measurement_id full compare implemented + runtime-validated; green against the real snapshot."
      },
      {
        "decision": "commits.timestamp as epoch microseconds",
        "verdict": "keep",
        "rationale": "Engine-independent + exact; e2e confirmed sub-second + pre-1970."
      },
      {
        "decision": "local postgres:16 testcontainer rehearsal (Docker-gated)",
        "verdict": "keep",
        "rationale": "Same engine as RDS; PR-3.4 used an equivalent Homebrew PG16 cluster when Docker was down."
      },
      {
        "decision": "004-as-master requires-superuser preflight guard",
        "verdict": "keep",
        "rationale": "Guard + fail-loud test shipped; PR-3.6 added the operator-facing README the PermissionError points to (cycle-1 should-fix closed)."
      },
      {
        "decision": "Prod historical load TIMING deferred Phase 3 -> Phase 5 (PR-5.0)",
        "verdict": "keep",
        "rationale": "PR-3.4 ran the real-snapshot LOCAL rehearsal with zero prod write; prod load remains PR-5.0/5.1."
      },
      {
        "decision": "trusted-input low-stakes review calibration; 3-vote Phase 3",
        "verdict": "keep",
        "rationale": "Appropriate; this cycle-2 review ran the full 3-vote (spec + correctness + maint)."
      }
    ]
  }
}
```

</details>

## Phase 3: Historical data load (DuckDB → Postgres) + value-verify — end-of-phase review (cycle 3) — accepted (3-vote)
**Cycle 3 (post PR-3.7 amend). Synthesizer output from /spiral:gauntlet (preset=phase-3, lenses=spec+correctness+maint, executor=claude). Verdict: ACCEPT — 0 must-fix, 1 should-fix (RESOLVED inline, commit 6275f272d), 1 nit (resolved). All cycle-2 nits confirmed CLOSED by PR-3.7. Code logic byte-identical to cycle-1's accept. Full JSON in the `<details>` block at the end.**

### Summary of changes
Phase 3 builds + validates the v3->v4 DuckDB->Postgres migration toolkit (atomic COPY loader; the PRIMARY per-measurement_id value-verify gate; a testcontainer rehearsal harness; a Python-writer cross-check; a requires-superuser bootstrap guard) and closes on PR-3.4's real-snapshot LOCAL rehearsal — GREEN against the real 4.2M-row v3 snapshot into local PG16, zero prod write. The one-shot prod load is deferred to PR-5.0 at the Phase-5 cutover. Doc amends: PR-3.6 (cycle-1 should-fixes: the requires-superuser README subsection + PR-3.1/PR-3.4 status annotations) and PR-3.7 (cycle-2 nits: directive-namespace alignment, load precondition doc, on-disk-snapshot runbook bullet + repointing stale prod-load refs to PR-5.0). The cycle-3 should-fix (3 sibling source-doc files with the same stale framing) was resolved inline at close. Code logic unchanged since cycle 1.

### Surprises and discoveries
- **The stale `prod load (PR-3.4)` framing PR-3.7 fixed in the README also lived in 3 sibling source/script files (postgres.rs, postgres_e2e.rs, cross_check_python_writer.py).** — Resolved inline at cycle-3 close (commit 6275f272d) — repointed all to PR-5.0 with PR-3.4 as the LOCAL rehearsal; applied directly rather than via a 4th amend cycle per the reviewers' 'do not spiral' guidance. (amend_plan: already-done)
- **vector_search_runs had 0 real rows in the PR-3.4 snapshot + was omitted from the cross-check envelope, so its paths are fixture+unit-only.** — Disclosed (PR-3.6) + tracked to PR-5.0; carry-forward. (amend_plan: already-done)

### Testing coverage assessment
**Tested (high confidence):**
- loader COPY-text pipeline (11 unit tests, high)
- value-verify discrimination across 6 tables (14 no-Docker tests, high)
- live PG16 e2e: counts + verify-clean + atomic rollback (2 testcontainer tests, high)
- requires-superuser preflight + marker detection (test_migrate_schema.py, high)
- Python cross-check discrimination, mutation-verified (test_post_ingest_postgres.py, high)
- real-data PR-3.4: 4.2M-row load + verify + cross-check clean (operational, high)
- PR-3.7 doc edits + cycle-3 repoint verified against live source (this review, high)

**Untested / under-covered (non-blocking):**
- vector_search_runs against REAL data — priority medium, fixture+unit only (0 real rows), DISCLOSED + deferred to PR-5.0
- verify-full TLS --ca-cert against live RDS — priority low, compile + local self-signed-cert run + runbook; needs live RDS

### Tradeoffs re-evaluation
- **One-shot historical load: retarget migrate/ (throwaway)** — _keep_: Validated clean against the real snapshot; removal trigger documented.
- **Execution model: operator-local master password, NO aws-sdk-rds** — _keep_: aws-sdk-rds absent; sync postgres suffices; the tokio-postgres swap annotated.
- **Value-column verify as PRIMARY gate** — _keep_: Per-measurement_id full compare, green against real data.
- **epoch-microsecond timestamps** — _keep_: Engine-independent + exact (sub-second + pre-1970 confirmed).
- **local postgres:16 testcontainer rehearsal** — _keep_: Same engine as RDS; PR-3.4 used an equivalent Homebrew PG16.
- **004-as-master requires-superuser preflight** — _keep_: Guard + test shipped; PR-3.6/3.7 completed the operator-facing docs.
- **Prod load TIMING deferred Phase 3 -> Phase 5 (PR-5.0)** — _keep_: PR-3.4 ran the zero-prod-risk rehearsal; the bundle is now internally consistent (all prod-load refs point at PR-5.0).
- **trusted-input low-stakes review calibration; 3-vote Phase 3** — _keep_: Appropriate; the cycle-3 should-fix was applied inline rather than spiraled, per the reviewers' guidance.

### Disagreements
None — all three lenses accepted.

**Dropped re-flags (carry-forward respected):**
- Cycle-2 doc nits (directive namespace, load doc-comment precondition, runbook acquisition note) — addressed by PR-3.7; all 3 confirmed CLOSED by the cycle-3 reviewers (PR-3.7)
- vector_search_runs real-data coverage — disclosed (PR-3.6) + tracked to PR-5.0; carry-forward (PR-3.4 status / PR-5.0)

### Findings (0 must-fix, 1 should-fix [applied], 1 nit [resolved])
| # | Severity | Kind | File:line | Description | Found by |
|---|----------|------|-----------|-------------|----------|
| 1 | should-fix | cross-bundle-consistency | `postgres.rs:16-17, postgres_e2e.rs:12, cross_check_python_writer.py:6,14` | PR-3.7 repointed the stale `prod load (PR-3.4)` references in migrate/README.md to PR-5.0, but the SAME pre-re-scope framing survived in 3 sibling source/script files (postgres.rs module doc, postgres_e2e.rs module doc, the cross-check docstring). RESOLVED inline at cycle-3 cl... | maint/claude,correctness/claude |

The 1 should-fix was applied inline at cycle-3 close to avoid an amend-spiral (both correctness + maint lenses explicitly advised 'do not spiral'). No open findings remain; the only tracked residual is vector_search_runs real-data coverage (PR-5.0).

<details><summary>Full Synthesizer Output JSON (gauntlet schema_version: 1) — cycle 3 raw archive</summary>

```json
{
  "schema_version": 1,
  "preset": "phase-3",
  "lenses_used": [
    "spec",
    "correctness",
    "maint"
  ],
  "cycle": 3,
  "overall": "accept",
  "must_fix_count": 0,
  "should_fix_count": 1,
  "nit_count": 1,
  "unified_findings": [
    {
      "severity": "should-fix",
      "kind": "cross-bundle-consistency",
      "file_line": "postgres.rs:16-17, postgres_e2e.rs:12, cross_check_python_writer.py:6,14",
      "description": "PR-3.7 repointed the stale `prod load (PR-3.4)` references in migrate/README.md to PR-5.0, but the SAME pre-re-scope framing survived in 3 sibling source/script files (postgres.rs module doc, postgres_e2e.rs module doc, the cross-check docstring). RESOLVED inline at cycle-3 close (commit 6275f272d): all three repointed so PR-3.4=LOCAL rehearsal, PR-5.0=prod load consistently across the bundle. Applied directly (mechanical, reviewer-specified) rather than via a 4th amend cycle, per both lenses' explicit 'do not spiral' guidance.",
      "recommended_fix": "APPLIED (6275f272d). fmt + ruff clean.",
      "found_by": [
        "maint/claude",
        "correctness/claude"
      ]
    }
  ],
  "disagreements": [],
  "dropped_re_flags": [
    {
      "topic": "Cycle-2 doc nits (directive namespace, load doc-comment precondition, runbook acquisition note)",
      "reason": "addressed by PR-3.7; all 3 confirmed CLOSED by the cycle-3 reviewers",
      "reference": "PR-3.7"
    },
    {
      "topic": "vector_search_runs real-data coverage",
      "reason": "disclosed (PR-3.6) + tracked to PR-5.0; carry-forward",
      "reference": "PR-3.4 status / PR-5.0"
    }
  ],
  "executive_summary": "Phase 3 CYCLE 3 (after the PR-3.7 doc-nit amend) is accepted by all three lenses with ZERO must-fix. PR-3.7 cleared the 3 cycle-2 nits (directive-namespace alignment, the `load` schema-precondition doc-comment, the on-disk-snapshot runbook bullet) and repointed 3 stale `prod load (PR-3.4)` references in migrate/README.md to PR-5.0; all reviewers verified the edits accurate against the live source and confirmed the cycle-2 nits CLOSED. The code logic is byte-identical to cycle-1's accept (cycles 2-3 added only documentation). The single cycle-3 should-fix — the same stale PR-3.4-as-prod-load framing surviving in 3 sibling source/script files (postgres.rs, postgres_e2e.rs, cross_check_python_writer.py) — was RESOLVED inline at the cycle-3 close (commit 6275f272d, fmt+ruff clean) rather than spawning a 4th amend cycle, exactly as both the correctness and maint lenses advised ('accept or defer, do not spiral'). No open findings remain. The only tracked residual is the vector_search_runs real-data coverage gap, disclosed and routed to PR-5.0. Verdict: ACCEPT, 0 must-fix, 1 should-fix (applied), 1 nit (resolved). Phase 3 is complete and internally consistent; recommend Proceed to Phase 4.",
  "phase_artifacts": {
    "summary": "Phase 3 builds + validates the v3->v4 DuckDB->Postgres migration toolkit (atomic COPY loader; the PRIMARY per-measurement_id value-verify gate; a testcontainer rehearsal harness; a Python-writer cross-check; a requires-superuser bootstrap guard) and closes on PR-3.4's real-snapshot LOCAL rehearsal — GREEN against the real 4.2M-row v3 snapshot into local PG16, zero prod write. The one-shot prod load is deferred to PR-5.0 at the Phase-5 cutover. Doc amends: PR-3.6 (cycle-1 should-fixes: the requires-superuser README subsection + PR-3.1/PR-3.4 status annotations) and PR-3.7 (cycle-2 nits: directive-namespace alignment, load precondition doc, on-disk-snapshot runbook bullet + repointing stale prod-load refs to PR-5.0). The cycle-3 should-fix (3 sibling source-doc files with the same stale framing) was resolved inline at close. Code logic unchanged since cycle 1.",
    "surprises": [
      {
        "what": "The stale `prod load (PR-3.4)` framing PR-3.7 fixed in the README also lived in 3 sibling source/script files (postgres.rs, postgres_e2e.rs, cross_check_python_writer.py).",
        "how_handled": "Resolved inline at cycle-3 close (commit 6275f272d) — repointed all to PR-5.0 with PR-3.4 as the LOCAL rehearsal; applied directly rather than via a 4th amend cycle per the reviewers' 'do not spiral' guidance.",
        "amend_plan": "already-done"
      },
      {
        "what": "vector_search_runs had 0 real rows in the PR-3.4 snapshot + was omitted from the cross-check envelope, so its paths are fixture+unit-only.",
        "how_handled": "Disclosed (PR-3.6) + tracked to PR-5.0; carry-forward.",
        "amend_plan": "already-done"
      }
    ],
    "coverage": {
      "tested_cases": [
        "loader COPY-text pipeline (11 unit tests, high)",
        "value-verify discrimination across 6 tables (14 no-Docker tests, high)",
        "live PG16 e2e: counts + verify-clean + atomic rollback (2 testcontainer tests, high)",
        "requires-superuser preflight + marker detection (test_migrate_schema.py, high)",
        "Python cross-check discrimination, mutation-verified (test_post_ingest_postgres.py, high)",
        "real-data PR-3.4: 4.2M-row load + verify + cross-check clean (operational, high)",
        "PR-3.7 doc edits + cycle-3 repoint verified against live source (this review, high)"
      ],
      "untested_cases": [
        "vector_search_runs against REAL data — priority medium, fixture+unit only (0 real rows), DISCLOSED + deferred to PR-5.0",
        "verify-full TLS --ca-cert against live RDS — priority low, compile + local self-signed-cert run + runbook; needs live RDS"
      ],
      "recommendations": "Coverage is strong; value-fidelity-critical paths runtime-validated against a real PG16 container AND a real 4.2M-row snapshot. The single real-data residual (vector_search_runs) is disclosed + tracked to PR-5.0. No Phase-3 tests owed."
    },
    "tradeoffs": [
      {
        "decision": "One-shot historical load: retarget migrate/ (throwaway)",
        "verdict": "keep",
        "rationale": "Validated clean against the real snapshot; removal trigger documented."
      },
      {
        "decision": "Execution model: operator-local master password, NO aws-sdk-rds",
        "verdict": "keep",
        "rationale": "aws-sdk-rds absent; sync postgres suffices; the tokio-postgres swap annotated."
      },
      {
        "decision": "Value-column verify as PRIMARY gate",
        "verdict": "keep",
        "rationale": "Per-measurement_id full compare, green against real data."
      },
      {
        "decision": "epoch-microsecond timestamps",
        "verdict": "keep",
        "rationale": "Engine-independent + exact (sub-second + pre-1970 confirmed)."
      },
      {
        "decision": "local postgres:16 testcontainer rehearsal",
        "verdict": "keep",
        "rationale": "Same engine as RDS; PR-3.4 used an equivalent Homebrew PG16."
      },
      {
        "decision": "004-as-master requires-superuser preflight",
        "verdict": "keep",
        "rationale": "Guard + test shipped; PR-3.6/3.7 completed the operator-facing docs."
      },
      {
        "decision": "Prod load TIMING deferred Phase 3 -> Phase 5 (PR-5.0)",
        "verdict": "keep",
        "rationale": "PR-3.4 ran the zero-prod-risk rehearsal; the bundle is now internally consistent (all prod-load refs point at PR-5.0)."
      },
      {
        "decision": "trusted-input low-stakes review calibration; 3-vote Phase 3",
        "verdict": "keep",
        "rationale": "Appropriate; the cycle-3 should-fix was applied inline rather than spiraled, per the reviewers' guidance."
      }
    ]
  }
}
```

</details>

## Deferred work

**2026-06-04 LEAN RE-PLAN PRUNE.** Per the course-correction, the backlog below is pruned to **data-correctness items only**. The test-hardening, doc-polish, least-privilege-cleanup, and infra-nit entries are **DROPPED** (not data-correctness; disproportionate to a trusted-input low-stakes dashboard) — they are NOT to be picked up. Concretely dropped: the PR-1.1 provision.sh nits (security-group/subnet/redirect/proxy-auth/teardown/engine-version/OIDC-thumbprint/README), the PR-1.2 concurrency/autocommit/ledger-fingerprint items, the PR-1.4 uv-Python-pin item, the PR-1.6 cycle-6/7 doc + password-guard + PROXY_ROLE_NAME test-hardening items, the Phase-1 phase-end nits, the PR-2.1 cycle-2 negative-test items, and the PR-2.2 cycle-4 destructive-scrub + cycle-7 real-deadlock/autocommit + cycle-15 (already-resolved) test-hardening items. **RETAINED (must actually happen, not data-correctness-but-real-blockers):** (1) the Phase-1 phase-end **operator pre-merge gate** — run the live OIDC `schema-deploy` apply against the public instance + confirm `status` clean before squash-merging `ct/bench-v4` → `develop`; (2) the **bench-v4 Python ruff line-length reconciliation** before the `develop` merge (code-quality merge blocker). Any genuinely data-correctness deferral added by future reviews appends below.

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
| PR-2.1 cycle-2 gauntlet (correctness) | `scripts/test_migrate_schema.py` (non-superuser-master test) | should-fix | The new test always re-creates `migrator` under the modeled master, so the "master lacks ADMIN on a pre-existing `migrator`" failure mode of the 004 self-grant (`GRANT migrator TO CURRENT_USER` -> InsufficientPrivilege) is not exercised. | The shipped code is correct on the documented single-bootstrap-master path (002+004 applied by the one master; precondition now documented in the 004 comment, PR-2.1 fix-commit 44245c4a8). The negative test pins an UNSUPPORTED misconfiguration (a different role re-running 004 against a pre-existing migrator). Fold into a follow-up test-hardening PR: create `migrator` as a separate role, then apply 004 as a master lacking ADMIN, asserting a clear error. |
| PR-2.1 cycle-2 gauntlet (correctness) | `scripts/test_migrate_schema.py` (non-superuser-master test) | nit | The test exercises only `createrole_self_grant=''` (self-grant branch fires) + the superuser no-op; the `createrole_self_grant='inherit'` no-op branch the 004 comment promises (auto-grant INHERIT TRUE -> `pg_has_role` guard skips, no self-grant) is not directly exercised on a non-superuser master. | Branch is conceptually validated (the guard is `IF NOT pg_has_role(...,'USAGE')`; an inherit-configured master has USAGE -> guard skips). Fold into the same follow-up test-hardening PR: a second master with `ALTER ROLE ... SET createrole_self_grant='inherit'`, asserting 004 applies with the membership-row count staying at the single auto-grant. |
| PR-2.2 cycle-4 gauntlet (fresh/codex) | `scripts/test_post_ingest_postgres.py` (schema_conn destructive-scrub guard) | should-fix | The `BENCH_TEST_PG_DSN` override's destructive-scrub guard treats any loopback/socket host as safe, but a localhost SSH tunnel to a real DB would pass the `_is_local_host` check and DROP the public schema. | Dev-only affordance (CI uses Docker testcontainers, never the override); the loopback guard already covers the common case, and a localhost-tunnel-to-prod is an exotic foot-gun. The robust fix (target an isolated test schema/database instead of scrubbing `public`, or require an explicit `BENCH_TEST_PG_ALLOW_DESTRUCTIVE=1` opt-in) is a larger test-infra change. Fold into the follow-up test-hardening PR. |
| PR-2.2 cycle-7 gauntlet (claude + fresh/codex) | `scripts/test_post_ingest_postgres.py` (retry + transaction-mode integration coverage) | should-fix | The `_retry_write_conflicts` real-conflict path is pinned only by mocked-`op` unit tests, and all ingest integration tests run against an `autocommit=True` fixture; no testcontainer test forces a REAL psycopg `DeadlockDetected` inside `with conn.transaction()` (two connections, reversed lock order) on a production-default `autocommit=False` connection and asserts retry + commit. correctness/claude VERIFIED both properties hold against a live PG16 container, so this pins (not fixes) correct behavior. | Container-only test (needs Docker). PR-2.3 (commit `8402c1990`) wired `scripts/` pytest into CI so these tests now RUN there (the prerequisite). The test ADDITIONS — (a) a two-connection reversed-order real-deadlock retry test and (b) an `autocommit=False` ingest fixture covering commit-on-success + rollback-on-validation-failure — are RE-MAPPED to the follow-up test-hardening PR: part (a) needs careful threaded 2-connection lock interleaving (deterministic-deadlock + flakiness risk) that warrants focused attention rather than bundling into the CI-wiring PR (keeps PR-2.3 within its 1-3 commit budget). **Re-mapped from PR-2.3 → follow-up test-hardening PR (CI-prerequisite delivered by PR-2.3).** |
| PR-2.2 cycle-15 gauntlet (fresh/codex) | `scripts/test_post_ingest_postgres.py:1178` (`test_server_mode_requires_benchmark_id` isolation) | should-fix | The test does not ISOLATE the missing-`--benchmark-id` rejection: `_main_server` checks `benchmark_id is None` (returns 2) BEFORE the token check, and the test does not set `INGEST_BEARER_TOKEN`, so if the benchmark-id check were deleted the test would still see `return 2` from the token path and pass for the wrong reason. **Test-discriminating-power gap, NOT a functional defect** — `_main_server` correctly rejects missing `--benchmark-id` today, and the cycle-15 accept verdict was not blocked by this. | Non-blocking should-fix on a pre-existing `--server` test, orthogonal to PR-2.2's git_show_field/Postgres core; surfaced at the PR-2.2 convergence (cycle-15) accept. Fold into PR-2.3 (the test-hardening PR): set a dummy `INGEST_BEARER_TOKEN` via `monkeypatch.setenv` (or call `_main_server` directly) so `benchmark_id=None` is the only failing condition, and assert stderr mentions `--benchmark-id`. **→ RESOLVED by PR-2.3 (commit `ed585f451`): token set so `benchmark_id=None` is the only failing condition + stderr asserted to name `--benchmark-id`; mutation-verified (deleting the benchmark-id check now fails the test).** |
| **PR-5.0 prod load (2026-06-05, was PR-3.4/PR-3.5 prod gates)** | (prod RDS `245040174862`/`us-east-1`) | **MOVED to Phase-5 cutover** (no longer Phase-3-blocking) | The one-shot PROD load + prod `verify` + prod cross-check were RE-SEQUENCED to the Phase-5 cutover (new PR-5.0) per the user's 2026-06-05 decision: seeding prod before Phase 4 builds the v4 reader is premature; load the freshest snapshot at cutover instead. NOT strictly operator-only — the agent HAS prod access (the `bench-prod` profile reaches `245040174862`; prod RDS is PubliclyAccessible) — but a prod data seed is a hard-to-reverse side-effect requiring explicit sign-off before the write. | See the PR-5.0 row in `Phases and PRs` + the `Prod historical load TIMING` Key decision. Phase 3 now closes on PR-3.4's REAL-snapshot LOCAL rehearsal (zero prod risk). RDS PITR (35-day) is the prod rollback. |
| PR-4.2 cycle-1 gauntlet (fresh + correctness) | `benchmarks-website/web/lib/db.test.ts` (CI enforcement) | should-fix | The new vitest suite (testcontainers roundtrip + `buildQuery`/`resolveSsl`/`requireEnv`/IAM units) is run by NO CI workflow: `web-deploy.yml` does not exist until PR-4.5, and the existing CI does not run `benchmarks-website/web` tests, so a `db.ts` regression could merge green. The testcontainers describe also self-skips without Docker. Same enforcement-gap class as the (resolved) PR-1.2/PR-1.5 "no CI runner" items, for the NEW web/ TS suite. | Wire a `benchmarks-website/web` CI job running `pnpm test` (Docker available so the testcontainers describe executes) as part of PR-4.5's CI workflow. **Resolved-by: PR-4.5.** |
| PR-4.3.c cycle-1 gauntlet (fresh+correctness/codex) | `web/lib/queries.ts` `groupNameQuery` + `web/lib/descriptions.ts` (statpopgen/polarsignals) | should-fix (v2→v4 display regression) | v2 (`src/config.js`) surfaced the group display names `Statistical and Population Genetics` / `PolarSignals Profiling` with descriptions, but the v4 read port (faithful to v3) special-cases only tpch/tpcds/clickbench in `groupNameQuery`, so statpopgen/polarsignals fall through to the legacy `dataset sf=N [storage]` name and their (already-ported) `descriptions.ts` cases are dead — those two group pages render without the v2 blurb. | FAITHFUL reproduction of v3/Axum: `server/src/api/groups.rs::group_name_query` has the identical tpch/tpcds/clickbench-only fall-through and `server/src/api/descriptions.rs` carries the identical dead cases. PR-4.3.c's approved acceptance is byte/semantic-equivalence to v3 (Phases-and-PRs row + the `Read-endpoint behavior-preservation is SEMANTIC equivalence` tradeoff), so restoring v2's names DIVERGES from v3 = a deliberate scope addition, out of scope for the faithful port. Resolve as a deliberate v4 enhancement (or upstream v3-source fix): add display-name cases for statpopgen/polarsignals in `groupNameQuery` so the descriptions attach, pinned by a `collectGroups` fixture for both suites. Needs a user call on whether v4 should restore v2 fidelity here vs. preserve the v3 behavior. |
| PR-4.3.c cycle-2 gauntlet (correctness/codex) | `web/lib/summary.ts` (all 4 summary paths) + `server/src/api/summary.rs` | should-fix (v3-source latent bug; preserved) | "Latest commit" selection is timestamp-only (`c.timestamp = MAX(ts)` for random-access + compression time/size; `row_number() ORDER BY timestamp DESC` for query summary), with no `commit_sha` tiebreaker. Two commits sharing a second-granularity git timestamp can tie at `MAX(ts)`, blending rows from multiple commits or picking nondeterministically instead of summarizing exactly one latest commit. | Preserved-v3-behavior (VERIFIED: the Rust v3 source has the identical timestamp-only selection in all three paths — summary.rs:56-103, :252-330, + the compression helpers — and the MF1 CTE fix preserved it exactly). PR-4.3.c acceptance = v3 semantic-equivalence, so adding `ORDER BY timestamp DESC, commit_sha DESC` + filter-by-commit_sha would DIVERGE from v3 across all 4 paths = a deliberate cross-substrate determinism improvement (and ideally an upstream v3-source fix too). Vanishingly unlikely in practice (trusted CI, develop commits minutes apart). Resolve as a deliberate "v4 correctness improvements over v3" effort with a same-second-two-commits regression test. Needs a user call (paired with the statpopgen/polarsignals item — both are real-latent-v3-bugs surfaced by the faithful-port reviews). |
| PR-4.3.c cycle-2 gauntlet (fresh/claude) | `web/lib/queries.ts` `groupNameQuery` + `web/lib/groups.test.ts` | should-fix (coverage) | `groupNameQuery`'s clickbench (`clickbench` → `Clickbench`), variant-append (` / variant`), and legacy-fallback branches have no direct test coverage; only the tpch + nvme + sf=1 + null-variant branch is exercised by the testcontainer fixture. | Faithful 1:1 port of `server/src/api/groups.rs::group_name_query`, whose own Rust tests share the same single-fixture gap, so parity with the source is preserved. Low priority under the trusted-input + faithful-port calibration. Resolve by adding a clickbench-group fixture and a variant-bearing tpch-group fixture to `groups.test.ts` (fold into the groupNameQuery v2-fidelity enhancement above, or a follow-up test-hardening pass). |
| PR-4.4.a cycle-1 gauntlet (fresh/codex) | `web/app/globals.css` mobile `@media (max-width:768px)` rule + `web/components/Header.tsx` | should-fix (bug; UI) | The ported mobile CSS hides `.repo-link-desktop` under 768px because v3's mobile nav (`.nav-controls-github` inside the hamburger panel) supplies the GitHub link there; PR-4.4.a renders only `.repo-link-desktop` and defers the mobile nav, so the GitHub link disappears on mobile viewports in the PR-4.4.a-only intermediate state. | The mobile nav (hamburger + `.nav-controls` + `.nav-controls-github`) is PR-4.4.b's scope; restoring the mobile GitHub affordance lands naturally there. The intermediate 4.4.a-only state is never deployed alone (Phase 4 ships as one cutover), so this is a transient gap, not a shipped regression. **Resolved-by: PR-4.4.b (render the static `.nav-controls-github` mobile fallback alongside the mobile-nav island).** |
| PR-4.4.a cycle-1 gauntlet (correctness/claude) | `web/lib/format.ts` `formatTimeNs` + `web/components/SummaryCard.tsx` `.toFixed(2)` ratio/score renders | should-fix (port-fidelity; rare) | JS `Number.prototype.toFixed` rounds half-away-from-zero; the v3 Rust originals (`format_time_ns`, `format!("{:.2}")`) round half-to-even. For an exactly-representable dyadic tie the rendered last digit diverges: a size ratio of exactly `0.125` renders `0.13x` here vs `0.12x` in v3 (and `12.5 ns` → `13 ns` vs `12 ns`). The underlying number is identical — only display rounding at exact ties differs. | Reachable only for exact dyadic-rational ratios (the `ns` tier always receives whole-integer `value_ns`, so no tie there; `geoMean`/quotient ratios essentially never land on an exact 3rd-decimal dyadic boundary). Single-display-digit divergence on a trusted-input low-stakes dashboard; deferred as a low-priority "v4 display-rounding parity (round-half-even)" item — pairs with the PR-4.3.c "v4 correctness improvements over v3 vs preserve-v3" decision flagged for the Phase-4 boundary. Resolve by a round-half-even helper applied before formatting + an exact-tie regression test (`0.125`→`0.12x`, `2.5`→`2 ns`), or record as an accepted display-parity tradeoff. |
| Phase-4 holistic review (correctness/codex) | `web/app/page.tsx:15` (landing-page caching) | should-fix | The landing RSC page is `force-dynamic` with no compensating cache layer, so every `/` render runs `collectGroups()` against Postgres. The `/api/*` routes are CDN-cached for 5 minutes (b53e07727), but the landing HTML (the highest-traffic path) is not. | Deploy-time decision that needs PR-4.5's Vercel config: either a `Cache-Control` header on `/` via `vercel.json` routes, or time-based revalidation once the database is reachable at build time. The page.tsx comment documents both options. **Resolved-by: PR-4.5.** |

## Accepted tradeoffs / r1 traps

- **Schema deploys have no manual-approval gate; PR merge is the authorization gate** (user decision 2026-05-29). A reviewed, merged PR is accepted as sufficient authorization to apply a migration to prod. We knowingly forgo (a) a GitHub Environment required-reviewer approval and (b) segregation of duties (a distinct approver from the author) and (c) merge-now/apply-later timing decoupling. Rationale: a manual approval only re-confirms the authorization already given at merge and does not verify execution safety; the real safety comes from migration testing. Reviewers must NOT re-flag the absence of an `environment:` gate or a manual approval step on `schema-deploy.yml`; it is an accepted, deliberate decision, not an oversight. (See the matching Key decision.)
- **Execution safety for additive migrations is the per-PR testcontainer test, not a prod dry-run gate** (user decision 2026-05-29). Additive DDL (CREATE TABLE/INDEX, ADD COLUMN) is validated by `scripts/test_migrate_schema.py` against `postgres:16-alpine` at PR time; a migration that cannot apply cannot merge. The heavier PITR-snapshot-restore-against-real-data test is reserved for data-affecting migrations only (tracked in Deferred work). Reviewers must NOT flag the lack of a prod-data migration test on additive-only migrations as a gap.
- **Duplicate JSON object keys in an ingest record collapse last-wins, matching v3 — `post-ingest.py` must NOT reject them** (PR-2.2 cycle-6 gauntlet triage; correctness/codex flagged this as a divergence, verified false). The v3 server parses the envelope via `serde_json::from_value` over an `axum::Json<serde_json::Value>` (`benchmarks-website/server/src/ingest.rs:79-83`); `serde_json::Value::Object` is last-wins on duplicate keys, so v3 silently keeps the last value. Python's `json.loads` (no `object_pairs_hook`) is ALSO last-wins, so the two substrates already agree. Adding an `object_pairs_hook` that rejects duplicate keys would make the Python writer STRICTER than v3 and CREATE a behavior divergence, violating the migration's Behavior-preservation invariant. Reviewers must NOT re-flag the absence of duplicate-key rejection in `read_records`; matching v3 means keeping last-wins. (The Rust bench producer never emits duplicate keys in steady state regardless.)
- **`scripts/post-ingest.py`'s PEP 723 block intentionally declares `dependencies = []`; `--postgres` runs from the project uv env, not standalone** (PR-2.2; flagged repeatedly as a should-fix, e.g. cycle-10 fresh/codex). The v3 `--server` path must stay standard-library-only so CI can invoke `python3 scripts/post-ingest.py` under a bare interpreter (the v3 path is in production until the Phase-5 cutover). Declaring `psycopg`/`boto3`/`xxhash` in the PEP 723 block would make `uv run` install them for EVERY invocation incl. `--server`, defeating that. The `--postgres` mode's deps are provided by the repo's uv workspace (the module docstring + the `pyproject.toml` dev-deps comment document this). Reviewers must NOT re-flag `dependencies = []` / "standalone `uv run --postgres` fails"; it is a deliberate decision, not an oversight.
- **IAM-token region precedence is explicit `--region` > boto3 session region > RDS-hostname-parsed region** (PR-2.2 deploy decision; documented in the `--region` help text since cycle 5; re-flagged cycle-10 fresh/codex as should-fix). `_rds_iam_token` resolves `region or session.region_name or _region_from_host(host)`. A region mismatch produces a wrong-region token that fails LOUD at connect (IAM auth rejects it), not a silent divergence. The boto3-session-before-host order matches the AWS-conventional default (the ambient session region is the operator's declared region); the host-parsed region is the last-resort fallback. Reviewers must NOT re-flag the session-over-host ordering; it is the accepted, documented precedence.
- **A per-app `benchmarks-website/web/.prettierrc.json` byte-identical to `vortex-web/.prettierrc.json` is COMPLIANT with the Next.js/TypeScript formatter-config BAN** (PR-4.1 cycle-1 gauntlet; Codex/fresh flagged it `must-fix`, adjudicated over-strict by the synthesizer + the two Claude lenses). The BAN ("Do not introduce a second formatter config -- reuse the existing `vortex-web/.prettierrc.json` shape (`singleQuote: true`, `trailingComma: "all"`, `printWidth: 100`, `tabWidth: 2`)") targets formatter-SHAPE divergence, NOT config-file count: it enumerates the exact shape values to match, which `web/.prettierrc.json` does byte-for-byte. `benchmarks-website/web/` is a separately-deployable pnpm/Vercel package; Prettier resolves config upward from the file's own directory tree, and `vortex-web/.prettierrc.json` lives in a SIBLING tree (not an ancestor), so it is undiscoverable from `web/` without either a fragile cross-package `--config ../../vortex-web/...` path or a repo-root shared config (a cross-cutting change out of Phase-4 scope). A self-contained byte-identical copy is the correct approach. Reviewers must NOT re-flag `web/.prettierrc.json` (or sibling per-app prettier configs landed in later Phase-4 PRs) as a must-fix; matching the enumerated shape IS the reuse the BAN requires. (Residual two-copy drift risk is real but non-blocking; a future repo-root hoist is the cleanup if drift ever surfaces.)
- **v4 read-service slugs are OPAQUE round-trippable tokens, NOT byte-identical to the Rust server's slugs** (PR-4.3.a cycle-1 gauntlet; both lenses raised cross-language slug byte-identity, mooted by the shard-endpoint deferral). The Next.js server both PRODUCES (`/api/groups`, `/api/group`) and CONSUMES (`/api/chart`, `/api/group`) every slug; the web-ui only ever round-trips them back unchanged, and the `/api/artifacts/.../shards/{i}` endpoint (the only consumer that keyed an in-memory artifact map by slug in the Rust `read_model.rs`) was deferred to PR-4.4 under the user's fork decision. So there is NO cross-language slug exchange and byte-identity with the Rust producer is NOT required — only internal round-trip consistency (`decode∘encode = id`) + a stable canonical encoding (pinned by the `slug.test.ts` golden vectors). Consequently two divergences from `slug.rs` are ACCEPTED and reviewers must NOT re-flag them: (a) Node `Buffer.from(x,'base64url')` decodes leniently (strips invalid chars / tolerates padding) where Rust `URL_SAFE_NO_PAD` rejects — harmless because any non-canonical input that survives still hits the per-variant field validation; (b) `VectorSearch.threshold` whole-number values serialize as `1` (JS `JSON.stringify`) vs `1.0` (Rust `serde_json`) — irrelevant absent a cross-language slug comparison. If PR-4.4 reintroduces a cross-language slug producer/consumer, revisit this entry.
- **Read-endpoint behavior-preservation is SEMANTIC equivalence, not byte-identity to the Axum JSON** (PR-4.3.b design decision, recorded before implementation). True byte-identity between the Rust read server and the Next.js port is IMPOSSIBLE for numeric values: `serde_json` renders a whole-number `f64` as `1500000.0` (always a decimal point) while JS `JSON.stringify` renders it as `1500000`. Chart series values (`value_ns`/`value_bytes`, materialized `as f64` in `charts.rs`) are exactly this case. The client (`chart-init.js`) parses the JSON, so `1500000.0` and `1500000` are the identical number to every consumer — the wire contract that matters is field names + structure + nesting + ordering + numeric VALUES, which the port preserves exactly. Accordingly the PR-4.3.b/4.3.c "snapshot test against fixtures" acceptance is implemented by SEEDING a known fixture into a local `postgres:16` testcontainer and asserting the assembled `ChartResponse`/`GroupsResponse` content + values (the same seed-then-assert-values shape the Axum `chart_api.rs`/`group_api.rs` tests use, NOT an `insta` byte-capture of Axum output). Reviewers must NOT flag "the JSON is not byte-identical to the Axum server" (e.g. `1500000` vs `1500000.0`, or key-insertion-order within an object); semantic equivalence + the snake_case field contract (`dto.rs`) is the preserved invariant. (Postgres `BIGINT` value columns are read via `::float8` so node-postgres returns a JS number matching the Rust `as f64`, avoiding the bigint-as-string default.)
- **`reqI32` in `web/lib/slug.ts` accepts whole-number JSON floats (`7.0`) that serde rejects for an `i32` field** (PR-4.3.b cycle-2 gauntlet; correctness/claude flagged as a nit, dismissed). The cycle-1 must-fix added `reqI32` (`Number.isInteger` + `[-2**31, 2**31-1]` range) so a forged non-i32 `query_idx` returns 400 instead of an unhandled 500. `reqI32` is strictly MORE lenient than serde in exactly one direction: a JSON number written with a decimal point (`7.0`) is rejected by serde's `i32` deserializer but `JSON.parse('7.0') === 7` passes `Number.isInteger`. This is only reachable via a FORGED slug (the canonical `chartKeyToSlug` always emits an integer literal, so the round-trip path never produces `7.0`), and the lenient outcome is a normal 200/404 with the correct integer bound cleanly to the Postgres `integer` column — NOT a 500 and not wrong data. Per the trusted-input + opaque-round-trippable-slug calibration this whole-number-float leniency is immaterial; reviewers must NOT re-flag it as a 400-vs-200 divergence. There is NO false-REJECT direction (every integer serde accepts is accepted), and `VectorSearch.threshold` (f64) is correctly left as `reqNumber`, unconstrained.
- **statpopgen/polarsignals query groups fall through to legacy naming so their v2 descriptions never attach — this faithfully reproduces v3 and reviewers must NOT re-flag it** (PR-4.3.c cycle-1 gauntlet triage; fresh/codex flagged must-fix, correctness/codex flagged should-fix, conservative-union elevated to must-fix; the synthesizer explicitly deferred the defect-vs-preserved-behavior call to Step 2.4 triage). `groupNameQuery` (`web/lib/queries.ts`) special-cases only tpch/tpcds/clickbench; statpopgen/polarsignals fall through to the legacy `dataset sf=N [storage]` name, so `groupDescription` never matches the `'Statistical and Population Genetics'` / `'PolarSignals Profiling'` cases in `web/lib/descriptions.ts`, which are therefore dead. This is a BYTE-FAITHFUL port of the Rust v3 source: `server/src/api/groups.rs::group_name_query` has the identical tpch/tpcds/clickbench-only fall-through, and `server/src/api/descriptions.rs` carries the identical dead `'Statistical and Population Genetics'` / `'PolarSignals Profiling'` cases. PR-4.3.c's APPROVED acceptance criterion is byte/semantic-equivalence to v3/Axum `group_api.rs` (the Phases-and-PRs row + the `Read-endpoint behavior-preservation is SEMANTIC equivalence` tradeoff above), so reproducing v3 exactly — including this inconsistency — is correct-as-approved; "fixing" it would DIVERGE from v3 and is a deliberate scope addition. v2's `src/config.js` DID surface those display names + descriptions, so the v2-fidelity restore is tracked as Deferred work for a separate deliberate v4-enhancement decision. Reviewers must NOT re-flag the statpopgen/polarsignals legacy-naming or the dead descriptions as a must-fix; matching v3 is the preserved invariant.
- **Summary "latest commit" selection is timestamp-only across all four summary paths, so same-second-timestamp commit ties can blend or pick nondeterministically — this faithfully reproduces v3 and reviewers must NOT re-flag it** (PR-4.3.c cycle-2 gauntlet; correctness/codex flagged must-fix, the other 3 lenses accepted; the synthesizer's own call said confirm v3-parity then reframe under this SEMANTIC-equivalence tradeoff). Git commit timestamps are second-granularity, so two commits could tie at `MAX(timestamp)`; the summaries select the latest commit by timestamp equality (`c.timestamp = MAX(ts)` for random-access + compression time/size) or a timestamp-only `row_number() ORDER BY timestamp DESC` (query summary), with NO `commit_sha` tiebreaker, so a tie can aggregate rows from multiple commits or pick one arbitrarily. This is a BYTE-FAITHFUL port of the Rust v3 source, VERIFIED across all three paths: `server/src/api/summary.rs:56-103` (random-access `c.timestamp = (SELECT MAX(c2.timestamp) ...)`), `:252-330` (query `row_number() OVER (... ORDER BY c.timestamp DESC)`, no `commit_sha`), and the compression-time/size helpers (`c.timestamp = MAX(ts)`). The PR-4.3.c MF1 CTE fix preserved `timestamp = MAX(timestamp)` EXACTLY and did NOT introduce or change the tie behavior. PR-4.3.c's approved acceptance is v3 semantic-equivalence, so adding a `commit_sha DESC` tiebreaker (the recommended fix) would DIVERGE from v3 across all four summary paths = a deliberate cross-substrate scope addition, tracked as Deferred work. The tie is also vanishingly unlikely in practice (trusted CI, develop commits minutes apart, both being the joint-latest with benchmark data). Reviewers must NOT re-flag the same-timestamp tie as a must-fix; matching v3 is the preserved invariant. (This is the SECOND cycle to surface a real-latent-v3-bug-but-preserve finding on PR-4.3.c; both are tracked in Deferred work for a possible deliberate "v4 correctness improvements over v3" effort — surface at the PR-completion / Phase-4 boundary.)
