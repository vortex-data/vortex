# bench-v4 CI emitter dual-write — design spec

**Status:** approved design (Phase 1 of big-plans)
**Work shape:** feature-integration (port + merge + provision)
**Spine:** `.big-plans/ct__bench-v4-emitters.md`
**Authoritative external plan:** `benchmarks-website/docs/runbooks/emitter-ingest-cutover.md` (verified 2026-06-19)
**Wire contract:** `benchmarks-website/CONTRACT.md` (SCHEMA_VERSION = 1)

## 1. Goal & premise

Make the Vortex monorepo CI emitters write benchmark results LIVE to the v4 RDS Postgres
("Path B / v4 dual-write") so the manual `vortex-bench-migrate` refresh is no longer needed.
The write is additive, best-effort (env-gated + `continue-on-error`), and cannot break the live
v2 (static S3 JSON) or v3 (DuckDB + `POST /api/ingest`) paths.

**Key premise — most code already exists, unmerged.** The v4 emitter was implemented on
`origin/ct/bench-v4` (commits `9a870091e` Phase 1 + `9a1824afa` Phase 2; tree at tip
`f9b36ae3f`) and, crucially, was authored against the CURRENT top-level `scripts/` layout. So
this is FINISH + MERGE + PROVISION, not write-from-scratch. `scripts/post-ingest.py` has an
identical blob at the merge-base (`8acef3aab`) and current `develop` tip — mainline has not
touched it in the 89 intervening commits — so the +934/-22 Postgres-writer diff applies cleanly.
The five follow-up commits after `9a1824afa` are website-repo Phases 3-5 and do NOT touch the
emitter, so no follow-up folding is needed for the emitter artifacts.

## 2. Architecture: dual-write, two paths from one emitter

Both paths originate from the same `vortex-bench --gh-json-v3` JSONL output (bare records, no
`measurement_id` on the wire). `scripts/post-ingest.py` dispatches on its mode:

- **Path A (v3, unchanged, hard-required):** `post-ingest.py --server $V3_INGEST_URL` wraps the
  records in a `{run_meta, commit, records}` envelope and POSTs to `{server}/api/ingest` with
  `Authorization: Bearer $INGEST_BEARER_TOKEN`. Stdlib-only (`urllib`/`json`/`subprocess`).
  **Preserved verbatim.**
- **Path B (v4, new, best-effort):** `post-ingest.py --postgres $DSN --region $REGION` connects
  to RDS as `bench_ingest` (RDS IAM auth, `sslmode=verify-full`), computes `measurement_id`
  client-side, and runs `INSERT ... ON CONFLICT (measurement_id) DO UPDATE` upserts in one
  transaction, then pings `POST {BENCH_SITE_BASE_URL}/api/revalidate` with
  `Authorization: Bearer $BENCH_REVALIDATE_TOKEN`. The record shapes are identical to Path A.

The v4 write is dormant until phase B sets `GH_BENCH_INGEST_ROLE_ARN`; every v4 workflow step is
gated on `vars.GH_BENCH_INGEST_ROLE_ARN != ''` with `continue-on-error: true`.

### 2.1 The `measurement_id` contract (load-bearing)

`scripts/_measurement_id.py` is a byte-for-byte Python port of the Rust reference in
`benchmarks-website/server/src/db.rs`, using the `xxhash` package's `XXH64` (seed 0) — NOT a
hand-rolled hash. The length-prefix / opt / i32 / f64 framing below is what the port must
reproduce around that hash:

- `hasher_for(tag)`: seed 0, write `tag` bytes, then `write_u8(0)` (per-table tag separator).
- `write_str(s)`: write `len(s) as u64`, then the UTF-8 bytes (length-prefixed).
- `write_opt_str(o)`: `Some` -> `write_u8(1)` + `write_str`; `None` -> `write_u8(0)`.
- `write_i32(v)`: `hasher.write_i32(v)`.
- `write_f64(v)`: `hasher.write_u64(v.to_bits())`.
- `finish`: `hasher.finish() as i64` (signed bit-cast — BIGINT is signed).

Per-table field order (the byte layout the port must match exactly):

| Table | Tag | Fields in order |
|---|---|---|
| `query_measurements` | `query_measurements` | commit_sha, dataset, dataset_variant(opt), scale_factor(opt), query_idx(i32), storage, engine, format |
| `compression_times` | `compression_times` | commit_sha, dataset, dataset_variant(opt), format, op |
| `compression_sizes` | `compression_sizes` | commit_sha, dataset, dataset_variant(opt), format |
| `random_access_times` | `random_access_times` | commit_sha, dataset, format |
| `vector_search_runs` | `vector_search_runs` | commit_sha, dataset, layout, flavor, threshold(f64) — `iterations` excluded |

Validated by `scripts/test_measurement_id.py` against the ported `scripts/measurement_id_golden.json`
(825-line cross-language golden vectors covering empty strings, Unicode, `query_idx` i32 bounds,
and float edge cases). The golden vectors ARE the contract — the test is wired as a required CI
check. The two ported files' docstrings reference `benchmarks-website/server/src/db.rs` and
`.../tests/measurement_id_golden.rs`, which no longer live in the monorepo — repoint those doc
references at the extracted `vortex-data/benchmarks-website` repo.

### 2.2 The v4 upsert (column lists, from `server/src/ingest.rs` + `migrations/001_initial_schema.sql`)

Commit dim first (`ON CONFLICT (commit_sha) DO UPDATE`), then 5 fact upserts keyed on
`measurement_id`:

- `query_measurements`: measurement_id, commit_sha, dataset, dataset_variant, scale_factor, query_idx, storage, engine, format, value_ns, all_runtimes_ns, peak_physical, peak_virtual, physical_delta, virtual_delta, env_triple
- `compression_times`: measurement_id, commit_sha, dataset, dataset_variant, format, op, value_ns, all_runtimes_ns, env_triple
- `compression_sizes`: measurement_id, commit_sha, dataset, dataset_variant, format, value_bytes
- `random_access_times`: measurement_id, commit_sha, dataset, format, value_ns, all_runtimes_ns, env_triple
- `vector_search_runs`: measurement_id, commit_sha, dataset, layout, flavor, threshold, value_ns, all_runtimes_ns, matches, rows_scanned, bytes_scanned, iterations, env_triple

Non-finite f64 (NaN/Inf) must be rejected loud, never written.

### 2.3 Third-party dependency handling (decided: port branch approach as-is)

The v4 `--postgres` path needs `psycopg[binary]`, `boto3` (RDS IAM token minting), and `xxhash`
(the `measurement_id` hash). **Mechanism (corrected from grill-me — the code disagrees with the
runbook):** the deps are supplied at the call site via
`uv run --no-project --with 'psycopg[binary]' --with boto3 --with xxhash scripts/post-ingest.py`,
NOT declared in `post-ingest.py`'s PEP-723 block (which stays `dependencies = []`). They are
**lazily imported only inside the `--postgres` branch**, so the v3 `--server` path and a bare
`import post_ingest` stay stdlib-only and dep-free even though the deps are no longer pinned in
the file. (The test invocations and the ci.yml jobs must likewise provide `xxhash` / the
testcontainer deps via `--with` or the workspace.)

## 3. Code-port scope — phase D (decided: everything incl. extras)

**IN (lands in `vortex-data/vortex`, ported from `f9b36ae3f`):**

- `scripts/post-ingest.py` — add `--postgres`/`--region` mode (IAM-auth upsert, one txn,
  NaN/Inf guard, `verify-full` TLS + post-connect `ssl_in_use` check, `bench_ingest` role
  enforcement, best-effort `refresh_site_cache`); keep the v3 `--server` path intact.
- `scripts/_measurement_id.py` (new) — the xxhash64 port (repoint docstring).
- `scripts/measurement_id_golden.json` (new) — golden vectors (required, the test fails without it).
- `scripts/test_measurement_id.py` (new) — golden-vector pytest (repoint docstring).
- `scripts/test_post_ingest_postgres.py` (new) — testcontainers Postgres writer/upsert tests.
- `scripts/cross_check_python_writer.py` (new) — extra cross-check utility (later-phase extra).
- `scripts/test_post_ingest_revalidate.py` (new) — revalidate-ping tests (later-phase extra).
- `.github/workflows/bench.yml` — insert best-effort v4 step block after the v3 ingest step.
- `.github/workflows/sql-benchmarks.yml` — same, additionally gated `inputs.mode == 'develop'`.
- `.github/workflows/v3-commit-metadata.yml` — same (empty.jsonl, commit-row upsert); ADD
  `id-token: write` permission (currently only `contents: read`).
- `.github/workflows/ci.yml` — wire the measurement_id pytest into the existing `python-test`
  job (no docker), and add a docker-gated `scripts-test` job for the testcontainer tests.
- `pyproject.toml` / `uv.lock` — regenerate `uv.lock` with `uv lock` (do NOT copy the branch
  blob; develop's lock diverged).

**OUT (the extracted `vortex-data/benchmarks-website` repo owns these; do not land in monorepo):**

- `migrations/*.sql`, `scripts/migrate-schema.py`, `scripts/test_migrate_schema.py`,
  `.github/workflows/schema-deploy.yml` — schema/role management is the website repo's job
  (`GitHubBenchmarkSchemaRole`); the monorepo emitter only CONNECTS as `bench_ingest`.
- All of `benchmarks-website/{server,migrate,web,infra}/**` — extracted repo.

### 3.1 Workflow v4 step shape (per runbook §2.D)

Each v4 step block: `aws-actions/configure-aws-credentials` assuming
`vars.GH_BENCH_INGEST_ROLE_ARN` (SHA-pinned action), download the RDS CA bundle, run
`post-ingest.py --postgres "$DSN" --region "$RDS_BENCH_REGION"`, then the revalidate ping to
`${BENCH_SITE_BASE_URL}/api/revalidate`. DSN:
`postgresql://bench_ingest@${RDS_BENCH_INSTANCE_ENDPOINT}:5432/${RDS_BENCH_DB_NAME}?sslmode=verify-full&sslrootcert=<ca-path>`.
Every v4 step: `if: vars.GH_BENCH_INGEST_ROLE_ARN != ''` + `continue-on-error: true`. `bench.yml`
and `sql-benchmarks.yml` already have `id-token: write` + a `configure-aws-credentials` step, so
the v4 step reuses that machinery (with the NEW ingest role ARN, distinct from the S3-scoped
`GitHubBenchmarkRole`). All edits must pass `yamllint --strict -c .yamllint.yaml`.

## 4. big-plans phase structure (decided: 4 phases, reordered D/A/C/B+soak)

### 4.0 Demo-safety model (grill-me outcome — HARD CONSTRAINT)

A live demo reads prod RDS in the hours after planning. Therefore:

- **No prod RDS writes until phase B** (the live cutover), which is hard-gated to AFTER the demo
  AND explicit user go-ahead. **Phase C also waits for after the demo** (it redeploys the v4
  site the demo may read). All external mutations (A, C, B) land post-demo; only the pure-code
  phase D runs during the demo window (zero prod interaction).
- **Pre-merge confidence is built WITHOUT prod or develop** (the user's "be confident before we
  merge" requirement): the **testcontainer Postgres tests** (`test_post_ingest_postgres.py`),
  run against a local throwaway Postgres pinned to **RDS's major version (16)**, exercise the
  full writer path — real upsert SQL, `measurement_id`, the single transaction, ON CONFLICT,
  NaN/Inf guard — and the **golden-vector test** proves `measurement_id` byte-parity with Rust.
  These are the load-bearing pre-merge gate (this is why "everything incl. extras" scope matters).
- The only thing local tests cannot prove — real-RDS IAM auth + `verify-full` TLS + role grant —
  is verified post-demo at phase B via a **read-only / rolled-back** `bench_ingest` connect
  (no data mutation), plus the first emitting run's `continue-on-error` logs.
- Reordered to **D -> A -> C -> B** (was A/D/C/B) so the next hours are spent only on code that
  cannot affect the demo; every prod-touching op is deferred behind a post-demo gate. The
  runbook's safety invariant still holds: do NOT set `GH_BENCH_INGEST_ROLE_ARN` until A exists
  AND D is merged — both precede phase B.

One code phase (D) bracketed by external-infra ops phases. **Ops phases (A, C, B) produce NO
in-repo diff** — they are direct CLI operations with side effects. Therefore ops phases:

- do NOT run through `subagent-driven-development`;
- are NOT gauntlet-reviewed and do NOT open a PR (there is no diff to review or merge);
- have **machine-checkable CLI exit criteria** (role exists / var set / token returns 200);
- use a **pre-action external-side-effect confirmation** as the human checkpoint before each
  mutating CLI op, and a post-action verification confirmation at the phase boundary.

This is the sanctioned big-plans ops-phase adaptation (it is not "skipping rigor" — the rigor
is the CLI verification + the confirmation gate; gauntlet has no artifact to review). The code
phase (D) runs the full standard machinery (SDD per sub-phase + gauntlet pr-2/3 + phase-end
gauntlet + PR + human gate + squash-merge).

**Phase order (reordered D -> A -> C -> B per §4.0):**

1. **Phase D (code): port the v4 dual-write emitter + workflows.** The one reviewable PR; merges
   with `GH_BENCH_INGEST_ROLE_ARN` unset, so the v4 code ships dormant-but-safe. Pre-merge
   confidence per §4.0 (testcontainer PG16 + golden). Zero prod interaction; demo-safe.
2. **Phase A (ops): provision the ingest IAM role.** Create `GitHubBenchmarkIngestRole` via
   `benchmarks-website/infra/provision.sh` `ensure_ingest_role()` (idempotent) or a surgical
   create. Trust `repo:vortex-data/vortex` (refs per the script — develop is what the live
   workflows run on); grant `rds-db:connect` as `bench_ingest` on the instance `DbiResourceId`.
   Record the ARN. Data-safe (IAM only) but an external mutation -> post-demo, confirm first.
3. **Phase C (ops, GATED): align the revalidate token.** Generate one fresh token, set on the
   v4 Vercel project (Production) AND the monorepo secret, redeploy v4 prod. Requires explicit
   user go-ahead (was deferred in a prior session by an auto-mode guard). Post-demo (redeploy).
4. **Phase B (ops, GATED — live cutover): flip the switch + repoint URLs + soak.** First, the
   read-only/rolled-back real-RDS `bench_ingest` verification (§4.0). Then set
   `GH_BENCH_INGEST_ROLE_ARN` (the gate) and repoint `BENCH_SITE_BASE_URL` +
   `BENCHMARKS_WEB_PROD_URL` to `https://benchmarks-website.vercel.app`; then await an emitting
   run and verify acceptance (§6). Requires explicit user go-ahead. Post-demo. Reversible by
   unsetting the var. This is the FIRST prod RDS write of the project.

Do NOT set `GH_BENCH_INGEST_ROLE_ARN` until A exists AND D is merged (both precede B by order).

## 5. Reference values (verified 2026-06-19)

- **AWS:** account `245040174862`, region `us-east-1`, profile `bench-prod` (IAM user
  `connor-aws-cli`). GitHub OIDC provider `token.actions.githubusercontent.com` exists.
- **RDS:** endpoint `vortex-bench-prod.c4f8qygk4xdp.us-east-1.rds.amazonaws.com:5432`, database
  `vortex_bench`. CA bundle (what the workflow ships): the GLOBAL bundle
  `https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem` (a superset; the runbook's
  region-specific `us-east-1-bundle.pem` also works, but the code uses global).
  `DbiResourceId` via `aws rds describe-db-instances --db-instance-identifier vortex-bench-prod
  --query 'DBInstances[0].DbiResourceId' --output text --profile bench-prod`.
- **IAM roles:** `GitHubBenchmarkIngestRole` (ABSENT, phase A creates it; ARN will be
  `arn:aws:iam::245040174862:role/GitHubBenchmarkIngestRole`); `GitHubBenchmarkSchemaRole`
  (exists). **DB roles:** `migrator` (DDL), `bench_ingest` (the v4 dual-write, `rds_iam`),
  `bench_read` (Vercel reader, password).
- **Monorepo (`vortex-data/vortex`) GitHub config:** already-correct vars
  `RDS_BENCH_DB_NAME=vortex_bench`, `RDS_BENCH_INSTANCE_ENDPOINT=vortex-bench-prod.c4f8qygk4xdp.us-east-1.rds.amazonaws.com`,
  `RDS_BENCH_REGION=us-east-1`,
  `GH_BENCH_SCHEMA_ROLE_ARN=arn:aws:iam::245040174862:role/GitHubBenchmarkSchemaRole`. To set in
  phase B: `GH_BENCH_INGEST_ROLE_ARN` (absent now); repoint `BENCH_SITE_BASE_URL` +
  `BENCHMARKS_WEB_PROD_URL` from the deleted `https://benchmarks-web.vercel.app` to
  `https://benchmarks-website.vercel.app`. Secret `BENCH_REVALIDATE_TOKEN` exists (set
  2026-06-16) but is re-aligned in phase C.
- **Vercel:** project `benchmarks-website`, team `vortex-data`, orgId
  `team_TkGBm7OlQtmqOFNpVNuaNpFX`, projectId `prj_AOss3j7VcSu5UoyBA1LIvj4G0DQ6`,
  `https://benchmarks-website.vercel.app`, `develop` = production. Production env is MISSING
  `BENCH_REVALIDATE_TOKEN` (phase C adds it). Local copy NOT linked — run
  `vercel link --scope vortex-data` (or pass `VERCEL_PROJECT_ID`) before any env op.

### 5.1 Phase B/C command reference (from runbook §2)

Phase B:
```
gh variable set GH_BENCH_INGEST_ROLE_ARN -R vortex-data/vortex --body 'arn:aws:iam::245040174862:role/GitHubBenchmarkIngestRole'
gh variable set BENCH_SITE_BASE_URL -R vortex-data/vortex --body 'https://benchmarks-website.vercel.app'
gh variable set BENCHMARKS_WEB_PROD_URL -R vortex-data/vortex --body 'https://benchmarks-website.vercel.app'
```
Phase C (one fresh token on both sides, then redeploy v4 prod):
```
TOKEN="$(python3 -c 'import secrets;print(secrets.token_urlsafe(48))')"
printf '%s' "$TOKEN" | vercel env add BENCH_REVALIDATE_TOKEN production
gh secret set BENCH_REVALIDATE_TOKEN -R vortex-data/vortex --body "$TOKEN"
```
A new Vercel env var only takes effect on the next production deploy.

## 6. Acceptance criteria (runbook §5)

After phase B, trigger/await an emitting workflow and confirm:

- the v4 step logs OIDC assume-role OK;
- the upsert reports `inserted`/`updated`;
- the revalidate ping returns `200 {revalidated:true}` (not 503/401 — 503 = Vercel prod missing
  `BENCH_REVALIDATE_TOKEN` or not redeployed since; 401 = token mismatch);
- `curl -s https://benchmarks-website.vercel.app/api/health` shows `row_counts.commits` and
  `latest_commit_timestamp` advancing to the just-run commit;
- the site shows the new commit with NO manual migration;
- the v3/v2 paths are unaffected (the workflow's v3 ingest step still succeeds).

## 7. Risks

0. **Demo-window prod-data corruption**: P=low-if-disciplined, impact=severe; the project must
   not write/alter prod RDS during the demo. Mitigation: §4.0 — no prod RDS write before phase B;
   C+B both hard-gated post-demo + go-ahead; only pure-code D runs in the demo window.
1. **Live-cutover blast radius (B)**: P=med; mitigation: best-effort + `continue-on-error` +
   env-gate; reversible by unsetting the var; watch the first run.
1a. **Testcontainer PG != RDS gap**: P=med, impact=moderate; local PG has no IAM auth / RDS TLS /
   role grant, so pre-merge tests cannot prove that binding. Mitigation: pin testcontainer to PG
   16 (RDS major), and verify the real-RDS path read-only/rolled-back at phase B before the flip.
2. **measurement_id port drift**: P=med, impact=severe; mitigation: golden-vector test wired as
   a required CI check, covering Unicode/float/i32 edges.
3. **Revalidate 503/401**: P=med, impact=minor; mitigation: phase C aligns one fresh token both
   sides + redeploys before B.
4. **AWS IAM-write reach unproven**: P=low; verify at phase A start.
5. **Vercel local copy not linked**: P=high, impact=minor; run `vercel link` before phase C env op.
6. **Workflow YAML re-anchor onto diverged develop**: P=low; `yamllint --strict` gates each edit.

## 8. Out of scope

- Making v4 primary, retiring v3, DNS cutover, v2 decommission — all later.
- Any code in `vortex-data/benchmarks-website` (read-only reference).
- RDS schema migrations / role management in the monorepo (website repo owns them).
- Changes to `vortex-bench/src/v3.rs` record shapes or the v2/v3 write paths.
- SCHEMA_VERSION bump (stays 1).
