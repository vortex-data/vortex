# bench-v4 CI emitter dual-write — big-plans spine

<!-- Link to the brainstorming design spec rather than duplicating it. -->
**Design spec:** `.big-plans/ct__bench-v4-emitters-design.md` (written by brainstorming in Step 1.2)
**Work shape:** feature-integration

---

## SESSION HANDOFF (2026-06-19) — READ FIRST ON RESUME

**Where we are:** Phase 1 (D, the CODE phase) is COMPLETE. The code shipped as a MINIMAL,
code-only PR **#8513** (branch `ct/bench-v4-emitter-dual-write`, draft -> develop): just
`post-ingest.py` `--postgres`, `_measurement_id.py`, and the 3 dormant workflow steps. The
orchestration scaffolding AND the test suite (testcontainer / revalidate / golden parity /
cross-check) were INTENTIONALLY dropped from the repo (user decision -- correctness testing lives
with the Rust source of truth in `vortex-data/benchmarks-website`). The `measurement_id` port was
verified once against the 65 golden vectors (all pass) before shipping. The earlier
include-scaffolding PR #8512 is CLOSED. The spine + design spec stay BRANCH-LOCAL on
`ct/bench-v4-emitters` and are NOT in #8513, so NO scaffolding-cleanup PR is needed at the end.

**#8513 merge:** the only failing required check is DCO (gmail-author / spiraldb-signoff mismatch,
same as every develop commit); the user admin-merges past it. Merging #8513 is INDEPENDENT of the
ops phases -- A can proceed before D merges; only phase B needs BOTH A done and D merged.

**DO NOT re-run the phase-end gauntlet on resume** -- it already passed (ledger `#### Phase 1
gate`, phase-4 / accepted).

**NEXT:** phase A (OPS) -- create `GitHubBenchmarkIngestRole`. Proceeding now per user direction
("immediately start building the roles"; pick up the pace, fewer per-step confirms). Phase A is
data-safe (creates an unused IAM role; no prod data/site touch). Phases C and B remain HARD-GATED:
post-demo + explicit go-ahead (they touch the prod site / first prod RDS write). See HARD
CONSTRAINTS below.

**HARD CONSTRAINTS still in force (design spec §4.0):**
- **DEMO SAFETY:** NO prod RDS writes until phase B. Phases C and B are GATED — post-demo +
  explicit user go-ahead. Only phase D (this code, done) was demo-safe. Confirm the demo is over
  before any prod-touching op (A is data-safe IAM-only but still an external mutation; C/B touch
  prod/site).
- **1PASSWORD must be UNLOCKED** for commit/push (SSH signing). It auto-locked mid-session and
  blocked a commit; if commits fail with `1Password: failed to fill whole buffer`, ask the user
  to unlock the desktop app.
- **Remaining phases after D merges:** A (create `GitHubBenchmarkIngestRole`, ops, data-safe),
  C (align `BENCH_REVALIDATE_TOKEN` on Vercel prod + monorepo secret + redeploy v4, ops, GATED,
  post-demo), B (set `GH_BENCH_INGEST_ROLE_ARN` + repoint `BENCH_SITE_BASE_URL`/`BENCHMARKS_WEB_PROD_URL`,
  live cutover + acceptance §6, ops, GATED, post-demo). All ops phases: direct CLI, no gauntlet/PR,
  pre-action confirm. See Orchestration notes + design spec §4-§6.

**Resume:** re-invoke `/spiral:big-plans` on branch `ct/bench-v4-emitters`; Phase 0 reads this
handoff + the Current Position below.

---

## Goal

Make the Vortex monorepo CI emitters write benchmark results LIVE to the v4 RDS Postgres
("Path B / v4 dual-write") so the manual `vortex-bench-migrate` refresh is no longer needed,
shipping the write as an additive, best-effort, env-gated step that can never break the live
v2/v3 paths.

## Architecture & key decisions

<!-- One-liners only. Full detail lives in the design spec. file:line anchors only for the
     most load-bearing items. The authoritative external plan is the runbook:
     benchmarks-website/docs/runbooks/emitter-ingest-cutover.md (verified 2026-06-19). -->

- **Architecture:** port the already-written, unmerged v4 emitter from `origin/ct/bench-v4`
  (commits `9a870091e` + `9a1824afa`, tree at tip `f9b36ae3f`) onto current `develop`; the
  feature is FINISH + MERGE + PROVISION, not write-from-scratch.
- **Decision: 4-phase structure, reordered D -> A -> C -> B** (grill-me, demo-driven) — Phase D
  (code: port emitter + workflows, full SDD + gauntlet + PR + merge, dormant) -> Phase A (ops:
  create IAM role) -> Phase C (ops, GATED: align revalidate token) -> Phase B (ops, GATED live
  cutover: set ARN var + repoint URLs + soak). Ops phases A/C/B are CLI ops with side effects and
  NO in-repo diff: they skip gauntlet + PR; exit criteria are CLI verification commands; the
  human checkpoint is a pre-action external-side-effect confirmation (see design spec § 4).
- **Decision: demo-safety model (HARD CONSTRAINT)** — NO prod RDS writes until phase B; phases C
  and B both hard-gated to AFTER the demo + explicit go-ahead (C redeploys the v4 site the demo
  reads). Only pure-code phase D runs in the demo window (zero prod interaction). Pre-merge
  confidence comes from testcontainer Postgres tests (pinned to RDS major version 16) + the
  golden-vector test, with NO prod/develop dependency; the real-RDS IAM/TLS path is verified
  read-only/rolled-back at phase B before the flip. See design spec § 4.0.
- **Decision: code-port scope = everything incl. extras** — the essentials (post-ingest.py
  `--postgres`, `_measurement_id.py` + golden.json + test, the 3 workflow v4 steps, ci.yml
  wiring) PLUS the testcontainer writer tests (`test_post_ingest_postgres.py`) PLUS the extras
  (`cross_check_python_writer.py`, `test_post_ingest_revalidate.py`). Migrations + migrate-schema
  are OUT (extracted website repo owns schema/roles). See design spec § 3.
- **Decision (Class B amendment, sub-phase 1.2): testcontainer test uses a self-contained schema
  fixture, migrations stay OUT.** The branch `test_post_ingest_postgres.py` depends on the real
  `migrations/` + `migrate-schema.py` + `benchmarks-website/server/src/schema.rs` (all out of
  monorepo scope), conflicting with "migrations OUT" + "testcontainer test IN". Resolved
  (user-chosen): a hand-written `scripts/_v4_schema_fixture.sql` creates the 6 tables; the
  `schema_conn` fixture applies it directly (no migrate-runner); the SCHEMA_VERSION lockstep
  sub-test self-checks `post-ingest.py`'s `SCHEMA_VERSION == 1`. See design spec § 3.
- **Decision: the v4 step is dormant until the switch is flipped.** Every v4 workflow step is
  gated on `vars.GH_BENCH_INGEST_ROLE_ARN != ''` with `continue-on-error: true`; merging D
  with the var unset ships dead-but-safe code. Setting the var (phase B) is the live cutover.
- **Decision: `measurement_id` is computed client-side by a Python xxhash64 port** that must
  reproduce the Rust reference (`benchmarks-website/server/src/db.rs`) byte-for-byte, validated
  against the ported `scripts/measurement_id_golden.json` golden vectors. The cross-language
  golden vectors ARE the contract.
- **Decision: the v3 `--server` path stays stdlib-only and intact;** the new v4 `--postgres`
  path uses third-party deps (`psycopg[binary]`, `boto3`, `xxhash`) supplied at the call site via
  `uv run --no-project --with ...` (NOT `post-ingest.py`'s PEP-723 block, which stays empty —
  corrected from the runbook by grill-me), lazily imported only inside the postgres branch, so
  importing the module or running `--server` never requires those deps. `_measurement_id.py` uses
  the `xxhash` package's XXH64. CA bundle is the global bundle.
- **Decision: sequence (reordered):** D (build + merge dormant) -> A (create role) -> C (align
  token, gated, post-demo, redeploy v4) -> B (set ARN var + repoint URLs, flips ON, post-demo) ->
  soak/acceptance. Do NOT set `GH_BENCH_INGEST_ROLE_ARN` until A exists and D is merged (both
  precede B by order).
- **Decision: `SCHEMA_VERSION` stays in lockstep at `1`** across `post-ingest.py` and the v4
  schema; no bump in this project.

## Out of scope

- No changes to the live v2 (static S3 JSON) or v3 (DuckDB + `/api/ingest`) write paths beyond
  leaving them fully intact; the v3 `--server` ingest step is preserved verbatim.
- No code lands in `vortex-data/benchmarks-website` — that repo (wire contract, golden vectors,
  IAM provisioner, revalidate endpoint, RDS schema/migrations) is READ-ONLY reference.
- No RDS schema migrations or `migrate-schema.py` in the monorepo — schema/role management is
  the extracted website repo's job (`schema-deploy.yml` + `GitHubBenchmarkSchemaRole`); the
  monorepo emitter only CONNECTS as the already-provisioned `bench_ingest` role.
- Making v4 primary, retiring v3, DNS cutover, v2 decommission — all explicitly later, not here.
- No changes to `vortex-bench/src/v3.rs` record shapes (the wire records are unchanged).

## Risks

0. **Demo-window prod-data corruption (next few hours)**: P=low-if-disciplined; impact=severe;
   mitigation: no prod RDS write before phase B; phases C and B hard-gated post-demo + go-ahead;
   only pure-code phase D runs during the demo window. Verified by failure isolation (the v4
   block is dormant until the ARN var is set, which is phase B).
1. **Live-cutover blast radius (phase B)**: P=med; impact=moderate; mitigation: best-effort +
   `continue-on-error` + env-gate means a v4 failure cannot fail a workflow (verified on-branch:
   all v4 steps carry the gate + `continue-on-error`, v3 runs first); B is reversible by
   unsetting `GH_BENCH_INGEST_ROLE_ARN`; watch the first emitting run before walking away.
2. **measurement_id port drift from the Rust reference**: P=med; impact=severe (silent wrong
   upsert keys -> duplicate/again rows); mitigation: the ported pytest asserts every golden
   vector byte-for-byte and is wired as a required CI check; covers Unicode + float + i32 edges.
3. **Revalidate 503/401 after cutover**: P=med; impact=minor (stale site, no data loss); cause:
   Vercel prod missing `BENCH_REVALIDATE_TOKEN` (503) or token mismatch (401); mitigation: phase
   C aligns one fresh token on both sides + redeploys before B flips the switch.
4. **AWS IAM-write reach unproven**: P=low; impact=moderate (phase A blocked); mitigation: verify
   at phase A start; `bench-prod` is an IAM user with sibling role-create precedent.
5. **Vercel local copy not linked**: P=high; impact=minor (phase C friction); mitigation: run
   `vercel link --scope vortex-data` (or pass `VERCEL_PROJECT_ID`) before any env op in C.
6. **Workflow YAML re-anchor onto diverged develop (`ci.yml`)**: P=low; impact=minor; mitigation:
   the v4 inserts target intact anchors; `yamllint --strict` gates every workflow edit.

---

## Current Position

```yaml
phase: "5: backfill — historical data load (OPS, DEFERRED)"
sub_phase: "5.1 re-run migrate"
task: null
status: awaiting-human-gate
last_gate: "2026-06-19 — CUTOVER LIVE + verified (A/C/D/B done; v4 dual-write writing on develop pushes); Phase 5 backfill deferred to the user's quiet-window schedule"
phase_entry_sha: 9e601fb05
```

---

## Phase Map

<!-- Phases reordered D -> A -> C -> B per the demo-safety model (design spec § 4.0).
     Phase 1 (D) is the ONLY code phase (standard SDD + gauntlet + PR + human gate + merge).
     Phases 2-4 (A/C/B) are OPS phases: no SDD, no gauntlet, no PR; direct CLI; exit = the
     CLI verification command; human checkpoint = pre-action confirmation + the phase-gate AUQ.
     See "Orchestration notes" below for the ops-phase protocol. -->

| Phase | Sub-phase | Scope (one line) | Exit criteria (command → expected) | Sub-phase gauntlet | Phase gauntlet | Task-plan pointer |
|---|---|---|---|---|---|---|
| 1: D — port v4 emitter (CODE) | 1.1 measurement_id contract | Port `_measurement_id.py` (xxhash XXH64) + `measurement_id_golden.json` + `test_measurement_id.py`; repoint the two docstrings off the extracted-repo paths | (phase-level — see exit row) | pr-2 | | `.big-plans/ct__bench-v4-emitters--1-1-measurement-id.plan.md` |
| *(phase 1 cont.)* | 1.2 Postgres writer | Add `--postgres`/`--region` mode to `post-ingest.py` (RDS IAM auth, `verify-full` TLS + `ssl_in_use` check, `bench_ingest` enforce, NaN/Inf guard, 5-table upsert + commit dim in one txn, best-effort revalidate) keeping `--server` stdlib-only intact; add `test_post_ingest_postgres.py` (testcontainers PG16), `test_post_ingest_revalidate.py`, `cross_check_python_writer.py` | | pr-3 | | `.big-plans/ct__bench-v4-emitters--1-2-postgres-writer.plan.md` |
| *(phase 1 cont.)* | 1.3 CI + workflow wiring | Wire the contract test into `ci.yml` `python-test` (docker-UNgated) + a docker-gated `scripts-test` job for the testcontainer tests; add the best-effort v4 step block to `bench.yml`, `sql-benchmarks.yml`, `v3-commit-metadata.yml` (+ `id-token: write` on the last); regenerate `uv.lock` with `uv lock` | | pr-3 | | `.big-plans/ct__bench-v4-emitters--1-3-ci-workflow-wiring.plan.md` |
| *(phase 1 exit)* | *(all sub-phases)* | v4 dual-write emitter + dormant workflow steps ported; v3 path intact | `uv run --no-project --with xxhash pytest scripts/test_measurement_id.py` → 0; `yamllint --strict -c .yamllint.yaml .github/workflows/bench.yml .github/workflows/sql-benchmarks.yml .github/workflows/v3-commit-metadata.yml .github/workflows/ci.yml` → 0; `python3 scripts/post-ingest.py --help` → 0 (stdlib-only); testcontainer writer tests green under docker | | phase-4 | |
| 2: A — provision IAM role (OPS) | 2.1 create ingest role | Create `GitHubBenchmarkIngestRole` (trust `repo:vortex-data/vortex` on develop; grant `rds-db:connect` as `bench_ingest` on the instance `DbiResourceId`) via `provision.sh` `ensure_ingest_role` or a surgical create; record the ARN. Pre-action confirm | (phase-level) | n/a (ops) | | n/a (ops — direct CLI) |
| *(phase 2 exit)* | *(ops)* | Ingest role exists with the rds-db:connect grant | `aws iam get-role --role-name GitHubBenchmarkIngestRole --profile bench-prod` → 0 AND `aws iam get-role-policy --role-name GitHubBenchmarkIngestRole --policy-name rds-db-connect-ingest --profile bench-prod` shows `rds-db:connect` for `bench_ingest` | | n/a (ops) | |
| 3: C — align revalidate token (OPS, GATED) | 3.1 set + align token | `vercel link --scope vortex-data`; generate one fresh token; `vercel env add BENCH_REVALIDATE_TOKEN production`; `gh secret set BENCH_REVALIDATE_TOKEN -R vortex-data/vortex`; redeploy v4 prod. GATED: explicit go-ahead + post-demo. Pre-action confirm | (phase-level) | n/a (ops) | | n/a (ops — direct CLI) |
| *(phase 3 exit)* | *(ops)* | Token aligned both sides + v4 redeployed | authed `POST {site}/api/revalidate` with the token → HTTP `200 {revalidated:true}` (not 503/401) | | n/a (ops) | |
| 4: B — flip switch + soak (OPS, GATED, live cutover) | 4.1 pre-flip verify + cutover | read-only/rolled-back `bench_ingest` RDS verify (IAM + `verify-full` TLS, no data write); then `gh variable set GH_BENCH_INGEST_ROLE_ARN` + repoint `BENCH_SITE_BASE_URL` + `BENCHMARKS_WEB_PROD_URL` to `https://benchmarks-website.vercel.app`. GATED: explicit go-ahead + post-demo. Pre-action confirm | (phase-level) | n/a (ops) | | n/a (ops — direct CLI) |
| *(phase 4 cont.)* | 4.2 soak + acceptance | trigger/await an emitting `develop` run; verify §5 acceptance (OIDC assume-role ok, upsert inserted/updated, revalidate 200, `/api/health` advances, v3 step still green) | | n/a (ops) | | n/a (ops — direct CLI) |
| *(phase 4 exit)* | *(ops)* | v4 dual-write live + acceptance green | `gh variable list -R vortex-data/vortex` shows `GH_BENCH_INGEST_ROLE_ARN` set + `BENCH_SITE_BASE_URL`=`https://benchmarks-website.vercel.app`; `curl -s https://benchmarks-website.vercel.app/api/health` `latest_commit_timestamp` advanced to a post-cutover commit; the emitting run's v3 step still succeeded | | n/a (ops) | |

---

## Orchestration notes (custom spine — READ ON RESUME)

This spine mixes one CODE phase with three OPS phases; the orchestrator handles them differently.

- **Phase 1 (D) — CODE — standard big-plans:** per sub-phase, `writing-plans` (JIT) ->
  `subagent-driven-development` -> `gauntlet` (pr-2/pr-3) checkpoint; phase-end `gauntlet`
  (phase-4); open the phase PR; mandatory human gate; squash-merge. Pre-merge confidence per
  design spec § 4.0 (testcontainer PG16 + golden, no prod/develop dependency).
- **Phases 2-4 (A/C/B) — OPS — adapted:** NO `writing-plans`, NO `subagent-driven-development`,
  NO `gauntlet`, NO PR (there is no in-repo diff to plan, review, or merge). Execute each
  sub-phase as direct CLI. Before EACH mutating external op (AWS / GitHub / Vercel), fire the
  pre-action external-side-effect confirmation. The phase exit criterion is the CLI verification
  command in the Phase Map. The phase-boundary human-gate AUQ still fires, but the user reviews
  the CLI verification output instead of a PR diff. Spine status for ops phases:
  `implementing` while running the CLI ops (resume = re-run idempotently), then
  `awaiting-human-gate` at the boundary (skip `reviewing`/`fixing` — no gauntlet). Do NOT route
  an ops phase's `implementing` status into the SDD/writing-plans loop.
- **GATED ops phases (C and B):** in addition to the per-op confirmation, both require explicit
  user go-ahead AND must wait until AFTER the demo (design spec § 4.0). C is the deferred
  `BENCH_REVALIDATE_TOKEN` un-deferral; B is the live cutover (first prod RDS write).
- **Wrap-up (after phase B):** the spine + design spec rode onto develop via phase D's
  squash-merge (big-plans' normal between-phase behavior; nothing sensitive — all infra values
  already exist in the repo's workflows/vars). Because the trailing ops phases A/C/B have NO PR,
  remove the scaffolding from develop with a dedicated `chore: remove big-plans spine` cleanup PR
  rather than relying on a final phase PR. The branch-local A/C/B `plan:` commits are the
  orchestration audit trail and are not merged.

---

## Reviewer context

### Project-specific BANS — constraints gauntlet reviewers MUST ENFORCE

<!-- Scoped so the feature's intended design (continue-on-error on the NEW v4 step; third-party
     deps on the NEW v4 path) is NOT flagged; the protective halves remain enforced. -->

- **commits**: do NOT omit the `Signed-off-by: Connor Tsui <connor@spiraldb.com>` DCO trailer on
  any commit — DCO is enforced repo-wide. (Source: root `CLAUDE.md` § Commits.)
- **commit messages**: do NOT include a `---` scissors line or backticks in a `git commit -m`
  body — the DCO pre-push hook false-positives on `---`, and backticks run as command
  substitution. Use `-F`/heredoc and drop scissors. (Source: project memory.)
- **`.github/`**: do NOT land workflow YAML that fails `yamllint --strict -c .yamllint.yaml`
  (double-quote when quoting, 1-space `{ }`, 0-space `[ ]`, 2-space inline comments, trailing
  newline, no trailing spaces). (Source: `.github/AGENTS.md`.)
- **actions**: do NOT add an unpinned or tag-pinned `uses:` — pin to a full commit SHA with a
  `# vN` comment, matching existing steps. (Source: existing workflows.)
- **Python**: do NOT introduce ruff `F,E,W,UP,I` failures or exceed line-length 120; new
  `scripts/*.py` are linted by the repo-wide `ruff format`/`ruff check`. (Source: `pyproject.toml`.)
- **SPDX**: do NOT add a new `scripts/*.py` or `.github/` file without the two SPDX header lines.
  (Source: `reuse.yml` + every existing file.)
- **v3 path protection**: do NOT add `continue-on-error` to (or otherwise weaken) the EXISTING
  v3 `--server` ingest step, and do NOT make the v3 `--server` path or a bare module import
  require third-party packages. (The NEW v4 step is intentionally `continue-on-error`; the NEW
  v4 path may lazily import third-party deps — those are design, not violations.)
- **secrets**: do NOT echo, print, or pass a secret/token as a CLI arg or log-visible value;
  secrets flow via `env:` and are read from the environment. (Source: existing workflows.)
- **SCHEMA_VERSION**: do NOT change `SCHEMA_VERSION` away from `1` — it must stay in lockstep
  across the emitter and the v4 schema. (Source: `CONTRACT.md`.)
- **comments**: do NOT use em dashes in comments/docs (use `--`); full-sentence comments, own
  lines, ~100-col. (Source: project memory + root `CLAUDE.md`.)

### Carry-forward (DO NOT re-flag)

#### Accepted tradeoffs

- **v4 emitter third-party deps**: the v4 `--postgres` path depends on third-party packages
  (`psycopg[binary]`, `boto3`, `xxhash`) supplied at the call site via `uv run --no-project --with`
  (NOT declared in `post-ingest.py`'s PEP-723 block, which stays `dependencies = []`), and lazily
  imported inside the postgres branch. Accepted: the v3 `--server` path stays stdlib-only.
- **v4 step best-effort by design**: every v4 workflow step is `continue-on-error: true` and
  env-gated. Accepted: a v4 failure intentionally does not fail the workflow (additive write).

#### Deferred work

- **Sub-phase 1.1**, `scripts/test_measurement_id.py:41`, **should-fix**: module-level
  `_load_port()` makes a missing `xxhash` a collection error rather than a skip. Deferral
  rationale: address in sub-phase 1.3 (CI + test wiring) — add `pytest.importorskip("xxhash")`
  and/or ensure CI supplies `xxhash` via `uv run --with`.
- **Sub-phase 1.1**, CI (no workflow runs the test), **should-fix**: the golden-parity test is
  not wired into CI. Deferral rationale: that IS sub-phase 1.3's explicit scope.
- **Sub-phase 1.1**, `scripts/measurement_id_golden.json`, **should-fix**: golden vectors lack
  NaN/Inf/-0.0 threshold and per-table empty-string coverage. Deferral rationale: OUT OF MONOREPO
  SCOPE — the golden JSON is generated by the Rust test in the `vortex-data/benchmarks-website`
  repo; coverage is added there and regenerated, never hand-edited in the monorepo.
- **Sub-phase 1.1**, `scripts/_measurement_id.py:15` + `scripts/test_measurement_id.py:34`,
  **nit**: docstring misattributes the `measurement_id_golden_vectors` test to `db.rs` (it lives
  in `tests/measurement_id_golden.rs`), plus a docstring phrasing fragment. Deferral rationale:
  cosmetic doc nits carried verbatim from the source branch; fold into sub-phase 1.3 if it
  touches these files.
- **Sub-phase 1.1**, `scripts/test_measurement_id.py` (`ensure_ascii`, 100-col), **nit**: add a
  comment explaining `ensure_ascii=False` is load-bearing for the multibyte guard; 3 lines exceed
  the 100-col style target (pass ruff's 120). Deferral rationale: cosmetic; optional cleanup in 1.3.
- **Sub-phase 1.2**, `scripts/cross_check_python_writer.py` -> `post-ingest.py:_upsert_commit`,
  **should-fix**: the cross-check harness raises a bare `KeyError` on an operator-supplied commit
  envelope missing an optional field (e.g. `message`). Deferral rationale: cross_check is operator
  tooling (the future ingest-cutover gate), not the production write path (which always supplies all
  9 commit fields via `build_commit`); harden with `commit.get(...)` when next exercised.
- **Sub-phase 1.2**, `scripts/post-ingest.py` write-conflict retry, **should-fix**: the
  deadlock/serialization retry is only unit-mocked (synthetic exceptions); no real-Postgres
  abort+transaction-re-entry test. Deferral rationale: the retry is a verbatim port of the v3
  server's tested logic; a real-conflict testcontainer test is valuable but non-blocking for the
  additive best-effort writer.
- **Sub-phase 1.2**, `scripts/_v4_schema_fixture.sql` header + `test_post_ingest_postgres.py`
  docstrings, **should-fix** (maint): the fixture/test docs under-describe migration 006 (omit the
  5 format/engine indexes, which ARE present in the fixture), reference `CONTRACT.md` without its
  `vortex-data/benchmarks-website/` path, and the "SCHEMA_VERSION / column-list contract" phrase
  implies an automated drift guard that does not exist. Deferral rationale: docs-only; the fixture
  CODE is column-by-column correct (opus-verified). Apply as a doc cleanup during phase-D
  finalization (before the phase-end gauntlet).
- **Sub-phase 1.3 — APPLY in phase-D finalization** (cheap, in-scope, clearly-correct should-fixes
  to land as a polish commit BEFORE the phase-end gauntlet, so the shipped artifact is clean):
  (a) `configure-aws-credentials` SHA mismatch — the 3 new v4 blocks pin `@99214aa6889…  # v6`
  while the repo's existing v3/S3 steps pin `@e7f100cf4c008…  # v6`; align the v4 blocks to the
  repo's existing `e7f100cf…` so each file is internally consistent (flagged by fresh + maint).
  (b) add `pytest.importorskip("xxhash")` to `scripts/test_measurement_id.py` (the 1.1-deferred
  local-dev-UX fix; CI already supplies xxhash via `--with`). (c) `scripts/post-ingest.py:~1000`
  error message says "set AWS_REGION" but boto3 reads `AWS_DEFAULT_REGION` — fix the message.
  (d) `scripts/test_post_ingest_revalidate.py` — assert the revalidate request method is POST
  (the 1.2-deferred nit).
- **Sub-phase 1.3**, **deferred / no-action** (nits + pre-existing): contract+revalidate step
  lives in the required `python-test` job (deliberate — required-check placement beats the
  build-coupling concern fresh raised); `sql-benchmarks.yml` has no `id-token: write` block
  (pre-existing pattern — its existing Setup-AWS-CLI step relies on the same caller-inherited
  permission; all callers grant it); redundant "Install uv" step in sql-benchmarks (harmless,
  idempotent, continue-on-error); v4 steps omit `--repo-url` (post-ingest.py's default is correct
  for the canonical repo); `--with testcontainers` unpinned; `empty.jsonl` created in both the v3
  and v4 steps of v3-commit-metadata (defensible — the v4 step may run when the v3 step skipped).
- **Ops (not code)**: making the new `scripts-test` job a REQUIRED check is a GitHub
  branch-protection setting (admin action), not a code change — track alongside the ops phases.
- **Phase 1 gate (phase-4) review** — new should-fixes (all non-blocking; refactor/hardening
  opportunities for a follow-up): (a) [arch] the 3-step v4 ingest block is copy-pasted across
  bench/sql-benchmarks/v3-commit-metadata (the SHA mismatch was a symptom) — consider extracting a
  `.github/actions/v4-ingest` composite action to collapse the DSN/CA-bundle/SHA to one site;
  (b) [arch/correctness] the record-kind->measurement_id dispatch + record-schema maps are restated
  across post-ingest.py, cross_check, and the tests — consolidate when the next fact table lands
  (the test maps are deliberately independent for verification value); (c) [correctness] commit-dict
  fields are accessed without the loud record-indexed validation convention (a malformed timestamp
  would raise a raw Postgres error; relates to the 1.2 cross_check KeyError item) — harden with the
  same convention when cross_check is next touched. Deferral rationale: the production path is safe
  (build_commit always supplies all fields; the workflow blocks are now SHA-consistent); these are
  maintainability refactors, not correctness gaps.
- **Phase 1 gate — no-action**: [correctness] `query_idx`/`iterations` validated as i32 (matching
  the v3 server's storage + the measurement_id hash) vs the producer's u32 wire type — not
  practically reachable. Doc nits (fold into a future doc pass): `measurement_id_golden.json` `note`
  + `_measurement_id.py` docstring still say `benchmarks-website/.../db.rs` without the
  `vortex-data/` org prefix and attribute the golden generator to `db.rs` vs `measurement_id_golden.rs`;
  the fixture header + test docstring omit the 006 engine/format indexes (present in the fixture);
  `post-ingest.py` comment references `migrations/004`; the v4 blocks lack a machine-checkable removal
  trigger comment.
- **Sub-phase 1.2**, multiple files, **nits** (~10): test rollback-assertion consistency +
  revalidate HTTP-method assertion (fresh); non-string `commit_sha` yields a misleading mismatch
  error (correctness); plan-cycle/PR labels in comments, `_load_module` 4x duplication, handler
  name vs kind inconsistency, v3-path removal marker (maint). Deferral rationale: minor polish
  carried from the verbatim port; not blocking.

---

## Verdict / Completion Ledger

<!-- Grows as sub-phases and phases complete. -->

### Phase 1: D — port v4 emitter (CODE)

#### Sub-phase 1.1: measurement_id contract

- **Shipped:** measurement_id Python port (`xxhash` XXH64) + 63-vector golden file +
  golden-parity pytest, extracted verbatim from `f9b36ae3f`, docstrings repointed at the
  extracted `vortex-data/benchmarks-website` repo; ruff-formatted to the repo's 120-col.
- **Gauntlet:** pr-2 / accepted (cycles: 1)
- **Deferred:** 5 items (see Carry-forward > Deferred work)

#### Sub-phase 1.2: Postgres writer

- **Shipped:** v4 `--postgres` IAM-auth upsert writer in `post-ingest.py` (verify-full TLS,
  NaN/Inf guard, 5-table + commit-dim upsert in one transaction, write-conflict retry, best-effort
  revalidate), v3 `--server` path kept stdlib-only; revalidate test + cross-check utility; adapted
  testcontainer test + self-contained `_v4_schema_fixture.sql`. 100 testcontainer tests pass vs
  postgres:16-alpine.
- **Gauntlet:** pr-3 / accepted (cycles: 1) — fresh + correctness (opus) + maint, zero must-fix.
- **Deferred:** 4 should-fix-class + ~10 nits (see Carry-forward > Deferred work)

#### Sub-phase 1.3: CI + workflow wiring

- **Shipped:** wired the contract + revalidate tests into the required `python-test` CI job and
  the testcontainer suite into a new docker-gated `scripts-test` job (all via `uv run --with`, no
  pyproject/uv.lock changes); added the dormant best-effort v4 ingest step block to bench.yml,
  sql-benchmarks.yml, v3-commit-metadata.yml (+ `id-token: write`), gated on
  `vars.GH_BENCH_INGEST_ROLE_ARN != ''` + `continue-on-error`.
- **Gauntlet:** pr-3 / accepted (cycles: 1) — fresh + correctness + maint, zero must-fix.
- **Deferred:** 4 apply-in-finalization should-fixes + several no-action nits/pre-existing
  (see Carry-forward > Deferred work).

#### Phase 1 gate

- **Phase-D finalization polish:** applied the 4 in-scope should-fixes (configure-aws-credentials
  SHA aligned to `e7f100cf`, `xxhash` importorskip, `AWS_DEFAULT_REGION` message, revalidate POST
  assertion) in commit `6fdd727f0`, reviewed by the phase-end gauntlet.
- **Exit criteria:** all PASS — measurement_id contract (65), yamllint --strict (4 workflows),
  `post-ingest.py --help` stdlib-only, testcontainer writer suite (100) under docker.
- **Gauntlet:** phase-4 / accepted (cycles: 1) — spec + correctness (opus) + maint + arch, zero
  must-fix. Correctness reviewer independently re-ran 172 tests (green); arch confirmed the v4
  path can never break the live v3/v2 path (dormant-but-ready). New should-fixes are refactor /
  doc items deferred to follow-ups (see Carry-forward > Deferred work).
- **Phase PR:** #8512 (draft -> develop; include-scaffolding form, user-chosen). Opened
  2026-06-19; CI running; the plan is to squash-merge on green.
- **Post-PR-open CI fixes** (commit `05b6b79f6`, config-only, no reviewed-code change, so the
  phase-4 gauntlet was NOT re-run): (a) `REUSE.toml` annotation licensing `.big-plans/**`
  CC-BY-4.0 (the include-scaffolding files lacked SPDX headers -> reuse-check); (b) `_typos.toml`
  ignore for the SQL verbs `UPDATEs`/`UPDATEd`/`INSERTs`/`INSERTed` in
  `cross_check_python_writer.py` docstrings (typos mis-split -> Spell Check). Verified locally
  (reuse lint compliant, typos exit 0). DCO fails on the gmail-author / spiraldb-signoff mismatch
  but is non-blocking here -- develop's own latest commit carries the identical mismatch and
  merged. Changelog label `changelog/ci` added to the PR.
- **RE-SCOPE (user decision, 2026-06-19):** #8512 CLOSED and superseded by **#8513**, a MINIMAL
  code-only PR from a fresh branch `ct/bench-v4-emitter-dual-write` off develop (commits
  `4723bcab1` writer+port, `aa5a15a39` workflow wiring). Ships ONLY `scripts/post-ingest.py`
  (`--postgres`), `scripts/_measurement_id.py`, and the 3 dormant workflow steps. DROPPED from the
  repo: the `.big-plans/` scaffolding, `measurement_id_golden.json`, `test_measurement_id.py`,
  `test_post_ingest_postgres.py` + `_v4_schema_fixture.sql`, `test_post_ingest_revalidate.py`,
  `cross_check_python_writer.py`, the `ci.yml` test wiring, and the `_typos.toml`/`REUSE.toml`
  changes (no longer needed). The `measurement_id` port was verified once against the 65 golden
  vectors (all pass) before shipping; the vectors + parity test live in `vortex-data/benchmarks-website`.
  #8513 exit checks pass locally (yamllint --strict on the 3 workflows, `post-ingest.py --help`
  stdlib-only, ruff check + format); only DCO is red (user admin-merges). Because the spine stays
  branch-local on `ct/bench-v4-emitters`, the planned scaffolding-cleanup PR is NO LONGER needed.

### Phase 2: A — provision IAM role (OPS)

#### Phase 2 (A) — create ingest role

- **Done (2026-06-19):** Created `GitHubBenchmarkIngestRole`
  (`arn:aws:iam::245040174862:role/GitHubBenchmarkIngestRole`) via a surgical `aws iam create-role`
  + `put-role-policy`, matching `provision.sh ensure_ingest_role()` exactly (no RDS/proxy/SG touch).
  The agent's Bash was blocked by the auto-mode classifier on the prod IAM mutation; the USER ran
  the two commands. The agent gathered/verified all values and confirmed the result.
- **Trust:** GitHub OIDC, `repo:vortex-data/vortex` on `develop` + `ct/bench-v4`,
  `aud=sts.amazonaws.com`. **Permission:** `rds-db:connect` on
  `arn:aws:rds-db:us-east-1:245040174862:dbuser:db-4VPTDACTRQHOS24WEIR3TNC2M4/bench_ingest`
  (instance `vortex-bench-prod`).
- **Exit criteria:** PASS -- `get-role` returns the ARN; `get-role-policy rds-db-connect-ingest`
  shows `rds-db:connect` for `bench_ingest`.
- **No gauntlet / no PR** (ops phase).

#### Phase 2 (A) gate

- **Demo confirmed over (2026-06-19);** user gave go-ahead for C/B. The harness auto-mode
  classifier hard-blocks the agent from prod mutations (Vercel env, GH secret/variable, AWS),
  and the user chose to run the ops themselves -- so C/B are operationalized as bash scripts the
  user runs (fish shell, so `#!/usr/bin/env bash` + `bash <script>`).

### Phase 3: C — align revalidate token (OPS)

#### Phase 3 (C) — DONE (2026-06-19)

- **Done:** fresh `BENCH_REVALIDATE_TOKEN` set on the v4 Vercel project (Production, Sensitive)
  AND the monorepo GH secret (same value); v4 prod redeployed via an empty commit pushed to
  benchmarks-website `develop` (the Git-connected production branch -- the dashboard Redeploy was
  "locked" for the user, and `vercel --prod` hit the 47k-file upload cap, so the Git push was the
  clean trigger).
- **Exit criteria:** PASS -- `POST https://benchmarks-website.vercel.app/api/revalidate` with the
  token returned `200 {revalidated:true}` (503 while the old build was live, then 200 once the new
  Production deployment went live).

### Phase 4: B — flip switch (OPS)

#### Phase 4 (B) — DONE / cutover verified LIVE (2026-06-19)

- **Done:** #8513 admin-merged to develop (merge commit `97850e9e0`). `bash cutover-phase-b.sh` set
  `GH_BENCH_INGEST_ROLE_ARN` (the on-switch) + repointed `BENCH_SITE_BASE_URL` at 19:28:39Z. First
  ARMED emitting run = `v3-commit-metadata` workflow_dispatch `27844693038` (+ the `geo` push run
  `27844711531`), both started AFTER the flip so the v4 step ran ACTIVE.
- **Acceptance — ALL PASS:**
  - v4 step ran (not skipped). **OIDC assume-role of `GitHubBenchmarkIngestRole` SUCCEEDED** (live
    proof phase A's role + trust policy work); connected to `vortex-bench-prod` as `bench_ingest`
    with `verify-full` TLS; `post-ingest.py --postgres` exited 0. Output `{"records":0,...}` is
    expected -- `v3-commit-metadata` ingests an empty `.jsonl` (registers the COMMIT dimension, not
    measurement rows); the heavier `bench.yml` / `sql-benchmarks.yml` develop-push runs carry the
    actual measurement records through the same (now-validated) path.
  - **v3 `--server` step still SUCCEEDED** -- v2/v3 paths intact.
  - **`/api/health` (reads v4 RDS) advanced:** `latest_commit_timestamp = 2026-06-19T19:29:07Z`
    (fresh, post-cutover); `row_counts` populated (commits 4638, query_measurements 4,981,532,
    compression_times 256,350, ...).
  - No revalidate-failure warning in the log (best-effort revalidate OK; phase C verified 200).
- **RESULT: the v4 dual-write is LIVE.** CI now writes new benchmark results to v4 on every develop
  push. The ONGOING manual `vortex-bench-migrate` refresh is retired (its one-time HISTORICAL use is
  the deferred Phase 5 backfill). Remaining: Phase 5 (historical backfill, user's schedule); the DNS
  cutover to make v4 primary stays OUT OF SCOPE and must follow Phase 5.

#### Post-merge CI scare — INVESTIGATED, NOT #8513 (2026-06-19, via ci-failure-analysis)

- User saw bench.yml jobs "failing" + `503` + `PutObject PreconditionFailed` shortly after the
  merge. Root cause is NOT #8513:
  - The failed bench jobs' failing step was the **benchmark step itself, conclusion `cancelled`**
    (external stop, GitHub Actions blip the user suspected), during 3 rapid develop pushes
    (97850e9e0 / dc3fa496d / 3f54d1f8c in ~3 min). `bench.yml` has NO concurrency block.
  - **`PutObject PreconditionFailed`** is from pre-existing `scripts/cat-s3.sh` (the v2 S3 results
    upload, last touched #7296 on 2026-04-06, NOT in the #8513 diff): it deliberately uses
    `--if-none-match "*"` / `--if-match "$etag"` optimistic concurrency and RETRIES on the
    precondition failure -- the expected signal when concurrent runs race the shared `data.json.gz`,
    by design.
  - **`503`** = the v4 revalidate ping (continue-on-error; 503 only before the phase-C redeploy, 200
    now) or transient S3 SlowDown that cat-s3.sh tolerates. Cannot fail the job.
  - Structural proof: the #8513 workflow diff is PURE ADDITION (bench.yml `@@ -128,6 +128,46`,
    sql-benchmarks `@@ -682,6 +682,45`; 0 lines removed, no `permissions:`/matrix/existing-step
    change); every added v4 step is env-gated + `continue-on-error` and runs AFTER the benchmark +
    v2-S3 + v3 steps; `post-ingest.py` does zero S3 PutObject (writes Postgres only). The user
    restarted the runs; a green re-run on the same code is the clincher.
- **Open soak item:** confirm a clean (uninterrupted) `bench.yml` / `sql-benchmarks.yml` run's v4
  step reports `inserted/updated > 0` (heavy measurement rows landing in v4, not just commit
  metadata). Note bench.yml / sql-benchmarks.yml lack an explicit `id-token: write` (deferred item);
  if the v4 OIDC step ever can't mint a token there, the v4 write silently no-ops (harmless to CI).

#### Post-cutover bug — v4 ingest sccache contamination (FIX in PR #8516, 2026-06-19)

- **Real bug, ours:** the v4 "Configure AWS credentials (OIDC)" step persists the assumed
  `GitHubBenchmarkIngestRole` (rds-db:connect only) as the job's AMBIENT AWS creds; the FOLLOWING
  "Install uv" step runs `spiraldb/actions/setup-uv` -> `uv sync` -> compiles `vortex-python` via
  **sccache (S3-backed on the `runs-on extras=s3-cache` bench runners)**, which then fails
  `s3:GetObject AccessDenied`. All v4 steps are `continue-on-error`, so CI stays GREEN, but the
  affected jobs' v4 write silently drops.
- **INTERMITTENT, not total** (corrects an earlier "broken" framing): the **commit-metadata**
  workflow runs on `ubuntu-latest` (NO S3 sccache) so its v4 ingest succeeds RELIABLY -> commit data
  lands every push (this is the "some commit data made it" the user observed). `bench.yml` /
  `sql-benchmarks.yml` (S3-cache runners) are intermittent -- MOST v4 ingests succeed (sccache server
  already up with original creds / vortex-python cached), a minority fail (e.g. `compress-bench`)
  when uv must drive sccache->S3 after the cred swap. So v4 is getting all commit data + most
  benchmark data; the gap is occasional silent drops.
- **Fix = install only the uv binary** (PR **#8516**, branch `ct/bench-v4-emitter-uv-fix`; the
  earlier "reorder" approach was reverted/superseded). Swap the 3 v4 "Install uv" steps from
  `spiraldb/actions/setup-uv` (which runs a full `uv sync --all-extras --dev` -> builds
  vortex-python via sccache) to `astral-sh/setup-uv@37802adc…  # v7.6.0` (the same binary spiraldb
  vendors), NO sync. The ingest runs `uv run --no-project --with`, so it only ever needed the uv
  BINARY -- **production proved this**: the user saw "Install uv" go RED while the ingest still
  SUCCEEDED (binary installs before the sync; the ingest never uses the synced workspace). Binary-
  only removes the wasteful workspace build AND the sccache->S3 dependency entirely, so step order
  no longer matters. The benchmark job's own non-v4 setup-uv is left as-is. **PR #8516 ALSO renames**
  `v3-commit-metadata.yml` -> `commit-metadata.yml` + name `v3 commit metadata` -> `Commit metadata`
  (feeds both backends; not a required check, referenced nowhere else). DCO is the lone red check
  (admin-merge). Phase 5's `--replace` backfill overwrites any intermittent-window gaps regardless.

- **Partial-charts concern — NOT the emitter (user-diagnosed, 2026-06-19):** the user saw some
  recent clickbench points missing on the v4 site. Diagnosed as a **benchmarks-website caching
  issue**: the "latest 100" view shows the recent data fine, but the "all data" view serves a
  lagging cache (a cache tag the `/api/revalidate` ping doesn't invalidate, or a longer TTL). The
  data IS in v4 (latest-100 proves it), so the cutover + dual-writes are correct. Fix lives in the
  website repo's `web/` cache/revalidation scope -- OUT OF SCOPE for this emitter project; no
  emitter action.

#### Cutover script archive (for Phase 5 / reference)

- **Token:** a fresh `BENCH_REVALIDATE_TOKEN` generated to `/tmp/revalidate_token` (64 chars,
  never printed).
- **Phase C script** `/tmp/cutover-phase-c.sh`: `vercel env add BENCH_REVALIDATE_TOKEN production`
  + `gh secret set BENCH_REVALIDATE_TOKEN -R vortex-data/vortex` (same token) -> prompt to redeploy
  v4 prod -> verify `POST $SITE/api/revalidate` returns 200. Vercel project
  org `team_TkGBm7OlQtmqOFNpVNuaNpFX` / proj `prj_AOss3j7VcSu5UoyBA1LIvj4G0DQ6`.
- **Phase B script** `/tmp/cutover-phase-b.sh` (gated on #8513 MERGED): `gh variable set
  GH_BENCH_INGEST_ROLE_ARN = arn:aws:iam::245040174862:role/GitHubBenchmarkIngestRole` (THE gate)
  + `gh variable set BENCH_SITE_BASE_URL = https://benchmarks-website.vercel.app` (was the deleted
  `benchmarks-web` project). `BENCHMARKS_WEB_PROD_URL` is absent in the monorepo -> no repoint.
- **Order:** C -> admin-merge #8513 (DCO is the lone red check) -> B -> trigger a NEW develop run
  (the merge commit's own run has the var still unset/dormant) -> acceptance (v4 step OIDC + upsert
  inserted/updated + revalidate 200; `/api/health` latest_commit_timestamp advances; v3 still green).
- **Revalidate contract** (verified in `web/app/api/revalidate/route.ts`): `POST`, `Authorization:
  Bearer <token>`, 200 `{revalidated:true}` / 401 mismatch / 503 unset.

### Phase 5: historical backfill (OPS, DEFERRED — added 2026-06-19 at user request)

#### Phase 5 (backfill) — one-time historical data load into v4 RDS

- **Why:** the A-D dual-write only makes CI write NEW results to v4 from the cutover forward. v4 is
  missing historical benchmark data (everything before its last migration + the gap up to cutover).
  This phase loads it so v4 is complete. Scope-expansion the USER surfaced; was implicitly "later".
- **Mechanism:** a ONE-TIME re-run of `vortex-bench-migrate` (the same tool the A-D project retired
  for the ONGOING refresh; still the right tool for a one-time historical load). Build
  `cargo build --release -p vortex-bench-migrate` (release, non-sandboxed); needs the sibling
  `server`; see [[project_bench_v4_migrate_tooling]] (base `ct/restore-bench-migrate` @ 684f96f36)
  + the migrate runbook.
- **Safety:** safe to run AFTER the cutover with the dual-write live -- both the emitter and migrate
  upsert on the same deterministic `measurement_id` (emitter Python port is golden-verified against
  the same Rust hash migrate uses), so the backfill fills historical rows and re-affirms overlapping
  ones to identical values. No dupes, no clobber, no gap.
- **When:** a quiet window with NO `develop` merges (hygiene -- clean before/after row-count check +
  no load contention; not strictly required for correctness given keyed upserts).
- **v4 is NOT primary yet** (`bench.vortex.dev` still serves complete v2/v3), so the temporary v4
  incompleteness does not hit users; the DNS cutover (make v4 primary) stays OUT OF SCOPE and must
  come AFTER this backfill completes.
- **Exit criteria:** v4 RDS row counts match the source (no gap vs v2/v3); `/api/health` row_counts
  reflect the full history.
- **No gauntlet / no PR** (ops phase).
