# bench-v4 CI emitter dual-write — big-plans spine

<!-- Link to the brainstorming design spec rather than duplicating it. -->
**Design spec:** `.big-plans/ct__bench-v4-emitters-design.md` (written by brainstorming in Step 1.2)
**Work shape:** feature-integration

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
phase: "1: D — port v4 emitter (CODE)"
sub_phase: null
task: null
status: reviewing
last_gate: null
phase_entry_sha: a3ffeeea8ad1c9147b31a0a4ece5233143975f32
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
  (psycopg + IAM-token minting) declared in `post-ingest.py`'s PEP-723 block and lazily imported
  inside the postgres branch. Accepted: the v3 `--server` path stays stdlib-only and unaffected.
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
