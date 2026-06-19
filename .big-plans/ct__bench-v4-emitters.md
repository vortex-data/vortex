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
phase: null
sub_phase: null
task: null
status: planning
last_gate: null
phase_entry_sha: null
```

---

## Phase Map

<!-- Empty until Step 1.4 decomposition (after brainstorming + grill-me). -->

(decomposition pending — Step 1.4)

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

- (none yet)

---

## Verdict / Completion Ledger

<!-- Grows as sub-phases and phases complete. -->

(none yet — execution has not started)
