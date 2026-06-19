# Sub-phase 1.3 — CI + workflow wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the new scripts/ tests into CI and add the best-effort v4 ingest step block to the three emitter workflows, leaving the v4 path DORMANT (gated on `vars.GH_BENCH_INGEST_ROLE_ARN != ''`).

**Architecture:** The v4 step blocks are ported verbatim from `f9b36ae3f` (already grill-me-correct: global RDS CA bundle, `uv run --no-project --with` deps, `continue-on-error` + env-gate). They insert at unchanged anchors on current develop (the v3 ingest step). The ci.yml test wiring is ADAPTED to our `--with` deps approach (NOT the branch's `--all-packages` + pyproject-deps approach): the cheap contract+revalidate tests run as a step in the existing required `python-test` job (no docker); the testcontainer test runs in a new lightweight docker-gated job. NO `pyproject.toml` / `uv.lock` changes (deps are supplied per-invocation via `--with`).

**Tech Stack:** GitHub Actions YAML, yamllint, uv (`uv run --no-project --with`), Docker (testcontainer job).

## Global Constraints

- All `.github/` edits MUST pass `uvx yamllint --strict -c .yamllint.yaml <files>` (double-quote when quoting, 1-space `{ }`, 0-space `[ ]`, >=2-space inline comments, trailing newline, no trailing spaces).
- All `uses:` action references MUST be SHA-pinned with a `# vN` comment. Reuse SHAs already present in the repo where possible.
- Every v4 step MUST keep `if: vars.GH_BENCH_INGEST_ROLE_ARN != ''` AND `continue-on-error: true` (the dormancy + best-effort safety property). Do NOT weaken or remove the existing v3 `--server` steps, and do NOT add `continue-on-error` to them.
- NO changes to `pyproject.toml` or `uv.lock` (deps via `--with`). Do NOT add the v4 deps to the workspace.
- This sub-phase touches ONLY: `.github/workflows/ci.yml`, `.github/workflows/bench.yml`, `.github/workflows/sql-benchmarks.yml`, `.github/workflows/v3-commit-metadata.yml`. Do NOT touch any scripts/ file (those are 1.1/1.2) or post-ingest.py.
- Commits: `git commit -F` with a heredoc (NEVER backticks or a `---` line); sign off `Signed-off-by: Connor Tsui <connor@spiraldb.com>`.

---

### Task 1: Wire the scripts/ tests into ci.yml

**Files:**
- Modify: `.github/workflows/ci.yml` (add one step to the `python-test` job; add a new `scripts-test` job)

- [ ] **Step 1: Add the contract + revalidate test step to the existing `python-test` job**

In `.github/workflows/ci.yml`, the `python-test` job (`name: "Python (test)"`, ~line 75) has a `Pytest - Vortex` step (~line 93) with `working-directory: vortex-python/`. AFTER that step's block (and any sibling step in the job, e.g. a basedpyright step at ~line 104, i.e. as the LAST step of the `python-test` job), add a new step that runs from the repo root (no `working-directory`):

```yaml
      - name: "Pytest - scripts (measurement_id + revalidate)"
        run: >-
          uv run --no-project --with pytest --with xxhash pytest
          scripts/test_measurement_id.py scripts/test_post_ingest_revalidate.py
```

This runs the two non-docker tests in the already-required `python-test` job (so they are required checks). It needs only `uv` (provided by the job's existing setup) + `xxhash` (via `--with`). Do NOT use `working-directory` (these paths are repo-root-relative). Match the surrounding indentation exactly.

- [ ] **Step 2: Add a new docker-gated `scripts-test` job for the testcontainer test**

Insert a new top-level job (2-space indent) into `.github/workflows/ci.yml` immediately BEFORE the `rust-docs:` job (~line 116). Copy the `actions/checkout@<sha>  # v6` line VERBATIM from an existing step in this same ci.yml (so the pinned SHA matches the repo); use the setup-uv action SHA `spiraldb/actions/.github/actions/setup-uv@a746510eafaa926484c354541cfc49b2ec06cc63  # 0.18.6` (the same SHA this PR introduces in the emitter workflows, Task 2):

```yaml
  scripts-test:
    name: "Python (scripts testcontainers)"
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@<COPY THE SHA + # v6 COMMENT FROM AN EXISTING ci.yml checkout STEP>
      - uses: spiraldb/actions/.github/actions/setup-uv@a746510eafaa926484c354541cfc49b2ec06cc63  # 0.18.6
      # The testcontainer suite MUST run in CI, not silently skip: a green job with
      # skipped tests would let a writer regression merge undetected. `docker info`
      # fails the job up front if the daemon is unavailable, and the CI env var makes
      # the testcontainer fixtures fail (not skip) per their _require_docker_for_testcontainers.
      - name: "Verify Docker is available for testcontainers"
        run: docker info
      - name: "Pytest - scripts testcontainers"
        run: >-
          uv run --no-project --with pytest --with "psycopg[binary]" --with boto3
          --with xxhash --with testcontainers pytest scripts/test_post_ingest_postgres.py
```

Note: this is a LIGHTWEIGHT job (ubuntu-latest + uv + docker), NOT the branch's `--all-packages` + setup-prebuild + large-runner version -- our tests need no Rust-workspace build. It names the explicit test file (NOT `pytest scripts/`) so it does not collect the pandas-dependent `scripts/tests/test_benchmark_reporting.py` or the contract/revalidate tests already covered in Step 1.

- [ ] **Step 3: Lint ci.yml**

```bash
uvx yamllint --strict -c .yamllint.yaml .github/workflows/ci.yml
```

Expected: exit 0 (no diagnostics). Fix any yamllint violations (indentation, quoting, comment spacing, trailing newline) and re-run.

- [ ] **Step 4: Confirm the wired commands actually pass locally (the exact commands CI will run)**

```bash
uv run --no-project --with pytest --with xxhash pytest scripts/test_measurement_id.py scripts/test_post_ingest_revalidate.py -q
docker info >/dev/null 2>&1 && echo "docker OK" || echo "DOCKER MISSING -- escalate"
uv run --no-project --with pytest --with "psycopg[binary]" --with boto3 --with xxhash --with testcontainers pytest scripts/test_post_ingest_postgres.py -q
```

Expected: the first command passes (measurement_id 65 + revalidate 7); the testcontainer command passes (100 tests) under docker. These are the literal commands the two CI steps run, so green here means green in CI.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ci.yml
git commit -F - <<'EOF'
ci: run the v4 emitter scripts/ tests

Add the measurement_id contract test + the revalidate test to the python-test
job (required, no docker), and a lightweight docker-gated scripts-test job for
the testcontainer Postgres writer suite. Deps are supplied per-invocation via
uv run --with (no pyproject/uv.lock changes).

Signed-off-by: Connor Tsui <connor@spiraldb.com>
EOF
```

---

### Task 2: Add the best-effort v4 ingest step to the three emitter workflows

**Files:**
- Modify: `.github/workflows/bench.yml`
- Modify: `.github/workflows/sql-benchmarks.yml`
- Modify: `.github/workflows/v3-commit-metadata.yml`

**Interfaces:** the v4 step runs `scripts/post-ingest.py --postgres` (sub-phase 1.2). The blocks are verbatim from `f9b36ae3f` and already correct.

- [ ] **Step 1: Extract the three branch files for reference**

```bash
git show f9b36ae3f:.github/workflows/bench.yml > /tmp/branch-bench.yml
git show f9b36ae3f:.github/workflows/sql-benchmarks.yml > /tmp/branch-sql.yml
git show f9b36ae3f:.github/workflows/v3-commit-metadata.yml > /tmp/branch-v3meta.yml
```

- [ ] **Step 2: Insert the v4 block into `bench.yml`**

In `.github/workflows/bench.yml`, the `Ingest results to v3 server` step ends at the `--repo-url "${{ github.server_url }}/${{ github.repository }}"` line (~line 129), followed by a blank line and `- name: Alert incident.io` (~line 131). From `/tmp/branch-bench.yml`, copy the v4 block (the three steps `Configure AWS credentials for v4 ingest (OIDC)` / `Install uv for v4 ingest` / `Ingest results to v4 Postgres (best-effort)` plus the leading comment block) VERBATIM and insert it between the v3 step and `- name: Alert incident.io`, preserving the blank-line spacing and indentation. Do NOT alter the v3 step. The block's gate is `if: vars.GH_BENCH_INGEST_ROLE_ARN != ''` + `continue-on-error: true` on every step.

- [ ] **Step 3: Insert the v4 block into `sql-benchmarks.yml`**

Same as Step 2, but in `.github/workflows/sql-benchmarks.yml` (v3 step `--repo-url` at ~line 683, `- name: Alert incident.io` at ~line 685). Use the block from `/tmp/branch-sql.yml`. NOTE: this block's gate additionally carries `inputs.mode == 'develop' &&` on each `if:` (matching the v3 step's mode guard) -- keep that verbatim.

- [ ] **Step 4: Insert the v4 block + `id-token` permission into `v3-commit-metadata.yml`**

In `.github/workflows/v3-commit-metadata.yml`:
1. The `permissions:` block (~line 11) currently is `contents: read` only. Add `id-token: write` ABOVE `contents: read` (from `/tmp/branch-v3meta.yml`):
   ```yaml
   permissions:
     id-token: write  # enables AWS-GitHub OIDC for the best-effort v4 ingest step
     contents: read
   ```
2. The `Ingest commit metadata to v3 server` step ends at `--repo-url ...` (~line 34, the last line of the file). Append the v4 block from `/tmp/branch-v3meta.yml` (the `Ingest commit metadata to v4 Postgres (best-effort)` variant -- note it writes an `empty.jsonl` and upserts the commit row only) after the v3 step.

- [ ] **Step 5: Lint all three workflows**

```bash
uvx yamllint --strict -c .yamllint.yaml .github/workflows/bench.yml .github/workflows/sql-benchmarks.yml .github/workflows/v3-commit-metadata.yml
```

Expected: exit 0. Fix any violations and re-run.

- [ ] **Step 6: Sanity-check the v4 gating is intact on every new step**

```bash
grep -c "GH_BENCH_INGEST_ROLE_ARN != ''" .github/workflows/bench.yml .github/workflows/sql-benchmarks.yml .github/workflows/v3-commit-metadata.yml
grep -c "continue-on-error: true" .github/workflows/bench.yml .github/workflows/sql-benchmarks.yml .github/workflows/v3-commit-metadata.yml
```

Expected: each file shows 3 gated steps (3 `GH_BENCH_INGEST_ROLE_ARN != ''` and 3 `continue-on-error: true`). Confirm the v3 `--server` steps were NOT modified (`git diff` shows only additions around them).

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/bench.yml .github/workflows/sql-benchmarks.yml .github/workflows/v3-commit-metadata.yml
git commit -F - <<'EOF'
ci: add best-effort v4 Postgres dual-write step to the emitter workflows

Insert the dormant, env-gated, continue-on-error v4 ingest step (OIDC
assume-role -> uv run post-ingest.py --postgres -> revalidate) after the v3
--server step in bench.yml, sql-benchmarks.yml, and v3-commit-metadata.yml
(the last also gains id-token: write). The block no-ops until
GH_BENCH_INGEST_ROLE_ARN is set and can never fail the job; the v3 path is
untouched.

Signed-off-by: Connor Tsui <connor@spiraldb.com>
EOF
```

---

## Self-Review

- **Spec coverage:** the contract + revalidate tests are wired into the required `python-test` job (Task 1 Step 1); the testcontainer test into a docker-gated job (Task 1 Step 2); the v4 best-effort step is added to all three emitter workflows with `id-token` on v3-commit-metadata (Task 2). No pyproject/uv.lock changes. Only the 4 workflow files touched.
- **Placeholder scan:** none -- exact anchors, exact blocks (from the branch + the embedded ci.yml additions), exact commands. The one intentional `<COPY THE SHA ...>` is an explicit instruction to match the repo's pinned checkout SHA.
- **Consistency:** the v4 step gate (`GH_BENCH_INGEST_ROLE_ARN != ''` + `continue-on-error`) is identical across all three workflows; the setup-uv SHA matches between Task 1's scripts-test job and Task 2's v4 blocks; the test commands in Task 1 match the locally-verified 1.2 commands.
