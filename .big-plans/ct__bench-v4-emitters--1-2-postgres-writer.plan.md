# Sub-phase 1.2 — Postgres writer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the v4 `--postgres` writer into `scripts/post-ingest.py` (keeping the v3 `--server` path stdlib-only and intact), plus its tests + cross-check utility, adapting the testcontainer test to a self-contained schema fixture.

**Architecture:** Extract-from-branch (`f9b36ae3f`) port. `post-ingest.py` applies cleanly (mainline never touched it). The testcontainer test is ADAPTED, not verbatim: it must not depend on the out-of-scope `migrations/` + `migrate-schema.py` + `benchmarks-website/server/src/schema.rs`; instead it applies a self-contained `scripts/_v4_schema_fixture.sql` and self-checks `SCHEMA_VERSION == 1`.

**Tech Stack:** Python 3.11+, `psycopg[binary]`, `boto3` (RDS IAM token), `xxhash`, `testcontainers`, pytest, `uv run --no-project --with ...`, ruff, Docker (local + CI for the testcontainer suite).

## Global Constraints

- Extract files from `f9b36ae3f` via `git show f9b36ae3f:<path> > <path>`. Do NOT re-author the writer logic.
- The v3 `--server` path MUST stay importable + runnable under bare `python3` (stdlib only). The v4 deps (`psycopg`, `boto3`, `xxhash`) are imported LAZILY inside the `--postgres` code path only; `post-ingest.py`'s PEP-723 block stays `dependencies = []`.
- Repoint cross-repo doc references: `benchmarks-website/server/src/<x>` and `benchmarks-website/AGENTS.md` no longer exist in this monorepo. Repoint to "the `vortex-data/benchmarks-website` repo's `<x>`". Do NOT change any code/logic/values.
- `migrations/` + `migrate-schema.py` + `schema.rs` stay OUT of the monorepo (the testcontainer test is adapted to a self-contained fixture instead — Task 3).
- SPDX headers on every new file (`# SPDX-License-Identifier: Apache-2.0` / `# SPDX-FileCopyrightText: Copyright the Vortex contributors`; SQL uses `--` comment SPDX lines).
- Python lint: ruff `F,E,W,UP,I` clean, line-length 120. A whitespace-only `ruff format` of extracted files to the repo's 120-col is expected and acceptable (no logic/value change). Comments use `--`, never em dashes.
- This sub-phase touches ONLY: `scripts/post-ingest.py`, `scripts/test_post_ingest_postgres.py`, `scripts/test_post_ingest_revalidate.py`, `scripts/cross_check_python_writer.py`, `scripts/_v4_schema_fixture.sql`. Do NOT touch `ci.yml`, any workflow (sub-phase 1.3), or the 1.1 files.
- Commits: `git commit -F` with a heredoc (NEVER backticks or a `---` line in the message); sign off `Signed-off-by: Connor Tsui <connor@spiraldb.com>`.

---

### Task 1: Port the `--postgres` writer into post-ingest.py

**Files:**
- Modify (full replace): `scripts/post-ingest.py` (340 lines on develop -> 1252 lines from `f9b36ae3f`)

**Interfaces:**
- Consumes: `scripts/_measurement_id.py` (ported in sub-phase 1.1) via the lazy `_measurement_id_module()` loader (`Path(__file__).parent / "_measurement_id.py"`).
- Produces (relied on by Tasks 2-3): `post_ingest.ingest_postgres(conn, commit, records) -> (inserted, updated)`, `post_ingest.connect_postgres(...)`, `post_ingest.refresh_site_cache(...)`, `post_ingest.SCHEMA_VERSION == 1`, and the `--server` / `--postgres` CLI dispatch.

- [ ] **Step 1: Replace post-ingest.py with the branch version**

```bash
cd "$(git rev-parse --show-toplevel)"
git show f9b36ae3f:scripts/post-ingest.py > scripts/post-ingest.py
```

- [ ] **Step 2: Verify the v3 `--server` path stays stdlib-only (the load-bearing safety property)**

```bash
python3 scripts/post-ingest.py --help
```

Expected: prints usage showing BOTH `--server` and `--postgres` options, exits 0, with NO ImportError. This proves module import + the help/v3 path need no third-party packages (psycopg/boto3/xxhash are lazily imported only inside the `--postgres` branch). If this errors with a missing third-party module, the lazy-import boundary is broken -- STOP and report (do not "fix" by adding deps to PEP-723).

- [ ] **Step 3: Repoint the cross-repo doc references**

Grep first: `grep -n "benchmarks-website" scripts/post-ingest.py` (expect ~9 hits at lines near 41/43/47/65-67/328/527/909). For EACH hit, repoint the path token: `benchmarks-website/server/src/<x>` -> ``the `vortex-data/benchmarks-website` repo's `server/src/<x>` ``, and `benchmarks-website/AGENTS.md` -> ``the `vortex-data/benchmarks-website` repo's `AGENTS.md` ``. These are doc comments only -- do NOT touch any code, the `SCHEMA_VERSION = 1` constant, or the wire logic. Re-run `python3 scripts/post-ingest.py --help` after editing to confirm nothing broke.

- [ ] **Step 4: Lint + format**

```bash
uvx ruff check scripts/post-ingest.py
uvx ruff format --check scripts/post-ingest.py || uvx ruff format scripts/post-ingest.py
uvx ruff check scripts/post-ingest.py && uvx ruff format --check scripts/post-ingest.py
python3 -m py_compile scripts/post-ingest.py
```

Expected: ruff check clean; after the conditional `ruff format`, both `ruff check` and `ruff format --check` exit 0; py_compile exits 0.

- [ ] **Step 5: Commit**

```bash
git add scripts/post-ingest.py
git commit -F - <<'EOF'
scripts: add v4 --postgres IAM-auth upsert writer to post-ingest.py

Port the v4 Postgres dual-write path (RDS IAM auth, verify-full TLS, NaN/Inf
guard, 5-table + commit-dim upsert in one transaction, best-effort revalidate)
from the v4 emitter branch, keeping the v3 --server path stdlib-only and intact.
v4 deps (psycopg, boto3, xxhash) are imported lazily inside the --postgres path.
Repoint docstrings to the extracted vortex-data/benchmarks-website repo.

Signed-off-by: Connor Tsui <connor@spiraldb.com>
EOF
```

---

### Task 2: Port the revalidate test + cross-check utility

**Files:**
- Create/Test: `scripts/test_post_ingest_revalidate.py` (262 lines; pure-stdlib, no Docker/psycopg)
- Create: `scripts/cross_check_python_writer.py` (247 lines; CLI cross-check utility)

**Interfaces:**
- Consumes: `scripts/post-ingest.py` (Task 1) loaded by sibling path (`SCRIPTS_DIR / "post-ingest.py"`).

- [ ] **Step 1: Extract both files from the branch**

```bash
git show f9b36ae3f:scripts/test_post_ingest_revalidate.py > scripts/test_post_ingest_revalidate.py
git show f9b36ae3f:scripts/cross_check_python_writer.py > scripts/cross_check_python_writer.py
```

- [ ] **Step 2: Run the revalidate test (pure stdlib + pytest)**

```bash
uv run --no-project --with pytest scripts/test_post_ingest_revalidate.py 2>/dev/null || uv run --no-project --with pytest --with xxhash pytest scripts/test_post_ingest_revalidate.py -q
```

Use this robust form: `uv run --no-project --with pytest pytest scripts/test_post_ingest_revalidate.py -q`. Expected: PASS. The test monkeypatches `urllib.request.urlopen` and exercises `refresh_site_cache` -- it loads `post-ingest.py` by path, whose module-level imports are stdlib, so no third-party deps are needed. If the import chain unexpectedly pulls a third-party module, add `--with xxhash` and report it as a concern (it would mean a module-level dep leaked).

- [ ] **Step 3: Sanity-check the cross-check utility compiles and lints**

```bash
python3 -m py_compile scripts/cross_check_python_writer.py
uvx ruff check scripts/cross_check_python_writer.py scripts/test_post_ingest_revalidate.py
uvx ruff format --check scripts/cross_check_python_writer.py scripts/test_post_ingest_revalidate.py || uvx ruff format scripts/cross_check_python_writer.py scripts/test_post_ingest_revalidate.py
uvx ruff check scripts/cross_check_python_writer.py scripts/test_post_ingest_revalidate.py && uvx ruff format --check scripts/cross_check_python_writer.py scripts/test_post_ingest_revalidate.py
```

Expected: py_compile exits 0; after the conditional format, ruff check + format --check both exit 0. (cross_check_python_writer.py is a CLI utility, not a test; py_compile + lint is the appropriate gate. Re-run the revalidate test if `ruff format` reformatted it.)

- [ ] **Step 4: Commit**

```bash
git add scripts/test_post_ingest_revalidate.py scripts/cross_check_python_writer.py
git commit -F - <<'EOF'
scripts: add revalidate test + python-writer cross-check utility

Port the pure-stdlib revalidate-hook test (asserts refresh_site_cache sends the
bearer header and swallows every failure so it can never change the ingest exit
code) and the cross_check_python_writer.py utility (confirms the Python writer
recomputes the same measurement_id as seeded Rust-loaded rows and UPDATEs rather
than duplicating) from the v4 emitter branch.

Signed-off-by: Connor Tsui <connor@spiraldb.com>
EOF
```

---

### Task 3: Self-contained schema fixture + adapted testcontainer test

**Files:**
- Create: `scripts/_v4_schema_fixture.sql` (the 6-table DDL, derived from the website repo's `migrations/001_initial_schema.sql`)
- Create/Test: `scripts/test_post_ingest_postgres.py` (1309 lines from the branch, ADAPTED)

**Interfaces:**
- Consumes: `post_ingest.ingest_postgres` / `connect_postgres` (Task 1); the schema fixture (this task).

- [ ] **Step 1: Create the self-contained schema fixture**

Read the reference DDL (READ-ONLY, does not land here): `/Users/connor/spiral/vortex-data/benchmarks-website/migrations/001_initial_schema.sql`. Create `scripts/_v4_schema_fixture.sql` containing the two SPDX `--` header lines, a comment noting it is a TEST-ONLY fixture mirroring that repo's `migrations/001_initial_schema.sql` (drift managed like the SCHEMA_VERSION / column-list cross-repo contract), then the six `CREATE TABLE IF NOT EXISTS` statements (`commits`, `query_measurements`, `compression_times`, `compression_sizes`, `random_access_times`, `vector_search_runs`) and their `CREATE INDEX IF NOT EXISTS` statements, copied verbatim (columns/types/PKs/nullability EXACTLY as in 001 -- `DOUBLE PRECISION`, `BIGINT[]`, `TIMESTAMPTZ`, etc.). Do NOT include the role/grant migrations (002-007); the test connects as the container superuser, so no roles are needed.

- [ ] **Step 2: Extract the testcontainer test from the branch**

```bash
git show f9b36ae3f:scripts/test_post_ingest_postgres.py > scripts/test_post_ingest_postgres.py
```

- [ ] **Step 3: Run it unmodified to confirm it fails on the out-of-scope deps (red)**

```bash
uv run --no-project --with pytest --with 'psycopg[binary]' --with boto3 --with xxhash --with testcontainers pytest scripts/test_post_ingest_postgres.py -q 2>&1 | tail -25
```

Expected: a collection/setup ERROR -- the module tries `_load_module("migrate-schema.py", ...)` (line ~77) and/or reads `migrations/` (line ~45), which do not exist in the monorepo. This confirms the adaptation in Step 4 is required.

- [ ] **Step 4: Adapt the test off the out-of-scope dependencies**

Grep the ported file for every out-of-scope reference: `grep -n "migrate_runner\|migrate-schema\|REPO_MIGRATIONS_DIR\|migrations\|schema.rs\|benchmarks-website" scripts/test_post_ingest_postgres.py`. Apply these edits (and any others the grep surfaces), changing ONLY schema-bootstrap plumbing, never the assertions/test bodies:

1. Remove the migrate-runner module load (line ~77: `migrate_runner = _load_module("migrate-schema.py", "migrate_schema")`) and the `REPO_MIGRATIONS_DIR = REPO_ROOT / "migrations"` binding (line ~45). Keep `REPO_ROOT` only if still used elsewhere; if it becomes unused, remove it (ruff F841/unused will flag).
2. Add a fixture-path constant near the other path constants, e.g. `_SCHEMA_FIXTURE = SCRIPTS_DIR / "_v4_schema_fixture.sql"`.
3. In the `schema_conn` fixture, replace `migrate_runner.apply(conn, REPO_MIGRATIONS_DIR)` (line ~285) with applying the fixture directly, e.g.:
   ```python
   with conn.cursor() as cur:
       cur.execute(_SCHEMA_FIXTURE.read_text())
   ```
   (The fixture's `CREATE TABLE/INDEX IF NOT EXISTS` are idempotent; the existing scrub-then-apply flow is preserved.)
4. Adapt the SCHEMA_VERSION lockstep test (lines ~849-857) that reads `benchmarks-website/server/src/schema.rs`: replace the file read + regex with a self-check that `post_ingest.SCHEMA_VERSION == 1` (the monorepo cannot read the website repo's `schema.rs`; the cross-repo lockstep is enforced by CONTRACT.md, not testable here). Keep the test function (rename its docstring to reflect the self-check) so the constant is still guarded.
5. Update the module docstring (lines ~5-9) to say the schema comes from the self-contained `_v4_schema_fixture.sql`, not the real `migrations/` via a migrate runner.

- [ ] **Step 5: Run the adapted testcontainer suite under Docker (green -- the load-bearing pre-merge gate)**

```bash
docker info >/dev/null 2>&1 && echo "docker OK" || echo "DOCKER MISSING -- escalate"
uv run --no-project --with pytest --with 'psycopg[binary]' --with boto3 --with xxhash --with testcontainers pytest scripts/test_post_ingest_postgres.py -q 2>&1 | tail -30
```

Expected: all tests PASS (the integration tests spin a `postgres:16-alpine` container, apply the fixture, and exercise the real upsert path; the mocked unit tests cover `connect_postgres`'s IAM/TLS/role enforcement). This is the load-bearing pre-merge confidence gate -- it proves the writer's upsert/transaction/measurement_id/NaN-guard behavior against a real Postgres with ZERO prod or develop dependency. If Docker is missing, escalate (do not let the suite silently skip -- this gate must actually run before merge).

- [ ] **Step 6: Lint + format the test and fixture**

```bash
uvx ruff check scripts/test_post_ingest_postgres.py
uvx ruff format --check scripts/test_post_ingest_postgres.py || uvx ruff format scripts/test_post_ingest_postgres.py
uvx ruff check scripts/test_post_ingest_postgres.py && uvx ruff format --check scripts/test_post_ingest_postgres.py
```

Expected: ruff check + format --check both exit 0. (The `.sql` fixture is not ruff-governed.) If `ruff format` changed the test, re-run Step 5 to confirm it still passes.

- [ ] **Step 7: Commit**

```bash
git add scripts/_v4_schema_fixture.sql scripts/test_post_ingest_postgres.py
git commit -F - <<'EOF'
scripts: add adapted testcontainer writer test + self-contained schema fixture

Port the testcontainer Postgres writer test and adapt it off the out-of-scope
migrations/ + migrate-schema.py + benchmarks-website/server/src/schema.rs: apply
a self-contained scripts/_v4_schema_fixture.sql (the 6-table DDL mirroring the
website repo's migrations/001) and self-check SCHEMA_VERSION == 1. The suite
spins postgres:16-alpine and exercises the real upsert/transaction path, the
load-bearing pre-merge confidence gate with no prod or develop dependency.

Signed-off-by: Connor Tsui <connor@spiraldb.com>
EOF
```

---

## Self-Review

- **Spec coverage:** post-ingest.py `--postgres` writer ported with v3 path intact (Task 1); revalidate test + cross-check utility ported (Task 2); testcontainer test adapted to the self-contained fixture + SCHEMA_VERSION self-check (Task 3). Out-of-scope (migrations/, migrate-schema.py, ci.yml, workflows) untouched.
- **Placeholder scan:** none -- every step has exact commands or exact edit instructions with line anchors.
- **Type consistency:** `ingest_postgres` / `connect_postgres` / `SCHEMA_VERSION` / `refresh_site_cache` names match across Task 1 (producer) and Tasks 2-3 (consumers); the fixture's table/column names match the upsert column lists in post-ingest.py and the golden `table` values from sub-phase 1.1.
