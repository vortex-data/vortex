# Read-path R1: recency-filter query_measurements chart reads on commit_timestamp — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `query_measurements` per-chart read seek the existing `idx_query_measurements_summary` index by filtering on the denormalized `commit_timestamp` instead of joining `commits` and scanning each chart's full ~18k-row history, so a bounded-window chart read returns its ~665-row window without the over-read (EXPLAIN-verified ~5x/chart, result-identical).

**Architecture:** One collector in `benchmarks-website/web/lib/queries.ts` (`collectQueryChart`) changes: (a) its data query drops `JOIN commits c`, orders by `q.commit_timestamp`, and uses a new `query_measurements`-scoped window filter that adds a sargable `q.commit_timestamp >= <cutoff>` predicate while keeping `q.commit_sha IN (last-n)` as the exact tie-trim; (b) its `buildEarliest` seed uses `MIN(q2.commit_timestamp)` directly instead of joining `commits`. The shared `factWindowFilter` and the other four collectors are untouched — only `query_measurements` carries `commit_timestamp`. No schema change; the index already exists (migrations 006/007). The change is behavior-preserving, so the test is a behavior-pinning equivalence guard (passes before and after; the "fix" is the query plan, not the result).

**Tech Stack:** TypeScript, `node-postgres`, Next.js route handlers, Vitest + `@testcontainers/postgresql`.

> **LOCAL-ENV NOTE (read before running anything):** Docker is **absent locally**, so the `@testcontainers/postgresql` suites in `queries.test.ts` are `describe.skipIf(!dockerAvailable())` and will **SKIP**, not run, on your machine. The DB assertions validate in **CI** when the branch is pushed (`web-deploy.yml` → "Check & Test"). Locally you verify only the static gates: `tsc --noEmit`, `eslint`, `prettier --check`. Do **not** expect a local red/green from the DB test; "verify it fails/passes" for DB tests means "confirm it compiles and reads correctly; CI runs it."

**Design spec (authoritative):** `.big-plans/ct__bench-v4-readpath-findings.md` (recommendation R1). Read it before starting.

---

## File Structure

- **Modify:** `benchmarks-website/web/lib/queries.ts`
  - Add one new helper `queryMeasurementWindowFilter` next to `factWindowFilter`.
  - Rewrite the data query + `buildEarliest` seed inside `collectQueryChart`.
  - Leave `factWindowFilter`, `seededCommitsInWindow`, and the other four collectors unchanged.
- **Modify (test):** `benchmarks-website/web/lib/queries.test.ts`
  - Add one `it(...)` to the existing `describe.skipIf(!dockerAvailable())('chartPayload (testcontainers Postgres)')` block asserting the full bounded-window (`?n=2`) payload.

---

## Task 1: Pin the bounded-window result with a full-payload equivalence test

This test asserts the exact `?n=2` payload for the fixture's TPC-H Q1 chart. It is a **behavior guard**: it must pass on the **current** code (in CI) and continue to pass after Task 2. It strengthens the existing `'caps commits with ?n …'` test (which only checks `commits.length` for `?n=1`) to a full commits + series + history equality for a window **smaller than** the fixture's commit count, which is exactly the boundary the `commit_timestamp` cutoff governs.

**Files:**
- Test: `benchmarks-website/web/lib/queries.test.ts` (add an `it` inside the existing `chartPayload (testcontainers Postgres)` describe block, near the existing `'caps commits with ?n …'` test ~line 170)

- [ ] **Step 1: Add the bounded-window equivalence test**

Add this test inside the `describe.skipIf(!dockerAvailable())('chartPayload (testcontainers Postgres)', …)` block (the fixture seeds three commits `'1'×40`@2026-04-23, `'2'×40`@2026-04-24, `'3'×40`@2026-04-25; Q1 has two series with values `1_000_000 + i*50_000` and `800_000 + i*50_000`). `parseCommitWindow`, `commitUrl`, and `QUERY_Q1` are already imported/defined in this file:

```ts
it('?n=2 selects exactly the two newest commits with correct values (commit_timestamp window)', async () => {
  // Pins the bounded-window result for a window SMALLER than the commit count
  // — the boundary the commit_timestamp cutoff + commit_sha tie-trim govern.
  // Must hold identically before and after the recency-filter refactor.
  const payload = await chartPayload(QUERY_Q1, parseCommitWindow('2'));
  expect(payload).toEqual({
    display_name: 'tpch sf=1 Q1 [nvme]',
    unit_kind: 'time_ns',
    history: { total_commits: 3, start_index: 1, loaded_commits: 2, complete: false },
    commits: [
      {
        sha: '2'.repeat(40),
        timestamp: '2026-04-24 12:00:00+00',
        message: 'second commit',
        url: commitUrl('2'.repeat(40)),
      },
      {
        sha: '3'.repeat(40),
        timestamp: '2026-04-25 12:00:00+00',
        message: 'third commit',
        url: commitUrl('3'.repeat(40)),
      },
    ],
    series: {
      'datafusion:vortex-file-compressed': [1_050_000, 1_100_000],
      'duckdb:parquet': [850_000, 900_000],
    },
    series_meta: {
      'datafusion:vortex-file-compressed': {
        engine: 'datafusion',
        format: 'vortex-file-compressed',
      },
      'duckdb:parquet': { engine: 'duckdb', format: 'parquet' },
    },
  });
});
```

- [ ] **Step 2: Verify the test compiles and lints (DB assertion runs in CI)**

Run:
```bash
cd benchmarks-website/web
npx tsc --noEmit
npx eslint lib/queries.test.ts
npx prettier --check lib/queries.test.ts
```
Expected: tsc clean, eslint clean, prettier reports the file formatted. (Vitest will SKIP the testcontainer block locally — no Docker. The DB assertion runs in CI on push.)

- [ ] **Step 3: Commit**

```bash
git add benchmarks-website/web/lib/queries.test.ts
git commit -F <msg-file>
```
Commit subject: `test: pin ?n=2 bounded-window query_measurements payload (PR-5.0.993)`
Include the DCO trailer `Signed-off-by: Connor Tsui <connor@spiraldb.com>` (write the message to a temp file and use `git commit -F`; the shell is fish and the repo has a DCO pre-push hook — do not inline backticks in `-m`).

---

## Task 2: Recency-filter `collectQueryChart` on `commit_timestamp` (drop the two `commits` joins)

**Files:**
- Modify: `benchmarks-website/web/lib/queries.ts`
  - Add `queryMeasurementWindowFilter` near `factWindowFilter` (~line 305).
  - Rewrite `collectQueryChart`'s `buildEarliest` seed and data query (~lines 329–365).

- [ ] **Step 1: Add the `query_measurements`-scoped window filter**

Insert this function immediately **after** `factWindowFilter` (do not modify `factWindowFilter`):

```ts
/**
 * Window filter for `query_measurements` charts. Unlike the shared
 * [`factWindowFilter`], this filters on the denormalized, indexed
 * `q.commit_timestamp` so the planner can seek `idx_query_measurements_summary`
 * (…, `commit_timestamp DESC`) instead of scanning the chart's full history and
 * post-filtering by `commit_sha`. The `>= cutoff` predicate (the timestamp of
 * the n-th newest commit) is the sargable lever; the `commit_sha IN (last-n)`
 * clause is kept as an exact tie-trim in case commits share the boundary
 * timestamp, so the result set is identical to [`factWindowFilter`]'s. Empty for
 * the unbounded `all` window. Only `query_measurements` carries
 * `commit_timestamp`, so this helper is not shared with the other collectors.
 */
function queryMeasurementWindowFilter(params: QueryParams, window: CommitWindow): string {
  const limit = commitWindowLimit(window);
  if (limit === null) {
    return '';
  }
  const n = params.bind(limit);
  return (
    ` AND q.commit_timestamp >= ` +
    `(SELECT min(timestamp) FROM ` +
    `(SELECT timestamp FROM commits ORDER BY timestamp DESC, commit_sha DESC LIMIT ${n}) w)` +
    ` AND q.commit_sha IN ` +
    `(SELECT commit_sha FROM commits ORDER BY timestamp DESC, commit_sha DESC LIMIT ${n})`
  );
}
```

Note: `params.bind(limit)` is called **once** and the returned `$n` placeholder is spliced into both subqueries — a positional parameter may legally appear multiple times in one statement, and this keeps the bind list to a single entry in textual order.

- [ ] **Step 2: Rewrite `collectQueryChart`'s seed and data query**

In `collectQueryChart`, replace the `buildEarliest` seed callback so it reads `MIN(q2.commit_timestamp)` directly (drop `JOIN commits c2`), and replace the data query so it drops `JOIN commits c USING (commit_sha)`, orders by `q.commit_timestamp`, and calls the new filter. The full updated function body for those two SQL blocks:

```ts
  const seeded = await seededCommitsInWindow(
    (p) =>
      `SELECT MIN(q2.commit_timestamp)
         FROM query_measurements q2
        WHERE q2.dataset = ${p.bind(dataset)}
          AND ${sargableDimEq(p, 'q2.dataset_variant', dataset_variant)}
          AND ${sargableDimEq(p, 'q2.scale_factor', scale_factor)}
          AND q2.storage = ${p.bind(storage)}
          AND q2.query_idx = ${p.bind(query_idx)}`,
    window,
  );
  if (seeded.commits.length === 0) {
    return null;
  }
  const acc = new SeriesAccumulator();
  acc.seedCommits(seeded.commits);

  const params = new QueryParams();
  const text = `
    SELECT q.commit_sha,
           q.engine, q.format, q.value_ns::float8 AS value
      FROM query_measurements q
     WHERE q.dataset = ${params.bind(dataset)}
       AND ${sargableDimEq(params, 'q.dataset_variant', dataset_variant)}
       AND ${sargableDimEq(params, 'q.scale_factor', scale_factor)}
       AND q.storage = ${params.bind(storage)}
       AND q.query_idx = ${params.bind(query_idx)}${queryMeasurementWindowFilter(params, window)}
     ORDER BY q.commit_timestamp, q.engine, q.format
  `;
```

Leave the rest of `collectQueryChart` (the row loop, name construction, `acc.finish(...)`) unchanged. Do **not** touch `collectCompressionTimeChart`, `collectCompressionSizeChart`, `collectRandomAccessChart`, or `collectVectorSearchChart` — they keep their `JOIN commits c` + `factWindowFilter(params, window)` because their tables have no `commit_timestamp`.

- [ ] **Step 3: Verify static gates (DB tests run in CI)**

Run:
```bash
cd benchmarks-website/web
npx tsc --noEmit
npx eslint lib/queries.ts
npx prettier --check lib/queries.ts
```
Expected: all clean. (The Task 1 equivalence test + the full existing `queries.test.ts`/`groups.test.ts` testcontainer suites validate the unchanged result in CI on push.)

- [ ] **Step 4: Sanity-check the diff against the scope contract**

Run `git diff benchmarks-website/web/lib/queries.ts` and confirm:
- `collectQueryChart` no longer contains `JOIN commits c` / `JOIN commits c2` and orders by `q.commit_timestamp`.
- `queryMeasurementWindowFilter` is the only new function; `factWindowFilter` is byte-identical to before.
- No other collector changed.

- [ ] **Step 5: Commit**

```bash
git add benchmarks-website/web/lib/queries.ts
git commit -F <msg-file>
```
Commit subject: `perf: recency-filter query_measurements reads on commit_timestamp (PR-5.0.993)`
Body should note: drops two `commits` joins; uses `idx_query_measurements_summary`; result-identical (the `commit_sha IN` tie-trim preserves exactness); `?n=all` path unchanged. Include `Signed-off-by: Connor Tsui <connor@spiraldb.com>`.

---

## Self-Review checklist (run before handing off to review)

1. **Spec coverage:** R1's two changes (data query + seed) are both in Task 2; the result-equivalence regression test is Task 1. R2 (collectGroupCharts single-group resolve) is explicitly out of scope. ✓
2. **Scope guard:** `factWindowFilter` and the four non-query collectors untouched (Task 2 Step 4 verifies). ✓
3. **Type consistency:** `queryMeasurementWindowFilter(params: QueryParams, window: CommitWindow): string` mirrors `factWindowFilter`'s signature; `commitWindowLimit`, `CommitWindow`, `QueryParams`, `sargableDimEq` are all already in scope in `queries.ts`. ✓
4. **`?n=all`:** `commitWindowLimit` returns `null` → helper returns `''` → unbounded full scan, unchanged. ✓
5. **Local-env honesty:** every "run the test" step states the DB suite skips locally and validates in CI. ✓

## Review

Review = `gauntlet pr-2` (single collector, behavior-preserving). The cumulative phase-so-far diff is reviewed; the key risks for the reviewer to probe are (a) result-equivalence vs the old `commit_sha IN` filter, (b) that the `commit_timestamp >= cutoff` predicate cannot drop a legitimate last-n commit, and (c) that no other collector regressed.
