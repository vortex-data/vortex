# Parallelize the Group-Bundle Query Fan-out (PR-5.0.991) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the cold-cache latency of `GET /api/group/{slug}?n=100` (measured 8-19s cold for the big groups) by running the per-chart queries concurrently instead of one-at-a-time, with identical output.

**Architecture:** `collectGroupCharts` (`benchmarks-website/web/lib/queries.ts`) currently builds a group's bundle with a sequential `await` loop: one `chartPayload` SQL query per chart (99 for TPC-DS, 43 for Clickbench), each waiting for the previous. Only a Vercel Data Cache MISS pays this (a hit returns the whole assembled bundle from one cache entry). Replace the loop with an order-preserving `Promise.all(group.charts.map(...))`, bounded by the existing pg pool (`lib/db.ts` `max: 8`). ~99 serial round-trips become ~13 concurrent waves of 8 -> roughly an 8x cold-time reduction. Output (chart set, order, flattened shape, null-skip) is unchanged.

**Tech Stack:** TypeScript, `pg` Pool, Next.js Data Cache, vitest (+ testcontainers Postgres for the order test).

**Why parallelize, not batch-into-one-SQL:** rewriting all five per-chart collectors into a single batched query is the bigger win but a much larger, riskier change across heterogeneous chart shapes. Parallelizing is ~6 lines, output-identical, and captures most of the gain; the SQL batch can follow later if needed.

---

## File Structure

- Modify: `benchmarks-website/web/lib/queries.ts` — `collectGroupCharts` (the loop at ~1143-1151).

No new files. The existing integration test `lib/groups.test.ts:183` ("collectGroupCharts inlines flattened chart payloads for one group") already pins chart order (`['Q1','Q2']`) + the flattened payload, so it guards the output-equivalence of this refactor. It runs in CI (testcontainers Postgres; Docker absent locally).

---

## Task 1: Parallelize the per-chart fan-out (order-preserving)

**Files:**
- Modify: `benchmarks-website/web/lib/queries.ts` (`collectGroupCharts`)

- [ ] **Step 1: Replace the sequential loop**

Replace:

```ts
  const charts: NamedChartResponse[] = [];
  for (const link of group.charts) {
    const chartKey = chartKeyFromSlug(link.slug);
    const chart = await chartPayload(chartKey, window);
    if (chart === null) {
      continue;
    }
    charts.push({ name: link.name, slug: link.slug, ...chart });
  }
```

with:

```ts
  // Fetch every chart's payload concurrently rather than in a sequential
  // `await` loop: on a Data Cache MISS a large group otherwise pays one SQL
  // round-trip per chart back-to-back (99 for TPC-DS), which dominates the
  // cold-start latency. `Promise.all` over `group.charts.map(...)` preserves
  // order, and the shared pool (`lib/db.ts`, `max: 8`) bounds concurrency so
  // this issues at most 8 in-flight queries, not one per chart at once.
  const settled = await Promise.all(
    group.charts.map(async (link): Promise<NamedChartResponse | null> => {
      const chart = await chartPayload(chartKeyFromSlug(link.slug), window);
      return chart === null ? null : { name: link.name, slug: link.slug, ...chart };
    }),
  );
  const charts = settled.filter((chart): chart is NamedChartResponse => chart !== null);
```

Rationale notes (do NOT paste into the file beyond the comment above):
- `map` + `Promise.all` + `filter` all preserve array order, so the bundle's chart order is identical to the sequential version (the existing test pins `['Q1','Q2']`).
- The `filter` type-guard reproduces the old `continue` null-skip: null charts drop out, surviving charts keep their relative order.
- Error semantics are equivalent: the old loop let a `chartPayload` throw propagate and fail the bundle; `Promise.all` rejects on the first rejection and fails the bundle the same way.
- Do NOT change `lib/db.ts` `poolMax` (raising it is a separate RDS-connection-limit tuning decision, out of scope).
- Leave the `collectGroups()` call at the top of `collectGroupCharts` as-is (a possible later optimization, out of scope here).

- [ ] **Step 2: Type-check**

Run (from `benchmarks-website/web/`): `npx tsc --noEmit`
Expected: clean (exit 0). (Confirms the `NamedChartResponse | null` map callback + the `chart is NamedChartResponse` type-guard line up.)

- [ ] **Step 3: Lint + format the touched file**

Run (from `benchmarks-website/web/`): `npx eslint lib/queries.ts && npx prettier --check lib/queries.ts`
Expected: clean. (If prettier reflows the new block, run `npx prettier --write lib/queries.ts` and re-stage.)

- [ ] **Step 4: Run the non-Docker unit tests that don't need a container**

Run (from `benchmarks-website/web/`): `npx vitest run lib/slug.test.ts lib/window.test.ts`
Expected: PASS (sanity that the module still imports/builds; the order-pinning `groups.test.ts` integration test needs Docker and runs in CI).

- [ ] **Step 5: Commit**

```bash
git commit -F - <<'EOF'
perf: parallelize the group-bundle query fan-out to cut cold latency (PR-5.0.991)

collectGroupCharts ran one SQL query per chart in a sequential await loop, so a
Data Cache miss on a large group (99 charts for TPC-DS) paid ~99 serial DB
round-trips back-to-back (8-19s cold). Issue them concurrently with an
order-preserving Promise.all bounded by the existing pool (max 8); output
(chart set, order, flattened shape, null-skip) is unchanged and pinned by the
existing collectGroupCharts integration test.

Signed-off-by: Connor Tsui <connor@spiraldb.com>
EOF
```

---

## Notes for the implementer

- This change touches `benchmarks-website/web/**`, so the push fires `web-deploy.yml` (full vitest incl. the testcontainers Postgres order test, `next build`, then production deploy). Locally, `tsc`/eslint/prettier + the non-Docker unit tests are sufficient; the integration order test runs in CI.
- No Rust touched -> do NOT run cargo/clippy.

## Self-Review checklist (completed by plan author)

- **Spec coverage:** sequential loop -> order-preserving `Promise.all` ✓; pool-bounded concurrency (no poolMax change) ✓; null-skip preserved via type-guard `filter` ✓; output/order identical (guarded by `groups.test.ts:200`) ✓; `collectGroups()` left as-is ✓.
- **Placeholder scan:** none — exact old/new blocks inline.
- **Type/identifier consistency:** `NamedChartResponse`, `chartPayload`, `chartKeyFromSlug` all already imported/used in this file; the map callback returns `NamedChartResponse | null` and the `filter` narrows to `NamedChartResponse`.
