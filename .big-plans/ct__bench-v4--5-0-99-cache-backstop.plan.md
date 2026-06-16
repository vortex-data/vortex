# Raise Vercel Data Cache Backstop (PR-5.0.99) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the Vercel Data Cache backstop from 1 hour to 24 hours so a CDN miss on the default `?n=100` window reads a still-warm Data Cache instead of paying the ~7.8s cold RDS fill — the user-confirmed cause of slow initial group opens on this low-traffic site.

**Architecture:** One constant in `benchmarks-website/web/lib/data-cache.ts` (`DATA_CACHE_BACKSTOP_SECONDS`) feeds every `unstable_cache` wrapper's `revalidate`. Change the constant `3600 -> 86400`, update its doc comment, and update the single test assertion that pins the literal value. No other code references the literal.

**Tech Stack:** Next.js `unstable_cache` (Vercel Data Cache), TypeScript, vitest.

**Tradeoff (accepted by the user 2026-06-15):** without the `POST /api/revalidate` ops wiring active, freshly ingested benchmark data can lag up to the backstop (24h). Benchmark data is low-frequency, trusted, and regenerable, so this is low-stakes; setting the ops wiring later (`BENCH_REVALIDATE_TOKEN` + `BENCH_SITE_BASE_URL`) restores immediate freshness via tag revalidation on each ingest, at which point the backstop is purely a safety net.

---

## File Structure

- Modify: `benchmarks-website/web/lib/data-cache.ts` — the `DATA_CACHE_BACKSTOP_SECONDS` constant + its doc comment.
- Modify: `benchmarks-website/web/lib/data-cache.test.ts` — the one assertion pinning the literal `3600`.

No new files. The change is intentionally minimal.

---

## Task 1: Raise the backstop constant + doc comment

**Files:**
- Modify: `benchmarks-website/web/lib/data-cache.ts:27-33`

- [ ] **Step 1: Update the doc comment + constant**

Replace this block:

```ts
/**
 * Backstop revalidation interval (seconds) for every cached read. The
 * post-ingest revalidate hook is the primary freshness mechanism; this bound
 * caps staleness at one hour if that hook ever fails to fire, so the layer
 * degrades to bounded staleness rather than serving stale data forever.
 */
export const DATA_CACHE_BACKSTOP_SECONDS = 3600;
```

with:

```ts
/**
 * Backstop revalidation interval (seconds) for every cached read. The
 * post-ingest revalidate hook is the primary freshness mechanism; this bound
 * caps staleness at twenty-four hours if that hook ever fails to fire, so the
 * layer degrades to bounded staleness rather than serving stale data forever.
 *
 * The window is one day rather than one hour because this is a low-traffic
 * site: a longer backstop keeps the default last-100 window warm across
 * overnight idle gaps, so a CDN miss reads this Data Cache instead of paying
 * the multi-second cold database fill. The freshness cost is bounded by the
 * revalidate hook, which flushes the tag on every ingest once its env is set.
 */
export const DATA_CACHE_BACKSTOP_SECONDS = 86400;
```

- [ ] **Step 2: Confirm nothing else hardcodes the old value**

Run: `grep -rn "3600" benchmarks-website/web/lib benchmarks-website/web/app`
Expected: no remaining `3600` in `lib/` or `app/` source (only the test assertion in Task 2, addressed next).

---

## Task 2: Update the test assertion

**Files:**
- Modify: `benchmarks-website/web/lib/data-cache.test.ts:55`

- [ ] **Step 1: Update the literal-value assertion**

Replace:

```ts
    expect(DATA_CACHE_BACKSTOP_SECONDS).toBe(3600);
```

with:

```ts
    expect(DATA_CACHE_BACKSTOP_SECONDS).toBe(86400);
```

(The assertion at `data-cache.test.ts:58`, `expect(call.options.revalidate).toBe(DATA_CACHE_BACKSTOP_SECONDS)`, references the constant and stays correct automatically — do not change it.)

- [ ] **Step 2: Run the data-cache test**

Run (from `benchmarks-website/web/`): `npx vitest run lib/data-cache.test.ts`
Expected: PASS (the TTL/tag test now asserts `86400`).

- [ ] **Step 3: Type-check**

Run (from `benchmarks-website/web/`): `npx tsc --noEmit`
Expected: clean (exit 0).

- [ ] **Step 4: Lint + format check the two touched files**

Run (from `benchmarks-website/web/`): `npm run lint && npm run format:check`
Expected: clean. (If `format:check` flags the edited comment wrapping, run `npm run format` and re-stage.)

- [ ] **Step 5: Commit**

```bash
git commit -F - <<'EOF'
perf: raise Vercel Data Cache backstop 1h->24h to cut cold group opens (PR-5.0.99)

On this low-traffic site the default ?n=100 group bundles fall out of the 1h
Data Cache between visits, so the first visitor pays the ~7.8s cold RDS fill.
A 24h backstop keeps the default window warm across overnight idle gaps, so a
CDN miss reads the Data Cache instead of the cold database. Freshness stays
bounded by the post-ingest revalidate hook once its env is wired.

Signed-off-by: Connor Tsui <connor@spiraldb.com>
EOF
```

---

## Notes for the implementer

- This change touches `benchmarks-website/web/**`, so the push fires `web-deploy.yml` (full vitest incl. the testcontainers Postgres suite, `next build`, then a production deploy). Locally, the narrow checks above (`vitest run lib/data-cache.test.ts`, `tsc --noEmit`, lint, format) are sufficient — the full suite + the Docker-gated integration tests run in CI. Do NOT run cargo/Rust checks (no Rust touched).
- Do not change `revalidate: DATA_CACHE_BACKSTOP_SECONDS` in `CACHE_OPTIONS` — it already reads the constant.
- Do not touch `READ_API_CACHE_CONTROL` in `lib/cache.ts` (the CDN `s-maxage`/`stale-while-revalidate`) — out of scope for this PR.

## Self-Review checklist (completed by plan author)

- **Spec coverage:** constant `3600 -> 86400` ✓; doc comment updated to reflect 24h + the cold-cache rationale ✓; test assertion updated ✓; no other reference to the literal ✓; CDN header untouched (out of scope) ✓.
- **Placeholder scan:** none — exact old/new strings inline.
- **Type/identifier consistency:** `DATA_CACHE_BACKSTOP_SECONDS` is the only identifier; `86400` = 24 × 3600.
