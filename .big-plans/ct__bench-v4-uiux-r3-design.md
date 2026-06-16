# PR-5.0.97 design: always-warm last-100 cache + full spinner coverage + fast Expand All (UI/UX round 3)

Date: 2026-06-12. Scoped with the user after PR-5.0.95 shipped, building on the read-only diagnosis
below (verified against the code, not speculation). This round adds a SERVER caching layer plus an
INGEST-side refresh hook, on top of client loading-model + spinner work — so unlike PR-5.0.9 /
PR-5.0.95 it is cross-cutting (client + server/auth + the production ingest script). Review:
**3-vote gauntlet preset pr-3**.

This document is authoritative for PR-5.0.97. The approved plan lives at
`~/.config/claude/plans/lets-continue-i-think-gleaming-meteor.md`.

## Problem (diagnosis, confirmed against the code)

PR-5.0.95 made group open lazy-hydrate only the ~visible charts (IntersectionObserver gating) and
added abort/timeout/retry + a spinner. The user reports the site is **still slow to load**, still
wants a spinner "when the data is loading", and wants **Expand All to load every chart's last 100
commits as fast as possible**. Three distinct, verified causes:

**1. Every CDN miss can pay the ~7.8s cold path; nothing keeps the last-100 payloads warm.**
- The read API connects to RDS per pool connection and mints a fresh RDS IAM auth token each time
  (`web/lib/db.ts` `passwordProvider` — tokens are ~15-min lived). On a cold Vercel function the
  first hit is Lambda cold start + IAM token mint + TLS connect + query ≈ **7.8s** (measured in
  PR-5.0.9). Warm it is ~0.2s MISS / 0.06s HIT.
- The only cache is the per-URL Vercel CDN: `READ_API_CACHE_CONTROL = 'public, s-maxage=300,
  stale-while-revalidate=86400'` (`web/lib/cache.ts`), applied to the 200s on `/api/chart`,
  `/api/group`, `/api/groups`. It is per-PoP and expires every 5 min, and on a low-traffic site the
  entries routinely fall out, so cold misses are the common case.
- **Nothing warms or refreshes any cache after an ingest** — no revalidation hook, no cron, no
  ISR/`unstable_cache`, no post-ingest warm pass. Pages are `force-dynamic`. The user's ask — "cache
  the last 100 commits always (and refresh on updates)" — has no mechanism today.

**2. Expand All still issues N per-chart requests per group; the bulk endpoint is unused.**
- The client fetches one `/api/chart/{slug}?n=100` per chart (`Chart.tsx:473`), gated by the
  PR-5.0.95 IntersectionObserver so OFF-screen charts do not even fetch until scrolled into view
  (`rootMargin '300px 0px'`, `hydrationQueue` concurrency 4). Expand All therefore loads only what is
  on screen; the rest stream in on scroll — the opposite of "load everything as fast as possible".
- A **bulk `GET /api/group/{slug}?n=100` endpoint already exists and is UNUSED by the client**
  (`web/app/api/group/[slug]/route.ts` → `collectGroupCharts`, `web/lib/queries.ts:1134`). It returns
  every chart in a group with full payloads inlined in ONE response.
- There is **no client-side payload cache** — closing and reopening a group refetches every chart
  (`Chart.tsx` `ensureInitialPayload` always fetches on a cache-less path).

**3. The spinner only covers the initial fetch; pre-hydration cards are blank.**
- PR-5.0.95's `.chart-spinner` renders only while `loading` is true during a `?n=100` fetch
  (`Chart.tsx` ~L1808 region). Before the IntersectionObserver fires (or before construction
  completes), the card is **blank white** — no placeholder at all. With lazy hydration this blank
  window is exactly what the user now perceives as "loading with no spinner".
- The spinner also **stops animating under `prefers-reduced-motion`** (the PR-5.0.95 guard replaces
  animation with a static state). If the user's OS has Reduce Motion on, they see no motion — and the
  blank pre-hydration card has nothing at all.

## User decisions (PINNED via AskUserQuestion, 2026-06-12)

- **Cache layer = Vercel Data Cache (`unstable_cache`) + a secret-protected `POST /api/revalidate`**
  called by `post-ingest.py`, plus a best-effort warm pass. Chosen over CDN-warming-only (per-PoP,
  5-min TTL → cold misses still hit RDS) and precomputed static JSON (new infra, second source of
  truth). This is the only option giving true refresh-on-update.
- **Spinner = RESPECT `prefers-reduced-motion`.** Keep the guard; under reduced motion show a STATIC
  spinner ring plus a visible "loading…" label (not nothing). Extend coverage to EVERY pre-data
  state, not just the in-flight fetch.

## Approved design

### A. Server "always-warm" Data Cache layer (new `web/lib/data-cache.ts`)

Wrap the **default-window (last-100) query path only** in `unstable_cache` (from `next/cache`; Next
15.5.19 — single-argument `revalidateTag(tag)`, not the canary two-arg form). One shared tag so a
single revalidation flushes the whole layer; a backstop TTL so a broken hook degrades to bounded
staleness, never forever-stale.

- `BENCH_DATA_TAG = 'bench-data'`, `DATA_CACHE_BACKSTOP_SECONDS = 3600`.
- `cachedDefaultGroupCharts(slug)` → `collectGroupCharts(groupKeyFromSlug(slug), { kind: 'last', n:
  DEFAULT_COMMIT_WINDOW })` (`queries.ts:1134`) — the hot path for the new client bundle fetch.
- `cachedDefaultChartPayload(slug)` → `chartPayload(key, { kind: 'last', n: DEFAULT_COMMIT_WINDOW })`
  (`queries.ts:579`) — for `/api/chart/{slug}?n=100` and the permalink page.
- `cachedGroups()` → `collectGroups` (`queries.ts:1096`); `cachedFilterUniverse()` →
  `collectFilterUniverse` (`queries.ts:1184`) — the landing page's per-request queries.
- Each wrapper passes `{ tags: [BENCH_DATA_TAG], revalidate: DATA_CACHE_BACKSTOP_SECONDS }`. Function
  ARGUMENTS are part of the cache key, so one `unstable_cache(fn, keyParts, opts)` per query covers
  all slugs. A cached `null` (404) is correct — a missing chart stays 404 until the next revalidate.
- **Routes / pages branch on the window**: in `web/app/api/chart/[slug]/route.ts` and
  `api/group/[slug]/route.ts`, after `parseCommitWindow` (`web/lib/window.ts`), use the cached
  function when `window.kind === 'last' && window.n === DEFAULT_COMMIT_WINDOW`; any other `?n=` keeps
  the existing direct query. `api/groups/route.ts` → `cachedGroups()`. `web/app/page.tsx` →
  `cachedGroups()` / `cachedFilterUniverse()`. `web/app/chart/[slug]/page.tsx` `getChart` →
  `cachedDefaultChartPayload` only when the parsed window is the default. Keep `force-dynamic`
  everywhere (it controls render mode, not `unstable_cache`) and keep the CDN headers
  (`READ_API_CACHE_CONTROL`) exactly as-is — worst-case staleness stays ≤5 min as today.
- Effect: even a CDN MISS no longer touches RDS for the default window — it reads the Data Cache
  (survives across invocations/regions; Vercel functions are single-region by default). The ~7.8s
  cold RDS path is eliminated for the last-100 window.
- **Risk (flag in PR description):** Vercel Data Cache item cap ≈ 2MB; the clickbench bundle ≈ 1.4MB
  uncompressed JSON — under the cap but close. If it ever trips, an over-limit item is silently not
  cached (degrades to today's direct query, no error); the fallback is to assemble
  `cachedDefaultGroupCharts` from per-chart cached entries.
- **Risk:** `unstable_cache` is superseded by `'use cache'` in later Next versions; fine on 15.5.
  Isolating all usage in `data-cache.ts` makes any future migration a one-file change.

### B. Refresh-on-update: `POST /api/revalidate` (new `web/app/api/revalidate/route.ts`)

- POST only. Auth: `authorization: Bearer <token>` compared against `process.env.BENCH_REVALIDATE_TOKEN`
  with `crypto.timingSafeEqual` over equal-length buffers (length-check first to avoid the throw).
- Missing env → `503 {error: 'not_configured'}` (**fail closed** — the route does nothing until the
  secret is wired). Bad/missing token → `401`. Success → `revalidateTag(BENCH_DATA_TAG)` → `200
  {revalidated: true}`.
- **Never** attach `READ_API_CACHE_CONTROL` — neither the 200 nor the errors may be CDN-cached.

### C. Post-ingest refresh + warm hook (`scripts/post-ingest.py`)

- New `refresh_site_cache(base_url, token, timeout)` using stdlib `urllib` + `concurrent.futures`:
  `POST {base}/api/revalidate` with the bearer token, then a best-effort warm pass — `GET
  {base}/api/groups`, parse the group slugs, `GET {base}/api/group/{slug}?n=100` for each (bounded
  concurrency 4, generous per-request timeout since the first warm request recomputes a whole
  bundle), plus `GET {base}/`. The warm pass pre-populates the freshly-invalidated Data Cache so even
  the first human request after an ingest is hot.
- **Every failure is caught, logged to stderr as a warning, and swallowed — the function returns
  `None` and can NEVER change the ingest exit code.** A cache refresh failing must not fail an ingest.
- Called at the end of the `--postgres` ingest path, after the write succeeds, **only when both
  `BENCH_SITE_BASE_URL` and `BENCH_REVALIDATE_TOKEN` are present** (silent no-op otherwise — the
  script stays inert until the env is wired).
- Workflows: two additive `env:` lines (`BENCH_SITE_BASE_URL: ${{ vars.BENCH_SITE_BASE_URL }}`,
  `BENCH_REVALIDATE_TOKEN: ${{ secrets.BENCH_REVALIDATE_TOKEN }}`) on the existing v4 Postgres ingest
  step in `bench.yml`, `sql-benchmarks.yml`, `v3-commit-metadata.yml`. Those steps are already
  `continue-on-error: true`. Keep the edits to 2-line additive insertions so PR-5.1's rewrite of
  these files carries them trivially. Run `yamllint --strict -c .yamllint.yaml` on the three files.

### D. Client: one group-bundle fetch + a session payload cache (fast Expand All)

- `web/lib/chart-store.ts`: add a payload cache + a bundle queue.
  - `const payloadCache = new Map<string, ChartResponse>()` with `getCachedPayload(slug)`,
    `primePayload(slug, payload)`, and a test-only `resetPayloadCache()`.
  - `export const bundleQueue: TaskQueue = makeQueue(BUNDLE_CONCURRENCY)` (new constant in
    `chart-format.ts`, value 3), reusing the existing `makeQueue` / `TaskQueue` machinery
    (`chart-store.ts:49-102`).
  - `ensureGroupBundle(groupSlug, priority): Promise<void>` with a per-group in-flight entry
    `{ entry, controller, failed }`: dedupe concurrent callers, bump priority via the existing
    `QueueEntry.priority` + `drain()` pattern, fetch `/api/group/{groupSlug}?n=100` with a fresh
    `AbortController` + `FETCH_TIMEOUT_MS` timer (mirror the per-chart pattern at `Chart.tsx`
    ~L485-503). On 200, `primePayload` every `charts[i].slug` + call `noteGroupSeries` per chart. On
    404, resolve as "bundle unavailable" (callers fall back per-chart). On failure, mark failed and
    clear the in-flight entry (no auto-retry). `abortGroupBundle(groupSlug)` aborts + clears the entry
    (idempotent; called on group close). Clearing the entry on abort lets a StrictMode remount or a
    reopen re-issue cleanly.
- `web/components/Chart.tsx`:
  - `ensureInitialPayload` (L450): first consult `getCachedPayload(this.slug)` — a synchronous hit
    seeds the EXISTING success path (normalize, `everWindowed`, `syncWindowChip`, `noteGroupSeries`)
    and resolves immediately. On a miss with a `groupSlug`: `await ensureGroupBundle(groupSlug,
    priority)` then re-check the cache; if still missing (bundle 404 / failed / slug absent), fall
    through to the EXISTING per-chart fetch unchanged. The permalink page (`groupSlug` undefined)
    keeps the per-chart path untouched.
  - Group-open effect (~L1744): on toggle-open kick `ensureGroupBundle(groupSlug, -index)` IMMEDIATELY
    (data arrives while the card is still off-screen) and set the loading state; keep the
    IntersectionObserver exactly as-is to gate `onGroupOpen` → `maybeConstruct` (the Chart.js CPU
    cost). `disarmHydration` additionally calls `abortGroupBundle(groupSlug)` alongside the existing
    `abortInFlightFetches()`.
  - Because the page-wide `index` (`data-chart-index`) is the priority, Expand All naturally drains
    bundles top-group-first at `BUNDLE_CONCURRENCY` — every chart's last-100 data loads eagerly,
    construction stays lazy on scroll. `retryInitialPayload` (L573) needs no change — it re-enters
    `ensureInitialPayload`, which re-checks the cache then re-attempts the bundle once (or falls back
    per-chart). `AbortError` stays the silent-cancellation sentinel everywhere.
- The payload cache is session-lifetime: close/reopen a group → zero fetches. Staleness after a
  server-side revalidation is accepted for an already-open tab (a refresh gets fresh data); a
  data-version invalidation is noted as future work, not built now.
- Payload math: clickbench ≈ 43 × ~34KB ≈ 1.4MB JSON uncompressed, but gzips to ~150-300KB on the
  wire and arrives as ONE request instead of 43 (whose 4-way concurrency serialized into ~11 waves).
  With the Data Cache behind it, one warm round-trip per group.

### E. Spinner coverage (every pre-data state)

- `web/components/Chart.tsx`: add `const [constructed, setConstructed] = useState(false)`, thread
  `setConstructed` through `CardCallbacks`, call it right after `state.chart = chart` in
  `maybeConstruct` (~L884). Render inside `.chart-wrap` (~L1941) whenever `!constructed && !error`:
  a `.chart-placeholder` with `role="status"`/`aria-live="polite"`, a spinner ring (`aria-hidden`),
  and a visible "loading…" label. This is **server-rendered**, so a pre-hydration card shows it with
  ZERO JS — never blank white. Keep the existing top-right `.chart-loading` pill (network signal;
  `Chart.loading.test.tsx` pins it) to avoid churning those assertions — pill = network in flight,
  placeholder = no chart yet.
- `web/app/globals.css` (near the existing spinner rules ~L1162-1214): `.chart-placeholder` centered
  in `.chart-wrap` (add `position: relative` to `.chart-wrap` if it lacks it), muted "pending" tint
  so the card reads as loading, reusing `@keyframes chart-spin`. Add `.chart-placeholder .chart-spinner`
  to the `@media (prefers-reduced-motion: reduce)` block so under reduced motion the **ring + label
  stay visible statically** (the user's chosen behavior) — never nothing.

## Out of scope (deferred; NOT data-correctness, so NOT added to the spine Deferred-work table)

- PR-5.1 (promote v4 `--postgres` ingest to required; prod RDS WRITE gate) — unchanged, queued after.
- Postgres read-path query optimization; caching the `?n=all` / non-default windows.
- Client data-version cache invalidation (an open tab keeps its session cache until refresh).
- Any visual/layout redesign beyond the placeholder + spinner coverage.

## Ops prerequisite (coordinate at execution; NOT a code blocker)

Generate the shared secret; set `BENCH_REVALIDATE_TOKEN` in the Vercel project env + as a GitHub
Actions secret, and `BENCH_SITE_BASE_URL` as an Actions var. Until set, the revalidate route 503s
fail-closed and the post-ingest hook is a silent no-op, so every piece degrades to current behavior
— the PR is safe to merge before the ops wiring lands.

## Test plan (web vitest jsdom/node + pytest; no Docker)

- **Data cache (A)**: new `web/lib/data-cache.test.ts` — `vi.mock('next/cache')`, assert each wrapper
  passes `tags: ['bench-data']` + the backstop TTL and that arguments key the cache. Route tests
  (node env, mock `@/lib/queries` + `@/lib/data-cache`): `?n=100` and missing `n` hit the cached
  function; `?n=all` / `?n=50` hit the direct function; 400/404 envelopes unchanged.
- **Revalidate route (B)**: new `web/app/api/revalidate/route.test.ts` — 503 when env unset; 401 on
  wrong/missing/length-mismatched token; 200 calls mocked `revalidateTag('bench-data')`; no
  `cache-control` header on any response.
- **Client bundle (D)**: extend `Chart.lazy-hydration.test.tsx` (MockIO + mocked fetch already there)
  — group open issues exactly ONE `/api/group/...?n=100` fetch for N islands; islands hydrate from it
  after IO fires; close aborts the bundle (signal aborted) and reopen re-issues; close/reopen AFTER
  success issues ZERO fetches (cache hit); a slug absent from the bundle falls back to one
  `/api/chart/...` fetch; a bundle 404 falls back per-chart; two groups opened together respect
  `BUNDLE_CONCURRENCY` + top-group priority. Reset module state per test (`resetPayloadCache`).
- **Spinner (E)**: extend `Chart.loading.test.tsx` — placeholder present in initial render (pre-fetch)
  and while fetch pending, removed after construction, not duplicated with the error state; extend
  `app/globals.spinner.test.ts` for `.chart-placeholder` + its reduced-motion static rule.
- **post-ingest (C)**: extend `scripts/test_post_ingest_postgres.py` (or a sibling
  `test_post_ingest_revalidate.py`) — `refresh_site_cache` sends the bearer header to
  `/api/revalidate`; a 500/timeout/connection error is swallowed (returns, no raise); `_main_postgres`
  exits 0 when refresh fails (mock `urllib.request.urlopen`); the hook is skipped when env is absent.
- **Regression**: the full existing suite (247) stays green, especially `Chart.lifecycle.test.tsx`
  (StrictMode double-mount vs the new bundle dedupe) and `server-smoke.test.ts` (`unstable_cache`
  under `next start`).
- **Gates**: `pnpm test`, `tsc --noEmit`, `pnpm build`, `pnpm lint`, `pnpm format:check`, `pytest`
  for `scripts/`, `yamllint --strict -c .yamllint.yaml` on the three workflows.

## Acceptance criteria (mirror the PR-5.0.97 spine row)

- The default `?n=100` window is served from the Vercel Data Cache (CDN misses no longer hit RDS for
  it); a `POST /api/revalidate` with the token flushes the tag (and post-ingest calls it + warms),
  giving true refresh-on-update; without the token it 401s, unconfigured it 503s.
- Expand All loads every chart's last-100 data eagerly via one `/api/group/{slug}?n=100` bundle per
  group (top-group-first), with Chart.js construction still lazy on scroll; close/reopen refetches
  nothing.
- Every pre-data card state shows a spinner placeholder (never blank white), with a static ring +
  visible label under `prefers-reduced-motion`.
- PR-5.0.9 / PR-5.0.95 behavior (opt-in full history, chip, dwell, IO gating, abort/timeout/retry) is
  unchanged.
- vitest + `tsc` + `next build` + eslint + prettier green; `pytest scripts/` green; yamllint green;
  3-vote gauntlet (pr-3) accepts.

## Sequencing

- Tasks A-C (server + ingest) and D-E (client) are independent work streams. Within the server
  stream, A → B → C is sequential (C calls B's endpoint; B invalidates A's tag). D and E are
  independent of each other. Ship as ONE PR with commits in that order so each commit is independently
  green.

## Process note

This round was scoped via parallel read-only exploration + a Plan-agent design pass, with the two
load-bearing decisions (cache mechanism, reduced-motion behavior) pinned via `AskUserQuestion` rather
than a fresh `brainstorming`/`grill-me` session. Execution slots this as PR-5.0.97, AHEAD of PR-5.1,
the same way PR-5.0.9 / PR-5.0.95 were inserted (the spine Amend flow).
