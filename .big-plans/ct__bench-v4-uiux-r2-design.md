# PR-5.0.95 design: lazy-hydration + resilient loading for large chart groups (UI/UX round 2)

Date: 2026-06-12. Scope agreed conversationally with the user after PR-5.0.9 shipped, building on
the live diagnosis below (verified against the code, not speculation). Like PR-5.0.9, scope is
pinned to the LOADING MODEL ONLY (plus one small new animation): no visual or layout redesign of the
charts themselves. Review: 2-vote gauntlet preset pr-2.

This document is authoritative for PR-5.0.95. The implementing conversation may optionally run
`superpowers:brainstorming` then `spiral:grill-me` on this doc to pin the open design decisions
(flagged inline) before writing code; otherwise implement directly, resolving the open decisions
with the recommendations given here.

## Problem (diagnosis, confirmed against the code)

PR-5.0.9 made full history opt-in, which removed the ~24MB `?n=all` warmup. But expanding a group
with many charts (e.g. clickbench, ~43 charts) still feels slow, and loading sometimes hangs. Two
distinct causes, both verified in `benchmarks-website/web/components/Chart.tsx`:

**1. Group open hydrates ALL charts at once, out of visual order, with no viewport gating.**
- Every chart island listens for its enclosing `<details>` `toggle` event; on open, all ~43 islands'
  `onGroupOpen` fire in the same tick (`Chart.tsx` `details` branch, ~L1615-1641), each scheduling a
  `?n=100` fetch on `hydrationQueue` (`HYDRATION_CONCURRENCY = 4`, `lib/chart-store.ts`).
- Order is NOT visual: `nextGroupOpenPriority()` increments per call in island-registration order
  and the queue drains highest-priority-first, so hydration tends to start from the BOTTOM/end of the
  group while the charts the user is looking at (the top) stay blank, then pop in out of order.
- No viewport gating: all 43 fetch on open even though only ~6 are visible. Cheap per fetch (34KB),
  but 43 × out-of-order × a cold Vercel function (first hit ~7.8s) feels slow. "Expand All"
  (`Header.tsx` `data-action="expand-all"`) is the pathological case: it opens every group and
  schedules every chart in every group at once.
- The permalink page ALREADY solves this with an `IntersectionObserver` (the `else` branch,
  `Chart.tsx` ~L1646-1658, `rootMargin: '150px 0px'`) that constructs a chart only when it scrolls
  near the viewport. The landing/`details` branch bypasses that and hydrates everything immediately.

**2. Fetches can hang with no recovery.**
- Both `fetch()` calls (`Chart.tsx` ~L463 initial `?n=100`, ~L550 full `?n=all`) pass NO
  `AbortSignal` and NO timeout. The controller has an `aborter` (`AbortController`, `Chart.tsx`
  ~L378) but it is wired only to DOM event listeners (wheel/strip pointers) and `destroy()`, NOT to
  the fetches. So a stalled request spins the "loading…" indicator forever, and closing/reopening a
  group does NOT cancel in-flight fetches — it piles more onto the server.
- The initial fetch never retries: the 4s auto-dismiss error path retries chart CONSTRUCTION, not the
  FETCH (`Chart.tsx` error-dismiss effect), so a failed/hung `?n=100` leaves that card stuck.
- The hang may also have a SERVER-side contributor (a 43-chart burst against the RDS pool
  `BENCH_DB_POOL_MAX = 8` plus Vercel function cold-starts). The client cannot tell which; see the
  pre-implementation investigation below.

**3. The loading state is a static text label**, not an animation: `{loading && <div
className="chart-loading">loading…</div>}` (`Chart.tsx` ~L1808). `.chart-loading` is styled in
`app/globals.css` (~L1163/1174) but has no animation. The user asked for a spinner.

## Approved design

### A. Viewport-based lazy hydration on the landing page (the highest-leverage change)

Gate the landing-page group charts' initial `?n=100` fetch + construct behind an
`IntersectionObserver`, reusing the proven permalink pattern. On group open, only charts near the
viewport hydrate; the rest hydrate as they scroll into view. Effect: top charts render first (visual
order), the clickbench burst drops from ~43 to ~6, server pressure (the likely hang cause) drops
sharply, and "Expand All" becomes cheap.

- In the mount effect's `details` branch (`Chart.tsx` ~L1615-1641): instead of calling
  `controller.onGroupOpen()` immediately on toggle-open AND on mount-if-open, arm an
  `IntersectionObserver` on the card when the group is open; when the card intersects, run the
  hydrate (`ensureInitialPayload` + `maybeConstruct` — i.e. what `onGroupOpen` does). Disconnect the
  observer when the group closes (the `toggle` handler) and on teardown (cleanups). Use a generous
  `rootMargin` (e.g. `200px 0px` to `400px 0px`) so charts hydrate slightly before they scroll in.
- Keep the PR-5.0.9 window chip + hover-dwell + interaction-promotion exactly as-is: they operate on
  an already-hydrated card, so they compose cleanly (a not-yet-hydrated card simply has no chip yet).
- **OPEN DECISION 1 — the group-summary bulk prefetch.** Today the summary `pointerenter`/`focusin`
  handler calls `controller.ensureInitialPayload(0, false)` for EVERY chart in the group
  (`Chart.tsx` ~L1628-1638), which would re-introduce the all-43 burst on hover and defeat lazy
  hydration. Recommended resolution: DROP the bulk summary-hover prefetch (or restrict it to the
  first N charts), and rely on the IntersectionObserver + the existing per-card hover-dwell. Decide
  this explicitly; if kept, it must be made viewport-aware so it cannot schedule off-screen charts.
- **OPEN DECISION 2 — keep `nextGroupOpenPriority` or simplify.** With IO gating, only visible charts
  schedule, so the burst is naturally small and order naturally visual; the reverse-order priority
  problem mostly disappears. Decide whether to keep the group-open priority stepping at all or
  simplify it (e.g. priority by viewport position). Low stakes either way.

### B. Fetch timeout + abort + retry (resilience; the "hangs" fix on the client side)

- Wire `this.aborter.signal` into BOTH `fetch()` calls so teardown/group-close cancels in-flight
  requests (frees server capacity; stops open/close from piling up load).
- Add a per-fetch timeout via a fresh `AbortController` (or `AbortSignal.timeout`) at a new constant
  (e.g. `FETCH_TIMEOUT_MS`, ~20000-30000) so a stalled request aborts instead of spinning forever.
  **OPEN DECISION 3** — exact timeout value + whether to use `AbortSignal.timeout` (cleaner) vs a
  manual `setTimeout`+abort (more control, must be cleared). Compose the per-fetch timeout signal
  with the controller's `aborter.signal` (e.g. `AbortSignal.any([aborter.signal, timeoutSignal])`).
- On initial-fetch timeout/failure, surface a RETRY affordance and make retry re-issue the FETCH
  (bounded), not just construction — mirror the PR-5.0.9 chip retry pattern (a clickable retry on the
  card's error/loading region). The full-history fetch already has the chip's retry; extend the same
  idea to the initial windowed fetch.
- Abort in-flight fetches on group close / IO disconnect / `destroy()`.

### C. Spinner animation

- Replace the static "loading…" text with a CSS spinner animation, and tie the chip's "loading"
  state (PR-5.0.9 `syncWindowChip` `data-state="loading"`) to a small inline spinner too for
  consistency. Add `@keyframes` + the spinner rule to `app/globals.css` near `.chart-loading`.
- **Respect `prefers-reduced-motion`**: under reduced motion, render a static indicator (no spin) so
  the change is accessible.
- Keep it lightweight and on-brand (match the existing token palette; this is NOT a visual redesign).

## Pre-implementation investigation (do this FIRST, ~10 min, read-only)

Before writing code, confirm whether the hangs are purely client-burst (which A fixes) or also need a
server-side fix. Open clickbench on the live staging site while watching Vercel function logs + RDS
`DatabaseConnections` (CloudWatch) / `pg_stat_activity` (via `bench_read`), OR time `/api/chart` for
~5 clickbench slugs cold vs warm. If a SPECIFIC chart query is slow (not just burst contention), note
it as separate follow-up work — do NOT expand PR-5.0.95's scope to server queries (that is read-path
perf, already covered by PR-5.1.5's machinery and out of this UI/UX round). `bench_read` reads are
authorized; any prod WRITE is not in scope here.

## Out of scope (deferred; not data-correctness, so NOT added to the spine Deferred-work table)

- Server-side query changes / `/api/chart` perf (separate from this UI/UX round; investigate per
  above and file separately if needed).
- Server-side `?n=all` downsampling.
- Any visual/layout redesign of the charts beyond the spinner animation.
- Changing the PR-5.0.9 chip / hover-dwell / interaction-promotion mechanics.
- Cold-start keep-warm infrastructure.

## Test plan (web vitest, jsdom/node; no Docker)

- **Lazy hydration**: on group open, only in-viewport charts schedule a `?n=100` fetch (mock
  `IntersectionObserver`; assert `hydrationQueue.schedule` / fetch count is bounded to the visible
  set, not the full group); an off-viewport chart schedules its fetch only when its observer fires.
- **Visual order**: a top-of-group chart hydrates before a bottom one (given the IO fires
  top-first).
- **Abort/timeout**: a fetch that never resolves is aborted at the timeout and the card surfaces a
  retry affordance (assert the fetch received an aborted signal); closing the group aborts in-flight
  fetches (assert `aborter`/signal aborted).
- **Retry**: clicking retry re-issues the initial `?n=100` fetch.
- **Spinner**: the loading state renders the spinner element; under a mocked
  `prefers-reduced-motion: reduce` the animation is disabled / static.
- **Regression**: all existing PR-5.0.9 tests (`Chart.loading.test.tsx`: warmup removal, chip states,
  dwell, terminal-404, race-guard) and the StrictMode lifecycle test stay green.
- **Gates**: web vitest + `tsc --noEmit` + `next build` + eslint + `prettier --check`.

## Acceptance criteria (mirror the PR-5.0.95 spine row)

- Expanding a ~43-chart group hydrates only the ~visible charts initially; the rest hydrate on
  scroll; the top charts render first (no more bottom-first / all-at-once burst).
- A stalled initial fetch times out and offers a retry rather than spinning forever; closing a group
  aborts its in-flight fetches.
- The loading state shows an animated spinner that respects `prefers-reduced-motion`.
- PR-5.0.9 behavior (opt-in full history, chip, dwell, terminal-404) is unchanged.
- vitest + `next build` + `tsc` + eslint + prettier green; 2-vote gauntlet (pr-2) accepts.

## Implementation sites (summary)

- `benchmarks-website/web/components/Chart.tsx`: IntersectionObserver-gated landing hydration in the
  mount effect `details` branch (reuse the `else`-branch IO shape); reconcile the summary-hover bulk
  prefetch (Open Decision 1); wire `aborter.signal` + a timeout into both `fetch()` calls; initial-
  fetch retry affordance; spinner markup for the loading state + chip loading state.
- `benchmarks-website/web/lib/chart-format.ts`: new constant(s) — `FETCH_TIMEOUT_MS` (and any IO
  `rootMargin` constant).
- `benchmarks-website/web/app/globals.css`: spinner `@keyframes` + rule + `prefers-reduced-motion`
  guard; tie `.chart-window-chip[data-state="loading"]` to the spinner.
- Tests under `benchmarks-website/web/components/` (extend `Chart.loading.test.tsx` and/or a new
  file; mock `IntersectionObserver` + fake timers for the timeout).

## Process note

This round was scoped conversationally with the user (the diagnosis above was presented and the three
improvements + the spinner were approved) rather than via a fresh `brainstorming` session. The open
decisions (1-3) are flagged for the implementing conversation; run `spiral:grill-me` on this doc if
extra rigor is wanted before implementing. Execution slots this as PR-5.0.95, AHEAD of PR-5.1, the
same way PR-5.0.9 was inserted.

## Pre-implementation investigation result (2026-06-12, read-only)

Resolved against the live site (`https://benchmarks-web.vercel.app`). The hangs are a CLIENT-side
burst plus cold-start plus no-recovery, NOT slow individual server queries, so the scope stays A+B+C
with no server-query work (exactly as designed):

- A single `/api/chart/{slug}?n=100` for clickbench is ~32KB and returns in 60-90ms even on a Vercel
  cache MISS.
- Firing ALL 43 clickbench charts concurrently against the warm function completes in ~0.75s wall
  with zero failures (slowest individual request ~93ms). The warm server absorbs the full burst
  trivially.
- Therefore the perceived slowness/hang is: (1) the cold Vercel function first-hit (~7.8s, measured
  in PR-5.0.9) amplified by hydrating all 43 charts out of visual order so the top charts the user is
  looking at resolve last; and (2) no timeout/abort/retry, so an unlucky stalled request spins
  "loading..." forever and group open/close piles more load on. A (lazy top-first hydration) and B
  (abort/timeout/retry) target exactly these. No specific chart query was slow; no server follow-up
  is filed.

## Open decisions RESOLVED (2026-06-12, pre-implementation)

These pin the three flagged decisions plus the B retry and C spinner mechanics, grounded in a read of
`Chart.tsx`, `lib/chart-store.ts`, `lib/chart-format.ts`, `app/globals.css`, and
`components/Chart.loading.test.tsx`.

- **Open Decision 1 (group-summary bulk prefetch): DROP it entirely.** The summary
  `pointerenter`/`focusin` handler at `Chart.tsx` ~L1627-1638 calls `ensureInitialPayload(0, false)`
  for every chart island, which would re-introduce the all-43 burst on a mere summary hover and
  defeat lazy hydration. With IO gating, opening the group already hydrates only the ~visible charts
  (each ~70ms), so the pre-warm's benefit is marginal and its cost is exactly the burst this PR
  removes. The per-card hover-dwell (`onCardHoverStart` -> `ensureFullHistory`, full `?n=all`) is a
  DIFFERENT mechanism and stays unchanged. Also drop the now-moot `+20` in `onGroupOpen` (`Chart.tsx`
  L518): it existed only to outrank the silent summary prefetch at priority 0, which is being removed.

- **Open Decision 2 (`nextGroupOpenPriority`): simplify to a visual-position priority.** Schedule
  each landing group-chart's initial `?n=100` fetch at a priority that puts TOP cards first. The
  island already receives its visual `index` within the group (rendered as `data-chart-index`), and
  the queue drains highest-priority-first, so scheduling at `priority = -index` (or any strictly
  decreasing function of `index`) makes the top card outrank lower ones and render first. Cross-group
  recency (the old `nextGroupOpenPriority` intent) is dropped: IO already bounds each group's burst to
  its visible cards, so whole-group races no longer happen. `nextGroupOpenPriority` /
  `GROUP_OPEN_PRIORITY_STEP` are used only at `Chart.tsx:517` (not in any test), so remove them if
  they become unused after this change; keep them only if still referenced.

- **Open Decision 3 (fetch timeout value + mechanism): manual per-fetch `AbortController` +
  `setTimeout`, `FETCH_TIMEOUT_MS = 30000`.** Do NOT use `AbortSignal.timeout` / `AbortSignal.any`:
  the test plan drives the timeout with fake timers, which a manual `setTimeout` makes deterministic
  and which `AbortSignal.timeout` does not. Per fetch: create a fresh `AbortController fc`; pass
  `fc.signal` to `fetch`; bridge the controller's existing `this.aborter.signal` to `fc.abort()` via a
  one-shot `'abort'` listener (and call `fc.abort()` immediately if `this.aborter.signal` is already
  aborted); arm `setTimeout(() => fc.abort(...), FETCH_TIMEOUT_MS)`; and in a `finally` clear the
  timer and remove the listener. Wire this into BOTH `fetch()` calls (the `?n=100` initial at
  `Chart.tsx` ~L463 and the `?n=all` full at ~L550) so group close / IO disconnect / `destroy()`
  cancels in-flight requests. 30s gives generous headroom over a genuinely cold-plus-contended
  first-hit (~7.8s) while still bounding a true hang, so a false abort of a slow-but-live request is
  very unlikely.

- **B retry affordance:** on initial-fetch failure or timeout, surface a CLICKABLE retry in the
  card's error region that re-issues the `?n=100` fetch (mirroring the PR-5.0.9 chip retry). It is
  user-initiated, so it is naturally bounded; there is no automatic fetch-retry loop. Note
  `.chart-error` currently has `pointer-events: none` (`globals.css` ~L1171), so the retry control
  must opt back into pointer events. The existing 4s auto-dismiss keeps retrying CONSTRUCTION only,
  unchanged.

- **C spinner:** render a CSS spinner inside `.chart-loading` (with visually-hidden "loading" text
  for accessibility) instead of the bare "loading..." string at `Chart.tsx` ~L1808; add the
  `@keyframes` + `.chart-spinner` rule near `.chart-loading` (`globals.css` ~L1174) and a
  `@media (prefers-reduced-motion: reduce)` block that renders a static indicator. Tie the chip's
  `data-state="loading"` (`globals.css` ~L1033) to a small inline spinner for consistency. This is a
  loading-state animation only, not a visual redesign.
