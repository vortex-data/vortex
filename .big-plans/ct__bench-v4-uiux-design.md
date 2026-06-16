# PR-5.0.9 design: opt-in full-history chart loading (UI/UX round)

Date: 2026-06-11. Approved by the user at the end of the brainstorming session. This is the design
for the UI/UX sub-PR the user queued ahead of PR-5.1 (see the spine's 2026-06-11 handoff sections).
Scope was pinned to "loading model only": no visual or layout redesign in this round.

## Problem

The v4 site already fetches a windowed `?n=100` per chart on group open (4-concurrent
`hydrationQueue`). The problem is what happens next: `onGroupOpen` in
`benchmarks-website/web/components/Chart.tsx` (the `ensureFullHistory` call after the initial
payload resolves) AUTOMATICALLY queues a background `?n=all` full-history fetch for EVERY chart in
the opened group through the 2-concurrent `fullHistoryQueue`. Nobody asked for that data; it is a
speculative warmup inherited from v3's shard-zero-then-warmup model.

Live measurements (2026-06-11, https://benchmarks-web.vercel.app, tpch SF=1 NVMe chart with
3,572 commits):

| Request | Cold function | Warm, CDN MISS | CDN HIT | Payload |
|---|---|---|---|---|
| `?n=100` | ~7.8s (one-time cold start) | 0.17-0.2s | 0.06s | 34KB |
| `?n=all` | (not separately measured) | 0.46-1.07s | 0.06s | 1.1MB |

Arithmetic of the warmup: opening the 22-chart tpch group queues ~24MB of full-history downloads;
"Expand All" (Header.tsx) opens every group and queues ~370 charts of `?n=all`, on the order of
hundreds of MB. These background fetches contend with the windowed fetches the user is actually
waiting on (server capacity, DB pool, bandwidth), which is the perceived "takes forever".

Two aggravating factors confirmed during investigation:

- The API cache policy is `s-maxage=300` with no stale-while-revalidate. On a low-traffic site the
  5-minute window nearly always lapses, so most visits MISS and the first request after idle also
  pays a multi-second Vercel cold start plus RDS connect.
- The server queries themselves are fast post-PR-5.1.5. This is a client fetch-orchestration
  problem, not a SQL problem.

## Key architectural fact (why late loading is safe here)

The user previously hit Chart.js jank ("slider becomes inaccurate, some things become sized
wrong") with load-small-then-load-more implementations. That failure is the axis re-base problem,
and v4 was specifically engineered around it: the windowed response carries
`history.total_commits` and `history.start_index`, and `normalizeChartPayload`
(`lib/chart-format.ts`) builds every chart on the FULL-length virtual x-axis from construction,
with `null` placeholders for the unloaded prefix. The slider max is the full history length from
the start (`syncSliderBounds`). When `?n=all` arrives, `replaceChartPayload` fills in the nulls;
nothing re-bases and the visible window is preserved. The interaction-promotion path also already
exists: `rangeTouchesUnloadedHistory` promotes the full fetch to `INTERACTION_FULL_PRIORITY` the
moment pan/zoom/slider touches the unloaded region. Both mechanisms are kept unchanged.

## Approved design

1. **Remove the automatic warmup.** Delete the `ensureFullHistory(priority)` call from
   `onGroupOpen` (Chart.tsx). Group open then costs only the windowed fetches (tpch: ~750KB
   total). Keep `fullHistoryQueue` and `FULL_HISTORY_CONCURRENCY = 2` as the bound for the opt-in
   fetches below. The permalink page (`app/chart/[slug]/page.tsx`) already upgrades only on
   interaction and needs no change.

2. **Window chip (always visible on windowed charts).** Each chart whose payload has
   `history.complete === false` shows a small per-card status chip: "latest 100 of 3,572". This
   signals at all times that the full view is not the default. Charts with complete history (fewer
   than 100 commits) show no chip. State machine:
   - windowed: "latest 100 of N"; on card hover the chip presents as the clickable action
     ("load all N").
   - loading: spinner in the chip while the full fetch is in flight (any trigger).
   - complete: "all N"; may settle to a quiet static label.
   - error: "retry" affordance (today a failed full fetch is console-only; the chip click retries).
   - Clicking the chip fetches at `INTERACTION_FULL_PRIORITY`.

3. **Hover intent, staged ("Both, staged" per user decision).** Hovering a chart card reveals the
   chip's action affordance immediately but fetches nothing. A continuous dwell of ~600ms on the
   same card starts a silent prefetch (chip shows the spinner) at a new mid-tier priority constant
   (above background 0, below `INTERACTION_FULL_PRIORITY`), so deliberate hovers have data ready
   by the time the user reaches for the slider, while mouse sweeps across the page fetch nothing.
   `pointerleave` cancels a pending dwell timer. Touch devices have no hover; the chip and the
   existing interaction promotion cover them. Full loads remain one chart at a time by
   construction (per-chart triggers plus the 2-concurrent queue bound).

4. **Interaction promotion kept unchanged.** Pan/zoom/slider touching the unloaded virtual region
   still fetches at top priority, so nobody hits a dead end at the 100-commit wall.

5. **CDN: add stale-while-revalidate.** Extend the API cache policy (`web/lib/cache.ts`, and the
   HTML-route rules in `web/vercel.json` if applicable) from `s-maxage=300` to
   `s-maxage=300, stale-while-revalidate=86400`. The CDN then serves the stale copy instantly and
   revalidates in the background. Benchmark data lands a few times a day, so day-scale staleness
   tolerance is acceptable. This makes both first paint and chip clicks feel instant for any
   recently-viewed chart and hides the cold-start path from most users. Verify the exact header
   Vercel's CDN honors (`Cache-Control` vs `Vercel-CDN-Cache-Control`) when implementing.

## Implementation sites

- `benchmarks-website/web/components/Chart.tsx`: remove the warmup call in `onGroupOpen`; add the
  chip DOM plus its state wiring (the per-card element registry and the `syncDownsampleBadge`
  pattern are the precedent); add card-level `pointerenter`/`pointerleave` dwell handling using
  the controller's existing AbortSignal listener pattern; route chip clicks and dwell fires into
  `ensureFullHistory`; update the stale docstrings (the "then queues the one-shot `?n=all`
  upgrade" header comment and the `ensureFullHistory` doc).
- `benchmarks-website/web/lib/chart-format.ts`: new constants (dwell ms, hover-prefetch priority);
  the existing `FULL_HISTORY_CONCURRENCY`, virtual-axis, and normalization code is unchanged.
- `benchmarks-website/web/lib/cache.ts` (+ `web/vercel.json` as needed): stale-while-revalidate.
- Tests under `benchmarks-website/web/` (vitest, node-env): see below.

## Out of scope (deliberately deferred, not data-correctness, so NOT added to the spine's
Deferred-work table)

- Viewport-based hydration on group open (windowed fetches are cheap enough at 34KB each).
- Server-side downsampling of `?n=all` payloads (1.1MB is acceptable for an explicit action).
- Any visual or layout redesign (user pinned scope to the loading model).
- Cold-start keep-warm infrastructure.

## Test plan

- Queue behavior: `onGroupOpen` schedules ZERO `fullHistoryQueue` entries (the discriminating
  test for the warmup removal); the hydration path is unchanged.
- Dwell: the prefetch fires after the dwell threshold and not before; `pointerleave` before the
  threshold cancels it; a second hover restarts cleanly.
- Chip: click promotes to `INTERACTION_FULL_PRIORITY`; state transitions windowed -> loading ->
  complete, and error -> retry; no chip when `history.complete` is true.
- Interaction promotion: existing `rangeTouchesUnloadedHistory` tests stay green.
- Existing tests that pin the old warmup ordering get updated to pin the NEW behavior (this is an
  intended behavioral change).
- Suite health: web vitest suite green, `next build`, `tsc`, `eslint`, `prettier`.
- Post-deploy manual verification: network profile of a tpch group open shows only `?n=100`
  requests (~750KB, no `?n=all` without intent); chip + dwell behave as designed; the API response
  carries the stale-while-revalidate directive and repeat visits serve from CDN.

## Acceptance criteria (mirrors the PR-5.0.9 spine row)

- No `?n=all` request is issued without per-chart user intent (hover dwell, chip click, or
  interaction touching unloaded history).
- Opening a 22-chart group transfers ~1MB or less of chart data (was ~24MB).
- Windowed charts visibly signal their window state via the chip; full loads show progress and
  surface failures with a retry.
- The virtual-axis upgrade path stays jank-free (visible window preserved across the fill-in).
- CDN policy includes stale-while-revalidate and repeat visits hit the CDN.
- Review: inner-loop 2-vote gauntlet (preset pr-2), per the project review calibration.

## Process note

The session handoff prescribed brainstorming then grill-me. Brainstorming ran to an approved
design; grill-me was skipped at user wrap-up. The load-bearing assumptions were instead verified
empirically during the session (live timing/payload measurements above, plus code reading of the
virtual-axis and promotion paths). If extra rigor is wanted, run spiral:grill-me on this document
before implementing; otherwise proceed to implementation.
