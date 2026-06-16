# Load-all (`?n=all`) cold-start + server-side downsampling — scoping for a fresh conversation (2026-06-15)

**Status: SCOPING ONLY — nothing implemented. The semantics are OPEN and to be DISCUSSED in a fresh
conversation (user's explicit request).** This doc frames the problem, the live measurements taken so
far, the candidate levers with their tradeoffs, and the open questions — so the next conversation can
have an informed discussion instead of re-deriving it.

## How this came up

After PR-5.0.993 (read-path R1) shipped and fixed the big-group cold *open* (`?n=100`, ~16s -> ~1.8s),
the user gave feedback (2026-06-15):
- The initial load is "a lot better" (R1 confirmed working).
- "Loading all chart data for a given chart is still pretty slow, mostly on cold start ... or rather
  cold start for a specific benchmark group? once I load all for the first chart the subsequent ones
  do seem to be faster."
- (The spinner complaint was a NON-issue: the spinner is gated off under `prefers-reduced-motion:
  reduce` by design (`globals.css:1235`); with Reduce motion OFF it spins fine. No work needed.)

The user is interested in **server-side downsampling** of `?n=all` but is "unsure what the semantics
are" — hence this scoping handoff.

## What R1 did and did NOT touch

R1 (`queryMeasurementWindowFilter` in `benchmarks-website/web/lib/queries.ts`) made BOUNDED windows
(`?n=100`, the default) sargable by filtering on the denormalized `commit_timestamp`, so a bounded
read seeks ~665 rows instead of scanning ~18k. **`?n=all` is deliberately UNCHANGED** — for the
unbounded window `commitWindowLimit` returns `null`, `queryMeasurementWindowFilter` returns `''`, and
the query is the same full-history scan as before. This is correct: "load all" genuinely needs every
row for that chart, so there is no over-read to remove (the over-read R1 killed was bounded-window
reads scanning full history; `?n=all` IS the full history).

## Measurements (live prod, 2026-06-15, function pre-warmed so this isolates the DB/payload layers)

Method: `?n=all` per single chart, CDN-busted with a `&z=<nanos>` param, function warmed first via
`/api/health`. `size_download` is uncompressed unless noted.

| Request | Time | Bytes | Note |
|---|---|---|---|
| appian chart1 (`?n=all`, cold group pages) | **0.53s** | 46,554 | first chart in the group |
| appian chart2 (`?n=all`, same group) | **0.095s** | 46,555 | ~6x faster — warm pages |
| appian chart3 (`?n=all`, same group) | 0.085s | 46,554 | warm |
| TPC-DS chart (`?n=all`, different group) | **0.95s** | 843,472 | cold again (diff group + bigger) |
| appian chart1 (`?n=all`, `Accept-Encoding: br`) | 0.22s | **14,371 on the wire** | brotli ~3.2x |

**Key findings:**
1. **The cold cost is the DB buffer-cache read of full history, NOT the payload.** chart1 and chart2
   ship the IDENTICAL 46.5KB, but chart1 (cold pages) is 0.53s and chart2 (warm pages) is 0.095s — the
   ~0.44s delta is purely the cold read of that chart's full-history pages into `shared_buffers`.
2. **Warming is per-GROUP** (matches the user): the first `?n=all` in a group warms the group's pages
   (charts in a group share dataset/storage and are co-located), so subsequent charts in the SAME
   group are ~6x faster; a DIFFERENT group is cold again. The function was already warm here, so this
   is DB-page warming, not Vercel function cold-start.
3. **The payload is already small on the wire** (brotli: 46.5KB -> 14.4KB; the 843KB TPC-DS chart is
   ~150-260KB brotli). So payload bytes are NOT the dominant cold cost on a normal connection.

## The crux for the downsampling idea

**Downsampling reduces the PAYLOAD, but the cold-start pain is the DB READ — and you must read all the
rows before you can downsample them.** So server-side downsampling would NOT speed up the cold start
the user is complaining about; it only shrinks the bytes sent to the client (after the DB read). The
two are different axes:

- **If the goal is the cold-start latency** (what the user described): downsampling does not help. The
  levers that would are (A) keep the group's pages warm, (B) cache `?n=all` responses so repeat loads
  skip the DB, (C) make the full-history read cheaper (index/heap), or (D) more DB cache/RAM so the
  full-history working set stays resident (the findings-doc "revisit RAM as query_measurements
  outgrows the ~1.8GB cache" watch-item — `?n=all` across all charts is a MUCH larger working set than
  the bounded-window reads R1 profiled, so RAM may actually matter HERE where it didn't there).
- **If the goal is payload/transfer** (slow connections, the original 2026-06-12 "1.1MB payload"
  framing): downsampling is the right lever — the client already LTTB-downsamples to ~500 points for
  display, so shipping ~1MB of raw points is mostly wasted. But it carries real semantic tradeoffs
  (below).

This DB-read-vs-payload distinction is the thing to settle first in the fresh conversation.

## Candidate levers (with tradeoffs) — to discuss, not yet decided

1. **Server-side `?n=all` downsampling (LTTB to N points/series).** Shrinks payload; the client already
   LTTBs to ~500. **Open semantics (the user's "unsure what the semantics are"):**
   - What does "load ALL data" MEAN if the server returns a downsampled view? Is a ~N-point summary
     acceptable, or do some users want exact per-commit values?
   - **Zoom**: the chart supports zooming into a sub-range; a global downsample to ~500-1000 points
     degrades zoomed-in detail. Options: downsample to a higher N (e.g. 2-4k) that keeps zoom usable;
     or viewport-aware re-fetch on zoom (more complex); or accept coarser zoom under "load all".
   - **Hover/tooltip exact values**: dropping points means hover only lands on retained points.
   - **Per-series vs shared x-axis**: charts have multiple series on a shared commit axis; LTTB picks
     different indices per series. Server-side must decide: downsample the shared commit axis (lose
     per-series fidelity) or per-series (bigger payload, union of axes). The client does per-series
     LTTB today.
   - Algorithm: LTTB (matches the client) is the natural choice.
   - Scope: ONLY `?n=all`; the default `?n=100` is bounded + cached and untouched.
2. **Cache `?n=all` responses** (currently force-dynamic / uncached; only `?n=100` is in the Vercel
   Data Cache). Would make repeat load-alls instant, but `?n=all` per chart is large and there are many
   charts — a lot of cache to hold; needs a sizing/eviction story. Could pair with a warm pass.
3. **Warm the group's pages** (extend the keep-warm to touch `?n=all` per group, or a periodic group
   scan). Directly targets the cold DB read, but the full-history working set is large and may exceed
   cache; warming all of it continuously is costly.
4. **Cheaper full-history read** — a covering index that makes the `?n=all` read index-only. The
   existing `idx_query_measurements_summary` INCLUDES `value_ns` and has the dims + `commit_timestamp
   DESC`, but the read also needs `commit_sha` (not in the index) for the x-axis, so it still heap-
   fetches. Adding `commit_sha` to the INCLUDE could make it index-only — at index-size cost. Needs an
   EXPLAIN check (like R1).
5. **Bigger DB instance / more RAM** — revisit ONLY if profiling shows the `?n=all` working set
   genuinely exceeds cache (physical I/O on the cold read). R1's profiling found near-zero physical I/O
   for bounded reads, but `?n=all` is a different, larger access pattern — re-measure ReadIOPS/
   ReadLatency during a cold `?n=all` fan-out before considering this.

## Suggested first steps for the fresh conversation (mirror R1's investigate-then-decide flow)

1. Profile a COLD `?n=all` read on live prod with EXPLAIN (ANALYZE, BUFFERS) + CloudWatch/PI: is the
   ~0.44s cold delta CPU (heap scan) or physical I/O (ReadIOPS/ReadLatency)? That decides whether the
   lever is index (4), RAM (5), warming (3), or caching (2).
2. Decide the PRIMARY goal: cold-start latency (DB read) vs payload bytes (downsampling). The user's
   words point at cold start, where downsampling does NOT help — so confirm what they actually want
   before building downsampling.
3. If downsampling is still wanted (for payload/slow-connection), resolve the open semantics above
   (zoom, hover, per-series axis, N) — this is a brainstorming/design discussion, ideal for the fresh
   conversation.
4. Then build via the established Amend flow (writing-plans -> SDD -> gauntlet pr-2 -> close -> push
   fires web-deploy.yml). Remember: Docker is absent locally so testcontainer tests validate in CI; do
   NOT ship unverifiable seeded tests (PR-5.0.993's CI-regression lesson — audit EVERY fixture).

## PROFILING RESULT (2026-06-16): cold `?n=all` IS physical-I/O-bound (answers step 1)

Profiled live prod (read-only, `bench-prod` AWS profile + PI + CloudWatch) per step 1. **The cold
`?n=all` cost is physical disk I/O from a working set that exceeds RAM, NOT CPU.** This is the
OPPOSITE of R1's bounded-window finding and CONFIRMS this doc's "RAM may actually matter HERE"
caveat. So for the cold-start goal the user described, the lever is RAM/index/caching, and
downsampling is OFF the table.

Method: warmed the Vercel function (`/api/health`), then a CDN-busted burst of `?n=all` GROUP bundles
across 12 big `query_measurements` groups (76s wall, 14:18:16..14:19:32Z). `?n=all` reads the DB
directly (uncached). TPC-DS (99 ch) alone = 25.6s / 83.7 MB; Clickbench (43 ch) = 9.2s.

Evidence during the burst:
- **PI db.load.avg by wait_event (1s):** TOTAL 3.3 avg / 8 peak active sessions on 2 vCPUs
  (oversubscribed). Dominant waits `IPC:BufferIO` 1.55 + `IO:DataFileRead` 1.05 (~2.6 AAS of I/O
  waiting) vs `CPU` only 0.54. About 80% of load is I/O-related, ~16% CPU.
- **PI top SQL:** the load is the `collectQueryChart` unbounded scan (`SELECT q.commit_sha,
  q.engine, q.format, q.value_ns FROM query_measurements q WHERE q.dataset=$1 ...`, no window
  filter) at ~2.8 of 3.3 AAS. Root cause confirmed.
- **CloudWatch (60s):** ReadIOPS 0 -> 1112/s, ReadThroughput 0 -> 13.5 MB/s, DiskQueueDepth 0 ->
  ~1.0, CPUUtilization 4.8% -> only 8.2%, ReadLatency ~1ms (gp3 is fast per-IO; the cost is read
  VOLUME). FreeableMemory stayed flat (~1.77 GB) under sustained 13.5 MB/s reads = pages CHURN
  through cache instead of accumulating = working set exceeds cache.
- **Sizing:** the whole DB is ~6 GB (FreeStorageSpace 13.94 GiB of 20 GiB allocated). Instance is
  db.t4g.medium (4 GiB RAM, shared_buffers ~1 GiB), so ~6 GB does NOT fit in cache. It WOULD fit
  entirely in a 16 GiB instance.

Lever re-assessment (was open; now evidence-ranked for the COLD-START goal):
- **RAM / bigger instance (lever 5): the root-cause fix, now JUSTIFIED.** The ~6 GB DB fits in a
  16 GiB instance (e.g. db.r7g.large / db.r6g.large), so the `?n=all` working set stays resident
  and cold reads vanish. Zero code, zero semantic tradeoff; costs more $/mo (an AWS cost/ops call).
  Even db.t4g.large (8 GiB) caches most of it.
- **Covering index / index-only scan (lever 4): partial, free, needs an EXPLAIN check.** Adding
  `commit_sha` to `idx_query_measurements_summary`'s INCLUDE could make the `?n=all` read index-only
  (the index is denser than the heap), cutting physical reads. Quantify with EXPLAIN (ANALYZE,
  BUFFERS) as `bench_read` before committing; the index grows on disk.
- **Cache `?n=all` (2) + warm pages (3): mitigations only.** They avoid the cold read on repeat or
  after-warm, but the FIRST cold load still pays, and continuous warming competes for the same
  scarce ~1 GB cache (you cannot keep a 6 GB DB warm in 1 GB). Most useful AFTER a RAM upsize or
  index fix.
- **Server-side downsampling (lever 1): does NOT help the cold start** (re-confirmed: you must read
  all rows from disk BEFORE downsampling, and the read IS the cost). It only shrinks PAYLOAD, a
  separate goal (slow connections) the user did not prioritize here.

DB profiling access: memory `project_bench_rds_profiling_access` (PI on; `bench_read` static pw in
Vercel env `BENCH_DB_PASSWORD`, needed for the EXPLAIN sizing of lever 4; the `vercel` CLI is not
installed locally).

## DECISION + EXECUTION (2026-06-16): RAM upsize to db.r7g.large

User chose the RAM upsize (the root-cause lever) over covering-index / cache-warm / downsampling /
stop. Target: db.t4g.medium (4 GiB) to db.r7g.large (2 vCPU / 16 GiB, memory-optimized Graviton3),
applied immediately (agent-run with the `bench-prod` creds, brief Single-AZ outage). No param-group
change needed: shared_buffers (`{DBInstanceClassMemory/32768}`) auto-scales to ~4 GiB and
effective_cache_size (`{/16384}`) to ~8 GiB, so the ~6 GB DB becomes fully resident. A manual
pre-resize snapshot is taken as insurance. Pricing: ~$47/mo to ~$174/mo (+~$127/mo). Validation:
re-run the cold `?n=all` burst + PI/CloudWatch and confirm ReadIOPS / IO-waits drop. Follow-up:
update `infra/provision.sh` + README cost table to reflect the new class.

### Validation (2026-06-16, post-resize on db.r7g.large)

Resize completed 14:45:40Z (db.t4g.medium to db.r7g.large; ~2 min Single-AZ reboot). shared_buffers
auto-scaled to ~4 GiB, effective_cache_size ~8 GiB. Re-profiled live prod:

- **Cross-group retention FIXED (the core complaint).** TPC-DS chart1 `?n=all` (843 KB) served
  ~0.25s; then after reading 6 OTHER big group bundles `?n=all` (cache pressure), the SAME chart
  re-read at ~0.24-0.26s, unchanged. On the old 4 GiB instance that pressure evicted the chart's
  pages ("different group is cold again"); on 16 GiB the working set stays resident.
- **Cold-ish per-chart read ~4x faster.** appian chart1 `?n=all` (46.5 KB, not in the warm-up
  bursts) read ~0.13s vs the old-instance cold 0.53s; warm ~0.10s (unchanged, always memory).
- **Instance healthy:** available, FreeableMemory ~9.9 GiB at rest (ample growth headroom), CPU idle.

Honest caveats:
- The very first read of never-cached data still pays a one-time disk fill (unavoidable without
  pre-warming), but it now STAYS cached (16 GiB holds the whole ~6 GB DB), so normal traffic + the
  keep-warm cron keep everything warm.
- A group `?n=all` BUNDLE (e.g. TPC-DS 99 charts, 83 MB) is still ~11s even warm, but that is
  app-layer cost (99 queries + 83 MB JSON serialize/transfer), NOT the DB read, and the client never
  requests the group `?n=all` bundle (it fetches per-chart). The per-chart `?n=all` the user
  complained about is now ~0.1-0.25s.
- Server-side downsampling remains UNBUILT and is only relevant if PAYLOAD / slow-connection becomes
  a goal (a separate axis); it does not affect this cold-start fix.

Pre-resize snapshot retained: `vortex-bench-prod-pre-r7g-resize-20260616`. Follow-up: reflect the new
class in `infra/provision.sh` + README cost table.

## Pointers
- Read path code: `benchmarks-website/web/lib/queries.ts` (`collectQueryChart`, `queryMeasurementWindowFilter`, `factWindowFilter`, `seededCommitsInWindow`), `lib/window.ts` (`commitWindowLimit`).
- Client LTTB: `lib/chart-format.ts` (`lttbIndices`); R1 findings: `.big-plans/ct__bench-v4-readpath-findings.md`.
- DB profiling access: memory `project_bench_rds_profiling_access` (PI enabled 2026-06-15; `bench_read` static pw).
- Everything else still HELD OFF per the user: PR-5.1 cutover (prod-write-gated), the develop rebase (#8362), the ops wiring.
