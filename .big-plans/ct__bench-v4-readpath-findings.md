# Read-path cold-fill investigation - findings + recommendations (2026-06-15)

Investigator: profiling session on live prod (`vortex-bench-prod`) + `benchmarks-web.vercel.app`.
Purpose: decide how (and whether) to fix slow first-open of big benchmark groups, and explicitly
answer "would a more powerful machine / more cores help". A **fresh agent will implement** the
accepted items below; this doc is the handoff. Nothing here was implemented.

## TL;DR (corrected conclusion - supersedes the old `finding_readpath_cold_fill_2026_06_15`)

The slow big-group open is **NOT** "DB-throughput-bound / concurrency ≈ 1 / needs more cores or
RAM". Measured root cause: **each per-chart query reads ~27× more rows than it needs** because the
recency window is applied via a `commits` join on `commit_sha` *after* a full-history scan, instead
of via the **denormalized, already-indexed `commit_timestamp`** column. That full-history scan
(~18k rows/chart × 99 charts × 2 queries) is the CPU load Performance Insights shows.

- **More cores: NO.** During the fan-out, RDS CPU is ~5% (60s avg) and **CPU credits never drop**;
  the DB CPU that exists is purely the over-read artifact. Fix the read and it evaporates.
- **More RAM / disk: NO (today).** `ReadIOPS` peaked at 17/min, `ReadLatency` ~0, `ReadThroughput`
  ~0.2 MB/s during a cold fan-out - essentially **zero physical I/O**; the working set is resident.
  RAM only becomes relevant if the table grows past the ~1.8 GB cache (future watch-item).
- **The fix is a *simplification*** (drops two `commits` joins, uses an existing index, no schema
  change), localized to one collector. Measured ~5× per-chart speedup, result-identical.

## Methodology

- AWS via the `bench-prod` CLI profile (acct 245040174862, us-east-1). See memory
  `project_bench_rds_profiling_access`.
- Read-only DB access as `bench_read` (static password) over TLS; all DB work was `SELECT` /
  `EXPLAIN (ANALYZE, BUFFERS)` - no writes.
- HTTP load via cache-bypassing windows: `?n=99..90` bypass both the Vercel Data Cache (only the
  default `?n=100` is cached) and the CDN (distinct query string), so each request runs a fresh
  server-side fan-out against RDS.
- Performance Insights was **enabled 2026-06-15** (7-day free retention) during this session.

## Instance facts

`vortex-bench-prod`: `db.t4g.medium` (2 vCPU, 4 GB), Postgres 16.4, gp3 20 GB / 3000 IOPS,
public endpoint. `shared_buffers` ≈ 910 MB, `effective_cache_size` ≈ 1.78 GB.
`query_measurements`: **4.85M rows, 2086 MB heap + 890 MB indexes**. `commits`: 4,545 rows (tiny).

## Measurements

### Resource envelope during a cold fan-out (CloudWatch + PI)
- CPU ~5% (60s avg), `CPUCreditBalance` pinned at max (576) - not CPU- or credit-bound.
- `ReadIOPS` ≤ 17/min, `ReadLatency` ~0, `ReadThroughput` ~0.2 MB/s - **~zero physical I/O**.
- PI `db.load.avg`: ~97% **CPU** wait, `IO:DataFileRead` ≈ 0. AAS peaked ~5.25 on 2 vCPU during
  the burst → the over-read makes the box briefly CPU-oversubscribed, but only because it scans
  ~18k rows/query.

### HTTP latency (cache-bypassing `?n=99..90`)
- Solo single chart: ~0.16 s. 8 concurrent: wall 0.40 s (eff. concurrency 2.2). 16 concurrent:
  wall 1.15 s (eff. concurrency 6.0). (Effective concurrency is 2–6, **not ≈1** as previously
  recorded; that HTTP number also includes Vercel multi-instance fan-out.)
- Full TPC-DS bundle (99 charts), warm RDS: **~4.7 s**. First hit of a cold window: **11.4 s** -
  that spike is a **Vercel function cold-start** (~6.7 s, the warmer's domain), confirmed by
  `ReadIOPS` staying ~0; it is NOT disk I/O.
- Cached (`?n=100` Data Cache hit): ~0.15 s.

### Per-chart query cost (EXPLAIN ANALYZE, BUFFERS - TPC-DS query_idx=1, n=99, warm)
| Query | Current | Optimized | Why |
|---|---|---|---|
| Seed `MIN` (chart's first commit) | **57 ms**, 18,199 buffers (18k-row heap scan + join to `commits`) | **8 ms**, 295 buffers, **index-only** | `MIN(commit_timestamp)` off the denormalized column, no join |
| Data query (the values) | **38 ms**, 18,004 buffers, 17,982 heap blocks → returns 665 | **9 ms**, 889 buffers, **665 heap blocks** | filter `commit_timestamp >= cutoff` (uses `idx_query_measurements_summary`) |
| Per chart total | ~95 ms | ~17 ms (~5×) | |

`× 99 charts`, CPU-bound on 2 cores ⇒ the ~4.7 s warm fan-out is dominated by these scans
(99 × 95 ms ÷ 2 cores ≈ 4.5 s). After the fix: ≈ 99 × 17 ms ÷ 2 ≈ 0.85 s.

### Result-equivalence (verified)
For query_idx ∈ {1,2,50,99}: current (`commit_sha IN last-99`), pure `commit_timestamp >= cutoff`,
and the combined form all return exactly **665 rows**. Only 1 commit sits at the boundary timestamp.
The combined form (`commit_timestamp >= cutoff AND commit_sha IN (last-99)`) is exact-correct AND
keeps the index plan (9 ms, 665 blocks). The pure `>=` form alone risks including extra rows only
if commits ever share an exact timestamp at the window boundary - keep the `commit_sha IN` tie-trim.

### Things measured and REJECTED
- **Single batched query for the whole group** (drop per-`query_idx` filter, partition in JS):
  measured **640 ms**, because without `query_idx` the planner uses the dims-only index and scans
  the group's entire 1.78M-row history, post-filtering to 65,835. Worse than 99 parallel
  index-optimal queries (~0.5 s wall) and adds JS-partitioning complexity. **Do not do this.**
- **More cores / bigger instance class**: rejected (see resource envelope).

### Other findings
- **All 99 charts in a group share the identical commit timeline** (same first/last commit, 2653
  commits each). So the 99 per-chart seed queries are redundant - but once the seed is 8 ms,
  deduping it into one group-level axis is **not worth the edge-case risk** (groups where charts
  diverge). Leave the per-chart seed.
- **`collectGroups()` costs ~0.74 s fresh** and the bundle re-runs it on every call to build **all
  ~17 groups' summaries** when it needs one (the `TODO(#7812)` in `collectGroupCharts`). Discovery
  itself is a skip-scan (~tens of ms, already optimized in PR-5.1.5); the summaries are the bulk.
- **Other fact tables** (`compression_times/sizes`, `random_access_times`, `vector_search_runs`)
  have **no `commit_timestamp` column** and only minimal indexes - so the recency-filter fix does
  NOT apply to them. They're tiny (≤25 charts, ≤248k rows) so their over-read is negligible. No
  action needed unless one ever grows large.

## Recommendations (prioritized)

### R1 - Recency-filter the `query_measurements` chart reads on `commit_timestamp` (PRIMARY)
Scope: `collectQueryChart` (data query) + its `buildEarliest` seed callback in
`benchmarks-website/web/lib/queries.ts`. For a bounded window:
- Data query: add `AND q.commit_timestamp >= (<cutoff: timestamp of the Nth-newest commit>)` and
  keep `AND q.commit_sha IN (<last-N commits>)` as the tie-trim. This lets the planner use
  `idx_query_measurements_summary`. The `commits` join for `c.timestamp` ordering can be dropped
  (order by `q.commit_timestamp` - denormalized, equal to `c.timestamp`).
- Seed `MIN`: replace `MIN(c.timestamp)` over the `commits` join with `MIN(q.commit_timestamp)`
  directly (index-only).
- Gated on a bounded window only; `?n=all` keeps the full scan (no cutoff) - unchanged.
- No schema change, no new index, uses existing `idx_query_measurements_summary`. Result-identical.
- Expected: per-chart ~95 ms → ~17 ms; cold TPC-DS bundle ~4.7 s (warm RDS) → ~1 s; kills the cold
  disk-read tail. Complexity-REDUCING (removes two joins).

### R2 - Resolve the single requested group in `collectGroupCharts` (SECONDARY)
Avoid re-running `collectGroups()` for all groups (the `TODO(#7812)`). Resolve just the requested
group's chart list + its one summary. Saves ~0.5–0.7 s per cold bundle. Moderate refactor; do only
if R1 alone leaves the bundle slower than desired.

### Rejected / not now
- Single batched group query (measured worse, more complex).
- More cores or RAM upsize (no CPU/credit/IO pressure today).
- Deduping the per-chart seed (cheap after R1; edge-case risk).
- Applying the fix to the small fact tables (negligible benefit).

### Future watch-items
- As `query_measurements` grows past the ~1.8 GB cache, physical I/O will start to matter - revisit
  RAM/instance sizing **then**, informed by PI (`IO:DataFileRead`) which is now enabled.
- `GET /api/group/{slug}?n=all` is a latent footgun (~48 MB / ~10 s full-group scan); the client
  never requests it, but the API allows it.

## Open items for the implementer
- Re-confirm the optimized plans on the actual code-generated SQL (the `QueryParams` builder), not
  just the hand-written SQL here.
- Decide the exact `cutoff` derivation (the Nth-newest commit's `timestamp`) and thread it through
  `factWindowFilter` / the seed builder.
- Add a regression test pinning result-equivalence (current vs optimized) for a representative chart.
- Review = `gauntlet pr-2` (single collector) or `pr-3` if R2 is bundled in.
