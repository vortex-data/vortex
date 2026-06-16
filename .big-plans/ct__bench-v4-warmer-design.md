# Warmer (#1) design spec — keep the Vercel function + DB pool warm

**Sub-PR:** PR-5.0.992 (the "warmer / #1"), inserted ahead of PR-5.1 via the big-plans Phase 5
Amend flow. **Review:** gauntlet `pr-2`. **Branch:** `ct/bench-v4`.

## Goal

Eliminate the dominant remaining cold-load cost for the *first* visitor to
`https://benchmarks-web.vercel.app`, by keeping the Vercel serverless **function instance** and its
pooled Postgres **connections** warm — not just the Vercel Data Cache (already handled by PR-5.0.97
+ PR-5.0.99).

## Context (already established — not re-derived here)

- PR-5.0.991 parallelized the per-chart query fan-out. The post-deploy measurement (2026-06-15)
  showed a cache-cold bundle on an *already-warm* function + connection is now ~0.8-1.5s, but the
  dominant remaining cost is the Vercel **function cold-start + cold DB connection** (RDS Proxy
  connect + IAM token mint + TLS): the first request to a freshly-spun-up instance is 6-11s, shared
  across URLs and chart-count-independent. A warm cache in front of a cold function still costs the
  first visitor several seconds.
- The existing GH Actions keep-warm cron (`.github/workflows/web-keep-warm.yml`, `*/5`) already pings
  the hot endpoints, but GitHub *scheduled* workflows run only from the **default** branch, so it is
  dormant on `ct/bench-v4` until merge. That dormancy is the entire reason this sub-PR exists.
- **Enabling fact:** this project's Vercel deploys are CLI-driven with the git integration disabled;
  pushes to `ct/bench-v4` run `vercel deploy --prebuilt --prod` — a **production** deploy. Vercel
  Cron Jobs run only against production deployments, so a `crons` entry in
  `benchmarks-website/web/vercel.json` fires **pre-merge** on this branch.

## Resolved decisions (from this brainstorming)

- **Vercel plan:** Pro (or above) — confirmed. Native cron sub-daily frequency is therefore allowed.
- **Mechanism:** Vercel-native cron in `vercel.json` (not an external pinger). It is in-repo, fires
  pre-merge given the prod-deploy fact, and needs no external account.
- **Cadence:** `*/2 * * * *` (every 2 minutes). Comfortably inside Vercel's idle-instance reclaim
  window; ~720 trivial invocations/day, negligible on Pro.
- **Warm target:** the existing public `GET /api/health`. No new endpoint and no `CRON_SECRET` (the
  route is already public read-only, consistent with the GH cron's no-secret design).
- **Connection lever:** raise pg `idleTimeoutMillis` to 5 min so pooled connections survive between
  `*/2` pings.
- **GH keep-warm cron:** unchanged — kept as a redundant warmer + uptime signal that activates at
  merge.

## Architecture

Two independent, complementary levers, plus one deliberate no-change:

1. **Function-instance warmth — Vercel cron.** A `crons` entry in `vercel.json` issues
   `GET /api/health` against the production deployment every 2 minutes. Pinging more often than
   Vercel reclaims an idle instance keeps a function instance alive.
2. **Connection warmth — pg `idleTimeoutMillis`.** pg's default idle timeout is 10s, so a pooled
   connection drops 10s after each ping; a user landing mid-gap re-pays the IAM-token + TLS connect
   even on a warm function. Raising it to 5 min (300000 ms) keeps the pool continuously warm between
   pings. `collectHealth` fans out a `Promise.all` of per-table `COUNT(*)` queries, so each ping
   exercises multiple pool connections at once — warming the same `max: 8` pool a cold-cache
   group-bundle fan-out (PR-5.0.991) will use.
3. **GH keep-warm cron — unchanged.** Redundant once the branch merges; it also doubles as an uptime
   signal. No edit.

## Components / changes (all under `benchmarks-website/web/`)

- **`vercel.json`** — add a top-level `crons` array:
  ```json
  "crons": [{ "path": "/api/health", "schedule": "*/2 * * * *" }]
  ```
  Keep the existing `$schema` and `headers` block unchanged.
- **`lib/db.ts`** — thread an `idleTimeoutMillis` through the config the same way `poolMax` already
  is: add the field to `DbConfig`, read `BENCH_DB_IDLE_TIMEOUT_MS` with a `300000` default in
  `readConfig()`, and pass it into `createPool`'s `new Pool({…})`. Default 300000 ms (5 min).
- No new endpoint, no new secret, no new env wiring required for the warmer to function (the
  `BENCH_DB_IDLE_TIMEOUT_MS` override is optional with a working default).

## Data flow

Vercel scheduler → `GET /api/health` (production) → `collectHealth()` checks out N pool connections
(warming the function + pool) → with `idleTimeoutMillis: 300000` those connections stay open until
the next ping re-uses them. A real first-visitor request then hits a warm function and a warm pool,
paying only the ~0.8-1.5s cache-cold query cost (PR-5.0.991), not the 6-11s cold-start.

## Error handling

A warm ping is best-effort. If the DB is unreachable, `/api/health` returns 500 and the cron
invocation is logged-failed in Vercel — no user impact, no special handling. Holding ≤8 idle
connections open for 5 min is negligible RDS load and well under RDS Proxy's ~30-min idle-client
timeout. IAM auth tokens authenticate only at connect time, so a long-lived connection is auth-safe
even past the ~15-min token validity window.

## Testing

- Unit-test the `idleTimeoutMillis` default/override resolution (mirroring the existing `resolveSsl`
  test style) and assert `createPool` threads the value into the pool's options.
- Add a small `vercel.json`-shape test asserting the cron entry targets `/api/health` on `*/2`
  (regression guard against accidental config edits).
- `tsc --noEmit`, eslint, prettier, and `next build` plus the full vitest suite (including the
  testcontainers Postgres integration suite) run in CI via `web-deploy.yml` on push.

## Deployment & post-deploy verification (REQUIRED — "deployed on this branch")

This sub-PR is not complete until the warmer is **live on `ct/bench-v4`**:

1. Push `ct/bench-v4` → `web-deploy.yml` runs Check & Test, then **Deploy Production**
   (`vercel deploy --prebuilt --prod`) so the new `vercel.json` `crons` block ships to the
   production deployment.
2. Confirm the production deploy succeeded (CI green) and `GET /api/health` responds 200 against
   `https://benchmarks-web.vercel.app`.
3. Confirm the Vercel cron is registered (Vercel dashboard → project → Cron Jobs, or the `vercel`
   CLI) and that the first scheduled `*/2` invocation of `/api/health` returns 200 in the Vercel
   cron logs. (Cron registration/logs live in the Vercel dashboard, which may be the operator's to
   check; the HTTP-level checks above are agent-runnable via `curl`.)
4. Optionally re-run the cold-start measurement a few minutes after the warmer is live to confirm
   the first-visitor path no longer pays the 6-11s cold-start.

## Out of scope

- Raising `poolMax` (a separate RDS-connection-limit tuning decision).
- Any change to the Data Cache backstop, CDN headers, or the `/api/revalidate` ops wiring.
- Editing or removing the GH keep-warm cron.
- The `?n=all` downsampling idea (dropped earlier as the wrong lever).

## Risks / known limits

Vercel may scale to multiple function instances under load; a single cron ping warms one, so the
warmer reduces but does not 100%-guarantee zero cold starts under concurrent multi-instance scaling.
On this low-traffic site there is effectively one instance, so it warms the typical first-visitor
path — which is exactly the reported complaint. This honest bound matches the spine's "never-cold"
framing.
