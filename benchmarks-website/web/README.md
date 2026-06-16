# Benchmarks web (v4 read service)

Next.js 15 (App Router) read service for `benchmarks.vortex.dev`, serving the benchmark charts
from the benchmarks Postgres database. This is the v4 frontend that replaces both the v2
Vite/React SPA (`../../src/`) and the v3 Axum server (`../server/`) at the Phase-5 cutover; until
then it runs behind a dev-only Vercel domain.

## Local development

```bash
pnpm install
pnpm dev          # needs BENCH_DB_* pointing at a database (see below)

pnpm format:check # prettier
pnpm lint         # eslint
pnpm build        # next build; deliberately works WITHOUT a database
pnpm test         # vitest; the Postgres integration suite needs a Docker daemon
```

`next build` never touches the database: every page and route is request-rendered
(`force-dynamic` or request-URL-dependent), so builds are reproducible with no `BENCH_DB_*`
configured. Keep it that way; the CI `test` job builds with no database on purpose.

## Database environment

Connection config is read by `lib/db.ts`:

| Variable | Required | Meaning |
|---|---|---|
| `BENCH_DB_HOST` | yes | Postgres host. |
| `BENCH_DB_NAME` | yes | Database name. |
| `BENCH_DB_USER` | yes | Role to connect as. |
| `BENCH_DB_PORT` | no (5432) | Port. |
| `BENCH_DB_PASSWORD` | no | Static password. When unset, each new connection authenticates with a freshly minted RDS IAM token instead. |
| `BENCH_DB_REGION` | for IAM | AWS region for the RDS IAM signer; required when no password is set. IAM token signing also needs AWS credentials in the runtime environment. |
| `BENCH_DB_SSL` | no (`verify-full`) | `verify-full` validates the certificate chain and hostname; `disable` is for local non-TLS containers only. Any other value fails loudly. |
| `BENCH_DB_CA` | prod | PEM contents of the Amazon RDS CA bundle; Node's trust store does not include the RDS roots, so `verify-full` against RDS fails without it. |
| `BENCH_DB_POOL_MAX` | no (8) | Max pool connections per serverless instance; the per-render summary fan-out (`SUMMARY_CONCURRENCY`) is sized to this default. |

## CDN caching

The read paths serve traffic through Vercel's CDN with a five-minute freshness window, matching
the v2 site's S3 refresh cadence:

- The data routes (`/api/groups`, `/api/group/*`, `/api/chart/*`) set
  `Cache-Control: public, s-maxage=300, stale-while-revalidate=300` on success responses
  (`lib/cache.ts`); error responses omit the header so they are never CDN-cached. `/api/health`
  is deliberately uncached so the liveness probe always reflects the live database.
- The HTML pages (`/` and `/chart/:slug`) cannot set response headers from a server component,
  and Next.js emits `Cache-Control: no-store` for `force-dynamic` pages, which takes precedence
  over config-file `Cache-Control` rules. Instead, `vercel.json` sets `Vercel-CDN-Cache-Control`
  on those routes: that header is consumed (and stripped) by Vercel's CDN alone at the highest
  precedence, so the CDN caches the rendered pages while browsers still revalidate every load.

Verify on a live deployment with `curl -sI <url> | grep -i x-vercel-cache` (expect `MISS` then
`HIT` within five minutes). The deploy workflow runs this probe automatically after each deploy.
When deployment protection returns 401/403 on a deployment URL the probe skips with a notice;
production avoids that blind spot automatically once `BENCHMARKS_WEB_PROD_URL` is set (the probe
then targets the public domain, where a 401/403 fails the run instead), so the manual check is
only needed for protected previews and for production while the var is unset. One deliberate
divergence from the API routes: the `vercel.json` header rules apply to every response status,
so an unknown `/chart/:slug` 404 can be CDN-cached for up to five minutes, which stays within
the site's five-minute staleness budget (transient 5xx responses are believed not CDN-cacheable
by Vercel; if one ever were, the same five-minute budget would bound it).

## Deploys

`.github/workflows/web-deploy.yml` runs the check suite on every PR touching
`benchmarks-website/web/**`, `migrations/**` (the integration suite applies that DDL to its
testcontainer), or the deploy tooling itself (the workflow and its `verify-cdn-cache` composite
action), then deploys via the Vercel CLI (`vercel pull` / `vercel build` /
`vercel deploy --prebuilt`): a preview deployment per same-repo PR, and a
production deployment on each push to the deploy branch (`ct/bench-v4` during the migration;
flips to `develop` when the migration branch squash-merges).

One-time operator setup:

1. Create the Vercel project: Framework Next.js, **Root Directory `benchmarks-website/web`**,
   and the GitHub integration **disabled** (deploys are CLI-driven from CI; the integration
   would double-deploy).
2. Set the GitHub repo secret `VERCEL_TOKEN` (a Vercel deploy token) and repo variables
   `VERCEL_ORG_ID` + `VERCEL_PROJECT_ID` (from the Vercel project settings). Optionally set the
   repo variable `BENCHMARKS_WEB_PROD_URL` to the public production URL, as a full
   `https://<domain>` with no trailing slash: deployment protection never covers the public
   domain, so the post-deploy CDN probe can verify caching through it even when deployment URLs
   are protected (and a 401/403 from it fails the deploy run rather than skipping).
3. Configure `BENCH_DB_*` on the Vercel project (Production and Preview environments). Two open
   wiring choices are deliberately left to this step, per environment:
   - **Endpoint**: the RDS Proxy (`vortex-bench-proxy.proxy-*.us-east-1.rds.amazonaws.com`) is
     VPC-internal, so plain Vercel functions cannot reach it; without VPC connectivity for the
     Vercel project, use the public RDS instance endpoint (the same endpoint CI writers use).
   - **Auth**: a static `BENCH_DB_PASSWORD` for the read-only `bench_read` role. This is the
     currently supported mode: migration `005_read_role.sql` creates `bench_read` with **NO
     `rds_iam` grant** (and idempotently revokes it if a pre-existing role carries it), because
     on RDS `rds_iam` membership forces IAM-only auth and the Vercel runtime has no AWS
     credentials to mint IAM tokens. IAM auth is therefore **not available for `bench_read` as
     shipped**: enabling it would require BOTH a follow-up migration granting `rds_iam` to a
     read role (which atomically disables that role's password auth) AND AWS credentials in the
     function runtime (for example Vercel's OIDC federation to an AWS role with
     `rds-db:connect`).
