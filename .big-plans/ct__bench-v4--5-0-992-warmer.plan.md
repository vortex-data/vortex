# Warmer (#1 / PR-5.0.992) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the Vercel serverless function instance and its pooled Postgres connections warm so the first visitor stops paying the 6-11s cold-start.

**Architecture:** Two small, independent changes under `benchmarks-website/web/`: (1) a Vercel-native cron in `vercel.json` pings the public `/api/health` route every 2 minutes (fires pre-merge because `ct/bench-v4` pushes are production Vercel deploys), keeping a function instance alive; (2) the pg pool's `idleTimeoutMillis` is raised from the 10s default to 5 min so pooled connections survive between pings. The existing GH keep-warm cron is left untouched.

**Tech Stack:** Next.js 15 (App Router), `pg`, Vitest, Vercel Cron Jobs.

**Authoritative design spec:** `.big-plans/ct__bench-v4-warmer-design.md` (read it before starting). Scope is exactly the two changes above plus their tests — nothing more.

**Out of scope (do NOT touch):** `poolMax`, the Data Cache backstop (`lib/data-cache.ts`), CDN headers (`lib/cache.ts`), the `/api/revalidate` ops wiring, the existing `.github/workflows/web-keep-warm.yml` GH cron, and `?n=all` downsampling. No new API endpoint, no `CRON_SECRET`.

**Working directory for all commands:** `benchmarks-website/web/`.

---

## File Structure

- **Modify** `benchmarks-website/web/lib/db.ts` — add an exported `resolveIdleTimeoutMillis()` (mirrors `resolveSsl`), add `idleTimeoutMillis` to `DbConfig`, set it in `readConfig()` via the resolver, thread it into `createPool`'s `new Pool({…})`.
- **Modify** `benchmarks-website/web/lib/db.test.ts` — add `resolveIdleTimeoutMillis` unit tests; update the `iamConfig` literal to satisfy the widened `DbConfig`; add a threading test via `getPool()`.
- **Modify** `benchmarks-website/web/vercel.json` — add a top-level `crons` array; leave `$schema` + `headers` unchanged.
- **Create** `benchmarks-website/web/lib/vercel-config.test.ts` — regression guard asserting the cron entry targets `/api/health` on `*/2 * * * *`.

---

## Task 1: Raise pg `idleTimeoutMillis` (connection-warmth lever)

**Files:**
- Modify: `benchmarks-website/web/lib/db.ts`
- Test: `benchmarks-website/web/lib/db.test.ts`

The current `lib/db.ts` reads `poolMax` inline as `Number(process.env.BENCH_DB_POOL_MAX ?? '8')` inside `readConfig()`, and `ssl` via the exported `resolveSsl()` helper. We add `idleTimeoutMillis` using the `resolveSsl` pattern (an exported, unit-tested resolver), because unlike `poolMax` this value gets dedicated tests.

- [ ] **Step 1: Write the failing resolver tests**

Add this `describe` block to `benchmarks-website/web/lib/db.test.ts` (after the existing `describe('resolveSsl', …)` block), and add `resolveIdleTimeoutMillis` to the existing import from `'./db'` (the import list at the top of the file):

```ts
describe('resolveIdleTimeoutMillis', () => {
  afterEach(() => {
    delete process.env.BENCH_DB_IDLE_TIMEOUT_MS;
  });

  it('defaults to 300000 ms (5 min) when unset', () => {
    delete process.env.BENCH_DB_IDLE_TIMEOUT_MS;
    expect(resolveIdleTimeoutMillis()).toBe(300000);
  });

  it('honors a numeric override', () => {
    process.env.BENCH_DB_IDLE_TIMEOUT_MS = '60000';
    expect(resolveIdleTimeoutMillis()).toBe(60000);
  });

  it('throws (fails loud) on a non-numeric value rather than silently using NaN', () => {
    process.env.BENCH_DB_IDLE_TIMEOUT_MS = 'soon';
    expect(() => resolveIdleTimeoutMillis()).toThrow(/BENCH_DB_IDLE_TIMEOUT_MS/);
  });

  it('throws on a negative value', () => {
    process.env.BENCH_DB_IDLE_TIMEOUT_MS = '-1';
    expect(() => resolveIdleTimeoutMillis()).toThrow(/BENCH_DB_IDLE_TIMEOUT_MS/);
  });
});
```

The import line near the top of `db.test.ts` becomes:

```ts
import {
  buildQuery,
  passwordProvider,
  requireEnv,
  resetPool,
  resolveIdleTimeoutMillis,
  resolveSsl,
  sql,
  type DbConfig,
} from './db';
```

- [ ] **Step 2: Run the resolver tests to verify they fail**

Run: `pnpm test -- lib/db.test.ts -t resolveIdleTimeoutMillis`
Expected: FAIL — `resolveIdleTimeoutMillis` is not exported from `./db` (import/type error or "not a function").

- [ ] **Step 3: Implement `resolveIdleTimeoutMillis` in `lib/db.ts`**

Add this exported function to `benchmarks-website/web/lib/db.ts`, immediately after the `resolveSsl` function (before `readConfig`):

```ts
/**
 * Resolves the pool's idle-connection timeout in milliseconds from
 * `BENCH_DB_IDLE_TIMEOUT_MS`. Defaults to 300000 (5 minutes) so a pooled
 * connection survives the keep-warm cron's two-minute ping gap instead of pg's
 * 10s default, which would otherwise drop the connection between pings and make
 * the next request re-pay the RDS IAM-token + TLS connect even on a warm
 * function instance. `0` is accepted and means pg never times out an idle
 * client. A non-numeric or negative value fails loud rather than silently
 * becoming `NaN`. Exported for unit testing the parsing and default.
 */
export function resolveIdleTimeoutMillis(): number {
  const raw = process.env.BENCH_DB_IDLE_TIMEOUT_MS ?? '300000';
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(
      `Invalid \`BENCH_DB_IDLE_TIMEOUT_MS\` \`${raw}\`; expected a non-negative number of milliseconds.`,
    );
  }
  return value;
}
```

- [ ] **Step 4: Run the resolver tests to verify they pass**

Run: `pnpm test -- lib/db.test.ts -t resolveIdleTimeoutMillis`
Expected: PASS (4 tests).

- [ ] **Step 5: Thread `idleTimeoutMillis` through the config into the pool**

In `benchmarks-website/web/lib/db.ts`:

(a) Add the field to the `DbConfig` interface (after the `poolMax: number;` line):

```ts
  /** Idle-connection timeout (ms) for the pg pool; see `resolveIdleTimeoutMillis`. */
  idleTimeoutMillis: number;
```

(b) Set it in `readConfig()` (add to the returned object, after the `poolMax: …` line):

```ts
    idleTimeoutMillis: resolveIdleTimeoutMillis(),
```

(c) Pass it into `createPool`'s `new Pool({…})` (add after the `max: config.poolMax,` line):

```ts
    idleTimeoutMillis: config.idleTimeoutMillis,
```

- [ ] **Step 6: Update the `iamConfig` test literal to satisfy the widened `DbConfig`**

The `iamConfig` literal in `db.test.ts` (in `describe('db IAM auth path …')`) now omits a required field. Add `idleTimeoutMillis` to it:

```ts
  const iamConfig: DbConfig = {
    host: 'proxy.example.us-east-1.rds.amazonaws.com',
    port: 5432,
    database: 'bench',
    user: 'bench_reader',
    region: 'us-east-1',
    ssl: false,
    poolMax: 4,
    idleTimeoutMillis: 300000,
    staticPassword: undefined,
  };
```

- [ ] **Step 7: Add the pool-threading test**

Add this `describe` block to `db.test.ts` (after the `resolveIdleTimeoutMillis` block). It drives the real env → `readConfig` → `createPool` path through the exported `getPool()` (no new export needed) and asserts the value lands on the pool's options. `getPool` must be added to the `'./db'` import. A static password is set so the IAM/Signer path is skipped, and `BENCH_DB_SSL=disable` avoids needing a CA; `new Pool()` is lazy so this never opens a socket. All env is snapshotted and restored, and the singleton pool is reset before and after so it cannot leak into (or inherit from) other tests.

```ts
describe('createPool threads idleTimeoutMillis into the pg Pool (via getPool)', () => {
  const ENV_KEYS = [
    'BENCH_DB_HOST',
    'BENCH_DB_NAME',
    'BENCH_DB_USER',
    'BENCH_DB_PASSWORD',
    'BENCH_DB_SSL',
    'BENCH_DB_IDLE_TIMEOUT_MS',
    'BENCH_DB_PORT',
    'BENCH_DB_REGION',
  ] as const;
  const saved: Record<string, string | undefined> = {};

  beforeEach(async () => {
    for (const k of ENV_KEYS) saved[k] = process.env[k];
    await resetPool();
  });

  afterEach(async () => {
    await resetPool();
    for (const k of ENV_KEYS) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
  });

  it('uses the resolved idleTimeoutMillis as the pool option', () => {
    process.env.BENCH_DB_HOST = 'localhost';
    process.env.BENCH_DB_NAME = 'bench';
    process.env.BENCH_DB_USER = 'bench_reader';
    process.env.BENCH_DB_PASSWORD = 'fixture-pw'; // skips the IAM/Signer path
    process.env.BENCH_DB_SSL = 'disable'; // avoids the BENCH_DB_CA requirement
    process.env.BENCH_DB_IDLE_TIMEOUT_MS = '123456';

    // `pg`'s Pool exposes the resolved construction options at runtime but the
    // types do not surface `options`, so read it through a narrow cast.
    const pool = getPool() as unknown as { options: { idleTimeoutMillis?: number } };
    expect(pool.options.idleTimeoutMillis).toBe(123456);
  });
});
```

Add `getPool` to the import from `'./db'` (alongside `resetPool`). Also ensure `beforeEach` is imported from `vitest` (the file currently imports `afterAll, afterEach, beforeAll, describe, expect, it, vi`):

```ts
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
```

- [ ] **Step 8: Run the full db test file**

Run: `pnpm test -- lib/db.test.ts`
Expected: PASS. (The `describe.skipIf(!dockerAvailable())` testcontainers block is skipped locally where Docker is absent; the resolver, IAM-path, buildQuery, resolveSsl, requireEnv, and the new threading tests all run and pass.)

- [ ] **Step 9: Commit**

```bash
git add benchmarks-website/web/lib/db.ts benchmarks-website/web/lib/db.test.ts
git commit -F- <<'EOF'
feat: raise pg idleTimeoutMillis to keep the pool warm between cron pings (PR-5.0.992)

Adds resolveIdleTimeoutMillis (BENCH_DB_IDLE_TIMEOUT_MS, default 300000 ms) and
threads it through DbConfig/readConfig into the pg Pool so pooled connections
survive the keep-warm cron's */2 ping gap instead of pg's 10s default.

Signed-off-by: Connor Tsui <connor@spiraldb.com>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

---

## Task 2: Vercel keep-warm cron (function-warmth lever)

**Files:**
- Modify: `benchmarks-website/web/vercel.json`
- Create: `benchmarks-website/web/lib/vercel-config.test.ts`

- [ ] **Step 1: Write the failing cron-shape regression test**

Create `benchmarks-website/web/lib/vercel-config.test.ts`:

```ts
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

// Read vercel.json from disk (not an import) so the assertion pins the actual
// shipped config the way Vercel reads it.
const vercelConfig = JSON.parse(
  readFileSync(fileURLToPath(new URL('../vercel.json', import.meta.url)), 'utf8'),
) as { crons?: Array<{ path: string; schedule: string }> };

describe('vercel.json keep-warm cron', () => {
  it('pings /api/health every 2 minutes to keep the function + DB pool warm', () => {
    expect(vercelConfig.crons).toContainEqual({
      path: '/api/health',
      schedule: '*/2 * * * *',
    });
  });
});
```

- [ ] **Step 2: Run the cron test to verify it fails**

Run: `pnpm test -- lib/vercel-config.test.ts`
Expected: FAIL — `vercel.json` has no `crons` key yet, so `vercelConfig.crons` is `undefined` and `toContainEqual` throws.

- [ ] **Step 3: Add the `crons` array to `vercel.json`**

Edit `benchmarks-website/web/vercel.json` to add a top-level `crons` array between `$schema` and `headers` (leave `$schema` and the entire `headers` block exactly as they are):

```json
{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "crons": [{ "path": "/api/health", "schedule": "*/2 * * * *" }],
  "headers": [
    {
      "source": "/",
      "headers": [
        {
          "key": "Vercel-CDN-Cache-Control",
          "value": "max-age=300, stale-while-revalidate=86400"
        }
      ]
    },
    {
      "source": "/chart/:slug",
      "headers": [
        {
          "key": "Vercel-CDN-Cache-Control",
          "value": "max-age=300, stale-while-revalidate=86400"
        }
      ]
    }
  ]
}
```

- [ ] **Step 4: Run the cron test to verify it passes**

Run: `pnpm test -- lib/vercel-config.test.ts`
Expected: PASS.

- [ ] **Step 5: Normalize formatting**

Run: `pnpm format`
Then confirm nothing else drifted: `git diff --stat` should show only `vercel.json` and `lib/vercel-config.test.ts` (plus Task 1's files if not yet committed). `pnpm format` is the project's prettier-write script and covers `vercel.json` + `lib/**`.

- [ ] **Step 6: Commit**

```bash
git add benchmarks-website/web/vercel.json benchmarks-website/web/lib/vercel-config.test.ts
git commit -F- <<'EOF'
feat: add a Vercel keep-warm cron pinging /api/health every 2 min (PR-5.0.992)

A vercel.json crons entry hits the public /api/health route every two minutes.
It fires pre-merge because ct/bench-v4 pushes are production Vercel deploys, and
collectHealth's COUNT(*) fan-out warms the pool a cold-cache group bundle uses.

Signed-off-by: Connor Tsui <connor@spiraldb.com>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

---

## Task 3: Full local verification

**Files:** none (verification only).

- [ ] **Step 1: Format check**

Run: `pnpm format:check`
Expected: PASS (all files prettier-clean).

- [ ] **Step 2: Lint**

Run: `pnpm lint`
Expected: PASS (eslint clean).

- [ ] **Step 3: Type-check via the production build**

Run: `pnpm build`
Expected: PASS. `next build` runs `tsc`-level checks and must succeed with NO database env (every route is request-rendered).

- [ ] **Step 4: Run the full test suite**

Run: `pnpm test`
Expected: PASS. The testcontainers Postgres `describe.skipIf(!dockerAvailable())` block self-skips locally (Docker absent) and runs in CI via `web-deploy.yml`; all unit tests (incl. the new resolver, threading, and cron-shape tests) pass.

- [ ] **Step 5: Confirm scope**

Run: `git diff --stat <branch-point>..HEAD`
Expected: exactly four paths changed — `lib/db.ts`, `lib/db.test.ts`, `lib/vercel-config.test.ts`, `vercel.json`. Nothing under `lib/data-cache.ts`, `lib/cache.ts`, `app/api/revalidate/**`, or `.github/workflows/web-keep-warm.yml`.

---

## Post-implementation (handled by the big-plans orchestrator, NOT by SDD)

After SDD reports all tasks complete, big-plans runs the sub-phase **gauntlet pr-2** review (Step 2.3), then the **close + push** (Step 2.4). The push to `ct/bench-v4` fires `web-deploy.yml` (Check & Test + Deploy Production = `vercel deploy --prebuilt --prod`), shipping the `crons` block + the raised `idleTimeoutMillis` to the production deployment. **Deployment on this branch is a required acceptance criterion** (per the design spec): after the deploy, verify `GET https://benchmarks-web.vercel.app/api/health` returns 200, the Vercel cron is registered, and its first `*/2` invocation succeeds (cron registration/logs live in the Vercel dashboard, which may be the operator's to check).

---

## Self-Review

- **Spec coverage:** vercel.json cron → Task 2; `idleTimeoutMillis` raise via DbConfig/readConfig/createPool mirroring `poolMax` → Task 1; resolver + threading + cron-shape tests → Tasks 1-2; local verification (`format:check`/`lint`/`build`/`test`) → Task 3; deployment-on-branch acceptance → Post-implementation note. No out-of-scope file is touched.
- **Placeholder scan:** none — every code/step shows complete content.
- **Type consistency:** `resolveIdleTimeoutMillis` (named identically everywhere); `DbConfig.idleTimeoutMillis` set in `readConfig` and consumed in `createPool` and the `iamConfig` literal; `BENCH_DB_IDLE_TIMEOUT_MS` env name consistent across resolver, tests, and the threading test; the cron object `{ path: '/api/health', schedule: '*/2 * * * *' }` is identical in `vercel.json` and the regression test.
