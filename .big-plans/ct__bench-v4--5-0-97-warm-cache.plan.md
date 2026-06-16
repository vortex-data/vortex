# PR-5.0.97: Always-Warm Last-100 Cache + Full Spinner Coverage + Fast Expand All — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the cold ~7.8s RDS path for the default `?n=100` window by caching it in Vercel's Data Cache (refreshed on ingest), make Expand All load every chart's last-100 fast via one bulk fetch per group into a session payload cache, and show a spinner placeholder for every pre-data card state.

**Architecture:** Two independent streams. Server stream: a new `web/lib/data-cache.ts` wraps the default-window queries in `unstable_cache` (tag `bench-data`, 1h backstop), a new `POST /api/revalidate` flushes the tag, and `scripts/post-ingest.py` calls it (best-effort) after each `--postgres` write. Client stream: `web/lib/chart-store.ts` gains a session payload `Map` filled by one `/api/group/{slug}?n=100` bundle fetch per group, and `web/components/Chart.tsx` gains a server-rendered `.chart-placeholder` for every pre-construction state. The authoritative design is `.big-plans/ct__bench-v4-uiux-r3-design.md`.

**Tech Stack:** Next.js 15.5.19 (App Router, `unstable_cache`/`revalidateTag` from `next/cache`), React 19, TypeScript, vitest 4 (jsdom + node), Python 3.11 stdlib (`urllib`, `concurrent.futures`), pnpm.

**Dependency ordering (explicit):**
- Server stream is sequential: **Task 1 → Task 2 → Task 3** (Task 3 calls Task 2's endpoint; Task 2 invalidates Task 1's tag).
- Client stream: **Task 4** and **Task 5** are independent of each other and of the server stream.
- **Task 6** (full-suite gate + PR description) runs last.
- Every task is independently committable and must leave `pnpm test` + `tsc` + `pnpm build` green.

**Project conventions (from CLAUDE.md / AGENTS.md):** full-sentence comments with periods, 100-column limit, **no em-dashes** in comments or docs, comments explain non-obvious logic only, tests use real assertions. Run gates that match the files touched; do not run Rust checks (this PR is TS + Python + YAML only).

---

## Task 1: Server Data-Cache layer (`web/lib/data-cache.ts` + route/page wiring)

**Files:**
- Create: `benchmarks-website/web/lib/data-cache.ts`
- Create: `benchmarks-website/web/lib/data-cache.test.ts`
- Modify: `benchmarks-website/web/app/api/chart/[slug]/route.ts`
- Modify: `benchmarks-website/web/app/api/group/[slug]/route.ts`
- Modify: `benchmarks-website/web/app/api/groups/route.ts`
- Modify: `benchmarks-website/web/app/page.tsx:43`
- Modify: `benchmarks-website/web/app/chart/[slug]/page.tsx:33-41`

**Context:** The default window is `{ kind: 'last', n: DEFAULT_COMMIT_WINDOW }` (`web/lib/window.ts:13`, value 100). `parseCommitWindow(null)` returns exactly that, so a missing `?n=` is the default. `chartPayload(key, window)` (`queries.ts:579`) and `collectGroupCharts(key, window)` (`queries.ts:1134`) take a key + window; `collectGroups()` (`queries.ts:1096`) and `collectFilterUniverse()` (`queries.ts:1184`) take no args. `groupKeyFromSlug` / `chartKeyFromSlug` come from `@/lib/slug`. Only the default window is cached; every other `?n=` keeps the direct query and rides the existing per-URL CDN cache.

- [ ] **Step 1: Write the failing test** (`benchmarks-website/web/lib/data-cache.test.ts`)

```ts
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// @vitest-environment node

import { afterEach, describe, expect, it, vi } from 'vitest';

// Capture the options unstable_cache is called with, and pass the wrapped
// function through unchanged so the cached wrappers still invoke the real query.
const cacheCalls: { keyParts: string[]; options: { tags?: string[]; revalidate?: number } }[] = [];
vi.mock('next/cache', () => ({
  unstable_cache: (
    fn: (...args: unknown[]) => unknown,
    keyParts: string[],
    options: { tags?: string[]; revalidate?: number },
  ) => {
    cacheCalls.push({ keyParts, options });
    return fn;
  },
}));

vi.mock('@/lib/queries', () => ({
  collectGroups: vi.fn(async () => [{ name: 'g', slug: 'gs', charts: [] }]),
  collectFilterUniverse: vi.fn(async () => ({ engines: [], formats: [] })),
  collectGroupCharts: vi.fn(async () => ({ name: 'g', charts: [] })),
  chartPayload: vi.fn(async () => ({ display_name: 'c' })),
}));

vi.mock('@/lib/slug', () => ({
  groupKeyFromSlug: (s: string) => ({ slug: s }),
  chartKeyFromSlug: (s: string) => ({ slug: s }),
  groupKeyToSlug: (k: { slug: string }) => k.slug,
}));

import {
  BENCH_DATA_TAG,
  DATA_CACHE_BACKSTOP_SECONDS,
  cachedDefaultChartPayload,
  cachedDefaultGroupCharts,
  cachedFilterUniverse,
  cachedGroups,
} from '@/lib/data-cache';

afterEach(() => {
  cacheCalls.length = 0;
});

describe('data-cache wrappers', () => {
  it('tags every wrapper with the shared bench-data tag and the backstop TTL', () => {
    expect(BENCH_DATA_TAG).toBe('bench-data');
    expect(DATA_CACHE_BACKSTOP_SECONDS).toBe(3600);
    for (const call of cacheCalls) {
      expect(call.options.tags).toEqual([BENCH_DATA_TAG]);
      expect(call.options.revalidate).toBe(DATA_CACHE_BACKSTOP_SECONDS);
    }
    expect(cacheCalls.length).toBeGreaterThanOrEqual(4);
  });

  it('invokes the wrapped query through the cached wrapper', async () => {
    await expect(cachedGroups()).resolves.toEqual([{ name: 'g', slug: 'gs', charts: [] }]);
    await expect(cachedFilterUniverse()).resolves.toEqual({ engines: [], formats: [] });
    await expect(cachedDefaultGroupCharts('gs')).resolves.toEqual({ name: 'g', charts: [] });
    await expect(cachedDefaultChartPayload('cs')).resolves.toEqual({ display_name: 'c' });
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd benchmarks-website/web && pnpm vitest run lib/data-cache.test.ts`
Expected: FAIL (`Cannot find module '@/lib/data-cache'`).

- [ ] **Step 3: Implement `benchmarks-website/web/lib/data-cache.ts`**

```ts
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { unstable_cache } from 'next/cache';

import {
  chartPayload,
  collectFilterUniverse,
  collectGroupCharts,
  collectGroups,
  type ChartResponse,
  type FilterUniverse,
  type Group,
  type GroupChartsResponse,
} from '@/lib/queries';
import { chartKeyFromSlug, groupKeyFromSlug } from '@/lib/slug';
import { DEFAULT_COMMIT_WINDOW } from '@/lib/window';

/**
 * The single revalidation tag shared by every cached read below. A successful
 * ingest flushes the whole layer with one [`revalidateTag`] call from
 * `POST /api/revalidate`, so newly ingested data shows up on the next request
 * rather than waiting out a TTL.
 */
export const BENCH_DATA_TAG = 'bench-data';

/**
 * Backstop revalidation interval (seconds) for every cached read. The
 * post-ingest revalidate hook is the primary freshness mechanism; this bound
 * caps staleness at one hour if that hook ever fails to fire, so the layer
 * degrades to bounded staleness rather than serving stale data forever.
 */
export const DATA_CACHE_BACKSTOP_SECONDS = 3600;

const CACHE_OPTIONS = { tags: [BENCH_DATA_TAG], revalidate: DATA_CACHE_BACKSTOP_SECONDS };

// The default last-100 group bundle, keyed by group slug. The slug is the cache
// key (an `unstable_cache` argument), so one wrapper covers every group. A
// `null` result (the group has no data) is cached too, which is correct: a
// missing group stays a 404 until the next ingest revalidates the tag.
const groupChartsCached = unstable_cache(
  async (slug: string): Promise<GroupChartsResponse | null> =>
    collectGroupCharts(groupKeyFromSlug(slug), { kind: 'last', n: DEFAULT_COMMIT_WINDOW }),
  ['data-cache:group-charts:n100'],
  CACHE_OPTIONS,
);

const chartPayloadCached = unstable_cache(
  async (slug: string): Promise<ChartResponse | null> =>
    chartPayload(chartKeyFromSlug(slug), { kind: 'last', n: DEFAULT_COMMIT_WINDOW }),
  ['data-cache:chart-payload:n100'],
  CACHE_OPTIONS,
);

const groupsCached = unstable_cache(
  async (): Promise<Group[]> => collectGroups(),
  ['data-cache:groups'],
  CACHE_OPTIONS,
);

const filterUniverseCached = unstable_cache(
  async (): Promise<FilterUniverse> => collectFilterUniverse(),
  ['data-cache:filter-universe'],
  CACHE_OPTIONS,
);

/** The default last-100 bundle for one group, served from the Data Cache. */
export function cachedDefaultGroupCharts(slug: string): Promise<GroupChartsResponse | null> {
  return groupChartsCached(slug);
}

/** The default last-100 payload for one chart, served from the Data Cache. */
export function cachedDefaultChartPayload(slug: string): Promise<ChartResponse | null> {
  return chartPayloadCached(slug);
}

/** Every group + chart link, served from the Data Cache. */
export function cachedGroups(): Promise<Group[]> {
  return groupsCached();
}

/** The filter chip universe, served from the Data Cache. */
export function cachedFilterUniverse(): Promise<FilterUniverse> {
  return filterUniverseCached();
}
```

Note: confirm `Group`, `FilterUniverse`, `ChartResponse`, `GroupChartsResponse` are exported from `@/lib/queries` (they are: `queries.ts` exports `Group` at L619, `GroupChartsResponse` at L643, `ChartResponse` is the chart payload type, and `FilterUniverse` is re-exported via `chart-format`). If `FilterUniverse` is only exported from `@/lib/chart-format`, import it from there instead.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd benchmarks-website/web && pnpm vitest run lib/data-cache.test.ts`
Expected: PASS.

- [ ] **Step 5: Wire the default-window branch into the chart route** (`benchmarks-website/web/app/api/chart/[slug]/route.ts`)

Replace the body's window+payload section so the default window reads the cache. The slug is already validated for the 400; the cached wrapper re-derives the key internally.

```ts
import { NextResponse } from 'next/server';

import { READ_API_CACHE_CONTROL } from '@/lib/cache';
import { cachedDefaultChartPayload } from '@/lib/data-cache';
import { chartPayload } from '@/lib/queries';
import { chartKeyFromSlug, type ChartKey } from '@/lib/slug';
import { DEFAULT_COMMIT_WINDOW, parseCommitWindow } from '@/lib/window';

// ... in GET, after the chartKeyFromSlug try/catch that yields `key`:
  const window = parseCommitWindow(new URL(request.url).searchParams.get('n'));
  // The default last-100 window is served from the Data Cache (warm across
  // invocations, refreshed on ingest); every other window keeps the direct
  // query and rides the per-URL CDN cache only.
  const payload =
    window.kind === 'last' && window.n === DEFAULT_COMMIT_WINDOW
      ? await cachedDefaultChartPayload(slug)
      : await chartPayload(key, window);
  if (payload === null) {
    return NextResponse.json({ error: 'not_found', message: 'chart not found' }, { status: 404 });
  }
  return NextResponse.json(payload, { headers: { 'cache-control': READ_API_CACHE_CONTROL } });
```

- [ ] **Step 6: Wire the default-window branch into the group route** (`benchmarks-website/web/app/api/group/[slug]/route.ts`)

```ts
import { READ_API_CACHE_CONTROL } from '@/lib/cache';
import { cachedDefaultGroupCharts } from '@/lib/data-cache';
import { collectGroupCharts } from '@/lib/queries';
import { groupKeyFromSlug, type GroupKey } from '@/lib/slug';
import { DEFAULT_COMMIT_WINDOW, parseCommitWindow } from '@/lib/window';

// ... after the groupKeyFromSlug try/catch that yields `key`:
  const window = parseCommitWindow(new URL(request.url).searchParams.get('n'));
  const payload =
    window.kind === 'last' && window.n === DEFAULT_COMMIT_WINDOW
      ? await cachedDefaultGroupCharts(slug)
      : await collectGroupCharts(key, window);
  if (payload === null) {
    return NextResponse.json({ error: 'not_found', message: 'group not found' }, { status: 404 });
  }
  return NextResponse.json(payload, { headers: { 'cache-control': READ_API_CACHE_CONTROL } });
```

- [ ] **Step 7: Wire `cachedGroups` into the groups route + the landing page, `cachedDefaultChartPayload` into the permalink page**

`benchmarks-website/web/app/api/groups/route.ts`: swap `collectGroups()` → `cachedGroups()` (import from `@/lib/data-cache`).

`benchmarks-website/web/app/page.tsx:43`:
```ts
import { cachedFilterUniverse, cachedGroups } from '@/lib/data-cache';
// ...
  const [groups, universe] = await Promise.all([cachedGroups(), cachedFilterUniverse()]);
```
(Drop the now-unused `collectFilterUniverse, collectGroups` import from `@/lib/queries`; keep `parseFilterCsv, singleSearchParam` from `@/lib/chart-format`.)

`benchmarks-website/web/app/chart/[slug]/page.tsx:33-41` (`getChart`): use the cache when the parsed window is the default.
```ts
import { cachedDefaultChartPayload } from '@/lib/data-cache';
import { DEFAULT_COMMIT_WINDOW, parseCommitWindow } from '@/lib/window';
// ...
const getChart = cache(async (slug: string, n: string | null): Promise<ChartResponse | null> => {
  let key: ChartKey;
  try {
    key = chartKeyFromSlug(slug);
  } catch {
    return null;
  }
  const window = parseCommitWindow(n);
  return window.kind === 'last' && window.n === DEFAULT_COMMIT_WINDOW
    ? cachedDefaultChartPayload(slug)
    : chartPayload(key, window);
});
```
Keep `export const dynamic = 'force-dynamic'` on both pages (it controls render mode, not `unstable_cache`).

- [ ] **Step 8: Add route-branch tests**

Extend or add `benchmarks-website/web/app/api/chart/[slug]/route.test.ts` and `.../group/[slug]/route.test.ts` (mirror the existing route-test setup if present; mock `@/lib/data-cache` and `@/lib/queries`). Assert: a request with no `n` and a request with `?n=100` call the cached function and NOT the direct query; `?n=all` and `?n=50` call the direct query and NOT the cached one; the 400 (bad slug) and 404 (null payload) envelopes are unchanged. If no route test file exists yet, create one following `data-cache.test.ts`'s node-env + `vi.mock` style.

- [ ] **Step 9: Run the gates for this task**

Run: `cd benchmarks-website/web && pnpm vitest run lib/data-cache.test.ts app/api && pnpm exec tsc --noEmit`
Expected: PASS. (`pnpm build` is run in Task 6 across the whole PR; run it here too if the server-smoke test is touched.)

- [ ] **Step 10: Commit**

```bash
git add benchmarks-website/web/lib/data-cache.ts benchmarks-website/web/lib/data-cache.test.ts \
  benchmarks-website/web/app/api/chart benchmarks-website/web/app/api/group \
  benchmarks-website/web/app/api/groups benchmarks-website/web/app/page.tsx \
  'benchmarks-website/web/app/chart/[slug]/page.tsx'
git commit -F - <<'EOF'
benchmarks-website: Vercel Data Cache layer for the default ?n=100 window (PR-5.0.97)

Signed-off-by: Connor Tsui <connor.tsui20@gmail.com>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

---

## Task 2: `POST /api/revalidate` endpoint (refresh-on-update)

**Files:**
- Create: `benchmarks-website/web/app/api/revalidate/route.ts`
- Create: `benchmarks-website/web/app/api/revalidate/route.test.ts`

**Context:** Bearer-token auth against `process.env.BENCH_REVALIDATE_TOKEN` with constant-time compare. Missing env → 503 fail-closed; bad token → 401; success → `revalidateTag(BENCH_DATA_TAG)` then 200. Never attach `READ_API_CACHE_CONTROL` (the response must not be CDN-cached). Depends on Task 1 (`BENCH_DATA_TAG`).

- [ ] **Step 1: Write the failing test** (`benchmarks-website/web/app/api/revalidate/route.test.ts`)

```ts
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// @vitest-environment node

import { afterEach, describe, expect, it, vi } from 'vitest';

const revalidateTag = vi.fn();
vi.mock('next/cache', () => ({ revalidateTag: (tag: string) => revalidateTag(tag) }));

import { POST } from '@/app/api/revalidate/route';

function post(token: string | null): Request {
  const headers = new Headers();
  if (token !== null) {
    headers.set('authorization', `Bearer ${token}`);
  }
  return new Request('http://localhost/api/revalidate', { method: 'POST', headers });
}

afterEach(() => {
  delete process.env.BENCH_REVALIDATE_TOKEN;
  revalidateTag.mockClear();
});

describe('POST /api/revalidate', () => {
  it('503s and does not revalidate when the token env is unset (fail closed)', async () => {
    const res = await POST(post('anything'));
    expect(res.status).toBe(503);
    expect(revalidateTag).not.toHaveBeenCalled();
    expect(res.headers.get('cache-control')).toBeNull();
  });

  it('401s on a missing or wrong token', async () => {
    process.env.BENCH_REVALIDATE_TOKEN = 'secret-token-value';
    expect((await POST(post(null))).status).toBe(401);
    expect((await POST(post('wrong'))).status).toBe(401);
    expect(revalidateTag).not.toHaveBeenCalled();
  });

  it('200s and revalidates the bench-data tag on the correct token', async () => {
    process.env.BENCH_REVALIDATE_TOKEN = 'secret-token-value';
    const res = await POST(post('secret-token-value'));
    expect(res.status).toBe(200);
    expect(revalidateTag).toHaveBeenCalledWith('bench-data');
    expect(res.headers.get('cache-control')).toBeNull();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd benchmarks-website/web && pnpm vitest run app/api/revalidate/route.test.ts`
Expected: FAIL (`Cannot find module '@/app/api/revalidate/route'`).

- [ ] **Step 3: Implement `benchmarks-website/web/app/api/revalidate/route.ts`**

```ts
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { timingSafeEqual } from 'node:crypto';

import { revalidateTag } from 'next/cache';
import { NextResponse } from 'next/server';

import { BENCH_DATA_TAG } from '@/lib/data-cache';

/**
 * `POST /api/revalidate` flushes the [`BENCH_DATA_TAG`] Data Cache entries so the
 * next read recomputes against freshly ingested data. `scripts/post-ingest.py`
 * calls this after a successful Postgres write. Auth is a bearer token compared
 * in constant time against `BENCH_REVALIDATE_TOKEN`; a missing env var fails
 * closed with `503` so an unconfigured deployment never silently accepts
 * unauthenticated revalidation. The response is never CDN-cached.
 */
export async function POST(request: Request): Promise<NextResponse> {
  const expected = process.env.BENCH_REVALIDATE_TOKEN;
  if (expected === undefined || expected === '') {
    return NextResponse.json({ error: 'not_configured' }, { status: 503 });
  }
  const header = request.headers.get('authorization');
  const provided = header?.startsWith('Bearer ') ? header.slice('Bearer '.length) : null;
  if (provided === null || !constantTimeEquals(provided, expected)) {
    return NextResponse.json({ error: 'unauthorized' }, { status: 401 });
  }
  revalidateTag(BENCH_DATA_TAG);
  return NextResponse.json({ revalidated: true }, { status: 200 });
}

/**
 * Constant-time string compare. `timingSafeEqual` throws on length mismatch, so
 * the length is checked first; returning early on a length difference leaks only
 * the token length, not its contents.
 */
function constantTimeEquals(a: string, b: string): boolean {
  const ab = Buffer.from(a);
  const bb = Buffer.from(b);
  if (ab.length !== bb.length) {
    return false;
  }
  return timingSafeEqual(ab, bb);
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd benchmarks-website/web && pnpm vitest run app/api/revalidate/route.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks-website/web/app/api/revalidate
git commit -F - <<'EOF'
benchmarks-website: POST /api/revalidate to flush the Data Cache on ingest (PR-5.0.97)

Signed-off-by: Connor Tsui <connor.tsui20@gmail.com>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

---

## Task 3: post-ingest refresh hook + workflow env wiring

**Files:**
- Modify: `scripts/post-ingest.py` (add `refresh_site_cache` + helpers, call from `_main_postgres`)
- Create: `scripts/test_post_ingest_revalidate.py`
- Modify: `.github/workflows/bench.yml:143-146`
- Modify: `.github/workflows/sql-benchmarks.yml` (the v4 Postgres step's `env:` map)
- Modify: `.github/workflows/v3-commit-metadata.yml` (the v4 Postgres step's `env:` map)

**Context:** `_main_postgres` (`post-ingest.py:1089`) runs the ingest then `print(...)` + `return 0`. The hook reads `BENCH_SITE_BASE_URL` + `BENCH_REVALIDATE_TOKEN` from the environment and is a no-op when either is absent. **Every failure is swallowed so it can never change the exit code.** The module is stdlib-only on the `--server` path; `urllib` + `concurrent.futures` are stdlib so the hook adds no dependency. The test loads the hyphenated module via `importlib` exactly like `test_post_ingest_postgres.py:66-78`.

- [ ] **Step 1: Write the failing test** (`scripts/test_post_ingest_revalidate.py`)

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Unit tests for post-ingest.py's best-effort site-cache refresh hook.

These are pure-stdlib tests (no Docker, no psycopg): they exercise
`refresh_site_cache` by monkeypatching `urllib.request.urlopen`, asserting the
bearer header is sent and that every failure is swallowed so the hook can never
change the ingest exit code.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def _load_module(filename: str, modname: str):
    path = SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(modname, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


post_ingest = _load_module("post-ingest.py", "post_ingest")


class _FakeResponse:
    def __init__(self, body: bytes = b"{}"):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_refresh_posts_revalidate_with_bearer(monkeypatch):
    calls: list[tuple[str, dict[str, str], bytes | None]] = []

    def fake_urlopen(req, timeout=None):
        calls.append((req.full_url, dict(req.headers), req.data))
        return _FakeResponse(b'{"groups": []}')

    monkeypatch.setattr(post_ingest.urllib.request, "urlopen", fake_urlopen)
    post_ingest.refresh_site_cache("https://example.test/", "tok", 5.0)

    revalidate = [c for c in calls if c[0].endswith("/api/revalidate")]
    assert revalidate, "expected a POST to /api/revalidate"
    # urllib title-cases header keys, so the bearer lives under "Authorization".
    assert revalidate[0][1].get("Authorization") == "Bearer tok"


def test_refresh_swallows_all_failures(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(post_ingest.urllib.request, "urlopen", boom)
    # Must not raise: a cache-refresh failure can never fail an ingest.
    assert post_ingest.refresh_site_cache("https://example.test", "tok", 5.0) is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/connor/spiral/vortex-data/vortex4 && uv run --no-project pytest scripts/test_post_ingest_revalidate.py -q`
Expected: FAIL (`AttributeError: module 'post_ingest' has no attribute 'refresh_site_cache'`).

- [ ] **Step 3: Implement the hook in `scripts/post-ingest.py`**

Add near the other helpers (after `parse_args`, before `_main_postgres`). `urllib.request` / `urllib.error` are already imported; add `from concurrent.futures import ThreadPoolExecutor` to the imports block.

```python
def _http(method: str, url: str, token: str | None, timeout: float) -> bytes:
    """Issue one HTTP request and return the body. Raises on any non-2xx or
    transport error; callers in `refresh_site_cache` swallow those."""
    headers = {"accept": "application/json"}
    if token is not None:
        headers["authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _warm_default_windows(base: str, timeout: float) -> None:
    """Best-effort warm pass: prime the freshly invalidated Data Cache for the
    landing page and every group's default last-100 bundle, so the first human
    request after an ingest is already hot. Each request is independent; one
    failure does not abort the others."""
    def warm(url: str) -> None:
        try:
            _http("GET", url, None, timeout)
        except Exception as exc:  # noqa: BLE001 -- warm is best-effort.
            print(f"warning: warm {url} failed: {exc}", file=sys.stderr)

    warm(f"{base}/")
    try:
        groups_body = _http("GET", f"{base}/api/groups", None, timeout)
        slugs = [g["slug"] for g in json.loads(groups_body).get("groups", []) if "slug" in g]
    except Exception as exc:  # noqa: BLE001
        print(f"warning: warm group discovery failed: {exc}", file=sys.stderr)
        return
    # A whole-bundle recompute is a few seconds cold, so warm with bounded
    # concurrency rather than one slow serial pass.
    with ThreadPoolExecutor(max_workers=4) as pool:
        pool.map(lambda s: warm(f"{base}/api/group/{s}?n=100"), slugs)


def refresh_site_cache(base_url: str, token: str, timeout: float) -> None:
    """Revalidate the site's Data Cache tag, then warm the default windows.
    BEST-EFFORT: every failure is logged to stderr and swallowed so a cache
    refresh can never change the ingest exit code."""
    base = base_url.rstrip("/")
    try:
        _http("POST", f"{base}/api/revalidate", token, timeout)
    except Exception as exc:  # noqa: BLE001 -- refresh must never raise into ingest.
        print(f"warning: cache revalidate failed: {exc}", file=sys.stderr)
    _warm_default_windows(base, timeout)
```

- [ ] **Step 4: Call the hook from `_main_postgres`** (`scripts/post-ingest.py`, after the success `print(...)`, before `return 0`)

```python
    print(
        json.dumps(
            {"records": len(records), "inserted": inserted, "updated": updated},
            separators=(",", ":"),
        )
    )
    # Best-effort site-cache refresh after a successful write. No-op unless both
    # env vars are set (so the script stays inert until the ops wiring lands),
    # and it can never fail the ingest.
    base_url = os.environ.get("BENCH_SITE_BASE_URL")
    revalidate_token = os.environ.get("BENCH_REVALIDATE_TOKEN")
    if base_url and revalidate_token:
        refresh_site_cache(base_url, revalidate_token, args.timeout)
    return 0
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd /Users/connor/spiral/vortex-data/vortex4 && uv run --no-project pytest scripts/test_post_ingest_revalidate.py -q`
Expected: PASS (2 tests). Then run the existing `python -m py_compile scripts/post-ingest.py` to confirm no syntax error.

- [ ] **Step 6: Add the env lines to the three workflows**

In each workflow's **v4 Postgres ingest step** `env:` map, add two additive lines. `bench.yml:143-146` becomes:
```yaml
        env:
          RDS_BENCH_INSTANCE_ENDPOINT: ${{ vars.RDS_BENCH_INSTANCE_ENDPOINT }}
          RDS_BENCH_DB_NAME: ${{ vars.RDS_BENCH_DB_NAME }}
          AWS_REGION: ${{ vars.RDS_BENCH_REGION }}
          BENCH_SITE_BASE_URL: ${{ vars.BENCH_SITE_BASE_URL }}
          BENCH_REVALIDATE_TOKEN: ${{ secrets.BENCH_REVALIDATE_TOKEN }}
```
Apply the same two-line addition to the corresponding v4 Postgres step's `env:` map in `sql-benchmarks.yml` and `v3-commit-metadata.yml` (find the step by its `--postgres` invocation; mirror the surrounding `${{ vars.* }}` / `${{ secrets.* }}` style). Keep the steps `continue-on-error: true` (already set).

- [ ] **Step 7: Lint the workflows**

Run: `cd /Users/connor/spiral/vortex-data/vortex4 && yamllint --strict -c .yamllint.yaml .github/workflows/bench.yml .github/workflows/sql-benchmarks.yml .github/workflows/v3-commit-metadata.yml`
Expected: no output (clean).

- [ ] **Step 8: Commit**

```bash
git add scripts/post-ingest.py scripts/test_post_ingest_revalidate.py \
  .github/workflows/bench.yml .github/workflows/sql-benchmarks.yml .github/workflows/v3-commit-metadata.yml
git commit -F - <<'EOF'
benchmarks-website: post-ingest site-cache revalidate + warm hook (PR-5.0.97)

Best-effort: POSTs /api/revalidate then warms the default windows after a
successful --postgres write. Swallows every failure so it can never change the
ingest exit code, and is a no-op unless BENCH_SITE_BASE_URL and
BENCH_REVALIDATE_TOKEN are both set.

Signed-off-by: Connor Tsui <connor.tsui20@gmail.com>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

---

## Task 4: Client group-bundle fetch + session payload cache (fast Expand All)

**Files:**
- Modify: `benchmarks-website/web/lib/chart-format.ts` (add `BUNDLE_CONCURRENCY`)
- Modify: `benchmarks-website/web/lib/chart-store.ts` (payload cache + `ensureGroupBundle` / `abortGroupBundle`)
- Modify: `benchmarks-website/web/components/Chart.tsx` (consult cache in `ensureInitialPayload`; kick bundle on group open; abort on close)
- Modify: `benchmarks-website/web/components/Chart.lazy-hydration.test.tsx` (bundle tests)

**Context:** `makeQueue` / `TaskQueue` / `QueueEntry` live in `chart-store.ts:36-99`; `noteGroupSeries(slug, meta)` at `chart-store.ts:303`. `ensureInitialPayload(priority, showLoading)` (`Chart.tsx:450`) currently always schedules a per-chart `/api/chart/{slug}?n=100` fetch; the per-fetch AbortController + `FETCH_TIMEOUT_MS` pattern is `Chart.tsx:474-503`. The group-open effect's `details` branch is `Chart.tsx:1744-1804` (`armHydration` → `onGroupOpen(priority)`, `disarmHydration` → `abortInFlightFetches()`). The bundle response shape is `GroupChartsResponse` (`queries.ts:643`): `{ name, summary?, description?, charts: NamedChartResponse[] }` where each chart is `ChartResponse & { name, slug }`. The IntersectionObserver continues to gate Chart.js **construction**; the bundle only changes how the **data** arrives.

- [ ] **Step 1: Write the failing tests** (extend `benchmarks-website/web/components/Chart.lazy-hydration.test.tsx`)

Add a `describe('PR-5.0.97 group-bundle hydration', ...)` block. Render the group with `groupSlug` set on each island (extend `renderGroup` to pass a `groupSlug` prop, e.g. `'g'`, and to mock `fetch` so a `/api/group/g?n=100` URL returns a bundle `{ name: 'g', charts: [{ name, slug, ...windowedPayload(3572) }, ...] }` for every island's slug). Assert:

```ts
// Group open issues exactly ONE /api/group/g?n=100 bundle fetch for N islands,
// and no per-chart /api/chart fetch (every island hydrates from the bundle).
const bundleCalls = fetchCalls.filter((u) => u.includes('/api/group/'));
const chartCalls = fetchCalls.filter((u) => u.includes('/api/chart/'));
expect(bundleCalls.length).toBe(1);
expect(chartCalls.length).toBe(0);
```

Also add cases (each fires the relevant MockIO instance(s) via `MockIO.instances[i].fire()`):
- After the bundle resolves, firing an island's IO constructs it (assert a chart canvas / construction effect, mirroring the existing lazy-hydration assertions).
- Closing the group (`details.open = false` + dispatch `toggle`) aborts the bundle: assert the bundle fetch received a signal that is now aborted (capture the `signal` from the `fetch` stub).
- Reopen AFTER the bundle succeeded issues ZERO new fetches (cache hit) — assert `fetchCalls.length` is unchanged after a close+reopen.
- A slug absent from the bundle falls back to exactly one `/api/chart/{thatSlug}?n=100` fetch.
- A bundle 404 (`{ ok: false, status: 404 }`) makes every island fall back to its own `/api/chart` fetch.
- Two groups opened together respect `BUNDLE_CONCURRENCY` and top-group priority (assert via a `bundleQueue.schedule` spy or by counting concurrent in-flight bundle fetches).

Reset module state between tests: import `resetPayloadCache` from `@/lib/chart-store` and call it in `beforeEach`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd benchmarks-website/web && pnpm vitest run components/Chart.lazy-hydration.test.tsx`
Expected: FAIL (no bundle behavior yet; `resetPayloadCache`/`bundleQueue` undefined).

- [ ] **Step 3: Add the `BUNDLE_CONCURRENCY` constant** (`benchmarks-website/web/lib/chart-format.ts`, near `HYDRATION_CONCURRENCY` at L33)

```ts
/** Per-tab cap for the per-group bundle fetches (`/api/group/{slug}?n=100`).
 * One in-flight bundle covers a whole group, so the cap bounds how many groups
 * fetch at once on Expand All without serializing the top groups. */
export const BUNDLE_CONCURRENCY = 3;
```

- [ ] **Step 4: Add the payload cache + bundle queue to `chart-store.ts`**

Add `BUNDLE_CONCURRENCY` to the `@/lib/chart-format` import and `FETCH_TIMEOUT_MS` too. Add, after the `fullHistoryQueue` export (L105):

```ts
import type { ChartResponse, GroupChartsResponse, SeriesTag } from '@/lib/queries';

/** Per-tab queue for the per-group `/api/group/{slug}?n=100` bundle fetches. */
export const bundleQueue: TaskQueue = makeQueue(BUNDLE_CONCURRENCY);

// Session-lifetime cache of the default last-100 chart payloads, keyed by chart
// slug and filled by `ensureGroupBundle`. Closing and reopening a group reads
// from here, so it never refetches. An open tab keeps these until reload; a
// server-side revalidation is picked up on the next full load (a data-version
// invalidation is future work, not built here).
const payloadCache = new Map<string, ChartResponse>();

/** The cached default payload for `slug`, or `undefined` on a miss. */
export function getCachedPayload(slug: string): ChartResponse | undefined {
  return payloadCache.get(slug);
}

/** Seed the cache for one chart slug (idempotent; last write wins). */
export function primePayload(slug: string, payload: ChartResponse): void {
  payloadCache.set(slug, payload);
}

/** Clear the cache and in-flight bundle map. TEST-ONLY: production never evicts
 * within a tab session. */
export function resetPayloadCache(): void {
  payloadCache.clear();
  inFlightBundles.clear();
}

interface BundleInFlight {
  entry: QueueEntry;
  controller: AbortController;
  promise: Promise<void>;
}

const inFlightBundles = new Map<string, BundleInFlight>();

/**
 * Fetch one group's default last-100 bundle (`/api/group/{slug}?n=100`) and
 * prime [`payloadCache`] for every chart in it. Concurrent callers for the same
 * group share one in-flight fetch (priority is bumped to the highest caller's).
 * A 404 or failure resolves without priming, so callers fall back to the
 * per-chart fetch. Never rejects: failures are swallowed here and surfaced as a
 * cache miss to the caller.
 */
export function ensureGroupBundle(groupSlug: string, priority: number): Promise<void> {
  const existing = inFlightBundles.get(groupSlug);
  if (existing) {
    if (priority > existing.entry.priority) {
      existing.entry.priority = priority;
      bundleQueue.drain();
    }
    return existing.promise;
  }
  const url = `/api/group/${encodeURIComponent(groupSlug)}?n=100`;
  const controller = new AbortController();
  const entry = bundleQueue.schedule(async () => {
    const timer = setTimeout(
      () => controller.abort(new DOMException('Fetch timed out', 'TimeoutError')),
      FETCH_TIMEOUT_MS,
    );
    try {
      const r = await fetch(url, {
        headers: { accept: 'application/json' },
        signal: controller.signal,
      });
      if (r.status === 404) {
        return null;
      }
      if (!r.ok) {
        throw new Error(`HTTP ${r.status}`);
      }
      return (await r.json()) as GroupChartsResponse;
    } finally {
      clearTimeout(timer);
    }
  }, priority);
  const promise = entry.promise
    .then((body) => {
      if (body !== null) {
        const bundle = body as GroupChartsResponse;
        for (const chart of bundle.charts) {
          // `NamedChartResponse` is `ChartResponse & { name, slug }`; the extra
          // keys are harmless in the cached payload.
          primePayload(chart.slug, chart);
          noteGroupSeries(groupSlug, chart.series_meta);
        }
      }
    })
    .catch((err: unknown) => {
      // A close/destroy abort is silent; a timeout or failure leaves the cache
      // unprimed so callers fall back per-chart. Surface non-abort failures for
      // debugging only.
      if (!(err instanceof DOMException && err.name === 'AbortError')) {
        console.warn('bench: group bundle fetch failed', err);
      }
    })
    .finally(() => {
      if (inFlightBundles.get(groupSlug)?.entry === entry) {
        inFlightBundles.delete(groupSlug);
      }
    });
  inFlightBundles.set(groupSlug, { entry, controller, promise });
  return promise;
}

/** Abort a group's in-flight bundle fetch (on group close) and drop its
 * in-flight entry so a reopen re-issues. Idempotent. */
export function abortGroupBundle(groupSlug: string): void {
  const inFlight = inFlightBundles.get(groupSlug);
  if (inFlight) {
    inFlight.controller.abort(new DOMException('group closed', 'AbortError'));
    inFlightBundles.delete(groupSlug);
  }
}
```

(`noteGroupSeries` is already defined in this module; reference it directly. `series_meta` is on `ChartResponse`.)

- [ ] **Step 5: Consult the cache + bundle in `ensureInitialPayload`** (`benchmarks-website/web/components/Chart.tsx:450`)

At the very top of `ensureInitialPayload`, after the `state.payload || state.disposed` guard, add a synchronous cache hit and a bundle attempt before the existing per-chart fetch. Import `ensureGroupBundle, getCachedPayload` from `@/lib/chart-store`.

```ts
  ensureInitialPayload(priority: number, showLoading: boolean): Promise<void> {
    const state = this.state;
    if (state.payload || state.disposed) {
      return Promise.resolve();
    }
    // Fast path: a sibling group-bundle fetch may have already cached this
    // chart's default payload. Seed from it synchronously (same steps as the
    // fetch success path) so no per-chart request is issued.
    const cached = getCachedPayload(this.slug);
    if (cached) {
      this.seedFromCachedPayload(cached);
      return Promise.resolve();
    }
    // On the landing page (a group slug is present), drive one bundle fetch per
    // group and hydrate from it. Only fall through to the per-chart fetch when
    // the bundle is unavailable (404 / failed / this slug missing).
    if (this.groupSlug) {
      if (showLoading) {
        this.cb.setLoading(true);
        this.cb.setRetryable(false);
      }
      const groupSlug = this.groupSlug;
      return ensureGroupBundle(groupSlug, priority).then(() => {
        if (state.disposed || state.payload) {
          return;
        }
        const fromBundle = getCachedPayload(this.slug);
        if (fromBundle) {
          this.seedFromCachedPayload(fromBundle);
          return;
        }
        // Bundle did not cover this chart: fall back to the per-chart fetch.
        return this.fetchInitialPayloadDirect(priority, showLoading);
      });
    }
    return this.fetchInitialPayloadDirect(priority, showLoading);
  }

  /** Seed state from a cached default payload (the bundle/cache hit path),
   * mirroring the per-chart fetch's success handler. */
  private seedFromCachedPayload(raw: ChartResponse): void {
    const state = this.state;
    if (state.fullLoaded) {
      return;
    }
    const normalized = normalizeChartPayload(raw);
    state.payload = normalized;
    state.fullLoaded = normalized.history.complete;
    if (!normalized.history.complete) {
      state.everWindowed = true;
    }
    this.syncWindowChip();
    this.cb.setLoading(false);
    if (this.groupSlug) {
      noteGroupSeries(this.groupSlug, normalized.series_meta);
    }
  }
```

Rename the existing per-chart fetch body (the code from `Chart.tsx:455` `if (state.initialFetchEntry) {` through the closing of the method at L567) into a new private method `fetchInitialPayloadDirect(priority, showLoading)` that contains the unchanged existing logic (the `initialFetchEntry` dedupe, the `hydrationQueue.schedule` fetch, the success/error handlers). The only change is the method boundary; the per-chart abort/timeout/retry behavior is preserved verbatim. `seedFromCachedPayload` does not call `maybeConstruct`; the caller (`onGroupOpen` / `retryInitialPayload`) already does after the promise resolves.

- [ ] **Step 6: Kick the bundle on group open + abort on close** (`benchmarks-website/web/components/Chart.tsx`, the `details` branch ~L1744-1804)

`onGroupOpen` already calls `ensureInitialPayload(priority, true)`, which now routes through the bundle, so the IntersectionObserver path is automatically bundle-backed once a card scrolls in. To make Expand All eager (load all groups' data immediately, not on scroll), kick the bundle from `armHydration` BEFORE the IO fires, while still gating construction on intersection:

```ts
      const priority = index === 0 ? 0 : -index;
      let io: IntersectionObserver | null = null;
      const armHydration = (): void => {
        if (io) {
          return;
        }
        // Start the group's bundle fetch immediately on open so every chart's
        // last-100 data loads eagerly (top-group-first by index priority), even
        // off-screen. Construction stays gated on intersection below.
        if (controllerHasGroup) {
          void ensureGroupBundle(groupSlug, priority);
        }
        if (typeof IntersectionObserver === 'undefined') {
          controller.onGroupOpen(priority);
          return;
        }
        io = new IntersectionObserver(/* ...unchanged... */);
        io.observe(card);
      };
      const disarmHydration = (): void => {
        io?.disconnect();
        io = null;
        controller.abortInFlightFetches();
        abortGroupBundle(groupSlug);
      };
```

where `groupSlug` is the island's `groupSlug` prop (guard the call so it only runs on the landing page where `groupSlug` is defined; bind `const controllerHasGroup = groupSlug !== undefined` and a non-null `groupSlug` local). Import `abortGroupBundle, ensureGroupBundle` from `@/lib/chart-store`. Note `abortGroupBundle` aborting on close is safe because each island calls it; the first wins and the rest no-op (idempotent). Because multiple islands in the same group each call `ensureGroupBundle(groupSlug, ...)`, the in-flight dedupe means only ONE fetch is issued per group (the test asserts exactly this).

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cd benchmarks-website/web && pnpm vitest run components/Chart.lazy-hydration.test.tsx components/Chart.lifecycle.test.tsx`
Expected: PASS (including the StrictMode double-mount lifecycle test, which exercises the bundle dedupe + abort/clear).

- [ ] **Step 8: Commit**

```bash
git add benchmarks-website/web/lib/chart-format.ts benchmarks-website/web/lib/chart-store.ts \
  benchmarks-website/web/components/Chart.tsx benchmarks-website/web/components/Chart.lazy-hydration.test.tsx
git commit -F - <<'EOF'
benchmarks-website: one group-bundle fetch + session payload cache for fast Expand All (PR-5.0.97)

Group open now issues a single /api/group/{slug}?n=100 bundle fetch feeding a
session-lifetime payload cache; charts hydrate from it (per-chart fetch is the
fallback for bundle 404s / missing slugs / the permalink page). Close/reopen
refetches nothing. IntersectionObserver still gates Chart.js construction.

Signed-off-by: Connor Tsui <connor.tsui20@gmail.com>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

---

## Task 5: Full pre-data spinner coverage

**Files:**
- Modify: `benchmarks-website/web/components/Chart.tsx` (`constructed` state + `.chart-placeholder`)
- Modify: `benchmarks-website/web/app/globals.css` (`.chart-placeholder` + reduced-motion)
- Modify: `benchmarks-website/web/components/Chart.loading.test.tsx`
- Modify: `benchmarks-website/web/app/globals.spinner.test.ts`

**Context:** Today a card is blank white until `maybeConstruct` builds the Chart.js instance (`state.chart = chart` at `Chart.tsx:884`). The only loading signal is the top-right `.chart-loading` pill, shown only while `loading` is true during a fetch (`Chart.tsx:1972-1977`). `.chart-wrap` (`globals.css:1050-1054`) is `position: relative; height: 420px`. The spinner CSS + `@keyframes chart-spin` + the `prefers-reduced-motion` guard already exist (`globals.css:1182-1214`). **User decision: respect `prefers-reduced-motion`** — under reduced motion the placeholder keeps a static ring + visible "loading…" label (never blank, never nothing).

- [ ] **Step 1: Write the failing tests**

Extend `benchmarks-website/web/components/Chart.loading.test.tsx`: assert a freshly-rendered card (before any fetch resolves) shows a `.chart-placeholder` inside `.chart-wrap` with `role="status"` and a visible label; assert it is still present while a fetch is pending; assert it is REMOVED after construction completes (drive a payload through and let `maybeConstruct` run, mirroring the file's existing construction setup); assert it does not appear alongside the `.chart-error` block when an error is shown.

Extend `benchmarks-website/web/app/globals.spinner.test.ts` (node-env CSS-text assertions, mirroring its existing keyframe checks): assert `globals.css` contains a `.chart-placeholder` rule, that `.chart-placeholder .chart-spinner` is included in the `@media (prefers-reduced-motion: reduce)` block (animation disabled), and that the placeholder retains a visible label / static ring under reduced motion (the reduced-motion block sets `animation: none` but does NOT `display: none` the ring or label).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd benchmarks-website/web && pnpm vitest run components/Chart.loading.test.tsx app/globals.spinner.test.ts`
Expected: FAIL.

- [ ] **Step 3: Add the `constructed` state + placeholder to `Chart.tsx`**

Add the state near the other `useState`s (`Chart.tsx:1633-1636`):
```ts
  const [constructed, setConstructed] = useState(false);
```
Thread `setConstructed` through `CardCallbacks` (`Chart.tsx:160-166`):
```ts
interface CardCallbacks {
  setY: (y: 'linear' | 'log') => void;
  setLoading: (on: boolean) => void;
  setError: (msg: string | null) => void;
  setRetryable: (on: boolean) => void;
  /** Flip once the Chart.js instance exists, so the pre-data placeholder hides. */
  setConstructed: (on: boolean) => void;
}
```
Pass it where the callbacks object is built (`Chart.tsx:1703`): `{ setY, setLoading, setError, setRetryable, setConstructed }`. In `maybeConstruct`, right after `state.chart = chart;` (`Chart.tsx:884`):
```ts
      state.chart = chart;
      this.cb.setConstructed(true);
```
Render the placeholder inside `.chart-wrap` (`Chart.tsx:1941-1943`) so it is server-rendered (no JS needed for the first paint) and disappears once constructed:
```tsx
      <div className="chart-wrap">
        {!constructed && !error && (
          <div className="chart-placeholder" role="status" aria-live="polite">
            <span className="chart-spinner" aria-hidden="true" />
            <span className="chart-placeholder-text">loading…</span>
          </div>
        )}
        <canvas data-chart-index={index} ref={canvasRef} />
      </div>
```
Keep the existing top-right `.chart-loading` pill (network-in-flight signal) unchanged so `Chart.loading.test.tsx`'s existing pill assertions stay green. The placeholder is the "no chart yet" signal; the pill is the "fetch in flight" signal.

- [ ] **Step 4: Add the `.chart-placeholder` CSS** (`benchmarks-website/web/app/globals.css`, near the spinner rules ~L1182)

```css
/* Pre-construction placeholder: fills the chart area so a card is never blank
   white before its Chart.js instance exists (server-rendered, hidden once the
   chart constructs). */
.chart-placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  color: var(--muted);
  background: color-mix(in srgb, var(--code-bg) 60%, transparent);
  font-size: 0.8rem;
  z-index: 1;
}
.chart-placeholder .chart-spinner {
  width: 1.4rem;
  height: 1.4rem;
  border-width: 3px;
}
```
Add `.chart-placeholder .chart-spinner` to the existing reduced-motion block (`globals.css:1209-1214`) so its animation is disabled but the static ring + label remain:
```css
@media (prefers-reduced-motion: reduce) {
  .chart-spinner,
  .chart-placeholder .chart-spinner,
  .chart-window-chip[data-state='loading']::before {
    animation: none;
  }
}
```
(Do NOT add `display: none` anywhere for the placeholder under reduced motion: the ring and the "loading…" label must stay visible.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd benchmarks-website/web && pnpm vitest run components/Chart.loading.test.tsx app/globals.spinner.test.ts`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add benchmarks-website/web/components/Chart.tsx benchmarks-website/web/app/globals.css \
  benchmarks-website/web/components/Chart.loading.test.tsx benchmarks-website/web/app/globals.spinner.test.ts
git commit -F - <<'EOF'
benchmarks-website: server-rendered spinner placeholder for every pre-data card state (PR-5.0.97)

A .chart-placeholder (spinner ring + "loading…" label) fills the chart area
until the Chart.js instance constructs, so pre-hydration cards are never blank.
Respects prefers-reduced-motion: under reduced motion the ring and label stay
visible statically (no animation).

Signed-off-by: Connor Tsui <connor.tsui20@gmail.com>
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
```

---

## Task 6: Full-suite gate pass + PR description notes

**Files:** none necessarily; fix any integration drift surfaced by the gates.

- [ ] **Step 1: Run the full web gate suite**

Run:
```bash
cd benchmarks-website/web
pnpm test
pnpm exec tsc --noEmit
pnpm build
pnpm lint
pnpm exec prettier --check .
```
Expected: all green. `pnpm build` exercises `server-smoke.test.ts` paths and confirms `unstable_cache` builds without a live DB (the data-cache wrappers must not run at build time on `force-dynamic` pages). If `next build` tries to prerender, confirm `export const dynamic = 'force-dynamic'` is still present on both pages.

- [ ] **Step 2: Run the Python + YAML gates**

Run:
```bash
cd /Users/connor/spiral/vortex-data/vortex4
uv run --no-project pytest scripts/test_post_ingest_revalidate.py -q
python -m py_compile scripts/post-ingest.py
yamllint --strict -c .yamllint.yaml .github/workflows/bench.yml .github/workflows/sql-benchmarks.yml .github/workflows/v3-commit-metadata.yml
```
Expected: all green.

- [ ] **Step 3: Confirm the acceptance criteria** (from the design doc; verify, do not just assert):
  - Default `?n=100` chart + group routes and both pages read through the Data Cache; non-default windows still hit the direct query (route-branch tests green).
  - `POST /api/revalidate` 503s unconfigured, 401s on a bad token, 200s + `revalidateTag('bench-data')` on the right token; no cache-control header on it.
  - Group open issues exactly one `/api/group/{slug}?n=100` per group; charts hydrate from the cache; close/reopen issues zero fetches; bundle 404 / missing slug falls back per-chart (lazy-hydration tests green).
  - Every pre-data card shows the placeholder; reduced motion keeps a static ring + label.
  - post-ingest refresh swallows all failures and is a no-op without the env vars.

- [ ] **Step 4: Record the deferred/risk notes for the PR body** (drafted via `spiral:pr-and-issue-voice` at close):
  - Risk: Vercel Data Cache ~2MB item cap vs the ~1.4MB clickbench bundle (under the cap; degrades to a direct query if ever exceeded).
  - Risk: `unstable_cache` is superseded by `'use cache'` in later Next; isolated in `data-cache.ts`.
  - Ops prerequisite: `BENCH_REVALIDATE_TOKEN` (Vercel env + GH secret) and `BENCH_SITE_BASE_URL` (GH var) must be set for refresh-on-update + warming to activate; until then the route 503s fail-closed and the hook no-ops (PR is safe to merge before the wiring).
  - Deferred (NOT data-correctness): client data-version invalidation for an open tab after a server revalidation.

- [ ] **Step 5: Final commit if any drift was fixed** (otherwise nothing to commit). Use a `benchmarks-website: PR-5.0.97 gate-pass fixes` subject with the standard sign-off trailer.

---

## Self-Review notes (author checklist, resolved)

- **Spec coverage:** A→Task 1; B→Task 2; C→Task 3; D→Task 4; E→Task 5; tests interleaved per task + Task 6 full gate. All six design areas mapped.
- **Type consistency:** `cachedDefaultGroupCharts(slug)` / `cachedDefaultChartPayload(slug)` / `cachedGroups()` / `cachedFilterUniverse()` and `BENCH_DATA_TAG` are used identically in Tasks 1-2. `ensureGroupBundle(groupSlug, priority)` / `abortGroupBundle(groupSlug)` / `getCachedPayload(slug)` / `primePayload` / `resetPayloadCache` / `bundleQueue` / `BUNDLE_CONCURRENCY` are consistent across Task 4 + its tests. `setConstructed` threads through `CardCallbacks` in Task 5.
- **No placeholders:** every code step shows the actual code; the only "follow the existing pattern" references are to concrete, named existing test files (the executor reads them in full).
- **Sequencing:** server stream 1→2→3 (3 calls 2; 2 invalidates 1's tag); client 4 and 5 independent; 6 last. Each task leaves the build green.
