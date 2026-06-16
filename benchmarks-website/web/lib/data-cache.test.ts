// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// @vitest-environment node

import { afterEach, describe, expect, it, vi } from 'vitest';

// Capture the options unstable_cache is called with, and pass the wrapped
// function through unchanged so the cached wrappers still invoke the real query.
// `vi.hoisted` lifts the capture array alongside the hoisted `vi.mock` factory,
// so it exists when `data-cache.ts` calls `unstable_cache` at import time.
const { cacheCalls } = vi.hoisted(() => ({
  cacheCalls: [] as { keyParts: string[]; options: { tags?: string[]; revalidate?: number } }[],
}));
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
    expect(DATA_CACHE_BACKSTOP_SECONDS).toBe(86400);
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
