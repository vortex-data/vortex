// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// @vitest-environment node

import { afterEach, describe, expect, it, vi } from 'vitest';

// `groupKeyFromSlug` validates the 400 path; a thrown error is a bad slug.
vi.mock('@/lib/slug', () => ({
  groupKeyFromSlug: (slug: string) => {
    if (slug === 'bad') {
      throw new Error('invalid');
    }
    return { slug };
  },
}));

type Bundle = { name: string; charts: never[] } | null;

const cachedDefaultGroupCharts = vi.fn<(slug: string) => Promise<Bundle>>(async () => ({
  name: 'cached',
  charts: [],
}));
vi.mock('@/lib/data-cache', () => ({
  cachedDefaultGroupCharts: (slug: string) => cachedDefaultGroupCharts(slug),
}));

const collectGroupCharts = vi.fn<(key: unknown, window: unknown) => Promise<Bundle>>(async () => ({
  name: 'direct',
  charts: [],
}));
vi.mock('@/lib/queries', () => ({
  collectGroupCharts: (key: unknown, window: unknown) => collectGroupCharts(key, window),
}));

import { GET } from '@/app/api/group/[slug]/route';

function get(slug: string, n: string | null): Promise<Response> {
  const query = n === null ? '' : `?n=${encodeURIComponent(n)}`;
  const request = new Request(`http://localhost/api/group/${slug}${query}`);
  return GET(request, { params: Promise.resolve({ slug }) });
}

afterEach(() => {
  cachedDefaultGroupCharts.mockClear();
  collectGroupCharts.mockClear();
  // Restore the default resolved values cleared mocks would otherwise lose.
  cachedDefaultGroupCharts.mockResolvedValue({ name: 'cached', charts: [] });
  collectGroupCharts.mockResolvedValue({ name: 'direct', charts: [] });
});

describe('GET /api/group/[slug] default-window branch', () => {
  it('reads the Data Cache (not the direct query) when no ?n= is given', async () => {
    const res = await get('ok', null);
    expect(res.status).toBe(200);
    expect(cachedDefaultGroupCharts).toHaveBeenCalledTimes(1);
    expect(cachedDefaultGroupCharts).toHaveBeenCalledWith('ok');
    expect(collectGroupCharts).not.toHaveBeenCalled();
  });

  it('reads the Data Cache when ?n=100 (the explicit default)', async () => {
    const res = await get('ok', '100');
    expect(res.status).toBe(200);
    expect(cachedDefaultGroupCharts).toHaveBeenCalledTimes(1);
    expect(collectGroupCharts).not.toHaveBeenCalled();
  });

  it('reads the direct query (not the cache) for ?n=all', async () => {
    const res = await get('ok', 'all');
    expect(res.status).toBe(200);
    expect(collectGroupCharts).toHaveBeenCalledTimes(1);
    expect(cachedDefaultGroupCharts).not.toHaveBeenCalled();
  });

  it('reads the direct query (not the cache) for a non-default ?n=50', async () => {
    const res = await get('ok', '50');
    expect(res.status).toBe(200);
    expect(collectGroupCharts).toHaveBeenCalledTimes(1);
    expect(cachedDefaultGroupCharts).not.toHaveBeenCalled();
  });

  it('400s on a malformed slug before either query runs', async () => {
    const res = await get('bad', null);
    expect(res.status).toBe(400);
    await expect(res.json()).resolves.toEqual({
      error: 'bad_request',
      message: 'invalid group slug',
    });
    expect(cachedDefaultGroupCharts).not.toHaveBeenCalled();
    expect(collectGroupCharts).not.toHaveBeenCalled();
  });

  it('404s when the cached default payload is null', async () => {
    cachedDefaultGroupCharts.mockResolvedValueOnce(null);
    const res = await get('ok', null);
    expect(res.status).toBe(404);
    await expect(res.json()).resolves.toEqual({ error: 'not_found', message: 'group not found' });
  });

  it('404s when the direct (non-default) payload is null', async () => {
    collectGroupCharts.mockResolvedValueOnce(null);
    const res = await get('ok', 'all');
    expect(res.status).toBe(404);
    await expect(res.json()).resolves.toEqual({ error: 'not_found', message: 'group not found' });
  });
});
