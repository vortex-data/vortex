// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// @vitest-environment node

import { afterEach, describe, expect, it, vi } from 'vitest';

// `chartKeyFromSlug` validates the 400 path; a thrown error is a bad slug.
vi.mock('@/lib/slug', () => ({
  chartKeyFromSlug: (slug: string) => {
    if (slug === 'bad') {
      throw new Error('invalid');
    }
    return { slug };
  },
}));

type Payload = { display_name: string } | null;

const cachedDefaultChartPayload = vi.fn<(slug: string) => Promise<Payload>>(async () => ({
  display_name: 'cached',
}));
vi.mock('@/lib/data-cache', () => ({
  cachedDefaultChartPayload: (slug: string) => cachedDefaultChartPayload(slug),
}));

const chartPayload = vi.fn<(key: unknown, window: unknown) => Promise<Payload>>(async () => ({
  display_name: 'direct',
}));
vi.mock('@/lib/queries', () => ({
  chartPayload: (key: unknown, window: unknown) => chartPayload(key, window),
}));

import { GET } from '@/app/api/chart/[slug]/route';

function get(slug: string, n: string | null): Promise<Response> {
  const query = n === null ? '' : `?n=${encodeURIComponent(n)}`;
  const request = new Request(`http://localhost/api/chart/${slug}${query}`);
  return GET(request, { params: Promise.resolve({ slug }) });
}

afterEach(() => {
  cachedDefaultChartPayload.mockClear();
  chartPayload.mockClear();
  // Restore the default resolved values cleared mocks would otherwise lose.
  cachedDefaultChartPayload.mockResolvedValue({ display_name: 'cached' });
  chartPayload.mockResolvedValue({ display_name: 'direct' });
});

describe('GET /api/chart/[slug] default-window branch', () => {
  it('reads the Data Cache (not the direct query) when no ?n= is given', async () => {
    const res = await get('ok', null);
    expect(res.status).toBe(200);
    expect(cachedDefaultChartPayload).toHaveBeenCalledTimes(1);
    expect(cachedDefaultChartPayload).toHaveBeenCalledWith('ok');
    expect(chartPayload).not.toHaveBeenCalled();
  });

  it('reads the Data Cache when ?n=100 (the explicit default)', async () => {
    const res = await get('ok', '100');
    expect(res.status).toBe(200);
    expect(cachedDefaultChartPayload).toHaveBeenCalledTimes(1);
    expect(chartPayload).not.toHaveBeenCalled();
  });

  it('reads the direct query (not the cache) for ?n=all', async () => {
    const res = await get('ok', 'all');
    expect(res.status).toBe(200);
    expect(chartPayload).toHaveBeenCalledTimes(1);
    expect(cachedDefaultChartPayload).not.toHaveBeenCalled();
  });

  it('reads the direct query (not the cache) for a non-default ?n=50', async () => {
    const res = await get('ok', '50');
    expect(res.status).toBe(200);
    expect(chartPayload).toHaveBeenCalledTimes(1);
    expect(cachedDefaultChartPayload).not.toHaveBeenCalled();
  });

  it('400s on a malformed slug before either query runs', async () => {
    const res = await get('bad', null);
    expect(res.status).toBe(400);
    await expect(res.json()).resolves.toEqual({
      error: 'bad_request',
      message: 'invalid chart slug',
    });
    expect(cachedDefaultChartPayload).not.toHaveBeenCalled();
    expect(chartPayload).not.toHaveBeenCalled();
  });

  it('404s when the cached default payload is null', async () => {
    cachedDefaultChartPayload.mockResolvedValueOnce(null);
    const res = await get('ok', null);
    expect(res.status).toBe(404);
    await expect(res.json()).resolves.toEqual({ error: 'not_found', message: 'chart not found' });
  });

  it('404s when the direct (non-default) payload is null', async () => {
    chartPayload.mockResolvedValueOnce(null);
    const res = await get('ok', 'all');
    expect(res.status).toBe(404);
    await expect(res.json()).resolves.toEqual({ error: 'not_found', message: 'chart not found' });
  });
});
