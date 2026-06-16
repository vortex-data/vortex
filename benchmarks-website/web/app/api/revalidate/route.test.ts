// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// @vitest-environment node

import { afterEach, describe, expect, it, vi } from 'vitest';

const revalidateTag = vi.fn();
vi.mock('next/cache', () => ({ revalidateTag: (tag: string) => revalidateTag(tag) }));

// Mock data-cache so the module-level `unstable_cache` calls in that file do not
// execute during this test (which only mocks `revalidateTag`, not `unstable_cache`).
vi.mock('@/lib/data-cache', () => ({ BENCH_DATA_TAG: 'bench-data' }));

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

  it('503s and does not revalidate when the token env is explicitly empty (fail closed)', async () => {
    process.env.BENCH_REVALIDATE_TOKEN = '';
    const res = await POST(post('anything'));
    expect(res.status).toBe(503);
    expect(revalidateTag).not.toHaveBeenCalled();
  });

  it('401s on a missing or wrong token', async () => {
    process.env.BENCH_REVALIDATE_TOKEN = 'secret-token-value';
    expect((await POST(post(null))).status).toBe(401);
    // Short wrong token (length mismatch, returns before timingSafeEqual).
    expect((await POST(post('wrong'))).status).toBe(401);
    // Same-length wrong token (exercises the timingSafeEqual rejection path).
    expect((await POST(post('secret-token-valuX'))).status).toBe(401);
    // Empty bearer token.
    expect((await POST(post(''))).status).toBe(401);
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
