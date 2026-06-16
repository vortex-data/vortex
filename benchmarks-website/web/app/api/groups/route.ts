// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { NextResponse } from 'next/server';

import { READ_API_CACHE_CONTROL } from '@/lib/cache';
import { cachedGroups } from '@/lib/data-cache';
import { type GroupsResponse } from '@/lib/queries';

/**
 * `GET /api/groups` returns every benchmark group in canonical `GROUP_ORDER`,
 * each with its chart links, an optional v2-compatible summary, and an optional
 * editorial description.
 *
 * The handler renders per request (keeping `next build` independent of a live
 * database); the edge CDN caches the 200 payload for five minutes via
 * [`READ_API_CACHE_CONTROL`].
 */
export async function GET(): Promise<NextResponse> {
  const groups = await cachedGroups();
  const body: GroupsResponse = { groups };
  return NextResponse.json(body, { headers: { 'cache-control': READ_API_CACHE_CONTROL } });
}
