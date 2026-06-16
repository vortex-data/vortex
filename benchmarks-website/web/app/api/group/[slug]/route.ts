// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { NextResponse } from 'next/server';

import { READ_API_CACHE_CONTROL } from '@/lib/cache';
import { cachedDefaultGroupCharts } from '@/lib/data-cache';
import { collectGroupCharts } from '@/lib/queries';
import { groupKeyFromSlug, type GroupKey } from '@/lib/slug';
import { DEFAULT_COMMIT_WINDOW, parseCommitWindow } from '@/lib/window';

/**
 * `GET /api/group/{slug}` returns the [`GroupChartsResponse`] for one group: its
 * name, optional summary + description, and every chart inside it with the full
 * payload inlined.
 *
 * Status mapping and error envelope match the Axum server: a malformed slug is
 * `400` `{error: 'bad_request', message}`, a well-formed slug for a group with
 * no data is `404` `{error: 'not_found', message}`, and a populated group is
 * `200`. `?n=` selects the per-chart commit window (`all` is uncapped; numeric
 * values are floored to 1 and clamped to `MAX_NUMERIC_COMMIT_WINDOW`). The 200
 * payload is CDN-cached per full URL for five minutes via
 * [`READ_API_CACHE_CONTROL`].
 */
export async function GET(
  request: Request,
  { params }: { params: Promise<{ slug: string }> },
): Promise<NextResponse> {
  const { slug } = await params;
  let key: GroupKey;
  try {
    key = groupKeyFromSlug(slug);
  } catch {
    return NextResponse.json(
      { error: 'bad_request', message: 'invalid group slug' },
      { status: 400 },
    );
  }
  const window = parseCommitWindow(new URL(request.url).searchParams.get('n'));
  // The default last-100 window is served from the Data Cache (warm across
  // invocations, refreshed on ingest); every other window keeps the direct
  // query and rides the per-URL CDN cache only.
  const payload =
    window.kind === 'last' && window.n === DEFAULT_COMMIT_WINDOW
      ? await cachedDefaultGroupCharts(slug)
      : await collectGroupCharts(key, window);
  if (payload === null) {
    return NextResponse.json({ error: 'not_found', message: 'group not found' }, { status: 404 });
  }
  return NextResponse.json(payload, { headers: { 'cache-control': READ_API_CACHE_CONTROL } });
}
