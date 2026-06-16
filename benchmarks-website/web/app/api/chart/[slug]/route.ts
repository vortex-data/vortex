// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { NextResponse } from 'next/server';

import { READ_API_CACHE_CONTROL } from '@/lib/cache';
import { cachedDefaultChartPayload } from '@/lib/data-cache';
import { chartPayload } from '@/lib/queries';
import { chartKeyFromSlug, type ChartKey } from '@/lib/slug';
import { DEFAULT_COMMIT_WINDOW, parseCommitWindow } from '@/lib/window';

/**
 * `GET /api/chart/{slug}` returns the [`ChartResponse`] JSON for one chart.
 *
 * Status mapping and error envelope match the Axum server: a malformed slug is
 * `400` `{error: 'bad_request', message}`, a well-formed slug for a chart with
 * no data is `404` `{error: 'not_found', message}`, and a populated chart is
 * `200`. `?n=` selects the commit window (`all` is uncapped; numeric values are
 * floored to 1 and clamped to `MAX_NUMERIC_COMMIT_WINDOW`). The 200 payload is
 * CDN-cached per full URL for five minutes via [`READ_API_CACHE_CONTROL`].
 */
export async function GET(
  request: Request,
  { params }: { params: Promise<{ slug: string }> },
): Promise<NextResponse> {
  const { slug } = await params;
  let key: ChartKey;
  try {
    key = chartKeyFromSlug(slug);
  } catch {
    return NextResponse.json(
      { error: 'bad_request', message: 'invalid chart slug' },
      { status: 400 },
    );
  }
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
}
