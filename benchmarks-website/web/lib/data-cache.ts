// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { unstable_cache } from 'next/cache';

import type { FilterUniverse } from '@/lib/chart-format';
import {
  chartPayload,
  collectFilterUniverse,
  collectGroupCharts,
  collectGroups,
  type ChartResponse,
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
 * caps staleness at twenty-four hours if that hook ever fails to fire, so the
 * layer degrades to bounded staleness rather than serving stale data forever.
 *
 * The window is one day rather than one hour because this is a low-traffic
 * site: a longer backstop keeps the default last-100 window warm across
 * overnight idle gaps, so a CDN miss reads this Data Cache instead of paying
 * the multi-second cold database fill. The freshness cost is bounded by the
 * revalidate hook, which flushes the tag on every ingest once its env is set.
 */
export const DATA_CACHE_BACKSTOP_SECONDS = 86400;

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
