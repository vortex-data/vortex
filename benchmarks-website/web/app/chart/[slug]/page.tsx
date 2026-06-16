// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { Metadata } from 'next';
import { notFound } from 'next/navigation';
import { cache } from 'react';

import { Chart } from '@/components/Chart';
import { Footer } from '@/components/Footer';
import { Header } from '@/components/Header';
import { parseFilterCsv, singleSearchParam, unitKindLabel } from '@/lib/chart-format';
import { cachedDefaultChartPayload } from '@/lib/data-cache';
import { chartPayload, collectFilterUniverse, type ChartResponse } from '@/lib/queries';
import { chartKeyFromSlug, type ChartKey } from '@/lib/slug';
import { DEFAULT_COMMIT_WINDOW, parseCommitWindow } from '@/lib/window';

// Rendered per request, like the landing page: the payload comes from Postgres
// at request time and `force-dynamic` keeps `next build` independent of a live
// database. Vercel's CDN caches the rendered page for five minutes via the
// `Vercel-CDN-Cache-Control` rule on `/chart/:slug` in `vercel.json` (see the
// landing page's note on why a plain `Cache-Control` rule cannot do this).
export const dynamic = 'force-dynamic';

type Params = Promise<{ slug: string }>;
type SearchParams = Promise<Record<string, string | string[] | undefined>>;

/**
 * Resolve the chart payload for this request, deduped across `generateMetadata`
 * and the page body via React's `cache` (both run in the same render pass).
 * Returns `null` for a malformed slug or a chart with no data; both render the
 * 404 page, matching v3's `chart_page` (which maps `ChartKey::from_slug`
 * failures to 404, unlike the JSON API's 400).
 */
const getChart = cache(async (slug: string, n: string | null): Promise<ChartResponse | null> => {
  let key: ChartKey;
  try {
    key = chartKeyFromSlug(slug);
  } catch {
    return null;
  }
  // The default last-100 window reads through the Data Cache (warm across
  // invocations, refreshed on ingest); every other window keeps the direct query.
  const window = parseCommitWindow(n);
  return window.kind === 'last' && window.n === DEFAULT_COMMIT_WINDOW
    ? cachedDefaultChartPayload(slug)
    : chartPayload(key, window);
});

/** The browser-tab title mirrors v3: `{display_name} - Vortex Benchmarks`. */
export async function generateMetadata({
  params,
  searchParams,
}: {
  params: Params;
  searchParams: SearchParams;
}): Promise<Metadata> {
  const { slug } = await params;
  const sp = await searchParams;
  const chart = await getChart(slug, singleSearchParam(sp.n));
  if (chart === null) {
    return {};
  }
  return { title: `${chart.display_name} - Vortex Benchmarks` };
}

/**
 * The chart permalink page (`/chart/{slug}`), the RSC port of v3's
 * `chart_page` + `chart.rs::chart_body`: the full header chrome, a meta line
 * (unit, series count, commit count), and a single chart card with the payload
 * inlined as a prop (no client fetch for the initial window; the island still
 * performs the one-shot `?n=all` upgrade on interaction with unloaded
 * history).
 *
 * `?n=` selects the server-side commit window exactly as on the API route.
 * Two small deliberate deviations from v3, both noted at the PR level: the
 * card keeps its title row (which on this page links to itself), and the
 * downsample badge lives in that title row where it actually updates; v3's
 * chart-page badge slot sat outside the card in the meta line, where
 * `chart-init.js` never found it, so it could never light up.
 */
export default async function ChartPage({
  params,
  searchParams,
}: {
  params: Params;
  searchParams: SearchParams;
}) {
  const { slug } = await params;
  const sp = await searchParams;
  // A malformed slug 404s before either query is issued (no needless universe
  // round-trip, and the 404 stands even with the database unreachable); for
  // well-formed slugs the chart payload and the filter universe are
  // independent queries run in parallel (getChart is React-cache'd, so the
  // generateMetadata dedupe is unaffected), matching the landing page.
  try {
    chartKeyFromSlug(slug);
  } catch {
    notFound();
  }
  // The filter universe is OPTIONAL chrome, matching v3's `.ok()` on
  // `cached_filter_universe`: a failed universe query renders the page without
  // the filter dropdown rather than failing an otherwise renderable chart.
  const [chart, universe] = await Promise.all([
    getChart(slug, singleSearchParam(sp.n)),
    collectFilterUniverse().catch(() => undefined),
  ]);
  if (chart === null) {
    notFound();
  }
  const initialEngines = parseFilterCsv(singleSearchParam(sp.engine));
  const initialFormats = parseFilterCsv(singleSearchParam(sp.format));

  const seriesCount = Object.keys(chart.series).length;
  const commitCount = chart.commits.length;
  return (
    <>
      <Header universe={universe} initialEngines={initialEngines} initialFormats={initialFormats} />
      <main>
        <p className="chart-meta">
          {'unit: '}
          <code>{unitKindLabel(chart.unit_kind)}</code>
          {' · '}
          {seriesCount} series · {commitCount} commit{commitCount !== 1 ? 's' : ''}
        </p>
        <Chart slug={slug} name={chart.display_name} index={0} initialPayload={chart} />
        <noscript>
          <p className="no-script">JavaScript is required to render the chart.</p>
        </noscript>
      </main>
      <Footer />
    </>
  );
}
