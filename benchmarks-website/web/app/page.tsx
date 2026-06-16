// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { Footer } from '@/components/Footer';
import { GroupNav } from '@/components/GroupNav';
import { GroupSection } from '@/components/GroupSection';
import { Header } from '@/components/Header';
import { parseFilterCsv, singleSearchParam } from '@/lib/chart-format';
import { cachedFilterUniverse, cachedGroups } from '@/lib/data-cache';

// Rendered per request, with CDN caching layered on by `vercel.json`: each
// render reads every group from Postgres via `collectGroups()`, and Vercel's
// CDN caches the response for five minutes via a `Vercel-CDN-Cache-Control`
// header rule on `/`, matching the `/api/*` routes' `READ_API_CACHE_CONTROL`
// cadence. A plain `Cache-Control` rule cannot express this: Next.js emits
// `Cache-Control: no-store` in the function response for `force-dynamic`
// pages, and function-emitted `Cache-Control` takes precedence over header
// rules from config files. `Vercel-CDN-Cache-Control` is consumed (and
// stripped) by Vercel's CDN alone at the highest precedence, so the config
// rule drives CDN caching while browsers still revalidate on every load.
// `force-dynamic` keeps `next build` independent of a live database.
export const dynamic = 'force-dynamic';

/**
 * The landing page: a server-rendered section per group in canonical
 * `GROUP_ORDER`, each with its summary card, per-group toolbar, and a grid of
 * chart-card client islands. `data-chart-index` is assigned page-wide so each
 * chart's index is unique across every group (matching v3's `landing_body`
 * counter).
 *
 * `?engine=` / `?format=` are the global filter's URL allowlists (CSV); they
 * seed the client filter store via the header's filter bar, exactly as v3's
 * `filter_state_script` bridge did.
 */
export default async function Home({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const initialEngines = parseFilterCsv(singleSearchParam(params.engine));
  const initialFormats = parseFilterCsv(singleSearchParam(params.format));

  const [groups, universe] = await Promise.all([cachedGroups(), cachedFilterUniverse()]);
  let nextIndex = 0;
  return (
    <>
      <Header universe={universe} initialEngines={initialEngines} initialFormats={initialFormats} />
      <GroupNav groups={groups.map((group) => ({ name: group.name, slug: group.slug }))} />
      <main>
        {groups.length === 0 ? (
          <p className="empty">No data ingested yet.</p>
        ) : (
          <>
            {groups.map((group) => {
              const startIndex = nextIndex;
              nextIndex += group.charts.length;
              return (
                <GroupSection
                  key={group.slug}
                  group={group}
                  startIndex={startIndex}
                  universe={universe}
                />
              );
            })}
            {/* v3's landing_body early-returns on an empty database, so the
                no-JS hint only renders alongside actual chart shells. */}
            <noscript>
              <p className="no-script">JavaScript is required to render the charts.</p>
            </noscript>
          </>
        )}
      </main>
      <Footer />
    </>
  );
}
