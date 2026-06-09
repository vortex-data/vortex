// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { Footer } from '@/components/Footer';
import { GroupSection } from '@/components/GroupSection';
import { Header } from '@/components/Header';
import { collectGroups } from '@/lib/queries';

// Rendered per request for now: the landing page reads every group from
// Postgres via `collectGroups()` on each render. The `/api/*` routes are
// CDN-cached for five minutes via `READ_API_CACHE_CONTROL`; the equivalent
// landing-page caching (a `Cache-Control` header on `/` via `vercel.json`, or
// time-based revalidation once the database is reachable at build time) is
// wired at the Vercel-deploy step (PR-4.5). `force-dynamic` keeps `next build`
// independent of a live database in the meantime.
export const dynamic = 'force-dynamic';

/**
 * The landing page: a server-rendered section per group in canonical
 * `GROUP_ORDER`, each with its summary card and a grid of (initially empty)
 * chart-card shells. `data-chart-index` is assigned page-wide so each chart's
 * index is unique across every group (matching v3's `landing_body` counter).
 */
export default async function Home() {
  const groups = await collectGroups();
  let nextIndex = 0;
  return (
    <>
      <Header />
      <main>
        {groups.length === 0 ? (
          <p className="empty">No data ingested yet.</p>
        ) : (
          <>
            {groups.map((group) => {
              const startIndex = nextIndex;
              nextIndex += group.charts.length;
              return <GroupSection key={group.slug} group={group} startIndex={startIndex} />;
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
