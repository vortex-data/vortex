// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { SpeedupSection } from '@/components/SpeedupSection';
import { collectSpeedupGroups } from '@/lib/synthesis';

// Rendered per request: the synthesized Vortex-vs-Parquet snapshot reads every
// group's full-history payloads from Postgres and reduces them to the latest
// commit's speedup distributions. `force-dynamic` keeps `next build` DB-free.
export const dynamic = 'force-dynamic';

/**
 * The Latest Commit tab (`/latest`): the synthesized head-to-head view, the
 * React port of `current.rs::current_body`. One section per group; each chart
 * distils a suite into a per-item Vortex-vs-Parquet speedup distribution with
 * swappable-format dropdowns.
 */
export default async function Latest() {
  const groups = await collectSpeedupGroups();
  if (groups.length === 0) {
    return <p className="empty">No data ingested yet.</p>;
  }
  return (
    <section className="current">
      <header className="current-intro">
        <h2 className="current-headline">Vortex vs Parquet, head to head.</h2>
        <div className="methodology">
          <p className="methodology-text">
            Each chart distils one benchmark suite into a single{' '}
            <strong>Vortex / Parquet ratio</strong> at the latest develop commit — geometric mean
            over the suite&rsquo;s items (queries, datasets, access patterns).{' '}
            <strong>1× is parity; above 1× means Vortex wins</strong> (faster for time, smaller for
            size). Swap either side with the dropdowns; <a href="/historic">Historic Data</a> plots
            the same number at every commit.
          </p>
        </div>
      </header>
      {groups.map((group) => (
        <SpeedupSection key={group.slug} group={group} />
      ))}
    </section>
  );
}
