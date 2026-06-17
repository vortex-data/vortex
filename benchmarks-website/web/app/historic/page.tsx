// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { GroupSection } from '@/components/GroupSection';
import { TpcSuiteSection } from '@/components/TpcSuiteSection';
import { clusterHistoricGroups } from '@/lib/historic-groups';
import { collectGroups } from '@/lib/queries';

// Rendered per request: reads every group from Postgres via `collectGroups()`,
// then clusters the TPC suites so storage / scale factor become in-place
// toggles. `force-dynamic` keeps `next build` DB-independent.
export const dynamic = 'force-dynamic';

/**
 * The Historic Data tab (`/historic`): the as-collected per-commit dashboard,
 * one section per group. TPC query suites are clustered into a single section
 * with storage (NVMe / S3) and scale-factor (SF=1 / SF=10 / …) toggle buttons;
 * everything else renders as a plain group. `data-chart-index` is assigned
 * page-wide so each chart's index is unique across every section.
 */
export default async function Historic() {
  const sections = clusterHistoricGroups(await collectGroups());
  if (sections.length === 0) {
    return <p className="empty">No data ingested yet.</p>;
  }
  let nextIndex = 0;
  return (
    <>
      {sections.map((section) => {
        const startIndex = nextIndex;
        if (section.kind === 'group') {
          nextIndex += section.group.charts.length;
          return (
            <GroupSection key={section.group.slug} group={section.group} startIndex={startIndex} />
          );
        }
        nextIndex += section.suite.pills.reduce((sum, pill) => sum + pill.charts.length, 0);
        return (
          <TpcSuiteSection key={section.suite.slug} suite={section.suite} startIndex={startIndex} />
        );
      })}
      {/* v3's landing_body early-returns on an empty database, so the no-JS hint
          only renders alongside actual chart shells. */}
      <noscript>
        <p className="no-script">JavaScript is required to render the charts.</p>
      </noscript>
    </>
  );
}
