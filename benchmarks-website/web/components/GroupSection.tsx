// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { Chart } from '@/components/Chart';
import { GroupToolbar } from '@/components/GroupToolbar';
import { SummaryCard } from '@/components/SummaryCard';
import type { FilterUniverse } from '@/lib/chart-format';
import type { Group } from '@/lib/queries';

/**
 * One group's landing-page section, the server-component port of
 * `server/src/html/landing.rs`'s per-group markup.
 *
 * The `<details>` disclosure wraps ONLY its `<summary>` header; the summary
 * card and the chart grid are siblings inside `.group-details`, so the CSS rule
 * `.group-disclosure:not([open]) ~ .chart-grid` hides the charts while the
 * group is collapsed (native `<details>` toggling, no JavaScript). The summary
 * card is intentionally outside that rule, so the at-a-glance rankings stay
 * visible whether or not the group is expanded.
 *
 * Each chart renders as a [`Chart`] client island carrying `data-chart-slug`
 * and a page-unique `data-chart-index`; the island lazily fetches
 * `/api/chart/[slug]` when this group's disclosure opens (it listens for the
 * native `toggle` event of the enclosing `details.group-disclosure`).
 *
 * The per-group toolbar (group Y override + series filter + reset) sits
 * between the summary card and the chart grid, exactly as in v3's
 * `landing_body`; CSS hides it while the disclosure is closed.
 *
 * `startIndex` is the running page-wide chart index of this group's first
 * chart, so `data-chart-index` is unique across every chart on the page.
 */
export function GroupSection({
  group,
  startIndex,
  universe,
}: {
  group: Group;
  startIndex: number;
  universe: FilterUniverse;
}) {
  const chartCount = group.charts.length;
  return (
    <section className="group-details" data-group-name={group.name} data-group-slug={group.slug}>
      <details className="group-disclosure">
        <summary className="group-summary">
          <span className="group-summary-row">
            <span className="group-name">{group.name}</span>
            {group.description !== undefined && (
              <span
                className="group-info-icon"
                tabIndex={0}
                role="note"
                aria-label={group.description}
                data-tooltip={group.description}
              >
                ⓘ
              </span>
            )}
            <span className="group-count">
              {chartCount} chart{chartCount !== 1 ? 's' : ''}
            </span>
          </span>
        </summary>
      </details>
      <SummaryCard summary={group.summary} />
      <GroupToolbar groupSlug={group.slug} universe={universe} />
      <div className="chart-grid">
        {group.charts.map((link, i) => (
          <Chart
            key={link.slug}
            slug={link.slug}
            name={link.name}
            index={startIndex + i}
            groupSlug={group.slug}
          />
        ))}
      </div>
    </section>
  );
}
