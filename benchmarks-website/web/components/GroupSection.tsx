// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { ChartCard } from '@/components/ChartCard';
import { SummaryCard } from '@/components/SummaryCard';
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
 * Each chart renders as an empty `.chart-card` shell carrying `data-chart-slug`
 * (the per-chart payload key) and a page-unique `data-chart-index`; the chart
 * client island added in PR-4.4.b hydrates the `<canvas>` from
 * `/api/chart/[slug]` on group-open. The per-group / per-chart toolbars, range
 * strip, and tooltip host are likewise deferred to PR-4.4.b.
 *
 * `startIndex` is the running page-wide chart index of this group's first
 * chart, so `data-chart-index` is unique across every chart on the page.
 */
export function GroupSection({ group, startIndex }: { group: Group; startIndex: number }) {
  const chartCount = group.charts.length;
  return (
    <section className="group-details" data-group-name={group.name} data-group-slug={group.slug}>
      <details className="group-disclosure">
        <summary className="group-summary">
          <span className="group-summary-row">
            <span className="group-name">{group.name}</span>
            <span className="group-count">
              {chartCount} chart{chartCount !== 1 ? 's' : ''}
            </span>
          </span>
        </summary>
      </details>
      <SummaryCard summary={group.summary} />
      <div className="chart-grid">
        {group.charts.map((link, i) => (
          <ChartCard key={link.slug} link={link} index={startIndex + i} />
        ))}
      </div>
    </section>
  );
}
