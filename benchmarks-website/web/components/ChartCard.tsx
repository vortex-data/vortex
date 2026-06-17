// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { HistoricChart } from '@/components/HistoricChart';
import type { ChartLink } from '@/lib/queries';

/**
 * One chart card: the title (linking to the chart permalink) and the robust
 * band chart. The [`HistoricChart`] island lazily fetches `/api/chart/[slug]`
 * when the card scrolls into view (so a collapsed group — or a hidden TPC
 * storage/SF panel — fetches nothing), preserving v4's lazy-on-expand load
 * model, then renders the median + p25–p75 band + outlier dots.
 *
 * Shared by [`GroupSection`] and the TPC suite section so both render charts
 * identically.
 */
export function ChartCard({ link, index }: { link: ChartLink; index: number }) {
  return (
    <section className="chart-card" data-chart-index={index} data-chart-slug={link.slug}>
      <h3 className="chart-card-title">
        <a href={`/chart/${link.slug}`}>{link.name}</a>
      </h3>
      <HistoricChart slug={link.slug} />
    </section>
  );
}
