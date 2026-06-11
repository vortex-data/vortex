// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { GroupToolbar } from '@/components/GroupToolbar';

const UNIVERSE = { engines: ['datafusion', 'duckdb'], formats: ['parquet', 'vortex'] };

describe('GroupToolbar (server-rendered markup)', () => {
  const html = renderToStaticMarkup(<GroupToolbar groupSlug="random_access" universe={UNIVERSE} />);

  it('renders the v3 group-toolbar contract: Y buttons, dropdown, reset', () => {
    expect(html).toContain('data-role="group-toolbar"');
    expect(html).toContain('data-group-y="linear"');
    expect(html).toContain('data-group-y="log"');
    expect(html).toContain('data-role="group-filter-dropdown"');
    expect(html).toContain('data-role="group-filter-trigger"');
    expect(html).toContain('data-role="group-filter-panel"');
    expect(html).toContain('data-role="group-toolbar-reset"');
    expect(html).toContain('Reset group');
  });

  it('ships linear highlighted as the resting group-Y default', () => {
    expect(html).toMatch(/toolbar-btn toolbar-btn--active[^>]*data-group-y="linear"/);
  });

  it('renders engine/format macro rows plus an empty lazy series row', () => {
    for (const value of [...UNIVERSE.engines, ...UNIVERSE.formats]) {
      expect(html).toContain(`data-value="${value}"`);
    }
    expect(html).toContain('data-role="group-series-chips"');
    // Three "all" chips: engine row, format row, series row.
    const allChips = html.match(/data-value="\*"/g);
    expect(allChips).toHaveLength(3);
    // Macro chips render inactive while no series is known (the v3
    // post-`syncGroupChipsUi` state): no known series, nothing to match.
    expect(html).not.toContain('filter-chip filter-chip--active');
  });
});
