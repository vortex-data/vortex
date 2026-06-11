// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { Chart } from '@/components/Chart';

describe('Chart (server-rendered card markup)', () => {
  const html = renderToStaticMarkup(
    <Chart slug="ra.eyJhIjoxfQ" name="gnomad" index={7} groupSlug="random_access" />,
  );

  it('renders the v3 chart-card contract attributes', () => {
    expect(html).toContain('class="chart-card"');
    expect(html).toContain('data-chart-index="7"');
    expect(html).toContain('data-chart-slug="ra.eyJhIjoxfQ"');
    expect(html).toContain('<canvas data-chart-index="7"');
  });

  it('renders the title permalink and the hidden downsample badge inside it', () => {
    expect(html).toContain('href="/chart/ra.eyJhIjoxfQ"');
    expect(html).toContain('data-role="downsample-badge"');
    expect(html).toMatch(/data-role="downsample-badge"[^>]*hidden/);
  });

  it('renders the per-card toolbar: throttled-input scope slider and Y buttons', () => {
    expect(html).toContain('id="scope-slider-7"');
    expect(html).toContain('data-role="scope-slider"');
    expect(html).toContain('type="range"');
    expect(html).toContain('data-y="linear"');
    expect(html).toContain('data-y="log"');
    // Linear ships highlighted, matching the v3 resting state.
    expect(html).toMatch(/toolbar-btn toolbar-btn--active[^>]*data-y="linear"/);
  });

  it('renders the tooltip host and the range strip with both handles', () => {
    expect(html).toContain('class="chart-tooltip-host"');
    expect(html).toContain('data-role="range-strip"');
    expect(html).toContain('data-role="range-window"');
    expect(html).toContain('data-role="range-handle-left"');
    expect(html).toContain('data-role="range-handle-right"');
  });
});
