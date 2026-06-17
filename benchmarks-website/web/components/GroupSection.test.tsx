// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { GroupSection } from '@/components/GroupSection';
import type { Group } from '@/lib/queries';

const RANDOM_ACCESS: Group = {
  name: 'Random Access',
  slug: 'random_access.eyJrIjoiUmFuZG9tQWNjZXNzR3JvdXAifQ',
  charts: [
    { name: 'gnomad', slug: 'ra.eyJhIjoxfQ' },
    { name: 'taxi', slug: 'ra.eyJhIjoyfQ' },
  ],
  summary: {
    type: 'randomAccess',
    title: 'Random Access Performance',
    rankings: [{ name: 'vortex', time: 1_500_000, ratio: 1 }],
    explanation: 'lower is better',
  },
  description: 'Tests selecting arbitrary row indices on NVMe',
};

describe('GroupSection', () => {
  it('renders the group-details shell with a collapsed disclosure', () => {
    const html = renderToStaticMarkup(<GroupSection group={RANDOM_ACCESS} startIndex={5} />);
    expect(html).toContain('class="group-details"');
    expect(html).toContain('data-group-name="Random Access"');
    expect(html).toContain(`data-group-slug="${RANDOM_ACCESS.slug}"`);
    // The disclosure is collapsed by default (no `open` attribute on the tag).
    expect(html).toContain('<details class="group-disclosure">');
    expect(html).toContain('<span class="group-name">Random Access</span>');
  });

  it('renders a pluralized chart count and no inline description', () => {
    const html = renderToStaticMarkup(<GroupSection group={RANDOM_ACCESS} startIndex={0} />);
    // The description lives on the Current page; the Historic section shows
    // neither a blurb nor the old hover info-icon.
    expect(html).not.toContain('class="group-blurb"');
    expect(html).not.toContain('class="group-info-icon"');
    expect(html).not.toContain('Tests selecting arbitrary row indices on NVMe');
    expect(html).toContain('2 charts');
  });

  it('renders the summary card inside the section', () => {
    const html = renderToStaticMarkup(<GroupSection group={RANDOM_ACCESS} startIndex={0} />);
    expect(html).toContain('class="benchmark-scores-summary"');
    expect(html).toContain('Random Access Performance');
  });

  it('renders one chart-card shell per chart with page-wide indices and permalinks', () => {
    const html = renderToStaticMarkup(<GroupSection group={RANDOM_ACCESS} startIndex={5} />);
    expect(html).toContain('class="chart-grid"');
    // First chart takes startIndex, second takes startIndex + 1.
    expect(html).toContain('data-chart-index="5"');
    expect(html).toContain('data-chart-slug="ra.eyJhIjoxfQ"');
    expect(html).toContain('data-chart-index="6"');
    expect(html).toContain('data-chart-slug="ra.eyJhIjoyfQ"');
    expect(html).toContain('href="/chart/ra.eyJhIjoxfQ"');
    expect(html).toContain('href="/chart/ra.eyJhIjoyfQ"');
    expect(html).toContain('<canvas');
  });

  it('uses the singular "chart" label for a single-chart group and omits a missing description', () => {
    const single: Group = {
      name: 'Vector Search',
      slug: 'vs.eyJrIjoiVmVjdG9yU2VhcmNoR3JvdXAifQ',
      charts: [{ name: 'recall', slug: 'vs.eyJhIjoxfQ' }],
    };
    const html = renderToStaticMarkup(<GroupSection group={single} startIndex={0} />);
    expect(html).toContain('<span class="group-count">1 chart</span>');
    // No description -> no blurb (and the info-icon is gone entirely).
    expect(html).not.toContain('class="group-blurb"');
    expect(html).not.toContain('class="group-info-icon"');
    // A group with no summary renders no summary card.
    expect(html).not.toContain('class="benchmark-scores-summary"');
  });
});
