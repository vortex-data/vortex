// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { Header } from '@/components/Header';

const UNIVERSE = { engines: ['datafusion', 'duckdb'], formats: ['parquet', 'vortex'] };

describe('Header (server-rendered chrome)', () => {
  const html = renderToStaticMarkup(
    <Header universe={UNIVERSE} initialEngines={[]} initialFormats={[]} />,
  );

  it('renders the hamburger toggle wired to the nav panel', () => {
    expect(html).toContain('data-role="nav-mobile-toggle"');
    expect(html).toContain('aria-controls="bench-nav-controls"');
    expect(html).toContain('aria-expanded="false"');
    expect(html).toContain('data-role="nav-controls"');
  });

  it('renders expand/collapse-all controls inside the nav panel', () => {
    expect(html).toContain('data-action="expand-all"');
    expect(html).toContain('data-action="collapse-all"');
    expect(html).toContain('Expand All');
    expect(html).toContain('Collapse All');
  });

  it('renders BOTH GitHub links: the desktop one and the mobile nav fallback', () => {
    // The mobile CSS hides `.repo-link-desktop` under 768px; the
    // `.nav-controls-github` link inside the hamburger panel covers mobile.
    expect(html).toContain('repo-link-desktop');
    expect(html).toContain('nav-controls-github');
    const matches = html.match(/href="https:\/\/github\.com\/vortex-data\/vortex"/g);
    expect(matches).toHaveLength(2);
  });

  it('renders the theme toggle with the v3 server-default label', () => {
    expect(html).toContain('data-role="theme-toggle"');
    expect(html).toContain('data-next-theme="light"');
    expect(html).toContain('theme-toggle-label');
  });

  it('renders the global filter dropdown with one chip per universe value', () => {
    expect(html).toContain('data-role="global-filter-bar"');
    expect(html).toContain('data-role="filter-trigger"');
    expect(html).toContain('data-role="filter-panel"');
    for (const value of [...UNIVERSE.engines, ...UNIVERSE.formats]) {
      expect(html).toContain(`data-value="${value}"`);
    }
    // One "all" reset chip per dimension row.
    const allChips = html.match(/data-value="\*"/g);
    expect(allChips).toHaveLength(2);
  });

  it('omits the filter dropdown when the universe is empty', () => {
    const bare = renderToStaticMarkup(
      <Header universe={{ engines: [], formats: [] }} initialEngines={[]} initialFormats={[]} />,
    );
    expect(bare).not.toContain('data-role="global-filter-bar"');
  });
});
