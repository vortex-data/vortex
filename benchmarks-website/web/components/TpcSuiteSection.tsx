// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

'use client';

import { useState } from 'react';

import { ChartCard } from './ChartCard';
import { SummaryCard } from './SummaryCard';
import type { TpcSuite } from '@/lib/historic-groups';

/**
 * A clustered TPC suite section, the React port of the landing TPC fan-out
 * (`landing.rs` + `current.rs::storage_toggle_pills` / `sf_toggle_pills`).
 *
 * One collapsible section with a storage row (NVMe / S3) and a scale-factor row
 * (SF=1 / SF=10 / …) of toggle buttons that swap which `(storage, scale-factor)`
 * panel's charts are visible — in place, no navigation. Every panel is rendered
 * up front but hidden until selected; combined with [`ChartCard`]'s lazy fetch,
 * an unselected (or collapsed) panel fetches nothing, so the page load stays
 * cheap and switching back to a panel reuses its already-built charts.
 */
function storageOrder(storage: string): number {
  return storage === 'nvme' ? 0 : storage === 's3' ? 1 : 2;
}

export function TpcSuiteSection({ suite, startIndex }: { suite: TpcSuite; startIndex: number }) {
  const initial = suite.pills.find((p) => p.current) ?? suite.pills[0];
  const [activeSlug, setActiveSlug] = useState(initial.slug);
  const active = suite.pills.find((p) => p.slug === activeSlug) ?? initial;

  // Picking a storage keeps the current SF when that combination exists, else
  // falls to the largest SF for that storage (mirrors the default-pick rule).
  const selectStorage = (storage: string) => {
    const same = suite.pills.find((p) => p.storage === storage && p.sfValue === active.sfValue);
    const target =
      same ??
      suite.pills
        .filter((p) => p.storage === storage)
        .sort((a, b) => Number(b.sfValue) - Number(a.sfValue))[0];
    if (target !== undefined) {
      setActiveSlug(target.slug);
    }
  };
  const selectSf = (sf: string) => {
    const same = suite.pills.find((p) => p.sfValue === sf && p.storage === active.storage);
    const target =
      same ??
      suite.pills
        .filter((p) => p.sfValue === sf)
        .sort((a, b) => storageOrder(a.storage) - storageOrder(b.storage))[0];
    if (target !== undefined) {
      setActiveSlug(target.slug);
    }
  };

  // Page-wide chart index base per panel (the cosmetic `data-chart-index`).
  let running = startIndex;
  const pillStart = new Map<string, number>();
  for (const pill of suite.pills) {
    pillStart.set(pill.slug, running);
    running += pill.charts.length;
  }

  const chartCount = active.charts.length;
  // The toggles live INSIDE the `<summary>`, so a click on one would otherwise
  // toggle the disclosure (the summary's default action). `preventDefault` on
  // the button click cancels that while still running our selection handler.
  const onPick = (e: React.MouseEvent, fn: () => void) => {
    e.preventDefault();
    e.stopPropagation();
    fn();
  };
  return (
    <section className="group-details" data-group-name={suite.name} data-group-slug={suite.slug}>
      <details className="group-disclosure">
        <summary className="group-summary">
          <span className="group-summary-row">
            <span className="group-name">{suite.name}</span>
            <span className="group-summary-controls">
              <span
                className="dim-toggle"
                data-role="storage-toggle"
                role="group"
                aria-label="Storage"
              >
                {suite.storages.map((s) => (
                  <button
                    key={s.value}
                    className={`dim-btn${s.value === active.storage ? ' dim-btn--active' : ''}`}
                    type="button"
                    data-storage={s.value}
                    aria-pressed={s.value === active.storage}
                    onClick={(e) => onPick(e, () => selectStorage(s.value))}
                  >
                    {s.label}
                  </button>
                ))}
              </span>
              <span
                className="dim-toggle"
                data-role="sf-toggle"
                role="group"
                aria-label="Scale factor"
              >
                {suite.scaleFactors.map((sf) => (
                  <button
                    key={sf.value}
                    className={`dim-btn${sf.value === active.sfValue ? ' dim-btn--active' : ''}`}
                    type="button"
                    data-sf={sf.value}
                    aria-pressed={sf.value === active.sfValue}
                    onClick={(e) => onPick(e, () => selectSf(sf.value))}
                  >
                    {sf.label}
                  </button>
                ))}
              </span>
            </span>
            <span className="group-count">
              {chartCount} chart{chartCount !== 1 ? 's' : ''}
            </span>
          </span>
        </summary>
      </details>
      <SummaryCard summary={active.summary} />
      <div className="history-fanout">
        <div className="history-sf-sets">
          {suite.pills.map((pill) => (
            <section
              key={pill.slug}
              className="speedup-sf"
              data-sf={pill.sfValue}
              data-storage={pill.storage}
              hidden={pill.slug !== activeSlug}
            >
              <div className="chart-grid">
                {pill.charts.map((link, i) => (
                  <ChartCard
                    key={link.slug}
                    link={link}
                    index={(pillStart.get(pill.slug) ?? 0) + i}
                  />
                ))}
              </div>
            </section>
          ))}
        </div>
      </div>
    </section>
  );
}
