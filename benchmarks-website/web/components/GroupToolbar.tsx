// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

'use client';

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react';

import { FilterIcon } from '@/components/FilterBar';
import type { FilterUniverse } from '@/lib/chart-format';
import {
  applyGroupMacro,
  clearGroupSeriesFilter,
  getGroupSnapshot,
  resetGroup,
  setGroupY,
  subscribeGroup,
  toggleGroupSeries,
} from '@/lib/chart-store';

/**
 * The per-group toolbar between a group's summary card and its chart grid, the
 * client port of `landing.rs::per_group_toolbar` plus the section-17 wiring of
 * `chart-init.js`: group-level Y-axis buttons on the left, a centered "Filter
 * series" dropdown, and a Reset button on the right. CSS hides the toolbar
 * while the enclosing `<details>` is closed, mirroring the chart-grid rule.
 *
 * The dropdown's engine/format chips are macros (one click bulk-toggles every
 * known series whose tag matches); the series chip row populates lazily via the
 * group store as charts in the group hydrate and surface their `series_meta`.
 *
 * Resolution layering (enforced by each chart island's `applyFilters`):
 * per-card legend overrides win, the per-group `hiddenSeries` filter hides
 * next, the global filter hides last. The Y broadcast skips charts whose
 * per-chart Y toolbar was clicked (sticky), and Reset intentionally clears
 * only the group state, never the per-card overrides, matching v3.
 */
export function GroupToolbar({
  groupSlug,
  universe,
}: {
  groupSlug: string;
  universe: FilterUniverse;
}) {
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Memoized on `groupSlug` so re-renders do not unsubscribe/re-subscribe the
  // group store (mirrors the chart island's subscriber).
  const subscribeToGroup = useCallback(
    (cb: () => void) => subscribeGroup(groupSlug, cb),
    [groupSlug],
  );
  const snapshot = useSyncExternalStore(
    subscribeToGroup,
    () => getGroupSnapshot(groupSlug),
    () => getGroupSnapshot(groupSlug),
  );

  // Close on outside click and on Escape, mirroring the global dropdown.
  useEffect(() => {
    if (!open) {
      return;
    }
    const onClick = (e: MouseEvent): void => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') {
        setOpen(false);
      }
    };
    document.addEventListener('click', onClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('click', onClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  // `null` (the resting default and the post-Reset state) renders as linear,
  // matching each chart's own default.
  const yVisual = snapshot.groupY === 'log' ? 'log' : 'linear';
  const knownLabels = Object.keys(snapshot.knownSeries).sort();
  const hiddenCount = snapshot.hiddenSeries.length;

  return (
    <section className="group-toolbar" data-role="group-toolbar">
      <div className="toolbar-group group-toolbar-y" role="group" aria-label="Group Y-axis scale">
        <span className="toolbar-label">Y</span>
        <button
          className={`toolbar-btn${yVisual === 'linear' ? ' toolbar-btn--active' : ''}`}
          type="button"
          data-group-y="linear"
          onClick={() => setGroupY(groupSlug, 'linear')}
        >
          linear
        </button>
        <button
          className={`toolbar-btn${yVisual === 'log' ? ' toolbar-btn--active' : ''}`}
          type="button"
          data-group-y="log"
          onClick={() => setGroupY(groupSlug, 'log')}
        >
          log
        </button>
      </div>
      <div
        className={`group-filter-dropdown${open ? ' group-filter-dropdown--open' : ''}`}
        data-role="group-filter-dropdown"
        ref={dropdownRef}
      >
        <button
          className="control-btn filter-trigger group-filter-trigger"
          type="button"
          data-role="group-filter-trigger"
          aria-haspopup="true"
          aria-expanded={open}
          onClick={(e) => {
            e.stopPropagation();
            setOpen((o) => !o);
          }}
        >
          <FilterIcon />
          <span>Filter series</span>
          {hiddenCount > 0 && (
            <span className="filter-badge" data-role="group-filter-badge">
              {hiddenCount}
            </span>
          )}
        </button>
        <div
          className="filter-panel group-filter-panel"
          data-role="group-filter-panel"
          hidden={!open}
        >
          <MacroRow
            label="Engine"
            dim="engine"
            universe={universe.engines}
            groupSlug={groupSlug}
            snapshotHidden={snapshot.hiddenSeries}
            knownSeries={snapshot.knownSeries}
          />
          <MacroRow
            label="Format"
            dim="format"
            universe={universe.formats}
            groupSlug={groupSlug}
            snapshotHidden={snapshot.hiddenSeries}
            knownSeries={snapshot.knownSeries}
          />
          <div className="global-filter-row group-series-row">
            <span className="global-filter-label">Series</span>
            <button
              className="filter-chip filter-chip--all"
              type="button"
              data-group-filter="series"
              data-value="*"
              aria-pressed="false"
              onClick={() => clearGroupSeriesFilter(groupSlug)}
            >
              all
            </button>
            {/* Series chips hydrate as charts in this group surface their
                `series_meta`; until then the row only shows the macros above. */}
            <div className="group-series-chips" data-role="group-series-chips">
              {knownLabels.map((label) => {
                const isActive = !snapshot.hiddenSeries.includes(label);
                return (
                  <button
                    key={label}
                    className={`filter-chip${isActive ? ' filter-chip--active' : ''}`}
                    type="button"
                    data-group-filter="series"
                    data-value={label}
                    aria-pressed={isActive}
                    onClick={() => toggleGroupSeries(groupSlug, label)}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>
      <button
        className="group-toolbar-reset"
        type="button"
        data-role="group-toolbar-reset"
        onClick={() => resetGroup(groupSlug)}
      >
        Reset group
      </button>
    </section>
  );
}

/**
 * An engine/format macro row inside the per-group filter panel. A macro chip is
 * active iff at least one known series matches this tag AND every match is
 * currently visible; with no matching series the chip is inert and renders
 * inactive so the dropdown does not falsely advertise irrelevant filters.
 */
function MacroRow({
  label,
  dim,
  universe,
  groupSlug,
  snapshotHidden,
  knownSeries,
}: {
  label: string;
  dim: 'engine' | 'format';
  universe: string[];
  groupSlug: string;
  snapshotHidden: string[];
  knownSeries: Record<string, { engine?: string; format?: string }>;
}) {
  return (
    <div className="global-filter-row group-macro-row">
      <span className="global-filter-label">{label}</span>
      <button
        className="filter-chip filter-chip--all"
        type="button"
        data-group-filter={dim}
        data-value="*"
        aria-pressed="false"
        onClick={() => clearGroupSeriesFilter(groupSlug)}
      >
        all
      </button>
      {universe.map((value) => {
        const matching = Object.keys(knownSeries).filter((l) => knownSeries[l]?.[dim] === value);
        const isActive = matching.length > 0 && matching.every((l) => !snapshotHidden.includes(l));
        return (
          <button
            key={value}
            className={`filter-chip${isActive ? ' filter-chip--active' : ''}`}
            type="button"
            data-group-filter={dim}
            data-value={value}
            aria-pressed={isActive}
            onClick={() => applyGroupMacro(groupSlug, dim, value)}
          >
            {value}
          </button>
        );
      })}
    </div>
  );
}
