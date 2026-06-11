// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

'use client';

import { useEffect, useRef, useState, useSyncExternalStore } from 'react';

import { seedActiveFromAllowlist, type FilterUniverse } from '@/lib/chart-format';
import {
  getGlobalFilterSnapshot,
  initGlobalFilter,
  subscribeGlobalFilter,
  toggleGlobalFilterValue,
} from '@/lib/chart-store';

/**
 * The global engine/format filter dropdown in the sticky header, the client
 * port of `server/src/html/filter.rs` plus the section-16 wiring of
 * `chart-init.js`. The trigger button shows a funnel icon plus a badge counting
 * how many chips are OFF (how many things the filter is hiding); the panel
 * holds one toggle-chip row per dimension.
 *
 * Chip semantics: a chip's active state mirrors the visibility of that
 * engine/format; with every chip in a row active no filter is applied for that
 * dimension. The "all" chip is a one-shot reset, never a current-state
 * indicator. Every change re-paints the chips (via the store subscription, here
 * and in every chart island) and syncs the URL `?engine=`/`?format=` allowlists
 * with `history.replaceState`, so a refresh or share preserves the view; the
 * params are omitted when a row is fully active so the no-filter URL is clean.
 */
export function FilterBar({
  universe,
  initialEngines,
  initialFormats,
}: {
  universe: FilterUniverse;
  /** URL `?engine=` allowlist parsed server-side; empty means no filter. */
  initialEngines: string[];
  /** URL `?format=` allowlist parsed server-side; empty means no filter. */
  initialFormats: string[];
}) {
  const [open, setOpen] = useState(false);
  const barRef = useRef<HTMLDivElement>(null);

  // Seed the tab-wide store from the server-provided universe + URL state.
  // Runs on mount (and again after a soft navigation re-mounts the header), so
  // the store tracks the most recently rendered page's URL.
  useEffect(() => {
    initGlobalFilter(universe, initialEngines, initialFormats);
    // The arrays are fresh per render but value-stable per page; re-seeding on
    // a literal identity change alone would clobber in-flight chip state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const snapshot = useSyncExternalStore(
    subscribeGlobalFilter,
    getGlobalFilterSnapshot,
    getGlobalFilterSnapshot,
  );

  // The chip rows render the PROPS universe (available on the server) with
  // active states from the store once it is seeded; before the mount effect
  // seeds it (server render + first client paint) the active sets fall back to
  // the same allowlist resolution the seeding applies, so the server markup
  // carries the correct chips and active states, matching v3's server-rendered
  // filter rows.
  const seeded = snapshot.universe.engines.length > 0 || snapshot.universe.formats.length > 0;
  const activeEngines = seeded
    ? snapshot.active.engines
    : seedActiveFromAllowlist(initialEngines, universe.engines);
  const activeFormats = seeded
    ? snapshot.active.formats
    : seedActiveFromAllowlist(initialFormats, universe.formats);

  // Close on outside click and on Escape, mirroring v3's document listeners.
  useEffect(() => {
    if (!open) {
      return;
    }
    const onClick = (e: MouseEvent): void => {
      if (barRef.current && !barRef.current.contains(e.target as Node)) {
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

  const hiddenCount =
    Math.max(0, universe.engines.length - activeEngines.length) +
    Math.max(0, universe.formats.length - activeFormats.length);

  const onChipClick = (dim: 'engine' | 'format', value: string): void => {
    toggleGlobalFilterValue(dim, value);
    syncFilterUrl();
  };

  return (
    <div
      className={`filter-dropdown${open ? ' filter-dropdown--open' : ''}`}
      data-role="global-filter-bar"
      ref={barRef}
    >
      <button
        className="control-btn filter-trigger"
        type="button"
        data-role="filter-trigger"
        aria-haspopup="true"
        aria-expanded={open}
        onClick={(e) => {
          e.stopPropagation();
          setOpen((o) => !o);
        }}
      >
        <FilterIcon />
        <span>Filters</span>
        {hiddenCount > 0 && (
          <span className="filter-badge" data-role="filter-badge">
            {hiddenCount}
          </span>
        )}
      </button>
      <div className="filter-panel" data-role="filter-panel" hidden={!open}>
        <FilterRow
          label="Engine"
          dim="engine"
          universe={universe.engines}
          active={activeEngines}
          onChipClick={onChipClick}
        />
        <FilterRow
          label="Format"
          dim="format"
          universe={universe.formats}
          active={activeFormats}
          onChipClick={onChipClick}
        />
      </div>
    </div>
  );
}

/** One row of chips inside the filter panel: the "all" reset chip plus one
 * toggle chip per universe value. */
function FilterRow({
  label,
  dim,
  universe,
  active,
  onChipClick,
}: {
  label: string;
  dim: 'engine' | 'format';
  universe: string[];
  active: string[];
  onChipClick: (dim: 'engine' | 'format', value: string) => void;
}) {
  return (
    <div className="global-filter-row">
      <span className="global-filter-label">{label}</span>
      <button
        className="filter-chip filter-chip--all"
        type="button"
        data-filter={dim}
        data-value="*"
        aria-pressed="false"
        onClick={() => onChipClick(dim, '*')}
      >
        all
      </button>
      {universe.map((value) => {
        const isActive = active.includes(value);
        return (
          <button
            key={value}
            className={`filter-chip${isActive ? ' filter-chip--active' : ''}`}
            type="button"
            data-filter={dim}
            data-value={value}
            aria-pressed={isActive}
            onClick={() => onChipClick(dim, value)}
          >
            {value}
          </button>
        );
      })}
    </div>
  );
}

/**
 * Mirror the active filter onto the URL as `?engine=`/`?format=` allowlists via
 * `history.replaceState`. A param is emitted only when its active set is a
 * strict subset of the universe; an all-active row leaves the URL clean.
 */
function syncFilterUrl(): void {
  if (!window.history?.replaceState) {
    return;
  }
  const { universe, active } = getGlobalFilterSnapshot();
  const url = new URL(window.location.href);
  syncDimensionUrl(url, 'engine', active.engines, universe.engines);
  syncDimensionUrl(url, 'format', active.formats, universe.formats);
  window.history.replaceState(null, '', url.toString());
}

function syncDimensionUrl(url: URL, paramName: string, active: string[], universe: string[]): void {
  if (active.length < universe.length) {
    url.searchParams.set(paramName, active.join(','));
  } else {
    url.searchParams.delete(paramName);
  }
}

/** Funnel icon used by the filter dropdown trigger (`render.rs::filter_icon`). */
export function FilterIcon() {
  return (
    <svg
      className="btn-icon"
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
    </svg>
  );
}
