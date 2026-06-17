// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

'use client';

import { useRef, useState } from 'react';

import { SpeedupChart } from './SpeedupChart';
import type { SpeedupGroup } from '@/lib/synthesis';

/**
 * One group's head-to-head section, the React port of
 * `current.rs::speedup_section`: a collapsible heading, a Speedup/Query sort
 * toggle that re-orders every chart in the group at once, and a grid of one
 * [`SpeedupChart`] per facet.
 *
 * Collapse is DOM-class-driven (toggling `current-group--collapsed` on the
 * section, which the CSS uses to hide the grid) so the header's Expand /
 * Collapse-All controls can drive every section uniformly without React state.
 */
export function SpeedupSection({ group }: { group: SpeedupGroup }) {
  const [sortMode, setSortMode] = useState<'speedup' | 'query'>('speedup');
  // The item hovered in any facet, mirrored as a gold highlight across every
  // panel in this group's grid (the synthesis cross-panel hover).
  const [hoverQuery, setHoverQuery] = useState<string | null>(null);
  const sectionRef = useRef<HTMLElement>(null);

  const orderNoun = group.facetedByEngine ? 'Query #' : 'Dataset';
  const magnitudeLabel = group.metric === 'smaller' ? 'Smaller' : 'Speedup';

  const toggleCollapse = (e: React.MouseEvent<HTMLButtonElement>) => {
    const sec = sectionRef.current;
    if (sec === null) {
      return;
    }
    const collapsed = sec.classList.toggle('current-group--collapsed');
    e.currentTarget.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
  };

  return (
    <section ref={sectionRef} className="current-group" id={group.anchor}>
      <header className="current-group-head">
        <h2 className="current-group-name">
          <button
            className="current-collapse-btn"
            type="button"
            data-role="current-collapse"
            aria-expanded="true"
            onClick={toggleCollapse}
          >
            <span className="current-collapse-caret" aria-hidden="true" />
            {group.name}
          </button>
          {group.description !== undefined && (
            <span className="current-group-desc">{group.description}</span>
          )}
        </h2>
        <div className="speedup-sort" data-role="speedup-sort" role="group" aria-label="Sort">
          <span className="speedup-sort-label">Sort</span>
          <button
            className={`speedup-sort-btn${sortMode === 'speedup' ? ' speedup-sort-btn--active' : ''}`}
            type="button"
            data-sort="speedup"
            aria-pressed={sortMode === 'speedup'}
            onClick={() => setSortMode('speedup')}
          >
            {magnitudeLabel}
          </button>
          <button
            className={`speedup-sort-btn${sortMode === 'query' ? ' speedup-sort-btn--active' : ''}`}
            type="button"
            data-sort="query"
            aria-pressed={sortMode === 'query'}
            onClick={() => setSortMode('query')}
          >
            {orderNoun}
          </button>
        </div>
      </header>
      <div className="speedup-grid">
        {group.facets.map((ed, i) => (
          <SpeedupChart
            key={`${ed.facet}-${i}`}
            data={ed}
            sortMode={sortMode}
            hoverQuery={hoverQuery}
            onHover={setHoverQuery}
          />
        ))}
      </div>
    </section>
  );
}
