// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

'use client';

import { useEffect, useRef, useState } from 'react';

/**
 * A group entry for the jump menu: its display name and the `data-group-slug`
 * carried by its landing-page `<section>` (see [`GroupSection`]).
 */
export interface GroupNavItem {
  name: string;
  slug: string;
}

/**
 * Expand and scroll the landing-page section for `slug` into view.
 *
 * Finds the group's `<section data-group-slug>`, opens its
 * `<details class="group-disclosure">` so the charts hydrate (setting `open`
 * fires the native `toggle` event the chart islands listen for), and
 * smooth-scrolls it into view; the section's `scroll-margin-top` keeps the
 * title clear of the sticky header. Smooth scrolling is skipped under
 * `prefers-reduced-motion`. Returns `true` when the section was found.
 *
 * Exported so the menu's jump behavior is unit-testable without rendering.
 */
export function jumpToGroup(slug: string, doc: Document = document): boolean {
  const section = Array.from(doc.querySelectorAll<HTMLElement>('[data-group-slug]')).find(
    (el) => el.dataset.groupSlug === slug,
  );
  if (section === undefined) {
    return false;
  }
  const disclosure = section.querySelector<HTMLDetailsElement>('details.group-disclosure');
  if (disclosure !== null) {
    disclosure.open = true;
  }
  const reduceMotion = doc.defaultView?.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  section.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' });
  return true;
}

/**
 * A left-side "Jump to group" menu: a fixed toggle button that opens a panel
 * listing every group, each a button that expands and scrolls to that group's
 * section (via [`jumpToGroup`]) and then closes the panel.
 *
 * Toggle-driven (not hover) so it works on touch and keyboard, mirroring the
 * header's hamburger nav ([`Header`]): `aria-expanded` / `aria-controls` on the
 * toggle, with outside-click and Escape closing the panel. Renders nothing when
 * there are no groups.
 */
export function GroupNav({ groups }: { groups: GroupNavItem[] }) {
  const [open, setOpen] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);

  // Close on outside click and Escape while open (the header nav's pattern).
  useEffect(() => {
    if (!open) {
      return;
    }
    const onClick = (e: MouseEvent): void => {
      const target = e.target as Node;
      if (!panelRef.current?.contains(target) && !toggleRef.current?.contains(target)) {
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

  if (groups.length === 0) {
    return null;
  }

  return (
    <nav className="group-nav" aria-label="Jump to group">
      <button
        className="control-btn group-nav-toggle"
        type="button"
        aria-haspopup="true"
        aria-expanded={open}
        aria-controls="group-nav-panel"
        ref={toggleRef}
        onClick={(e) => {
          e.stopPropagation();
          setOpen((o) => !o);
        }}
      >
        <ListIcon />
        <span>Groups</span>
      </button>
      <div
        className={`group-nav-panel${open ? ' group-nav-panel--open' : ''}`}
        id="group-nav-panel"
        ref={panelRef}
      >
        <p className="group-nav-heading">Jump to group</p>
        <ul className="group-nav-list">
          {groups.map((group) => (
            <li key={group.slug}>
              <button
                className="group-nav-link"
                type="button"
                onClick={() => {
                  jumpToGroup(group.slug);
                  setOpen(false);
                }}
              >
                {group.name}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
}

/** A list/menu glyph for the toggle, matching the header icons' stroke style. */
function ListIcon() {
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
      <path d="M8 6h13" />
      <path d="M8 12h13" />
      <path d="M8 18h13" />
      <path d="M3 6h.01" />
      <path d="M3 12h.01" />
      <path d="M3 18h.01" />
    </svg>
  );
}
