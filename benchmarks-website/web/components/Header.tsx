// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

'use client';

import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';

import { FilterBar } from '@/components/FilterBar';
import type { FilterUniverse } from '@/lib/chart-format';

/**
 * The sticky site header, the full client port of
 * `server/src/html/render.rs::site_header` plus the section-18 header controls
 * of `chart-init.js`: the hamburger-toggled mobile nav panel (expand/collapse
 * all, the global filter dropdown, and the mobile-only GitHub link), the
 * theme-aware logo, the desktop GitHub link, and the theme toggle.
 *
 * Expand/Collapse All writes `details.open` on every group disclosure
 * directly; the native `toggle` event (which also fires for scripted changes)
 * is how each chart island learns its group opened, so no shared React tree is
 * needed between the header and the islands.
 *
 * The filter dropdown renders only when the universe has at least one chip,
 * matching v3's `show_filters` guard. Pages without chart data (or without the
 * universe fetched) pass an empty universe and get the chrome-only header.
 *
 * Theming: the html `data-theme` attribute plus `localStorage["bench-theme"]`
 * are the source of truth (seeded pre-paint by the theme-bootstrap inline
 * script in the root layout). The server renders the v3 default label "Light";
 * the mount effect immediately corrects the label to name the NEXT theme, the
 * exact v3 `updateThemeButtons` sequencing.
 */
export function Header({
  universe,
  initialEngines,
  initialFormats,
}: {
  universe?: FilterUniverse;
  initialEngines?: string[];
  initialFormats?: string[];
}) {
  const [navOpen, setNavOpen] = useState(false);
  const [nextTheme, setNextTheme] = useState<'light' | 'dark'>('light');
  const navRef = useRef<HTMLDivElement>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);

  // Label/attribute sync on mount (the bootstrap script may have set a stored
  // theme before hydration) and on every toggle.
  useEffect(() => {
    setNextTheme(effectiveTheme() === 'light' ? 'dark' : 'light');
  }, []);

  // Close the mobile nav panel on outside click and on Escape.
  useEffect(() => {
    if (!navOpen) {
      return;
    }
    const onClick = (e: MouseEvent): void => {
      const target = e.target as Node;
      if (!navRef.current?.contains(target) && !toggleRef.current?.contains(target)) {
        setNavOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') {
        setNavOpen(false);
      }
    };
    document.addEventListener('click', onClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('click', onClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [navOpen]);

  const showFilters =
    universe !== undefined && (universe.engines.length > 0 || universe.formats.length > 0);

  const onThemeToggle = (): void => {
    const next = effectiveTheme() === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    try {
      localStorage.setItem('bench-theme', next);
    } catch {
      // Private browsing modes may reject storage writes; the in-page theme
      // still applies for this visit.
    }
    setNextTheme(next === 'light' ? 'dark' : 'light');
  };

  return (
    <header className="sticky-header">
      <div className="header-content">
        <div className="header-left">
          <button
            className="control-btn nav-mobile-toggle"
            type="button"
            data-role="nav-mobile-toggle"
            aria-haspopup="true"
            aria-expanded={navOpen}
            aria-controls="bench-nav-controls"
            aria-label="Toggle navigation menu"
            ref={toggleRef}
            onClick={(e) => {
              e.stopPropagation();
              setNavOpen((o) => !o);
            }}
          >
            <HamburgerIcon />
          </button>
          <Link className="logo-link" href="/" aria-label="bench.vortex.dev home">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img className="site-logo logo-light" src="/Vortex_Black_NoBG.png" alt="Vortex" />
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img className="site-logo logo-dark" src="/Vortex_White_NoBG.png" alt="Vortex" />
          </Link>
          <h1 className="site-title">Vortex Benchmarks</h1>
        </div>
        <div className="header-center">
          <div
            className={`nav-controls${navOpen ? ' nav-controls--open' : ''}`}
            id="bench-nav-controls"
            data-role="nav-controls"
            aria-label="Benchmark group controls"
            ref={navRef}
          >
            <button
              className="control-btn"
              type="button"
              data-action="expand-all"
              onClick={() => setAllGroups(true)}
            >
              <ChevronsDownIcon />
              <span>Expand All</span>
            </button>
            <button
              className="control-btn"
              type="button"
              data-action="collapse-all"
              onClick={() => setAllGroups(false)}
            >
              <ChevronsUpIcon />
              <span>Collapse All</span>
            </button>
            {showFilters && (
              <FilterBar
                universe={universe}
                initialEngines={initialEngines ?? []}
                initialFormats={initialFormats ?? []}
              />
            )}
            {/* Mobile-only GitHub link rendered inside the hamburger panel;
                hidden on desktop via CSS, where `.repo-link-desktop` covers the
                wide-viewport case. */}
            <a
              className="repo-link nav-controls-github"
              href="https://github.com/vortex-data/vortex"
              rel="noopener noreferrer"
              target="_blank"
            >
              <GitHubIcon />
              <span>GitHub</span>
            </a>
          </div>
        </div>
        <div className="header-right">
          <a
            className="repo-link repo-link-desktop"
            href="https://github.com/vortex-data/vortex"
            rel="noopener noreferrer"
            target="_blank"
          >
            <GitHubIcon />
            <span>GitHub</span>
          </a>
          <button
            className="control-btn theme-toggle"
            type="button"
            data-role="theme-toggle"
            data-next-theme={nextTheme}
            aria-label={`Switch to ${nextTheme} mode`}
            onClick={onThemeToggle}
          >
            <SunIcon />
            <MoonIcon />
            <span className="theme-toggle-label">{nextTheme === 'dark' ? 'Dark' : 'Light'}</span>
          </button>
        </div>
      </div>
    </header>
  );
}

/** The forced or media-derived effective theme (`chart-init.js::effectiveTheme`). */
function effectiveTheme(): 'light' | 'dark' {
  const forced = document.documentElement.getAttribute('data-theme');
  if (forced === 'light' || forced === 'dark') {
    return forced;
  }
  if (window.matchMedia?.('(prefers-color-scheme: dark)').matches) {
    return 'dark';
  }
  return 'light';
}

/** Open or close every group disclosure. Writing `details.open` fires each
 * disclosure's `toggle` event, which is what triggers the chart islands'
 * group-open hydration. */
function setAllGroups(open: boolean): void {
  document.querySelectorAll<HTMLDetailsElement>('details.group-disclosure').forEach((d) => {
    d.open = open;
  });
}

function HamburgerIcon() {
  return (
    <svg
      className="btn-icon"
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M4 6h16" />
      <path d="M4 12h16" />
      <path d="M4 18h16" />
    </svg>
  );
}

/** Inline GitHub mark, ported from `render.rs::github_icon`. */
function GitHubIcon() {
  return (
    <svg
      className="github-logo"
      viewBox="0 0 16 16"
      width="16"
      height="16"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

function ChevronsDownIcon() {
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
      <path d="m7 6 5 5 5-5" />
      <path d="m7 13 5 5 5-5" />
    </svg>
  );
}

function ChevronsUpIcon() {
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
      <path d="m17 18-5-5-5 5" />
      <path d="m17 11-5-5-5 5" />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg
      className="btn-icon theme-icon theme-icon-light"
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
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2" />
      <path d="M12 20v2" />
      <path d="m4.93 4.93 1.41 1.41" />
      <path d="m17.66 17.66 1.41 1.41" />
      <path d="M2 12h2" />
      <path d="M20 12h2" />
      <path d="m6.34 17.66-1.41 1.41" />
      <path d="m19.07 4.93-1.41 1.41" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg
      className="btn-icon theme-icon theme-icon-dark"
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
      <path d="M20.99 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 20.99 12.79z" />
    </svg>
  );
}
