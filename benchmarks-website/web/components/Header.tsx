// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useCallback, useEffect, useState } from 'react';

/**
 * The sticky site header, the React/client-island port of
 * `server/src/html/render.rs::site_header` plus the header-control behaviour
 * from `static/chart-init.js` (`initHeaderControls` / `bindMobileNav`).
 *
 * Interactivity lives here as a client component: the primary nav's active
 * state (`usePathname`), the light/dark theme toggle (persisted to
 * `localStorage["bench-theme"]` and reflected on `documentElement[data-theme]`,
 * matching the pre-paint bootstrap in `layout.tsx`), the mobile hamburger
 * panel, and Expand/Collapse-All over the page's `<details>` disclosures and
 * `.current-group` sections.
 *
 * Theme changes also emit a `bench:themechange` event so chart client islands
 * can re-bake their canvas palettes (the synthesis `recolor*Charts` hook).
 */
export function Header() {
  const pathname = usePathname();
  // Overview is `/`, Latest is `/latest`, Historic is `/historic`. The
  // Expand/Collapse-All actions only apply to the two data tabs.
  const active: 'overview' | 'latest' | 'historic' | 'other' =
    pathname === '/'
      ? 'overview'
      : pathname.startsWith('/latest')
        ? 'latest'
        : pathname.startsWith('/historic')
          ? 'historic'
          : 'other';
  const showGroupControls = active === 'latest' || active === 'historic';

  // `next` is the theme the toggle switches TO; the canonical brand surface is
  // dark, so the deterministic server/first-client render advertises "Light".
  const [nextTheme, setNextTheme] = useState<'light' | 'dark'>('light');
  const [navOpen, setNavOpen] = useState(false);

  const effectiveTheme = (): 'light' | 'dark' => {
    const forced = document.documentElement.getAttribute('data-theme');
    if (forced === 'light' || forced === 'dark') {
      return forced;
    }
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  };

  // Reconcile the toggle's advertised target with the real resolved theme once
  // mounted (the server render cannot know `localStorage` / `prefers-color`).
  useEffect(() => {
    setNextTheme(effectiveTheme() === 'light' ? 'dark' : 'light');
  }, []);

  const toggleTheme = useCallback(() => {
    const target = effectiveTheme() === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', target);
    try {
      localStorage.setItem('bench-theme', target);
    } catch {
      /* private mode / storage disabled — theme still applies for this page */
    }
    setNextTheme(target === 'light' ? 'dark' : 'light');
    window.dispatchEvent(new CustomEvent('bench:themechange', { detail: { theme: target } }));
  }, []);

  const setAllGroups = useCallback((open: boolean) => {
    document.querySelectorAll<HTMLDetailsElement>('details.group-disclosure').forEach((d) => {
      d.open = open;
    });
    document.querySelectorAll<HTMLElement>('.current-group').forEach((sec) => {
      sec.classList.toggle('current-group--collapsed', !open);
      sec
        .querySelector('[data-role="current-collapse"]')
        ?.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  }, []);

  // Close the mobile nav on outside-click / Escape, matching `bindMobileNav`.
  useEffect(() => {
    if (!navOpen) {
      return;
    }
    const onClick = (e: MouseEvent) => {
      const target = e.target;
      if (!(target instanceof Node)) {
        return;
      }
      const inNav = document.getElementById('bench-nav-controls')?.contains(target) ?? false;
      const onToggle =
        target instanceof Element && target.closest('[data-role="nav-mobile-toggle"]') !== null;
      if (!inNav && !onToggle) {
        setNavOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
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
          <div
            className={`nav-controls${navOpen ? ' nav-controls--open' : ''}`}
            id="bench-nav-controls"
            data-role="nav-controls"
            aria-label="Site navigation"
          >
            <nav className="site-nav" aria-label="Primary">
              <NavLink href="/" label="Overview" current={active === 'overview'} />
              <NavLink href="/latest" label="Latest Commit" current={active === 'latest'} />
              <NavLink href="/historic" label="Historic Data" current={active === 'historic'} />
            </nav>
            <a
              className="repo-link nav-controls-github"
              href="https://github.com/vortex-data/vortex"
              rel="noopener noreferrer"
              target="_blank"
            >
              <GithubIcon />
              <span>GitHub</span>
            </a>
          </div>
        </div>
        <div className="header-right">
          {showGroupControls && (
            <div className="page-actions" data-role="page-actions">
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
            </div>
          )}
          <a
            className="repo-link repo-link-desktop"
            href="https://github.com/vortex-data/vortex"
            rel="noopener noreferrer"
            target="_blank"
          >
            <GithubIcon />
            <span>GitHub</span>
          </a>
          <button
            className="control-btn theme-toggle"
            type="button"
            data-role="theme-toggle"
            data-next-theme={nextTheme}
            aria-label={`Switch to ${nextTheme} mode`}
            onClick={toggleTheme}
          >
            <SunIcon />
            <MoonIcon />
            <span className="theme-toggle-label">{nextTheme === 'light' ? 'Light' : 'Dark'}</span>
          </button>
        </div>
      </div>
    </header>
  );
}

function NavLink({ href, label, current }: { href: string; label: string; current: boolean }) {
  return (
    <Link
      className={`site-nav-link${current ? ' site-nav-link--active' : ''}`}
      href={href}
      aria-current={current ? 'page' : undefined}
    >
      {label}
    </Link>
  );
}

const STROKE = {
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 2,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
  'aria-hidden': true,
};

function HamburgerIcon() {
  return (
    <svg className="btn-icon" viewBox="0 0 24 24" width={18} height={18} {...STROKE}>
      <path d="M4 6h16" />
      <path d="M4 12h16" />
      <path d="M4 18h16" />
    </svg>
  );
}

function GithubIcon() {
  return (
    <svg
      className="github-logo"
      viewBox="0 0 16 16"
      width={16}
      height={16}
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
    </svg>
  );
}

function ChevronsDownIcon() {
  return (
    <svg className="btn-icon" viewBox="0 0 24 24" width={16} height={16} {...STROKE}>
      <path d="m7 6 5 5 5-5" />
      <path d="m7 13 5 5 5-5" />
    </svg>
  );
}

function ChevronsUpIcon() {
  return (
    <svg className="btn-icon" viewBox="0 0 24 24" width={16} height={16} {...STROKE}>
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
      width={16}
      height={16}
      {...STROKE}
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
      width={16}
      height={16}
      {...STROKE}
    >
      <path d="M20.99 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 20.99 12.79z" />
    </svg>
  );
}
