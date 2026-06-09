// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import Link from 'next/link';

/**
 * The sticky site header, the static-chrome subset of the server-component port
 * of `server/src/html/render.rs::site_header`: the theme-aware logo, the site
 * title, and the desktop GitHub link.
 *
 * The interactive controls (mobile nav toggle, expand/collapse-all, the global
 * filter dropdown, and the theme toggle) are deferred to PR-4.4.b's client
 * islands; until then, theming follows `prefers-color-scheme` via the global
 * CSS and per-group expand/collapse is the native `<details>` behavior.
 *
 * Two `<img>` logos are rendered (black for light, white for dark); the global
 * CSS shows the one matching the active theme via the `.logo-light` /
 * `.logo-dark` classes. Plain `<img>` (rather than `next/image`) matches v3 and
 * lets the CSS size the logo by height.
 */
export function Header() {
  return (
    <header className="sticky-header">
      <div className="header-content">
        <div className="header-left">
          <Link className="logo-link" href="/" aria-label="bench.vortex.dev home">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img className="site-logo logo-light" src="/Vortex_Black_NoBG.png" alt="Vortex" />
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img className="site-logo logo-dark" src="/Vortex_White_NoBG.png" alt="Vortex" />
          </Link>
          <h1 className="site-title">Vortex Benchmarks</h1>
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
        </div>
      </div>
    </header>
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
