// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// `no-page-custom-font` targets per-page `<link>` fonts injected outside a
// Pages-Router `_document` (which load for only one page). Here the fonts live
// in the App-Router ROOT layout, so they load globally for every route exactly
// as the rule intends; the warning is a Pages-Router false positive. We keep
// the external `<link>` fonts (Geist + Funnel Display) to mirror v2/v3 verbatim
// rather than pull `next/font` + the `geist` package into this shell PR.
/* eslint-disable @next/next/no-page-custom-font */

import type { Metadata, Viewport } from 'next';
import type { ReactNode } from 'react';

import './globals.css';

export const metadata: Metadata = {
  title: 'Vortex Benchmarks',
  description: 'Continuous benchmark results for Vortex.',
  // Theme-aware favicons, ported from `render.rs::favicon_links`: the black
  // sigil on light-mode tabs, the white sigil on dark-mode tabs, with the dark
  // sigil as the unmediated fallback (and the apple-touch icon).
  icons: {
    icon: [
      { url: '/icon-light.png', media: '(prefers-color-scheme: light)' },
      { url: '/icon-dark.png', media: '(prefers-color-scheme: dark)' },
      { url: '/icon-dark.png' },
    ],
    apple: '/icon-dark.png',
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
};

// The pre-paint theme bootstrap, ported verbatim from
// `render.rs::theme_bootstrap_script`: apply any stored theme choice before
// first paint so a dark-mode visitor never flashes the light theme. Inline (not
// a module) and in `<head>` so it runs before the body renders. The stored
// theme lands as a `data-theme` attribute the server never rendered, so the
// root `<html>` carries `suppressHydrationWarning` (attribute-level, one
// element deep) to keep dev hydration checks quiet for themed visitors. The
// script's bare catch is deliberate and v3-byte-identical: localStorage
// access throws in some private-browsing modes, and the correct fallback is
// silently keeping the default prefers-color-scheme theme.
const THEME_BOOTSTRAP = `(function(){try{var t=localStorage.getItem("bench-theme");if(t==="light"||t==="dark"){document.documentElement.dataset.theme=t;}}catch(e){}})();`;

/**
 * Root layout: the `<html>`/`<body>` shell plus the global stylesheet, the
 * pre-paint theme bootstrap, and the external web fonts (Geist sans + mono for
 * body/metrics, Funnel Display for headings), mirroring v2's `index.html` /
 * v3's `render.rs::web_font_links`. React 19 hoists the `<link>` tags into
 * `<head>`.
 */
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOTSTRAP }} />
      </head>
      <body>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Funnel+Display:wght@300;400;500;600;700;800&display=swap"
        />
        <link
          rel="stylesheet"
          href="https://unpkg.com/geist@1.3.0/dist/fonts/geist-sans/style.css"
        />
        <link
          rel="stylesheet"
          href="https://unpkg.com/geist@1.3.0/dist/fonts/geist-mono/style.css"
        />
        {children}
      </body>
    </html>
  );
}
