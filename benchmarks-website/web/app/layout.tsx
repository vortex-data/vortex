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

import { Footer } from '@/components/Footer';
import { Header } from '@/components/Header';
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

// Pre-paint theme bootstrap, ported verbatim from
// `render.rs::theme_bootstrap_script`: read the persisted choice and reflect it
// on `<html data-theme>` before first paint so a returning visitor's chosen
// theme never flashes the default. Kept inline (not a component) so it runs
// before the body renders.
const THEME_BOOTSTRAP = `(function(){try{var t=localStorage.getItem("bench-theme");if(t==="light"||t==="dark"){document.documentElement.dataset.theme=t;}}catch(e){}})();`;

/**
 * Root layout: the `<html>`/`<body>` shell, the global stylesheet, the external
 * web fonts (Geist sans + mono for body/metrics, Funnel Display for headings),
 * the theme bootstrap, and the shared chrome (`Header` / `Footer`) wrapping the
 * per-route `<main>`. Mirrors v3's `render.rs::render_page`.
 */
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    // `suppressHydrationWarning`: the theme bootstrap below sets `data-theme` on
    // `<html>` from localStorage before React hydrates, so the client `<html>`
    // attributes intentionally differ from the (theme-less) server render. The
    // flag scopes the suppression to this one element's attributes.
    <html lang="en" suppressHydrationWarning>
      <body>
        <script dangerouslySetInnerHTML={{ __html: THEME_BOOTSTRAP }} />
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
        <Header />
        <main>{children}</main>
        <Footer />
      </body>
    </html>
  );
}
