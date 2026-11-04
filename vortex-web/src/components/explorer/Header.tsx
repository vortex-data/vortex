// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { ReactNode } from 'react';
import { Icon } from '../Icon';
import { ThemePicker } from '../ThemePicker';

const LINKS = [
  { label: 'Docs', href: 'https://docs.vortex.dev' },
  { label: 'Bench', href: 'https://bench.vortex.dev' },
  { label: 'Explorer', href: '/', current: true },
  { label: 'Blog', href: 'https://vortex.dev/blog' },
  { label: 'Slack', href: 'https://vortex.dev/slack' },
];

function Links() {
  return LINKS.map((l) => (
    <a
      key={l.label}
      className="vx-label vx-nav-link"
      href={l.href}
      {...(l.current ? { 'aria-current': 'page' as const } : {})}
    >
      {l.label}
    </a>
  ));
}

interface HeaderProps {
  /** The open file's name and facts, beside the wordmark. */
  file?: ReactNode;
  /** Controls for the open file, such as the view switch. With them, the links move to the menu. */
  tools?: ReactNode;
  /** Buttons after the theme switch, such as Close. */
  end?: ReactNode;
  /** The tools' twin rows in the Menu's panel (vx-header-menu-spill), shown below 640px, where
   *  the tools leave the row. */
  menu?: ReactNode;
}

/** The design system's header, as on the other Vortex sites: wordmark, links, and theme. */
export function Header({ file, tools, end, menu }: HeaderProps) {
  return (
    <header className={`vx-header is-solid ${tools ? 'has-tools' : 'has-nav'}`}>
      <div className="vx-header-start">
        <a href="/" aria-label="Vortex Explorer home">
          <img
            className="vx-header-mark is-dark"
            src="/vortex_wordmark_dark_theme.svg"
            alt="Vortex"
            width={83}
            height={24}
          />
          <img
            className="vx-header-mark is-light"
            src="/vortex_wordmark.svg"
            alt="Vortex"
            width={83}
            height={24}
          />
        </a>
        <p className="vx-label vx-header-label">Explorer</p>
        {file}
      </div>
      <nav className="vx-header-nav" aria-label="Primary">
        <Links />
      </nav>
      {tools && <div className="vx-header-tools">{tools}</div>}
      <div className="vx-header-end">
        <ThemePicker />
        {end}
        <details className="vx-header-menu">
          <summary className="vx-btn vx-btn-secondary is-mono">
            <Icon name="menu" className="is-open" />
            <Icon name="close" className="is-close" />
            <span className="vx-header-menu-text">Menu</span>
          </summary>
          <nav className="vx-header-menu-panel" aria-label="Primary">
            <Links />
            {menu}
          </nav>
        </details>
      </div>
    </header>
  );
}
