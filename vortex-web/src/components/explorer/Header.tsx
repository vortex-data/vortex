// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { type ReactNode, useEffect, useRef } from 'react';
import { Icon } from '../Icon';
import { MarkGithub } from '../MarkGithub';
import { MarkSlack } from '../MarkSlack';
import { MarkX } from '../MarkX';

const REPO = 'vortex-data/vortex';
const REPO_URL = `https://github.com/${REPO}`;
// Read at build time (vite.config.ts); '' when GitHub did not answer.
const STARS = import.meta.env.VITE_GITHUB_STARS;
// vortex.dev's redirect to the Slack invite.
const SLACK_URL = 'https://vortex.dev/slack';
const X_URL = 'https://x.com/vortexdotdev';

const LINKS = [
  { label: 'Docs', href: 'https://docs.vortex.dev' },
  { label: 'Benchmarks', href: 'https://bench.vortex.dev' },
  { label: 'Explorer', href: import.meta.env.BASE_URL, current: true },
  { label: 'Blog', href: 'https://vortex.dev/blog' },
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

/**
 * The Menu: the links behind a button on a narrow screen. It is a <details>, so it opens without
 * scripts. It closes on Escape and on a click outside it.
 */
function Menu({ spill }: { spill: ReactNode }) {
  const menu = useRef<HTMLDetailsElement>(null);

  useEffect(() => {
    const el = menu.current;
    if (!el) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || !el.open) return;
      el.open = false;
      el.querySelector('summary')?.focus();
    };
    const onPointer = (e: PointerEvent) => {
      if (el.open && !el.contains(e.target as Node)) el.open = false;
    };
    document.addEventListener('keydown', onKey);
    document.addEventListener('pointerdown', onPointer);
    return () => {
      document.removeEventListener('keydown', onKey);
      document.removeEventListener('pointerdown', onPointer);
    };
  }, []);

  return (
    <details ref={menu} className="vx-header-menu">
      <summary className="vx-btn vx-btn-secondary is-mono">
        <Icon name="menu" className="is-open" />
        <Icon name="close" className="is-close" />
        <span className="vx-header-menu-text">Menu</span>
      </summary>
      <nav className="vx-header-menu-panel" aria-label="Primary">
        <Links />
        {spill}
      </nav>
    </details>
  );
}

interface HeaderProps {
  /** The Explorer's own controls, such as the open file and the view switch. */
  tools?: ReactNode;
  /** The tools' twin rows in the Menu's panel (vx-header-menu-spill), shown below 640px, where
   *  the tools leave the row. */
  menu?: ReactNode;
}

/**
 * The design system's Header card, the one header every Vortex surface draws: the wordmark, the
 * links, the Explorer's tools, then the end group: the socials (X, Slack, and the GitHub button),
 * and on a narrow screen the Menu. The theme switch is in the Footer. Below 960px X and Slack move
 * to twin rows in the Menu's panel, and the GitHub button and the Menu are icon squares; below
 * 640px GitHub follows. The wordmark points back to vortex.dev.
 */
export function Header({ tools, menu }: HeaderProps) {
  return (
    <header className="vx-header">
      <div className="vx-header-start">
        <a href="https://vortex.dev" aria-label="Vortex home">
          <img
            className="vx-header-mark is-dark"
            src={`${import.meta.env.BASE_URL}vortex_wordmark_dark_theme.svg`}
            alt="Vortex"
            width={83}
            height={24}
          />
          <img
            className="vx-header-mark is-light"
            src={`${import.meta.env.BASE_URL}vortex_wordmark.svg`}
            alt="Vortex"
            width={83}
            height={24}
            loading="lazy"
          />
        </a>
      </div>
      <nav className="vx-header-nav" aria-label="Primary">
        <Links />
      </nav>
      {tools && <div className="vx-header-tools">{tools}</div>}
      <div className="vx-header-end">
        <div className="vx-header-social">
          <a
            className="vx-nav-link vx-header-social-link"
            href={X_URL}
            aria-label="Vortex on X"
            title="Vortex on X"
          >
            <MarkX />
          </a>
          <a
            className="vx-nav-link vx-header-social-link"
            href={SLACK_URL}
            aria-label="Vortex on Slack"
            title="Vortex on Slack"
          >
            <MarkSlack />
          </a>
          <a
            className="vx-btn vx-btn-secondary is-mono vx-header-github"
            href={REPO_URL}
            aria-label={`${STARS ? `${STARS} stars, ` : ''}${REPO} on GitHub`}
            title={`${REPO} on GitHub`}
          >
            <MarkGithub />
            {STARS && (
              <span className="vx-btn-count">
                <Icon name="star" />
                {STARS}
              </span>
            )}
          </a>
        </div>
        <Menu
          spill={
            <>
              {menu}
              <a
                className="vx-label vx-nav-link vx-header-menu-spill is-social"
                href={REPO_URL}
                aria-label={`${REPO} on GitHub`}
              >
                GitHub
                <MarkGithub />
              </a>
              <a
                className="vx-label vx-nav-link vx-header-menu-spill is-social is-icon"
                href={SLACK_URL}
                aria-label="Vortex on Slack"
              >
                Slack
                <MarkSlack />
              </a>
              <a
                className="vx-label vx-nav-link vx-header-menu-spill is-social is-icon"
                href={X_URL}
                aria-label="Vortex on X"
              >
                X
                <MarkX />
              </a>
            </>
          }
        />
      </div>
    </header>
  );
}
