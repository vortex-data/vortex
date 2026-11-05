// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { ThemePicker } from '../ThemePicker';

const LINKS = [
  { href: 'https://vortex.dev/charter.pdf', label: 'Charter' },
  { href: 'https://lfprojects.org/policies/code-of-conduct/', label: 'Code of conduct' },
  { href: 'https://github.com/vortex-data/vortex/blob/develop/SECURITY.md', label: 'Security' },
  { href: 'https://zoom-lfx.platform.linuxfoundation.org/meetings/vortex', label: 'Calendar' },
];

/**
 * The design system's Footer card, the one footer every Vortex surface draws: The Linux
 * Foundation's logo (the system's CSS inverts it in Light), the line LF Projects requires, the
 * credit, the project's links, and the theme switch, which the system's CSS moves up to the
 * mark's line on a phone. The Explorer shows the footer on its start page only. Each run of text
 * is one string, so React writes it as one text node, as the docs do; a `{' '}` splits it, and
 * Chromium then sets the glyphs a pixel apart from the docs' and the site's footer.
 */
export function Footer() {
  return (
    <footer className="vx-footer">
      <img
        className="vx-footer-mark"
        src={`${import.meta.env.BASE_URL}lf_white.svg`}
        width={1000}
        height={60}
        alt="The Linux Foundation"
        loading="lazy"
      />
      <p className="vx-footer-text">
        {
          'Copyright © Vortex a Series of LF Projects, LLC. For web site terms of use, trademark policy and other project policies please see '
        }
        <a className="vx-link" href="https://lfprojects.org/policies">
          lfprojects.org/policies
        </a>
        {'. '}
        <span className="vx-footer-credit">
          {'Donated by '}
          <a className="vx-link" href="https://spiraldb.com">
            Spiral
          </a>
          .
        </span>
      </p>
      <nav className="vx-footer-nav" aria-label="Project">
        {LINKS.map(({ href, label }) => (
          <a key={href} className="vx-label-sm vx-nav-link" href={href}>
            {label}
          </a>
        ))}
      </nav>
      <ThemePicker />
    </footer>
  );
}
