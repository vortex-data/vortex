// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { Footer } from './Footer';
import { Header } from './Header';

/**
 * The one 404 every Vortex surface serves (the design system's Status), under the shared header and
 * footer: the code, what it means, and the ways back, over a DotField. The Explorer does not load
 * the mark's renderer, so the mark's cell shows the static symbol.
 */
export function NotFound() {
  return (
    <div className="x-app is-page">
      <Header />
      <main className="x-not-found">
        <div className="vx-status has-mark has-dot-field">
          <div className="vx-dot-field" aria-hidden="true" />
          <div className="vx-status-text">
            <h1 className="vx-status-code">404</h1>
            <p className="vx-status-sentence">Page not found.</p>
            <div className="vx-btn-row">
              <a className="vx-btn vx-btn-primary" href="https://vortex.dev">
                Go to the homepage
              </a>
              <a className="vx-btn vx-btn-secondary" href={import.meta.env.BASE_URL}>
                Open the Explorer
              </a>
              <a className="vx-btn vx-btn-secondary" href="https://docs.vortex.dev">
                Read the docs
              </a>
            </div>
          </div>
          <div className="vx-status-mark" aria-hidden="true">
            <img
              className="x-not-found-symbol"
              src={`${import.meta.env.BASE_URL}logo-icon.svg`}
              alt=""
              width={188}
              height={188}
            />
          </div>
        </div>
      </main>
      <Footer />
    </div>
  );
}
