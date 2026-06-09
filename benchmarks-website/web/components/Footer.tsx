// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/**
 * The build-SHA footer, the server-component port of
 * `server/src/html/render.rs::site_footer`.
 *
 * The SHA is sourced from `VERCEL_GIT_COMMIT_SHA` (the same env var `lib/health.ts`
 * reports as `build_sha`), so the footer and the health probe stay in lockstep
 * with the deployed commit. When the SHA is the literal `"unknown"` (a build
 * outside a git checkout, e.g. local dev), the short SHA renders without a
 * commit link, matching v3.
 */
export function Footer() {
  const fullSha = process.env.VERCEL_GIT_COMMIT_SHA ?? 'unknown';
  const shortSha = fullSha.slice(0, 7);
  const commitUrl = `https://github.com/vortex-data/vortex/commit/${fullSha}`;
  return (
    <footer className="site-footer">
      <span className="site-footer-label">build </span>
      {fullSha === 'unknown' ? (
        <code className="site-footer-sha">{shortSha}</code>
      ) : (
        <a className="site-footer-sha" href={commitUrl} rel="noopener noreferrer" target="_blank">
          <code>{shortSha}</code>
        </a>
      )}
    </footer>
  );
}
