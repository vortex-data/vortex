// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

/** vortex.dev's formatStars: "950", "3.3k", "12k". */
function formatStars(n: number): string {
  if (n < 1000) return `${n}`;
  const k = n < 9950 ? Math.round(n / 100) / 10 : Math.round(n / 1000);
  return `${k}k`;
}

/**
 * The repository's star count for the header's GitHub button, read once per build, as vortex.dev
 * and the docs read it, so every surface draws the same button. Any failure (offline, a timeout, a
 * rate limit) gives '', and the button shows the mark alone.
 */
async function githubStars(): Promise<string> {
  try {
    const res = await fetch('https://api.github.com/repos/vortex-data/vortex', {
      headers: {
        Accept: 'application/vnd.github+json',
        'X-GitHub-Api-Version': '2022-11-28',
        // A token, where the build has one, lifts the API's limit of 60 requests an hour.
        ...(process.env.GITHUB_TOKEN
          ? { Authorization: `Bearer ${process.env.GITHUB_TOKEN}` }
          : {}),
      },
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return '';
    const count: unknown = ((await res.json()) as { stargazers_count?: unknown }).stargazers_count;
    return typeof count === 'number' && Number.isInteger(count) && count >= 0
      ? formatStars(count)
      : '';
  } catch {
    return '';
  }
}

export default defineConfig(async () => ({
  // The docs' Vercel project serves the Explorer under /explorer/, and serves that directory at the
  // root of explore.vortex.dev. Every other address there gets the docs' 404, the one 404 the
  // monorepo builds (docs/vercel-output-config.json).
  base: '/explorer/',
  plugins: [react()],
  define: { 'import.meta.env.VITE_GITHUB_STARS': JSON.stringify(await githubStars()) },
}));
