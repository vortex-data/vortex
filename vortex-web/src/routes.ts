// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import type { MainView } from './components/explorer/MainArea';

export type DeepLink =
  | { kind: 'file'; source: string }
  | { kind: 'compare'; baselineSource: string; candidateSource: string }
  | { kind: 'error'; message: string };

export function viewForPathname(pathname: string): MainView {
  if (pathname === '/compare') return 'compare';
  if (pathname === '/swimlane') return 'swimlane';
  return 'details';
}

export function resolveDeepLink(pathname: string, searchParams: URLSearchParams): DeepLink | null {
  if (pathname === '/file') {
    const source = searchParams.get('url');
    return source
      ? { kind: 'file', source }
      : { kind: 'error', message: 'A file link requires a URL.' };
  }

  if (pathname !== '/compare') return null;

  const baselineSource = searchParams.get('baseline');
  const candidateSource = searchParams.get('candidate');
  if (!baselineSource && !candidateSource) return null;
  if (!baselineSource || !candidateSource) {
    return {
      kind: 'error',
      message: 'A comparison link requires both baseline and candidate URLs.',
    };
  }
  return { kind: 'compare', baselineSource, candidateSource };
}
