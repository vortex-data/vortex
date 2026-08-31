// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { resolveDeepLink, viewForPathname } from './routes';

describe('Vortex Web routes', () => {
  const routes = [
    ['/', 'details'],
    ['/file', 'details'],
    ['/swimlane', 'swimlane'],
    ['/compare', 'compare'],
    ['/unknown', 'details'],
  ] as const;
  routes.forEach(([pathname, expected]) => {
    it(`maps ${pathname} to the ${expected} view`, () => {
      assert.equal(viewForPathname(pathname), expected);
    });
  });

  it('resolves an individual remote file deep link', () => {
    const source = 'https://objects.example/file.vortex?signature=a+b';
    const params = new URLSearchParams({ url: source });

    assert.deepEqual(resolveDeepLink('/file', params), { kind: 'file', source });
  });

  it('resolves both files in a comparison deep link', () => {
    const baselineSource = 'https://objects.example/previous.vortex?signature=old';
    const candidateSource = 'https://objects.example/new.vortex?signature=new';
    const params = new URLSearchParams({ baseline: baselineSource, candidate: candidateSource });

    assert.deepEqual(resolveDeepLink('/compare', params), {
      kind: 'compare',
      baselineSource,
      candidateSource,
    });
  });

  it('rejects incomplete deep links', () => {
    assert.deepEqual(resolveDeepLink('/file', new URLSearchParams()), {
      kind: 'error',
      message: 'A file link requires a URL.',
    });
    assert.deepEqual(
      resolveDeepLink(
        '/compare',
        new URLSearchParams({ baseline: 'https://objects.example/previous.vortex' }),
      ),
      {
        kind: 'error',
        message: 'A comparison link requires both baseline and candidate URLs.',
      },
    );
  });
});
