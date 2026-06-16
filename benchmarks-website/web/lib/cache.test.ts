// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { describe, expect, it } from 'vitest';

import { READ_API_CACHE_CONTROL } from './cache';

describe('READ_API_CACHE_CONTROL', () => {
  it('keeps a 5-minute fresh window but allows day-scale stale-while-revalidate', () => {
    expect(READ_API_CACHE_CONTROL).toContain('s-maxage=300');
    expect(READ_API_CACHE_CONTROL).toContain('stale-while-revalidate=86400');
  });
});
