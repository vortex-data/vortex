// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { describe, expect, it } from 'vitest';
import {
  commitWindowLimit,
  commitWindowUrlValue,
  DEFAULT_COMMIT_WINDOW,
  MAX_NUMERIC_COMMIT_WINDOW,
  parseCommitWindow,
} from './window';

describe('parseCommitWindow', () => {
  it('defaults to the 100-commit window for null/undefined', () => {
    expect(parseCommitWindow(undefined)).toEqual({ kind: 'last', n: DEFAULT_COMMIT_WINDOW });
    expect(parseCommitWindow(null)).toEqual({ kind: 'last', n: DEFAULT_COMMIT_WINDOW });
  });

  it('parses "all" case-insensitively and trims surrounding whitespace', () => {
    expect(parseCommitWindow('all')).toEqual({ kind: 'all' });
    expect(parseCommitWindow('ALL')).toEqual({ kind: 'all' });
    expect(parseCommitWindow(' all ')).toEqual({ kind: 'all' });
  });

  it('parses a plain numeric window', () => {
    expect(parseCommitWindow('50')).toEqual({ kind: 'last', n: 50 });
  });

  it('floors 0 to 1 and clamps large values to the max', () => {
    expect(parseCommitWindow('0')).toEqual({ kind: 'last', n: 1 });
    expect(parseCommitWindow('99999')).toEqual({ kind: 'last', n: MAX_NUMERIC_COMMIT_WINDOW });
  });

  it('falls back to the default for malformed values', () => {
    expect(parseCommitWindow('banana')).toEqual({ kind: 'last', n: DEFAULT_COMMIT_WINDOW });
    expect(parseCommitWindow('')).toEqual({ kind: 'last', n: DEFAULT_COMMIT_WINDOW });
    expect(parseCommitWindow('-5')).toEqual({ kind: 'last', n: DEFAULT_COMMIT_WINDOW });
    expect(parseCommitWindow('5.5')).toEqual({ kind: 'last', n: DEFAULT_COMMIT_WINDOW });
  });

  it('treats a value overflowing u32 as malformed (matches Rust parse::<u32>)', () => {
    expect(parseCommitWindow('99999999999999')).toEqual({
      kind: 'last',
      n: DEFAULT_COMMIT_WINDOW,
    });
  });
});

describe('commitWindowLimit', () => {
  it('returns the count for a bounded window and null for all', () => {
    expect(commitWindowLimit({ kind: 'last', n: 42 })).toBe(42);
    expect(commitWindowLimit({ kind: 'all' })).toBeNull();
    expect(commitWindowLimit(parseCommitWindow(undefined))).toBe(100);
  });
});

describe('commitWindowUrlValue', () => {
  it('renders the URL value', () => {
    expect(commitWindowUrlValue({ kind: 'last', n: 100 })).toBe('100');
    expect(commitWindowUrlValue({ kind: 'all' })).toBe('all');
  });
});
