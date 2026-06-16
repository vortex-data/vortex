// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// @vitest-environment node

import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';

const css = readFileSync(join(__dirname, 'globals.css'), 'utf8');

describe('PR-5.0.95 spinner CSS', () => {
  it('defines a spin keyframes animation and a spinner rule', () => {
    expect(css).toMatch(/@keyframes\s+chart-spin/);
    expect(css).toMatch(/\.chart-spinner\b/);
  });

  it('disables the spinner animation under prefers-reduced-motion: reduce', () => {
    const reduced = css.match(
      /@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{(?:[^{}]|\{[^{}]*\})*\}/g,
    );
    expect(reduced).not.toBeNull();
    expect(reduced!.join('\n')).toMatch(/\.chart-spinner[\s\S]*animation:\s*none/);
  });
});

describe('PR-5.0.97 chart-placeholder CSS', () => {
  it('defines a .chart-placeholder rule', () => {
    expect(css).toMatch(/\.chart-placeholder\b/);
  });

  it('styles the .chart-placeholder-text label', () => {
    expect(css).toMatch(/\.chart-placeholder-text\s*\{[^}]*\}/);
  });

  it('includes .chart-placeholder .chart-spinner in the prefers-reduced-motion block', () => {
    const reduced = css.match(
      /@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{(?:[^{}]|\{[^{}]*\})*\}/g,
    );
    expect(reduced).not.toBeNull();
    const block = reduced!.join('\n');
    expect(block).toMatch(/\.chart-placeholder\s+\.chart-spinner/);
    expect(block).toMatch(/animation:\s*none/);
  });

  it('does NOT display:none the ring or label under reduced motion', () => {
    const reduced = css.match(
      /@media\s*\(prefers-reduced-motion:\s*reduce\)\s*\{(?:[^{}]|\{[^{}]*\})*\}/g,
    );
    expect(reduced).not.toBeNull();
    const block = reduced!.join('\n');
    // The ring and label must stay visible; display:none is forbidden here.
    expect(block).not.toMatch(/\.chart-placeholder[\s\S]*display\s*:\s*none/);
    expect(block).not.toMatch(/\.chart-placeholder-text[\s\S]*display\s*:\s*none/);
  });
});
