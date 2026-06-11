// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { renderToStaticMarkup } from 'react-dom/server';
import { describe, expect, it } from 'vitest';

import { SummaryCard } from '@/components/SummaryCard';
import type { Summary } from '@/lib/summary';

function render(summary?: Summary): string {
  return renderToStaticMarkup(<SummaryCard summary={summary} />);
}

describe('SummaryCard', () => {
  it('renders nothing when there is no summary', () => {
    expect(render(undefined)).toBe('');
  });

  it('renders a randomAccess card with ranks, ns times, and ratios', () => {
    const html = render({
      type: 'randomAccess',
      title: 'Random Access Performance',
      rankings: [
        { name: 'vortex', time: 1_500_000, ratio: 1 },
        { name: 'parquet', time: 3_000_000, ratio: 2 },
      ],
      explanation: 'Random access time | Ratio to fastest (lower is better)',
    });
    expect(html).toContain('class="benchmark-scores-summary"');
    expect(html).toContain('<h3 class="scores-title">Random Access Performance</h3>');
    expect(html).toContain('#1');
    expect(html).toContain('vortex');
    expect(html).toContain('1.50 ms');
    expect(html).toContain('1.00x');
    expect(html).toContain('#2');
    expect(html).toContain('parquet');
    expect(html).toContain('3.00 ms');
    expect(html).toContain('2.00x');
    expect(html).toContain('Random access time | Ratio to fastest (lower is better)');
  });

  it('renders nothing for a randomAccess card with no rankings', () => {
    expect(render({ type: 'randomAccess', title: 't', rankings: [], explanation: 'e' })).toBe('');
  });

  it('renders both compression speedups when present', () => {
    const html = render({
      type: 'compression',
      title: 'Compression Throughput vs Parquet',
      compressRatio: 2.5,
      decompressRatio: 1.8,
      datasetCount: 3,
      explanation: 'higher is better',
    });
    expect(html).toContain('Write Speed (Compression)');
    expect(html).toContain('2.50x');
    expect(html).toContain('Scan Speed (Decompression)');
    expect(html).toContain('1.80x');
  });

  it('omits a missing compression speedup row', () => {
    const html = render({
      type: 'compression',
      title: 't',
      compressRatio: 2.5,
      datasetCount: 1,
      explanation: 'e',
    });
    expect(html).toContain('Write Speed (Compression)');
    expect(html).not.toContain('Scan Speed (Decompression)');
  });

  it('renders nothing for a compression card with neither ratio', () => {
    expect(render({ type: 'compression', title: 't', datasetCount: 0, explanation: 'e' })).toBe('');
  });

  it('renders min/mean/max for a compressionSize card', () => {
    const html = render({
      type: 'compressionSize',
      title: 'Compression Size Summary',
      minRatio: 0.3,
      meanRatio: 0.45,
      maxRatio: 0.6,
      datasetCount: 4,
      explanation: 'lower is better',
    });
    expect(html).toContain('Min Size Ratio');
    expect(html).toContain('0.30x');
    expect(html).toContain('Mean Size Ratio');
    expect(html).toContain('0.45x');
    expect(html).toContain('Max Size Ratio');
    expect(html).toContain('0.60x');
  });

  it('renders a queryBenchmark card with scores and total runtimes', () => {
    const html = render({
      type: 'queryBenchmark',
      title: 'Performance Summary',
      rankings: [{ name: 'vortex:vortex-file', score: 1.0, totalRuntime: 5_000_000_000 }],
      explanation: 'lower is better',
    });
    expect(html).toContain('#1');
    expect(html).toContain('vortex:vortex-file');
    expect(html).toContain('1.00x');
    expect(html).toContain('5.00 s');
  });
});
