// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { describe, expect, it } from 'vitest';

import type { CommitPoint, NamedChartResponse, SeriesTag, UnitKind } from './queries';
import {
  buildFacets,
  buildHistory,
  facetGeomeans,
  facetOf,
  formatValue,
  geomeanOf,
  latestValue,
  mult,
  pooledGeomean,
} from './synthesis';

function commit(i: number): CommitPoint {
  const sha = String(i).padStart(2, '0') + '0'.repeat(38);
  return {
    sha,
    timestamp: `2026-05-${String(i).padStart(2, '0')} 12:00:00+00`,
    message: `commit ${i}`,
    url: `https://example/commit/${sha}`,
  };
}

function chart(
  name: string,
  unit: UnitKind,
  commits: CommitPoint[],
  series: Record<string, (number | null)[]>,
  meta: Record<string, SeriesTag>,
): NamedChartResponse {
  return {
    name,
    slug: name,
    display_name: name,
    unit_kind: unit,
    history: {
      total_commits: commits.length,
      start_index: 0,
      loaded_commits: commits.length,
      complete: true,
    },
    commits,
    series,
    series_meta: meta,
  };
}

/** A two-commit query chart with duckdb/datafusion × vortex/parquet series. */
function queryChart(name: string, latest: Record<string, number>): NamedChartResponse {
  const cs = [commit(1), commit(2)];
  const series: Record<string, (number | null)[]> = {};
  const meta: Record<string, SeriesTag> = {};
  for (const [key, v] of Object.entries(latest)) {
    const [engine, format] = key.split(':');
    series[key] = [v * 1.1, v]; // older then latest
    meta[key] = { engine, format };
  }
  return chart(name, 'time_ns', cs, series, meta);
}

describe('latestValue', () => {
  it('returns the last finite value, skipping trailing nulls', () => {
    expect(latestValue([1, 2, null])).toBe(2);
    expect(latestValue([5])).toBe(5);
    expect(latestValue([null, null])).toBeUndefined();
    expect(latestValue([])).toBeUndefined();
  });
});

describe('facetOf', () => {
  it('uses the engine when present', () => {
    expect(facetOf('duckdb:parquet', 'duckdb')).toBe('duckdb');
  });
  it('falls back to the op suffix', () => {
    expect(facetOf('parquet:encode', undefined)).toBe('encode');
  });
  it('is empty with neither', () => {
    expect(facetOf('parquet', undefined)).toBe('');
  });
});

describe('buildFacets — query suite (engine facets)', () => {
  const charts = [
    queryChart('Q1', {
      'duckdb:vortex-file-compressed': 100,
      'duckdb:parquet': 200,
      'datafusion:vortex-file-compressed': 150,
      'datafusion:parquet': 300,
    }),
    queryChart('Q2', {
      'duckdb:vortex-file-compressed': 100,
      'duckdb:parquet': 400,
      'datafusion:vortex-file-compressed': 100,
      'datafusion:parquet': 100,
    }),
  ];
  const { facets, facetedByEngine } = buildFacets(charts);

  it('splits into one facet per engine, sorted', () => {
    expect(facetedByEngine).toBe(true);
    expect(facets.map((f) => f.facet)).toEqual(['datafusion', 'duckdb']);
  });
  it('defaults to Vortex vs Parquet', () => {
    expect(facets[0].defaultA).toBe('vortex-file-compressed');
    expect(facets[0].defaultB).toBe('parquet');
    expect(facets[0].metric).toBe('faster');
  });
  it('collects one item row per chart with the latest per-format value', () => {
    const duck = facets.find((f) => f.facet === 'duckdb')!;
    expect(duck.queries.map((q) => q.query)).toEqual(['Q1', 'Q2']);
    expect(duck.queries[0].v).toEqual({ 'vortex-file-compressed': 100, parquet: 200 });
  });
  it('orders formats Vortex-first', () => {
    expect(facets[0].formats.map((f) => f.id)).toEqual(['vortex-file-compressed', 'parquet']);
    expect(facets[0].formats.map((f) => f.label)).toEqual(['Vortex', 'Parquet']);
  });
});

describe('facetGeomeans', () => {
  it('computes the geomean of parquet/vortex ratios with a win count', () => {
    const charts = [
      queryChart('Q1', { 'duckdb:vortex-file-compressed': 100, 'duckdb:parquet': 200 }), // 2x
      queryChart('Q2', { 'duckdb:vortex-file-compressed': 100, 'duckdb:parquet': 400 }), // 4x
    ];
    const [duck] = facetGeomeans(charts);
    expect(duck.facet).toBe('duckdb');
    expect(duck.geomean).toBeCloseTo(Math.sqrt(2 * 4), 10); // geomean(2,4)
    expect(duck.wins).toBe(2);
    expect(duck.total).toBe(2);
  });
  it('counts a loss (ratio < 1) as a non-win', () => {
    const charts = [
      queryChart('Q1', { 'duckdb:vortex-file-compressed': 200, 'duckdb:parquet': 100 }), // 0.5x
    ];
    const [duck] = facetGeomeans(charts);
    expect(duck.wins).toBe(0);
    expect(duck.geomean).toBeCloseTo(0.5, 10);
  });
});

describe('buildFacets — compression (op facets) and size (smaller)', () => {
  it('facets by operation for compression times', () => {
    const c = chart(
      'taxi',
      'time_ns',
      [commit(1)],
      {
        'vortex-file-compressed:encode': [100],
        'parquet:encode': [200],
        'vortex-file-compressed:decode': [50],
        'parquet:decode': [120],
      },
      {
        'vortex-file-compressed:encode': { format: 'vortex-file-compressed' },
        'parquet:encode': { format: 'parquet' },
        'vortex-file-compressed:decode': { format: 'vortex-file-compressed' },
        'parquet:decode': { format: 'parquet' },
      },
    );
    const { facets, facetedByEngine } = buildFacets([c]);
    expect(facetedByEngine).toBe(false);
    expect(facets.map((f) => f.facet)).toEqual(['decode', 'encode']);
    expect(facets.every((f) => f.metric === 'faster')).toBe(true);
  });
  it('reports the smaller metric for byte units (no facet)', () => {
    const c = chart(
      'taxi',
      'bytes',
      [commit(1)],
      { 'vortex-file-compressed': [400], parquet: [800] },
      {
        'vortex-file-compressed': { format: 'vortex-file-compressed' },
        parquet: { format: 'parquet' },
      },
    );
    const { facets } = buildFacets([c]);
    expect(facets).toHaveLength(1);
    expect(facets[0].facet).toBe('');
    expect(facets[0].metric).toBe('smaller');
  });
});

describe('buildHistory', () => {
  it('emits one line per non-Parquet format with per-commit geomeans', () => {
    const c = chart(
      'taxi',
      'time_ns',
      [commit(1), commit(2)],
      {
        'duckdb:vortex-file-compressed': [50, 100],
        'duckdb:parquet': [100, 400], // parquet/vortex = 2 then 4
      },
      {
        'duckdb:vortex-file-compressed': { engine: 'duckdb', format: 'vortex-file-compressed' },
        'duckdb:parquet': { engine: 'duckdb', format: 'parquet' },
      },
    );
    const hist = buildHistory([c]);
    expect(hist).not.toBeNull();
    expect(hist!.commits).toHaveLength(2);
    expect(hist!.lines).toHaveLength(1);
    expect(hist!.lines[0].format).toBe('vortex-file-compressed');
    expect(hist!.lines[0].engine).toBe('duckdb');
    expect(hist!.lines[0].speedups).toEqual([2, 4]);
  });
  it('returns null when there is no Parquet baseline', () => {
    const c = chart(
      'taxi',
      'time_ns',
      [commit(1)],
      { 'duckdb:vortex-file-compressed': [100] },
      { 'duckdb:vortex-file-compressed': { engine: 'duckdb', format: 'vortex-file-compressed' } },
    );
    expect(buildHistory([c])).toBeNull();
  });
});

describe('pooledGeomean / geomeanOf / formatValue / mult', () => {
  it('pools by item count', () => {
    // facet A: geomean 2 over 1 item; facet B: geomean 8 over 1 item -> geomean(2,8)=4
    const pooled = pooledGeomean([
      { facet: 'a', geomean: 2, wins: 1, total: 1 },
      { facet: 'b', geomean: 8, wins: 1, total: 1 },
    ]);
    expect(pooled).toBeCloseTo(4, 10);
  });
  it('geomeanOf ignores non-positive / non-finite', () => {
    expect(geomeanOf([4, 9])).toBeCloseTo(6, 10);
    expect(geomeanOf([0, -1, NaN])).toBeNull();
  });
  it('formats values by unit', () => {
    expect(formatValue(1_500_000_000, 'time_ns')).toBe('1.5 s');
    expect(formatValue(2_500_000, 'time_ns')).toBe('2.5 ms');
    expect(formatValue(1024, 'bytes')).toBe('1.0 KiB');
  });
  it('formats multipliers', () => {
    expect(mult(1.3)).toBe('1.30×');
    expect(mult(23.4)).toBe('23×');
  });
});
