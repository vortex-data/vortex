// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { describe, expect, it } from 'vitest';
import {
  chartKeyFromSlug,
  chartKeyToSlug,
  groupKeyFromSlug,
  groupKeyToSlug,
  type ChartKey,
  type GroupKey,
} from './slug';
import { familyForChartKind, familyForGroupKind } from './families';

const CHART_KEYS: ChartKey[] = [
  {
    k: 'QueryMeasurement',
    dataset: 'tpch',
    dataset_variant: null,
    scale_factor: '1',
    storage: 'nvme',
    query_idx: 7,
  },
  { k: 'CompressionTime', dataset: 'taxi', dataset_variant: 'partitioned' },
  { k: 'CompressionSize', dataset: 'taxi', dataset_variant: null },
  { k: 'RandomAccess', dataset: 'taxi' },
  { k: 'VectorSearch', dataset: 'cohere-large-10m', layout: 'partitioned', threshold: 0.75 },
];

const GROUP_KEYS: GroupKey[] = [
  { k: 'QueryGroup', dataset: 'tpch', dataset_variant: null, scale_factor: '1', storage: 'nvme' },
  { k: 'CompressionTimeGroup' },
  { k: 'CompressionSizeGroup' },
  { k: 'RandomAccessGroup' },
  { k: 'VectorSearchGroup', dataset: 'cohere', layout: 'partitioned' },
];

describe('chart slug round-trip', () => {
  it.each(CHART_KEYS)('round-trips $k', (key) => {
    expect(chartKeyFromSlug(chartKeyToSlug(key))).toEqual(key);
  });

  it.each(CHART_KEYS)('prefixes $k with its family chart prefix', (key) => {
    const prefix = chartKeyToSlug(key).split('.')[0];
    expect(prefix).toBe(familyForChartKind(key.k).chartSlugPrefix);
  });

  it('preserves explicit null Option fields (not omitted) and a multibyte dataset', () => {
    const key: ChartKey = {
      k: 'QueryMeasurement',
      dataset: 'tp…h',
      dataset_variant: null,
      scale_factor: null,
      storage: 's3',
      query_idx: 22,
    };
    expect(chartKeyFromSlug(chartKeyToSlug(key))).toEqual(key);
  });

  it('rejects a malformed chart slug', () => {
    expect(() => chartKeyFromSlug('not-a-slug')).toThrow();
    expect(() => chartKeyFromSlug('qm.****')).toThrow();
  });
});

describe('group slug round-trip', () => {
  it.each(GROUP_KEYS)('round-trips $k', (key) => {
    expect(groupKeyFromSlug(groupKeyToSlug(key))).toEqual(key);
  });

  it.each(GROUP_KEYS)('prefixes $k with its family group prefix', (key) => {
    const prefix = groupKeyToSlug(key).split('.')[0];
    expect(prefix).toBe(familyForGroupKind(key.k).groupSlugPrefix);
  });

  it('uses a group prefix distinct from the same family chart prefix', () => {
    const chartPrefix = chartKeyToSlug({
      k: 'CompressionTime',
      dataset: 'tpch',
      dataset_variant: null,
    }).split('.')[0];
    const groupPrefix = groupKeyToSlug({ k: 'CompressionTimeGroup' }).split('.')[0];
    expect(chartPrefix).not.toBe(groupPrefix);
  });

  it('rejects a malformed group slug', () => {
    expect(() => groupKeyFromSlug('not-a-slug')).toThrow();
    expect(() => groupKeyFromSlug('qmg.****')).toThrow();
  });
});

describe('canonical slug encoding', () => {
  // Pins the docstring's contract: `k` first, fields in Rust declaration order,
  // Option fields emitted as explicit `null`. `toEqual` round-trip alone is
  // key-order-insensitive and would miss a reorder; this asserts exact bytes.
  it('encodes a QueryMeasurement chart key as exact canonical JSON', () => {
    const [prefix, payload] = chartKeyToSlug({
      k: 'QueryMeasurement',
      dataset: 'tpch',
      dataset_variant: null,
      scale_factor: '1',
      storage: 'nvme',
      query_idx: 7,
    }).split('.');
    expect(prefix).toBe('qm');
    expect(Buffer.from(payload, 'base64url').toString('utf8')).toBe(
      '{"k":"QueryMeasurement","dataset":"tpch","dataset_variant":null,"scale_factor":"1","storage":"nvme","query_idx":7}',
    );
  });

  it('encodes a QueryGroup key as exact canonical JSON', () => {
    const payload = groupKeyToSlug({
      k: 'QueryGroup',
      dataset: 'tpch',
      dataset_variant: null,
      scale_factor: '1',
      storage: 'nvme',
    }).split('.')[1];
    expect(Buffer.from(payload, 'base64url').toString('utf8')).toBe(
      '{"k":"QueryGroup","dataset":"tpch","dataset_variant":null,"scale_factor":"1","storage":"nvme"}',
    );
  });
});

describe('slug decode validates full payload shape', () => {
  // Forge a slug from an arbitrary payload (any prefix; the prefix is ignored
  // on decode, so the JSON body is the only thing under test).
  const forge = (obj: unknown): string =>
    `x.${Buffer.from(JSON.stringify(obj), 'utf8').toString('base64url')}`;

  it('rejects a known-discriminant chart payload missing a required field', () => {
    expect(() => chartKeyFromSlug(forge({ k: 'RandomAccess' }))).toThrow(/dataset/);
    expect(() => chartKeyFromSlug(forge({ k: 'QueryMeasurement', dataset: 'tpch' }))).toThrow(
      /storage|query_idx/,
    );
  });

  it('rejects a required field of the wrong type', () => {
    expect(() =>
      chartKeyFromSlug(
        forge({
          k: 'QueryMeasurement',
          dataset: 'tpch',
          dataset_variant: null,
          scale_factor: '1',
          storage: 'nvme',
          query_idx: '7',
        }),
      ),
    ).toThrow(/query_idx/);
  });

  it('rejects a query_idx that is not a 32-bit integer (serde i32 parity)', () => {
    // The Rust `query_idx` is an `i32`, so serde rejects a non-integer or
    // out-of-range value as a malformed slug. Without this the forged value
    // would survive decode and only fail at the Postgres `integer` bind.
    const base = {
      k: 'QueryMeasurement',
      dataset: 'tpch',
      dataset_variant: null,
      scale_factor: '1',
      storage: 'nvme',
    };
    expect(() => chartKeyFromSlug(forge({ ...base, query_idx: 1.5 }))).toThrow(/query_idx/);
    expect(() => chartKeyFromSlug(forge({ ...base, query_idx: 2_147_483_648 }))).toThrow(
      /query_idx/,
    );
    expect(() => chartKeyFromSlug(forge({ ...base, query_idx: -2_147_483_649 }))).toThrow(
      /query_idx/,
    );
    // A valid i32 still round-trips.
    expect(chartKeyFromSlug(forge({ ...base, query_idx: 2_147_483_647 })).k).toBe(
      'QueryMeasurement',
    );
  });

  it('rejects an overflowing threshold literal (serde f64 parity)', () => {
    // serde rejects an overflowing JSON number literal ("number out of range")
    // while `JSON.parse` overflows it to `Infinity`. Forge the payload as a raw
    // string: `JSON.stringify(Infinity)` would write `null` and mask the case.
    const raw = '{"k":"VectorSearch","dataset":"cohere","layout":"flat","threshold":1e400}';
    const forged = `x.${Buffer.from(raw, 'utf8').toString('base64url')}`;
    expect(() => chartKeyFromSlug(forged)).toThrow(/threshold/);
  });

  it('treats an absent Option field as null (serde parity)', () => {
    expect(chartKeyFromSlug(forge({ k: 'CompressionTime', dataset: 'taxi' }))).toEqual({
      k: 'CompressionTime',
      dataset: 'taxi',
      dataset_variant: null,
    });
  });

  it('rejects a known-discriminant group payload missing a required field', () => {
    expect(() => groupKeyFromSlug(forge({ k: 'VectorSearchGroup', dataset: 'cohere' }))).toThrow(
      /layout/,
    );
  });
});
