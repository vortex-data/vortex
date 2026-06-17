// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { describe, expect, it } from 'vitest';
import { groupDescription } from './descriptions';

// Pure string logic, so these run without Docker. They mirror the Rust
// `descriptions.rs` unit tests verbatim (the v2 contract).
describe('groupDescription', () => {
  it('returns the editorial static descriptions', () => {
    expect(groupDescription('Random Access')).toBe(
      'Point lookups — selecting specific rows by position from an NVMe file. What feature ' +
        'stores, vector retrieval, and per-record serving actually do.',
    );
    expect(groupDescription('Compression')).toBe(
      'Encode and decode throughput (MB/s) for Vortex vs Parquet (zstd page compression). ' +
        'Encode gates ingestion; decode gates every scan after.',
    );
    expect(groupDescription('Compression Size')).toBe(
      'Compressed file size per format across a fixed set of datasets. A faster format that ' +
        'bloats on disk just trades one bill for another.',
    );
    expect(groupDescription('Clickbench')).toBe(
      "ClickHouse's 43-query analytical suite over real web-analytics data — the field's " +
        'standard test for single-table scans, filters, and aggregations.',
    );
    expect(groupDescription('Statistical and Population Genetics')).toBe(
      'Population-genetics queries over the gnomAD dataset — DuckDB-only, exercising the ' +
        'deeply-nested array operations real genomics pipelines run on.',
    );
    expect(groupDescription('PolarSignals Profiling')).toBe(
      'Scan-layer benchmark modeled on PolarSignals/Parca: projection and filter pushdown ' +
        'over deeply-nested profile schemas — the shape continuous-profiling backends actually read.',
    );
  });

  it('synthesizes TPC-H descriptions with the scale-bytes annotation', () => {
    expect(groupDescription('TPC-H (NVMe) (SF=1)')).toBe(
      'TPC-H — 22 analytical queries against the canonical OLAP star schema — at scale factor 1 (~1 GB), on local NVMe.',
    );
    expect(groupDescription('TPC-H (S3) (SF=10)')).toBe(
      'TPC-H — 22 analytical queries against the canonical OLAP star schema — at scale factor 10 (~10 GB), against S3.',
    );
  });

  it('synthesizes TPC-DS descriptions', () => {
    expect(groupDescription('TPC-DS (NVMe) (SF=1)')).toBe(
      'TPC-DS — the broader 99-query analytical suite (larger schemas, skewed distributions) — at scale factor 1, on local NVMe.',
    );
  });

  it('renders TPC-H at an unknown scale factor without the bytes annotation', () => {
    // SF=5 is not in the scale-bytes table, so the `(~N GB)` suffix is dropped
    // (matching the `("TPC-H", None)` arm) rather than the name failing to match.
    expect(groupDescription('TPC-H (NVMe) (SF=5)')).toBe(
      'TPC-H — 22 analytical queries against the canonical OLAP star schema — at scale factor 5, on local NVMe.',
    );
  });

  it('returns null for groups with no canonical description', () => {
    expect(groupDescription('cohere-large-10m / partitioned')).toBeNull();
    expect(groupDescription('Made-up benchmark')).toBeNull();
  });

  it('returns null for malformed TPC names', () => {
    // No `(NVMe)` / `(S3)` storage prefix.
    expect(groupDescription('TPC-H something else')).toBeNull();
    // `SF=` with no digits.
    expect(groupDescription('TPC-H (NVMe) (SF=)')).toBeNull();
  });
});
