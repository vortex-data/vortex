// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { describe, expect, it } from 'vitest';
import { groupDescription } from './descriptions';

// Pure string logic, so these run without Docker. They mirror the Rust
// `descriptions.rs` unit tests verbatim (the v2 contract).
describe('groupDescription', () => {
  it('returns the v2 static descriptions verbatim', () => {
    expect(groupDescription('Random Access')).toBe(
      'Tests performance of selecting arbitrary row indices from a file on NVMe storage',
    );
    expect(groupDescription('Compression')).toBe(
      'Measures encoding and decoding throughput (MB/s) for Vortex files and Parquet files ' +
        '(with zstd page compression)',
    );
    expect(groupDescription('Compression Size')).toBe(
      'Compares compressed file sizes and compression ratios across different encoding strategies',
    );
    expect(groupDescription('Clickbench')).toBe(
      "ClickHouse's analytical benchmark suite testing real-world query patterns on web " +
        'analytics data',
    );
    expect(groupDescription('Statistical and Population Genetics')).toBe(
      'A suite of Statistical and Population genetics queries using the gnomAD dataset',
    );
    expect(groupDescription('PolarSignals Profiling')).toBe(
      'Profiling data benchmark modeled on PolarSignals/Parca, exercising scan-layer performance ' +
        'with projection and filter pushdown on deeply nested schemas',
    );
  });

  it('synthesizes TPC-H descriptions with the scale-bytes annotation', () => {
    expect(groupDescription('TPC-H (NVMe) (SF=1)')).toBe(
      'TPC-H benchmark queries on local NVMe storage at SF=1 (~1GB of data)',
    );
    expect(groupDescription('TPC-H (S3) (SF=10)')).toBe(
      'TPC-H benchmark queries against S3 storage at SF=10 (~10GB of data)',
    );
    expect(groupDescription('TPC-H (NVMe) (SF=100)')).toBe(
      'TPC-H benchmark queries on local NVMe storage at SF=100 (~100GB of data)',
    );
    expect(groupDescription('TPC-H (S3) (SF=1000)')).toBe(
      'TPC-H benchmark queries against S3 storage at SF=1000 (~1TB of data)',
    );
  });

  it('synthesizes TPC-DS descriptions without the scale-bytes annotation', () => {
    expect(groupDescription('TPC-DS (NVMe) (SF=1)')).toBe(
      'TPC-DS benchmark queries on local NVMe storage at SF=1',
    );
    expect(groupDescription('TPC-DS (NVMe) (SF=10)')).toBe(
      'TPC-DS benchmark queries on local NVMe storage at SF=10',
    );
  });

  it('renders TPC-H at an unknown scale factor without the bytes annotation', () => {
    // SF=5 is not in the scale-bytes table, so the `(~NGB of data)` suffix is dropped
    // (matching the Rust `("TPC-H", None)` arm) rather than the name failing to match.
    expect(groupDescription('TPC-H (NVMe) (SF=5)')).toBe(
      'TPC-H benchmark queries on local NVMe storage at SF=5',
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
