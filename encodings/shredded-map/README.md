# vortex-shredded-map

A `Map<Utf8, V>` encoding for observability labels (Prometheus/Thanos series labels,
OpenTelemetry attributes, log fields). Keys that occur in many rows move into dedicated columns;
the rest stay in a residual map.

## Layout

`ShreddedMap` keeps the logical dtype of the map it encodes (UTF-8 keys, `keys_sorted = true`) and
stores:

- a non-nullable **`fields` struct** with one row-aligned nullable field per shredded key, in key
  order. A non-null value at row `i` means row `i` contains `(key, value)`;
- a **residual** `Map` of the same dtype holding every other entry, which also carries the outer
  validity;
- an optional **repeat hint**, a bit per row marking rows equal to their predecessor.

`shred` picks columns by how many rows hold a key (`ShredOptions`):

- every key on at least `sparse_below` of the rows (default 80%) becomes a **dense** column, a
  `Dict` of per-row codes into the column's distinct values;
- of the remaining keys on at least `min_frequency` of the rows (default 1%), the
  `max_sparse_columns` most common (default 64) become **sparse** columns: a null-filled `Sparse`
  array holding only the present rows, with patch chunk offsets so one row is found without a
  search over all present rows;
- everything else stays in the residual.

Only the first entry of a key in a row moves to a column, and only when its value is non-null.
Decoding merges each row's present columns with its residual entries in key order, shredded entries
first. Together these rules make the round trip exact for duplicate keys and null values. Slice,
take and filter push down into every child.

`encode` goes one step further when rows repeat: it returns `Dict(row codes, ShreddedMap of
distinct rows)`, the shape of a time-series database's series index. Every operation runs once per
distinct row and expands to rows by gathering list-view `(offset, size)` pairs.

`compress_shredded` and `compress_encoded` compress child by child through the dictionary and
sparse layers. A generic compressor would canonicalize its input and undo the layout.

For union values, `labels` defines `Union<str: Utf8, int: I64, float: F64, bool: Bool>`, and a
column whose values all select one variant stores that variant directly ("typed" shredding).

## Shredding contract

A `ShreddedMap` is one node of a shredding contract meant to be shared with variant shredding:

```text
Node {
  fields:   Struct { k1: Child, k2: Child, ... }   // nullable fields, any encoding
  residual: Map<Utf8, V>                           // entries of this node that were not shredded
}
Child = Utf8 column | Struct column (a nested Node)
```

- **Exclusive.** For every shredded key `k`, a row's first non-null value of `k` lives only in
  `fields.k`. A null in `fields.k` means `k` is absent or null in that row, so readers never consult
  the residual for a shredded key. The residual holds unshredded keys, null-valued entries and later
  duplicates of a key.
- **Selection** per node, configurable through `ShredOptions`: a key needs `min_frequency` of the
  rows to be shredded, is dense at `sparse_below` presence and sparse below it, and at most
  `max_sparse_columns` sparse keys are kept.
- **Version 1** shreds string leaves only. Nested objects (struct-column children), a depth limit
  and a key whose value is an object in some rows and a string in others (an error) apply once
  nested inputs are shredded; a `Map<Utf8, Utf8>` has one level and can never conflict. Typed
  leaves (narrowest integer or float) are deferred.

## Operations

`ops` implements each operation over the canonical `Map` (the baseline), over `ShreddedMap`, and
over whatever `encode` returns:

| op | `ops::map` | `ops::shredded` |
| --- | --- | --- |
| label names per row (`List<Utf8>`) | zero-copy keys | merge |
| distinct label names | hash every key | column keys + residual |
| one label as a string | scan every row | column read (residual for rare keys) |
| all values as strings (`Map<Utf8, Utf8>`) | format every value | format per column, then merge |
| decompress into a canonical `Map` | n/a | column-major merge, or row-major when most rows repeat |
| project a key subset into a new map | filter entries | pick columns + filter residual |

## Correctness

`src/tests.rs` holds a proptest that generates label maps from a skewed vocabulary, including
empty, long and non-ASCII keys, duplicate keys, null values, null rows, mixed value types and
rows sharing entries. Each map is shredded with random options and checked against a plain Rust
model: decode through `to_map`, `execute` and `scalar_at`, every operation for the shredded, encoded
and canonical-map implementations, and slice, take and filter.

## Datasets and benchmarks

The benchmarks read JSON lines of `{"labels": {...}}` (`LO2_PATH`, `LO2_MODE=series` for one row
per line). Datasets used during development:

| dataset | rows | distinct keys | notes |
| --- | --- | --- | --- |
| [LO2v2](https://zenodo.org/records/18937117) Prometheus series | 205k | 86 | `scripts/lo2_to_jsonl.py` converts the raw range-query JSON |
| flaws.cloud CloudTrail events | 586k | 869 | flattened request parameters, long tail of keys |
| OpenStack OSProfiler traces | 435k | 239 | span metadata |
| OpenTelemetry demo spans | 532k | 140 | span attributes |
| OTRF APT29 Windows event logs | 440k | 378 | event fields |
| Online Boutique spans | 1M | 6 | fixed keys, nothing to shred |

```bash
LO2_PATH=/path/data.jsonl LO2_MODE=series cargo bench -p vortex-shredded-map --bench prom_labels
LO2_PATH=/path/data.jsonl LO2_MODE=series cargo run --release -p vortex-shredded-map --example prom_labels_report
```

`prom_labels_report` prints sizes against Parquet and checks every label of every row against the
canonical map. `shred_sweep` sweeps the shredding thresholds, and `layout` prints the compressed
layout column by column.

Compressed sizes (MiB), `Map<Utf8, Utf8>`:

| dataset | Parquet zstd(3) | `Map` + BtrBlocks | + compact | shredded + BtrBlocks | + compact | `encode` + compact |
| --- | --- | --- | --- | --- | --- | --- |
| LO2 | 0.47 | 2.91 | 1.37 | 0.72 | 0.18 | 0.015 |
| CloudTrail | 27.44 | 61.13 | 29.95 | 32.19 | 24.53 | 23.62 |
| OpenStack | 5.29 | 27.10 | 7.70 | 10.37 | 3.26 | 2.40 |
| OpenTelemetry | 1.81 | 39.22 | 5.19 | 3.63 | 1.22 | 0.71 |
| APT29 | 9.44 | 47.11 | 14.87 | 15.04 | 4.60 | 3.01 |
| Online Boutique | 0.30 | 6.15 | 1.57 | 1.53 | 0.80 | 0.31 |

## Key-set layout and map compressor schemes

`keyset::KeySetMap` is a second map layout: one copy of each distinct key set, plus a per-row key
set id and value offset, so keys are stored once per set rather than once per entry. It supports
the same operations as the other layouts.

`scheme` registers the layouts as BtrBlocks map schemes (`KEYSET_SCHEME`, `KEYSET_ROWS_SCHEME`,
`SHREDDED_SCHEME`), so a compressor with them registered picks the smallest layout for a map
column by itself. Compact sizes (MiB):

| dataset | `Map` | key sets | `encode` | BtrBlocks pick |
| --- | --- | --- | --- | --- |
| LO2 | 1.37 | 1.22 | 0.015 | 0.015 |
| CloudTrail | 29.95 | 27.97 | 23.62 | 23.58 |
| OpenStack | 7.70 | 6.04 | 2.40 | 2.32 |
| OpenTelemetry | 5.19 | 3.57 | 0.71 | 0.70 |
| APT29 | 14.87 | 11.82 | 3.01 | 2.98 |
| Online Boutique | 1.57 | 1.48 | 0.31 | 0.31 |

## Single-label lookups

`point::ShreddedProbe` and `point::MapProbe` read one label of one row without decoding anything
else, and keep what they decode between reads: run ends, sparse chunk offsets, and the 1024-value
blocks of children that reads have touched. `MapProbe` resolves a key to its dictionary codes once
and then compares integer codes. `point_bench` compares them with a row scan over Arrow.

CloudTrail, 586k rows, compressed with BtrBlocks, per lookup over 2000 random rows:

| key | Arrow | `ShreddedProbe` warm | `MapProbe` warm |
| --- | --- | --- | --- |
| `userAgent` (dense, run-end codes) | 0.86 µs | 0.49 µs | 3.3 µs |
| `errorMessage` (sparse, 50%) | 0.48 µs | 0.46 µs | 2.4 µs |
| `requestParameters.policyArn` (sparse, 5%) | 0.37 µs | 0.14 µs | 3.0 µs |
| `eventName` (dense) | 0.74 µs | 1.2 µs | 2.1 µs |

## Filter, then project

`ops::query` answers `SELECT k1, ..., kn WHERE labels[k] = v`, one `Utf8` array per projected
key:

- `query::shredded` compares only the filter key's column (on its dictionary, then codes) and
  filters only the projected columns, keeping their sparse and dictionary layers. Dictionaries
  decode only the 1024-value blocks the selected rows reference.
- `query::map` scans rows once, stopping at the filter key in each row, comparing dictionary codes
  or string view prefixes instead of strings.

`query_bench` picks each dataset's queries from the data (the most common key with 5 to 10,000
distinct values, filtered on values near 0.5%, 5% and 30% of rows, projecting 5 keys of mixed
frequency) and checks every result against a row scan over Arrow. Milliseconds, best of 5, for the
most selective filter:

| dataset | rows matched | Arrow | `Map` | `Map` + BtrBlocks | shredded | shredded + BtrBlocks | shredded + compact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LO2 | 972 | 1.7 | 2.5 | 3.7 | 0.21 | 0.66 | 1.3 |
| CloudTrail | 2,170 | 24.3 | 29.7 | 87.1 | 0.60 | 4.1 | 10.6 |
| OpenStack | 855 | 31.9 | 48.5 | 47.6 | 1.1 | 2.7 | 11.6 |
| OpenTelemetry | 2,743 | 46.3 | 72.4 | 68.9 | 0.55 | 2.6 | 4.4 |
| APT29 | 292 | 21.8 | 33.2 | 92.1 | 0.48 | 1.6 | 7.1 |
| Online Boutique | 3,089 | 15.6 | 28.8 | 24.3 | 0.66 | 2.0 | 18.0 |

## Squeeze

BtrBlocks' compact preset compresses strings with zstd level 3 in frames of 8192 values and has no
zstd for integers. `squeeze::squeeze` re-encodes every integer and string node of a compressed
layout with high-level zstd over large frames wherever that is smaller, narrowing dictionary codes
first. Row-dictionary codes follow repeating trace shapes that zstd's long-range matching captures.
It trades random access within a frame for size.

`best_size` searches shredding options and layouts, squeezes the smallest, and checks every row of
the result. Smallest layout found (MiB):

| dataset | Parquet zstd(3) | Parquet zstd(22) | Vortex | vs zstd(3) | vs zstd(22) |
| --- | --- | --- | --- | --- | --- |
| LO2 | 0.469 | 0.349 | 0.014 | 32.5× smaller | 24.2× smaller |
| CloudTrail | 27.44 | 25.56 | 23.07 | 1.19× smaller | 1.11× smaller |
| OpenStack | 5.29 | 4.05 | 2.00 | 2.64× smaller | 2.03× smaller |
| OpenTelemetry | 1.81 | 1.20 | 0.59 | 3.08× smaller | 2.05× smaller |
| APT29 | 9.44 | 6.94 | 2.52 | 3.75× smaller | 2.76× smaller |
| Online Boutique | 0.298 | 0.163 | 0.052 | 5.7× smaller | 3.1× smaller |

`size_breakdown` prints where the bytes of a layout go.
