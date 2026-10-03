# vortex-shredded-map

A `Map<Utf8, V>` encoding for observability labels (Prometheus/Thanos series labels,
OpenTelemetry attributes, log fields). Keys that occur in many rows move into dedicated columns;
the rest stay in a residual map.

## Layout

`ShreddedMap` keeps the logical dtype of the map it encodes (UTF-8 keys, `keys_sorted = true`) and
stores:

- one row-aligned nullable **column per shredded key**. A non-null value at row `i` means row `i`
  contains `(key, value)`;
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
