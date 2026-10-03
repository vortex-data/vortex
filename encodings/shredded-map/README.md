# vortex-shredded-map

A `Map<Utf8, V>` encoding that shreds frequently occurring keys into dedicated columns, aimed at
observability labels (Prometheus/Thanos series labels, OpenTelemetry attributes).

## Layout

`ShreddedMap` keeps the logical dtype of the map it encodes (UTF-8 keys, `keys_sorted = true`) and
stores:

- one row-aligned nullable **column per shredded key**. A non-null value at row `i` means row `i`
  contains `(key, value)`;
- a **residual** `Map` of the same dtype with every other entry, which also carries the outer
  validity.

A key is shredded when at least `min_frequency` of the rows hold a non-null value for it (default
5%, at most `max_columns`). For a union value dtype, a column whose values all select the same
variant stores that variant's child directly ("typed" shredding). For example `job` becomes a
`Utf8?` column and `container_number` an `I64?` column, instead of a sparse union.

Decoding merges each row's present columns with its residual entries in key order. Only the first
entry of a key in a row moves to a column, and only when its value is non-null. Equal keys decode
shredded-first. Together these rules make the round trip exact for duplicate keys and null values.
Slice, take and filter push down into every child.

`labels` defines the label value type `Union<str: Utf8, int: I64, float: F64, bool: Bool>`.
`LabelValue::infer` picks the narrowest variant that formats back to exactly the original string,
so `"200"` becomes an int, while `"007"`, `"False"` and `"1e-05"` stay strings.

`ops` implements each operation twice, once over the canonical `Map` (the baseline) and once over
`ShreddedMap`:

| op | `ops::map` | `ops::shredded` |
| --- | --- | --- |
| label names per row (`List<Utf8>`) | zero-copy keys | merge |
| distinct label names | hash every key | column keys + residual |
| one label as a string | scan every row | column read (residual for rare keys) |
| all values as strings (`Map<Utf8, Utf8>`) | format every value | format per column, then merge |
| decompress into a canonical `Map` | n/a | merge + gather |
| project a key subset into a new map | filter entries | pick columns + filter residual |

## Correctness

`src/tests.rs` contains a proptest that generates label maps from a skewed vocabulary. The maps
include empty, long and non-ASCII keys, duplicate keys, null values, null rows and mixed value types.
Each map is shredded with random options and checked against a plain Rust model. The checks cover:

- decode through `to_map`, `execute` and `scalar_at`;
- every operation, for both the shredded and the canonical-map implementations;
- slice, take and filter.

The `prom_labels_report` example also checks every label of every row of the real dataset against
the canonical map.

## Row runs and dictionary columns

Label maps are usually written series by series, so consecutive rows repeat the same labels
(about 92 times per run in the benchmark data). Three things exploit this:

- `shred` detects rows equal to their predecessor, by shared list-view range or by content. Such
  rows share residual entries and column codes.
- With `ShredOptions::dictionary` (the default), each column is a `Dict` of per-row codes into the
  column's distinct values.
- `encode` goes one step further and returns `Dict(row codes, ShreddedMap of distinct runs)`, the
  same shape as a time-series database's series index. Every operation runs once per run and then
  expands to rows by gathering list-view `(offset, size)` pairs, so rows that share labels also
  share their decoded entries.

`compress_shredded` and `compress_encoded` compress child by child. A generic compressor would
otherwise canonicalize its input and undo the layout.

## Results (string to string, 1M sample rows, 10.8k series)

Sizes:

| format | size |
| --- | --- |
| Parquet zstd(3) `Map<Utf8,Utf8>` | 0.46 MiB |
| Vortex `Map` + BtrBlocks | 13.58 MiB |
| Vortex `Map` + BtrBlocks compact (zstd/pco) | 1.06 MiB |
| shredded + BtrBlocks | 0.22 MiB |
| shredded + compact | 0.09 MiB |
| encoded + BtrBlocks | 0.18 MiB |
| encoded + compact | 0.05 MiB |

Compression takes 0.51 s for the plain map, 0.12 s for the shredded map and 0.035 s for the
encoded map. With one row per series (205k rows, no repeats), shredded + BtrBlocks is 1.41 MiB,
against 2.91 MiB for the plain map, and shredded + compact is 0.43 MiB, against 0.47 MiB for
Parquet.

Median times, every Vortex result fully materialized:

| op | Arrow | Map | Map + BtrBlocks | shredded | encoded | encoded + BtrBlocks |
| --- | --- | --- | --- | --- | --- | --- |
| decompress to a map | n/a | n/a | 39 ms | 98 ms | 7.9 ms | 8.6 ms |
| label names per row | 0.4 µs | 3.8 ms | 43 ms | 83 ms | 7.0 ms | 7.6 ms |
| distinct label names | 60 ms | 64 ms | 100 ms | 3.8 ms | 0.08 ms | 0.13 ms |
| one label, every row (`job`) | 33 ms | 32 ms | 71 ms | 0.8 ms | 1.2 ms | 1.7 ms |
| one label, 18% of rows (`image`) | 27 ms | 30 ms | 70 ms | 2.8 ms | 4.7 ms | 7.8 ms |
| one label, 4% of rows (`mode`, residual) | 24 ms | 29 ms | 67 ms | 8.3 ms | 1.8 ms | 2.8 ms |
| project 3 common keys to a map | 133 ms | 84 ms | 125 ms | 21 ms | 5.7 ms | 6.5 ms |
| project 2 rare keys to a map | 53 ms | 52 ms | 87 ms | 21 ms | 5.7 ms | 6.1 ms |

## Dataset

[LO2v2](https://zenodo.org/records/18937117) (`light-oauth2-metrics.zip`, 78 MB zipped, 1.7 GB of
raw Prometheus range-query JSON). It holds real cAdvisor, node_exporter and Go runtime metrics
from a microservice testbed: 205,104 series, 18.96M samples and 86 distinct label keys.
`__name__`, `instance` and `job` are on every series, 11 cAdvisor `container_label_*` keys are on
about 18%, and the rest is a long tail.

```bash
python3 scripts/lo2_to_jsonl.py <unzipped>/LO2_run_1739743201 /path/lo2_series.jsonl
LO2_PATH=/path/lo2_series.jsonl LO2_ROWS=1000000 cargo bench -p vortex-shredded-map --bench prom_labels
LO2_PATH=/path/lo2_series.jsonl cargo run --release -p vortex-shredded-map --example prom_labels_report
```

Rows are samples (long format, one label map per sample). Set `LO2_MODE=series` for one row per
series, and `LO2_VALUES=union` for `Map<Utf8, Union<str, int, float, bool>>` values.
