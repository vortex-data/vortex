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
series.
