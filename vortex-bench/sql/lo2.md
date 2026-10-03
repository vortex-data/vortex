# LO2 Prometheus metrics benchmark

A benchmark that runs real PromQL over real Prometheus data, translated to SQL.

## Data

The data is the sample run from [LO2v2](https://zenodo.org/records/18937117), a data set of
logs and metrics from the light-oauth2 microservice under Locust load, published by
researchers at the Universities of Oulu and Helsinki under CC BY 4.0. The run has 54 test cases
back to back over about 90 minutes. Prometheus scraped node_exporter and cAdvisor on the single
Docker Compose host, and the authors exported every metric with `query_range` at a one second
step.

The harness downloads the 78 MB archive and converts it to Prometheus layout:

- One row per sample, with columns `labels`, `ts` and `value`. That gives 3,833 series and
  about 19 million samples.
- `labels` is a struct with one nullable, dictionary-encoded string field per label key in the
  data, 86 in all. Each series has nulls for the labels it does not carry. cAdvisor series
  carry up to 18 labels, including Docker Compose labels with long hash values.
- `ts` is int64 milliseconds, the type Prometheus uses for sample timestamps.
- Rows are sorted by label set and then by time, the order of a compacted Prometheus block.

## Queries

The queries in [`lo2.sql`](./lo2.sql), numbered from Q0 in file order, are taken from four
sources that are widely used or used to benchmark Prometheus. Each query quotes its original
PromQL in a comment.

| Source | Queries |
|---|---|
| [node-mixin](https://github.com/prometheus/node_exporter/tree/master/docs/node-mixin), shipped with node_exporter | Recording rules and one alert, Q0 to Q9 |
| [awesome-prometheus-alerts](https://github.com/samber/awesome-prometheus-alerts) host rules, as run by VictoriaMetrics' prometheus-benchmark | Alerts, Q10 to Q16 |
| [Node Exporter Full](https://grafana.com/grafana/dashboards/1860), Grafana dashboard 1860 | Panels, Q17 to Q22 |
| [prombench](https://github.com/prometheus/test-infra/tree/master/prombench), the Prometheus release benchmark | Load generator queries, Q23 to Q27 |

These are the queries from those sources whose metrics exist in the data. The translation
rules are in the header of `lo2.sql`. In short:

- Instant queries are evaluated at one fixed time. Dashboard panels run as a one hour range
  query at a one minute step.
- `rate()` is the increase over the samples in the window divided by their time span. It omits
  Prometheus' extrapolation to the window edges and its counter-reset handling.
- Series are keyed by the labels that vary within each metric. Thresholds and `and` become
  filters and joins, and vector matching on identical labels becomes a join or a pivot.

The host was healthy during the run, so the alert queries return no rows. They still read and
filter the same data a Prometheus server would to evaluate them.

Every query runs unchanged on DataFusion and DuckDB, and both return the same results.

The harness lives in [`src/lo2`](../src/lo2).

## CI variant

CI runs this suite only under the `action/bench-sql-extended` label, on DataFusion and DuckDB
over Parquet and Vortex. It does not run on `develop` or under the other benchmark labels.

## Running locally

```bash
vx-bench run lo2 --engine datafusion,duckdb --format parquet,vortex
```
