# S3 latency benchmark

Measures object store request latency for the requests a Vortex scan issues, and fits a latency
model from the measurements. It answers questions such as how long a 64 KiB footer read takes, how
throughput scales with requests in flight, and how much coalescing saves on a projection.

## Dataset

`prepare` uploads synthetic files with the shape of Vortex files written by the default strategy:
per-row-group column chunks with optional validity buffers, per-column zone maps, a footer, a
postscript and the `VTXF` end-of-file marker. The bytes are random; only offsets and sizes matter.
The CI dataset is 100 files of 20 MiB.

## Workloads

`run` records one CSV row per request with its size, offset, requests in flight, time to first
byte and total time. It runs these phases:

- `cold`: the first requests of the process, including DNS, TCP and TLS setup.
- `size_sweep`: serial reads from 1 B to 16 MiB at random offsets.
- `suffix`: suffix reads (`bytes=-N`) against the equivalent bounded tail reads, and HEAD.
- `concurrency`: 4 KiB, 64 KiB, 1 MiB and 8 MiB reads at 1 to 256 requests in flight.
- `scan`: the request stream of `vortex-file` scans. Each file is opened with a HEAD, the 64 KiB
  read from the end of the file and a second footer read when the footer is larger. Segment
  reads are coalesced like `vortex-io` (merge within 1 MiB, up to 16 MiB) and issued with up to
  192 requests in flight per file. Scenarios cover full scans, projections with and without
  coalescing, filter then project, point lookups, and overlapping concurrent scans.

## Running it

Run the `S3 Latency` workflow from the Actions tab. It runs the benchmark 10 times by default,
each on a fresh `c8gd.8xlarge` in `us-east-1`. The job summary shows the report, and two
artifacts hold the results:

- `s3-latency-model`: `model.json`, the latency model described below.
- `s3-latency-report`: the Markdown report and every request as `all-requests.csv.gz`.

Datasets are deleted once they are older than three hours.

Locally, against any S3-compatible store configured through the usual `AWS_*` variables, or a
`file://` directory:

```bash
cargo run --release -p s3-latency-bench -- prepare --url s3://bucket/prefix/
cargo run --release -p s3-latency-bench -- run --url s3://bucket/prefix/ --output results/run-1
python3 scripts/s3-latency-analyze.py results --report report.md --model model.json
```

## Latency model

`model.json` (schema `vortex-s3-latency-model/1`) models one request as

```text
latency = first_byte + bytes / min(per_connection_bytes_per_s, nic_bytes_per_s / active_bodies)
```

where `active_bodies` is the number of requests transferring data at the same time. Its
`request_model` holds:

| field | meaning |
|---|---|
| `first_byte_s.inverse_cdf` | `[quantile, seconds]` pairs to sample the time to first byte from |
| `head_s.inverse_cdf` | the same for HEAD requests |
| `per_connection_bytes_per_s` | transfer rate of one connection |
| `nic_bytes_per_s` | aggregate transfer rate of the instance |
| `small_read_extra_first_byte_s` | extra first-byte time for 1, 2 and 3 byte reads |
| `fresh_object` | first-byte multiplier for objects read within `warm_after_age_s` of being written |

The model is fitted from the microbenchmark phases only. The analysis then replays every
measured scan through a simulator driven by the model and reports predicted against measured
wall time under `validation`, so a model that stops describing the scans shows up there. The
remaining keys hold the raw measurements: `fits` has linear fits of the serial and concurrent
reads, and `measurements` has the per-size, per-concurrency and per-scenario tables shown in the
report.

Known limits: the model splits bandwidth evenly between transfers, so it underestimates p99
latency when the network is saturated or when thousands of small requests run per second. It was
measured on a single instance type.
