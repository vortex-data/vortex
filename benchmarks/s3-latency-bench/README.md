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

## Simulated S3

`run --url memory:///` generates the same dataset in an in-memory `object_store`, and
`--latency-model model.json` wraps the store so every request behaves like S3 under the model:
GETs wait a first-byte latency sampled from the model's distribution (HEADs from theirs), then
stream at `min(per_connection, nic / active_transfers)`. `--latency-mode throttle` instead uses
`object_store`'s `ThrottledStore` with the model's median first byte and per-connection bandwidth,
a fixed-delay baseline with no distribution and no shared network cap.

```bash
cargo run --release -p s3-latency-bench -- run --url memory:///s3-latency/ \
  --latency-model model.json --output sim/run-1
python3 scripts/s3-latency-analyze.py sim --compare model.json
```

`--compare` adds a section putting the simulated serial, HEAD, concurrency and scan measurements
next to the real ones the model was fitted from. In CI, the `simulate` and `compare` jobs do this
for both modes after every run and post the result to the PR (`s3-latency-sim` artifact).

## Full scans of TPC-H and TPC-DS

`scan` fully scans every table of a real Vortex dataset through any of these stores: each table's
files are opened with `vortex-file` and every chunk is decoded to canonical arrays, so the runtime
covers both IO and decompression. Generate data with `data-gen`, then scan it from S3 or from a
simulated store:

```bash
cargo run --release --bin data-gen -- tpch --formats vortex --opt scale-factor=10
cargo run --release -p s3-latency-bench -- scan --url memory:///tpch/ \
  --load-from vortex-bench/data/tpch/10.0/vortex-file-compressed \
  --latency-model model.json --dataset tpch-sf10 --output scans/model/tpch.json
python3 scripts/s3-scan-report.py scans
```

In CI the scan jobs do this for TPC-H and TPC-DS at SF=10 on real S3 and both simulated stores, and
post the per-table runtimes to the PR (`s3-latency-scan` artifact).

## TPC-H and TPC-DS queries on simulated S3

The crate is also a library: `s3_latency_bench::simulated::from_env` builds simulated S3 from
`VORTEX_SIMULATED_S3_MODEL` (a model file), `VORTEX_SIMULATED_S3_DIR` (a local copy of the dataset)
and `VORTEX_SIMULATED_S3_MODE` (`model` or `throttle`). `datafusion-bench` registers it in place of
the real bucket when these are set, so the same queries run unchanged against the same `s3://`
paths:

```bash
VORTEX_SIMULATED_S3_MODEL=model.json VORTEX_SIMULATED_S3_DIR=vortex-bench/data/tpch/10.0 \
  cargo run --release --bin datafusion-bench -- tpch --formats vortex \
  --opt scale-factor=10 --opt remote-data-dir=s3://bucket/tpch/ -d gh-json -o queries/model/tpch.json
python3 scripts/s3-query-report.py queries
```

In CI the query jobs run every TPC-H and TPC-DS query this way on real S3 and both simulated
stores, and post per-query runtimes to the PR (`s3-latency-queries` artifact).

Known limits: the model splits bandwidth evenly between transfers, so it underestimates p99
latency when the network is saturated or when thousands of small requests run per second. It was
measured on a single instance type.
