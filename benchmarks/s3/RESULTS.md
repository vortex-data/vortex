<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# S3 request latency results

Results of the `S3 Measurements` workflow
([run 41102574447](https://github.com/vortex-data/vortex/actions/runs/41102574447), 2026-10-30,
commit `44d2073`): one window of 102,000 hot and 10,000 cold ranged GETs, with 0 errors, from an
on-demand `c8gd.16xlarge` in `us-east-1a` reading `vortex-ci-benchmark-datasets` in `us-east-1`.
The raw JSONL and `tails.csv` are in that run's `s3-measurements-1` artifact, which expires on
2026-11-29.

- **Hot** reads come from the 20-second warmup and three rounds of 1,700 requests for each size and
  concurrency, over four fresh 16 MiB objects that the sweep reads repeatedly.
- **Cold** reads touch each of 2,000 objects per size in `s3-measurements/cold-v1/` exactly once, at
  concurrency 4, with no HEAD or warmup. In this run the objects had been written 27 to 29 minutes
  earlier and never read.

Times are in milliseconds and cover the SDK operation: response headers, first body byte and the full
body. The quantiles use the same linear interpolation as `measure.py`. With 20,400 requests per hot size,
p99.9 rests on about 20 requests. With 2,000 per cold size it rests on about 2, so treat cold p99.9 as
approximate.

## Total latency by size

Hot reads are pooled over concurrency 1, 4, 16 and 64.

| Size | Hot n | Hot p50 | Hot p90 | Hot p99 | Hot p99.9 | Hot max | Cold n | Cold p50 | Cold p90 | Cold p99 | Cold p99.9 | Cold max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 KiB | 20,400 | 26.6 | 33.7 | 62.1 | 156.5 | 308.9 | 2,000 | 69.3 | 106.4 | 154.7 | 253.0 | 796.0 |
| 64 KiB | 20,400 | 27.2 | 36.8 | 69.6 | 155.5 | 290.8 | 2,000 | 88.7 | 130.2 | 177.6 | 227.8 | 278.7 |
| 256 KiB | 20,400 | 28.0 | 37.4 | 67.1 | 220.4 | 556.7 | 2,000 | 93.7 | 138.8 | 182.2 | 213.9 | 222.1 |
| 1 MiB | 20,400 | 32.4 | 47.7 | 88.1 | 245.5 | 313.8 | 2,000 | 120.5 | 163.4 | 216.0 | 314.7 | 406.6 |
| 8 MiB | 20,400 | 91.1 | 155.0 | 357.0 | 742.7 | 1273.2 | 2,000 | 234.3 | 327.2 | 502.4 | 707.1 | 799.6 |

## Time to first byte at the same concurrency (4)

| Size | Hot p50 | Hot p99 | Hot p99.9 | Cold p50 | Cold p99 | Cold p99.9 |
|---|---|---|---|---|---|---|
| 4 KiB | 26.6 | 56.3 | 96.1 | 69.3 | 154.7 | 253.0 |
| 64 KiB | 25.1 | 57.3 | 148.0 | 87.3 | 176.3 | 226.1 |
| 256 KiB | 25.6 | 62.9 | 102.3 | 84.8 | 161.8 | 184.1 |
| 1 MiB | 25.9 | 57.6 | 123.5 | 103.8 | 180.6 | 293.1 |
| 8 MiB | 25.4 | 59.9 | 185.9 | 112.0 | 206.5 | 309.9 |

## Observations

- Cold reads wait 3 to 4 times longer for the first byte than hot ones at the median (69 to 112 ms
  against 25 to 26 ms), and about 3 times longer at p99 (155 to 207 ms against 56 to 63 ms).
- Hot first-byte time does not depend on range size, but cold first-byte time grows with it, from
  69 ms at 4 KiB to 112 ms at 8 MiB.
- For hot ranges up to 1 MiB, p99 is 2 to 3 times the median and p99.9 another 2 to 3 times p99.
- The 8 MiB tail at concurrency 64 (p99 516 ms, p99.9 983 ms) comes from 64 streams sharing the
  runner's bandwidth, not from time to first byte, whose p99 stays at 68 ms.
- Every cold object was read for the first time, while hot reads reuse four objects, so this run
  cannot separate "not read for a while" from "never read before". A later run re-reads the same cold
  objects after hours or days, which separates the two.

## Hot latency by size and concurrency

| Size | Concurrency | n | First byte p50 | First byte p99 | First byte p99.9 | Total p50 | Total p99 | Total p99.9 | Total max |
|---|---|---|---|---|---|---|---|---|---|
| 4 KiB | 1 | 5,100 | 26.7 | 60.7 | 246.4 | 26.7 | 60.7 | 246.4 | 283.7 |
| 4 KiB | 4 | 5,100 | 26.6 | 56.3 | 96.1 | 26.6 | 56.3 | 96.1 | 308.9 |
| 4 KiB | 16 | 5,100 | 26.3 | 57.1 | 77.4 | 26.3 | 57.1 | 77.4 | 244.7 |
| 4 KiB | 64 | 5,100 | 26.8 | 71.7 | 90.4 | 26.8 | 71.7 | 90.4 | 161.4 |
| 64 KiB | 1 | 5,100 | 25.2 | 60.4 | 95.2 | 27.2 | 65.5 | 113.4 | 261.7 |
| 64 KiB | 4 | 5,100 | 25.1 | 57.3 | 148.0 | 27.1 | 67.9 | 162.7 | 290.8 |
| 64 KiB | 16 | 5,100 | 25.5 | 60.5 | 149.3 | 27.6 | 67.7 | 224.2 | 280.4 |
| 64 KiB | 64 | 5,100 | 25.2 | 72.0 | 125.2 | 27.1 | 78.0 | 141.2 | 272.1 |
| 256 KiB | 1 | 5,100 | 25.4 | 58.4 | 161.0 | 28.3 | 65.6 | 271.8 | 556.7 |
| 256 KiB | 4 | 5,100 | 25.6 | 62.9 | 102.3 | 28.6 | 69.2 | 132.8 | 272.9 |
| 256 KiB | 16 | 5,100 | 24.7 | 53.8 | 80.0 | 27.1 | 61.9 | 99.8 | 235.1 |
| 256 KiB | 64 | 5,100 | 25.3 | 62.6 | 243.2 | 28.3 | 71.9 | 266.9 | 382.8 |
| 1 MiB | 1 | 5,100 | 24.9 | 80.1 | 155.9 | 31.3 | 107.0 | 246.5 | 313.8 |
| 1 MiB | 4 | 5,100 | 25.9 | 57.6 | 123.5 | 33.0 | 78.9 | 247.7 | 298.4 |
| 1 MiB | 16 | 5,100 | 25.0 | 58.8 | 99.6 | 32.0 | 76.2 | 138.1 | 265.0 |
| 1 MiB | 64 | 5,100 | 25.9 | 71.1 | 130.7 | 33.6 | 86.4 | 228.8 | 312.1 |
| 8 MiB | 1 | 5,100 | 25.4 | 62.1 | 242.4 | 89.1 | 208.4 | 379.0 | 519.5 |
| 8 MiB | 4 | 5,100 | 25.4 | 59.9 | 185.9 | 88.8 | 194.0 | 324.1 | 525.6 |
| 8 MiB | 16 | 5,100 | 26.1 | 82.5 | 220.5 | 90.0 | 290.9 | 381.9 | 471.7 |
| 8 MiB | 64 | 5,100 | 26.6 | 67.6 | 122.6 | 112.8 | 515.9 | 983.2 | 1273.2 |

## Use in the learned S3 model

`learn.py` in [#10975](https://github.com/vortex-data/vortex/pull/10975) accepts these traces
as they are. It trains only on hot (`phase: "measure"`) GETs and ignores cold ones. Training on this
run plus the two earlier one-window runs (40928994391 and 41062421085, 3,840 requests each) gives
109,680 hot GETs, with 5,340 to 5,556 per size and in-flight band. Each band keeps 256 bulk buckets
plus every observation in its top 0.1%, so the profile reproduces the hot p99.9 tail.

The model cannot represent cold reads yet, because a profile has no notion of whether an object was
read before. Supporting them would need:

1. `learn.py` to learn a second set of size bands from `phase: "cold"` GETs.
2. The profile to carry those bands.
3. The RAM-backed endpoint to draw the first GET of each object path from the cold bands and later
   GETs from the hot ones.
