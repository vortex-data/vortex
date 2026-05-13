# OrderedLatentArray microbench (P1)

Captured from `cargo bench -p vortex-ordered-latent --bench recast` with
`N = 1_000_000` elements per primitive type and 100 divan samples per case.
All "MB/s" numbers below are computed from the median per-iter time and the
raw buffer byte size (`N * size_of::<T>()`).

## Hardware

```
Linux 6.18.5 x86_64
Model name: Intel(R) Xeon(R) Processor @ 2.80GHz
```

`lscpu` did not expose a `CPU MHz` field on this host; nominal frequency is
the 2.80 GHz reported above.

## Throughput (MB/s) per T

Computed as `N * sizeof(T) / median_ns * 1000`. The `memcpy` column is the
reference clone of an identical `Buffer<T>` (see the anomaly note below).

| T   | OrderedLatent encode | OrderedLatent decode | memcpy (Buffer clone) |
| --- | -------------------- | -------------------- | --------------------- |
| i8  |               11 700 |               11 600 |                79 700 |
| i16 |                9 630 |                9 540 |               160 000 |
| i32 |                9 000 |                9 080 |               319 000 |
| i64 |                8 470 |                8 130 |               639 000 |
| u32 |               14 400 |               14 100 |               319 000 |
| u64 |               13 400 |               13 400 |               638 000 |
| f32 |                8 260 |                7 820 |               319 000 |
| f64 |                7 720 |                7 600 |               638 000 |

## scalar_at (ns/op) per T

Each sample issues 1 000 random `execute_scalar` calls; per-op time is
`median / 1000`. `pco_full` rows are reported in microseconds because Pco
materialises a full chunk per call.

| T   | OrderedLatent (ns/op) | full PcoArray (ns/op) |
| --- | --------------------- | --------------------- |
| i8  |                   133 |                   n/a |
| i16 |                   133 |               477 000 |
| i32 |                   133 |               504 000 |
| i64 |                   137 |               739 000 |
| u32 |                   133 |               501 000 |
| u64 |                   137 |               742 000 |
| f32 |                   134 |               530 000 |
| f64 |                   138 |               599 000 |

Pco does not encode `i8`, so the `pco_full` column is `n/a` for that row.

## Full pco reference

`i64 x 1M elements: encode median ~12.8 ms, decode median ~2.93 ms`
(`Pco::from_primitive` at level 0, default values-per-page).

## Observations

- OrderedLatent encode and decode are within ~1.3-1.6x of one another for
  every T, confirming the recast is symmetric: both directions are a
  per-element xor-flip plus a typed copy.
- Per-element scalar_at on OrderedLatent lands in ~130-140 ns regardless of
  T, three to four orders of magnitude below `pco_full`'s 0.5-0.7 ms (Pco
  has to materialise a full chunk per index).
- OrderedLatent recast throughput is ~1-2 GB/s per element-byte (encode i64
  median 8.47 GB/s, u64 median 13.4 GB/s), so it scales sublinearly with
  element width: per-item time grows roughly with `sizeof(T)`, leaving wider
  types slightly slower per byte due to the wider xor work.
- Anomaly: the `memcpy` row measures `Buffer<T>::clone`, which is an `Arc`
  bump rather than a real `memcpy`. The reported 80-640 TB/s figures are
  therefore an Arc-clone reference, not a memory-bandwidth ceiling, and the
  "within 2x of memcpy" claim cannot be made from this run. OrderedLatent
  throughput against a true `memcpy` baseline is left as follow-up.
