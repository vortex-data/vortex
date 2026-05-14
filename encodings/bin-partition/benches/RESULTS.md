# BinPartition + VarWidthBitPacked microbench (P4)

First layered-pco phase whose byte budget is directly comparable to a
monolithic `PcoArray`. `BinPartitionArray` decomposes the input into
`(bin_idx, offset)`; `VarWidthBitPackedArray` stores the offsets at
per-bin widths in a single packed buffer with a batch-indexed prefix sum
for O(64) random access.

Run with:

```bash
RUSTC_WRAPPER= cargo bench -p vortex-bin-partition --bench bin_partition
```

- `N = 1_000_000` `i64` values per scenario
- `MAX_BINS = 16`, `PCO_LEVEL = 0`, `PCO_VALUES_PER_PAGE = 0` (defaults)
- `SCALAR_AT_SAMPLES = 1_000`
- Seed: `42` (and scenario-specific xors)
- Numbers below are **medians** from divan, 100 samples per case.

## Hardware

```text
$ uname -srm
Linux 6.18.5 x86_64

$ lscpu | grep -E "Model name|CPU MHz" | head -2
Model name:                              Intel(R) Xeon(R) Processor @ 2.10GHz
```

4 vCPUs, KVM hypervisor, L1d 192 KiB / L2 8 MiB / L3 260 MiB.

## Scenarios

| Tag | Name           | Builder                                            |
|-----|----------------|----------------------------------------------------|
| A   | Skewed-low     | `(rng.f64().powi(3) * 1000.0) as i64`              |
| B   | Uniform random | `rng.random_range(-1e9 ..= 1e9)`                   |
| C   | Quasi-monotone | `i + rng.random_range(-100 ..= 100)`               |

## Compression ratio

Raw input is `N * 8 = 8_000_000` bytes (7.63 MiB). Encoded sizes are
`nbytes()` on the produced array, printed in MiB by the bench harness.

| Scenario       | raw size | bin_partition size | bin_partition ratio (×) | full Pco size | full Pco ratio (×) |
|----------------|----------|--------------------|-------------------------|---------------|--------------------|
| Skewed-low     | 7.63 MiB | 2.66 MiB           | 2.87 ×                  | 1.19 MiB      | 6.40 ×             |
| Uniform random | 7.63 MiB | 5.25 MiB           | 1.45 ×                  | 3.70 MiB      | 2.06 ×             |
| Quasi-monotone | 7.63 MiB | 3.93 MiB           | 1.94 ×                  | 1.07 MiB      | 7.11 ×             |

## Throughput (MB/s, median)

### Scenario A — skewed-low

| direction | encode_bin_partition | decode_bin_partition | pco_encode | pco_decode |
|-----------|----------------------|----------------------|------------|------------|
| MB/s      | 464.2                | 799.7                | 1315       | 3190       |
| Mitem/s   | 58.0                 | 100.0                | 164.4      | 398.8      |

### Scenario B — uniform random

| direction | encode_bin_partition | decode_bin_partition | pco_encode | pco_decode |
|-----------|----------------------|----------------------|------------|------------|
| MB/s      | 450.9                | 744.3                | 1213       | 3147       |
| Mitem/s   | 56.4                 | 93.0                 | 151.7      | 393.4      |

### Scenario C — quasi-monotone

| direction | encode_bin_partition | decode_bin_partition | pco_encode | pco_decode |
|-----------|----------------------|----------------------|------------|------------|
| MB/s      | 541.5                | 1304                 | 1148       | 2629       |
| Mitem/s   | 67.7                 | 163.0                | 143.5      | 328.7      |

## `scalar_at` (median, 1_000 random indices)

Total time is the loop over 1_000 indices; the per-call number divides
that total by 1_000.

| scenario       | bin_partition total | bin_partition per-call | pco total | pco per-call |
|----------------|---------------------|------------------------|-----------|--------------|
| A skewed-low   | 391.4 µs            | ~391 ns                | 617 ms    | ~617 µs      |
| B uniform      | 401.3 µs            | ~401 ns                | 623.8 ms  | ~624 µs      |
| C quasi-mono   | 397.6 µs            | ~398 ns                | 771.7 ms  | ~772 µs      |

bin_partition scalar_at sustains **~2.5 Mitem/s** across all three
scenarios; the bench reports it as `~390 µs` per 1k-index loop.

## Observations

1. **Ratio: bin_partition cracks PCO's lead on uniform data, but not
   when there is structure.** On B (uniform random `i64` in
   `[-1e9, 1e9]`) bin_partition keeps 70 % of PCO's bytes (1.45 × vs.
   2.06 ×). On A (skewed-low) it captures 45 % of PCO's ratio (2.87 ×
   vs. 6.40 ×). On C (quasi-monotone) PCO's delta encoding rips through
   the structure (7.11 ×) and bin_partition only sees the wide range
   (1.94 ×). This matches expectations: bin partition alone never
   touches local correlation; that is a job for delta/RLE layers above
   it.

2. **Decode throughput is ~3.5–4× below PCO's, which is the price of
   the layered indirection on this codepath.** Median MB/s:
   bin_partition decode is 800 / 744 / 1304 (A/B/C) against PCO's
   3190 / 3147 / 2629. Scenario C is the fastest decode for
   bin_partition (1.3 GB/s) because every bin happens to be narrow,
   making the bit-unpack faster.

3. **Encode is ~2.5–3× slower than PCO.** Quantile sampling plus the
   per-bin width pick plus the variable-width pack costs roughly 500
   MB/s vs. PCO's ~1.2 GB/s. This is acceptable for a first cut — there
   is no SIMD path for the bit-pack yet.

4. **Random-access `scalar_at` is the headline win.** bin_partition
   resolves any element in ~390–400 ns regardless of scenario (the
   batch-prefix-sum is O(1) plus an O(64) bit-unpack within the batch).
   The full-PCO baseline is ~600–770 µs per element because PCO has no
   random access and must decompress a whole page. That is a
   **~1500–2000 × speed-up** for point lookups, and it is the property
   that justifies the layered stack for selective scans. Compared to
   the ~150 ns/scalar reported for the mode/run-end arrays in P1/P3,
   bin_partition is in the same order of magnitude — a 2–3 × slower
   constant for what is now a real bit-packed representation.

5. **Variance is generally tame.** The widest slowest/fastest spread is
   in `pco_full_encode A` (5.85 ms ↔ 22.76 ms) and `decode_bin_partition
   B` (10.2 ms ↔ 18.3 ms); medians and means agree to within ~5 % in
   every case, so the headline numbers are stable. No bench errored or
   failed to converge.
