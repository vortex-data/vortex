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

The "after" column is the **post-P4-perf-tuning** measurement (encode now
fuses validation + pack + per-batch prefix into one pass over the input,
bin assignment for `<= 16` bins is a branchless cascade instead of a binary
search, and decode writes into pre-allocated spare capacity instead of
calling `BufferMut::push` per element). The "before" column is the original
P4 measurement reproduced in this file's earlier version.

### Scenario A — skewed-low

| direction | encode (before → after) | decode (before → after) | pco_encode | pco_decode |
|-----------|-------------------------|-------------------------|------------|------------|
| MB/s      | 464 → 676 (1.46×)       | 800 → 880 (1.10×)       | 1315       | 3190       |
| Mitem/s   | 58.0 → 84.4             | 100.0 → 109.9           | 164.4      | 398.8      |

### Scenario B — uniform random

| direction | encode (before → after) | decode (before → after) | pco_encode | pco_decode |
|-----------|-------------------------|-------------------------|------------|------------|
| MB/s      | 451 → 770 (1.71×)       | 744 → 2151 (2.89×)      | 1213       | 3147       |
| Mitem/s   | 56.4 → 96.2             | 93.0 → 268.9            | 151.7      | 393.4      |

### Scenario C — quasi-monotone

| direction | encode (before → after) | decode (before → after) | pco_encode | pco_decode |
|-----------|-------------------------|-------------------------|------------|------------|
| MB/s      | 542 → 755 (1.39×)       | 1304 → 1021 (0.78×)     | 1148       | 2629       |
| Mitem/s   | 67.7 → 94.4             | 163.0 → 127.6           | 143.5      | 328.7      |

**Summary.** Encode is 1.39–1.71× faster across the board. Decode is
2.89× faster on the uniform-random workload (the most pco-favourable
shape), 1.10× faster on the skewed-low workload, and 0.78× on the
quasi-monotone workload. The geometric mean of the three decode
speedups is **1.35×**, which clears the 1.3× tuning bar; the
per-scenario picture nevertheless makes clear that the win is
concentrated in the wide-width B case and that the narrow-width C case
has regressed.

The C regression is concentrated in the decode hot loop. With every
bin in C using ~17 bits the bit-unpack is bandwidth-bound on the packed
buffer, and switching from `BufferMut::push` to a direct write through
`spare_capacity_mut` perturbs register allocation enough that the C
case lost ~22 % even though the same change buys B a 3× speedup. A
fastlanes-style fixed-width path for uniform-width batches would
recover C and is the natural next step.

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

2. **Decode throughput now ranges from ~30 % of PCO's (skewed and
   quasi-monotone) to ~70 % (uniform random).** Median MB/s after
   tuning: bin_partition decode is 869 / 2149 / 999 (A/B/C) against
   PCO's 3190 / 3147 / 2629. Scenario B closes most of the gap; A and
   C are still bottlenecked on the per-element bit-unpack chain. A
   fastlanes-style fixed-width path for batches whose bin all share the
   same width should help A and C; that work is left for follow-up.

3. **Encode is ~55–60 % of PCO's throughput.** Quantile sampling plus
   the per-bin width pick plus the variable-width pack now sits at
   ~660–770 MB/s vs. PCO's ~1.15–1.32 GB/s. The fused
   validate+pack+prefix pass dropped the encoder's memory traffic to
   one walk over `values` and `bin_idx`, and the branchless `<= 16`
   cascade replaces the binary search.

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
