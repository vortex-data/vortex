# ConsecutiveDeltaArray microbench (P3)

Captured from `cargo bench -p vortex-consecutive-delta --bench
consecutive_delta` with `N = 1_000_000` `i64` elements and 100 divan
samples per case. Two inputs are exercised in parallel:

- **A (favorable)**: monotone-with-jitter timestamps,
  `x[i] = 1_700_000_000_000 + i * 1000 + noise` for `noise ∈ [-50, 50]`.
  After delta the values are ~`1000 ± 50` — the regime where pco's auto
  mode would also pick a delta path.
- **B (control)**: uniform random `i64`. Deltas have full high-bit
  entropy; nothing structural to exploit.

All "MB/s" numbers below are computed from the median per-iter time and
the raw buffer byte size (`N * size_of::<i64>() = 8 MB`).

## Hardware

```
Linux 6.18.5 x86_64
Model name: Intel(R) Xeon(R) Processor @ 2.80GHz
```

`lscpu` did not expose a `CPU MHz` field on this host; the nominal
frequency is the 2.80 GHz reported above.

## Throughput (MB/s) — Scenario A (monotone timestamps, favorable)

| stage  | ConsecutiveDelta | full Pco |
| ------ | ---------------: | -------: |
| encode |            3 821 |    1 129 |
| decode |            3 939 |    3 013 |

## Throughput (MB/s) — Scenario B (random i64, control)

| stage  | ConsecutiveDelta | full Pco |
| ------ | ---------------: | -------: |
| encode |            3 829 |      644 |
| decode |            3 928 |    2 768 |

## scalar_at (µs/op) — Scenario A

Each sample issues 1 000 random `execute_scalar` calls; per-op time is
`median / 1000`.

| array            | µs/op | relative |
| ---------------- | ----: | -------: |
| ConsecutiveDelta | 132.2 |     1.0x |
| full Pco         | 648.2 |     4.9x |

## scalar_at (µs/op) — Scenario B

| array            | µs/op | relative |
| ---------------- | ----: | -------: |
| ConsecutiveDelta | 129.7 |     1.0x |
| full Pco         | 717.8 |     5.5x |

The 1 000 indices are drawn uniformly from `[0, N)`, so on average each
ConsecutiveDelta call replays ~`N/2 = 500_000` wrapping adds from the
seed before returning.

## Observations

- **The random-access cliff is real but does not cross pco within
  N = 1 M.** ConsecutiveDelta scalar_at is `O(i)`: at the average index
  (`~500 000`) it costs ~132 µs in both scenarios. Full pco scalar_at
  costs ~648 µs (A) / ~718 µs (B), a fixed page-decode cost independent
  of index. A simple linear extrapolation
  (`t(i) ≈ 132 µs · (i / 500 000)`) puts the crossover at
  `i ≈ 2.45 M` for scenario A and `i ≈ 2.71 M` for scenario B — i.e.
  outside this 1 M-element array. Even at the *worst* index `N - 1`,
  our layered cost (~264 µs extrapolated) stays under pco's per-call
  decode. The cliff is visible — every doubling of `N` doubles our
  midpoint cost while pco's stays flat — but it does not actually win
  for pco at this `N` because pco's page-decode constant is ~5x larger
  than the prefix-sum work for half of our array. Crossover happens at
  roughly `N ≈ 5 M` for scenario A.

- **Delta decode is faster than full pco on both scenarios.** On the
  favorable scenario A, layered decode is ~1.31x faster than full-pco
  decode (3.94 GB/s vs 3.01 GB/s). On the unfavorable scenario B the
  gap widens slightly (3.93 GB/s vs 2.77 GB/s, ~1.42x), because pco's
  full decode pays the bin-partition + tANS + page-assembly overhead
  proportional to entropy and B's deltas have nothing to compress;
  meanwhile our decode is one tight prefix-sum loop regardless of
  input statistics. The structural decode kernel pays no entropy
  premium.

- **Delta encode is dramatically faster, and the gap widens on the
  control.** On A, layered encode is ~3.4x faster than full pco
  (3.82 GB/s vs 1.13 GB/s). On B the ratio explodes to ~5.9x
  (3.83 GB/s vs 0.64 GB/s) because pco's encode-side mode/delta
  detection runs over a sample and then bins + tANS-codes high-entropy
  output, whereas our encode is a single `wrapping_sub` pass that does
  not care about entropy. This is exactly the "structural layers are
  cheap, entropy is expensive" split the design doc argues for.

- **The ConsecutiveDelta kernel is bandwidth-bound, not arithmetic-
  bound.** Encode and decode throughput both sit at ~3.8–3.9 GB/s on
  this host, essentially the same number across scenarios A and B and
  across encode vs decode. A single read + write + one wrapping
  add/sub is the whole inner loop, and the rate is what the L3-resident
  8 MB buffer's read/write pair will sustain. Higher orders or input
  validity plumbing will only ever subtract from this number.

- **PCo encode is the slowest single operation in the matrix and is
  ~2x slower on B than on A.** Full pco encode on the control jumps
  from 7.1 ms (A) to 12.4 ms (B), reflecting that pco's auto-mode
  spends more cycles when the post-delta latent does not compress —
  it tries more modes, picks Classic, and the entropy code still has
  to encode high-entropy output. ConsecutiveDelta encode is unchanged
  (2.09 ms vs 2.09 ms). The layered split lets a future selection
  policy skip the entropy stage entirely when it would not help,
  which is the central premise of the Fast-RA profile.
