# FloatQuantArray microbench (P2c)

Captured from `cargo bench -p vortex-float-quant --bench float_quant` with
`N = 1_000_000` `f64` elements, `k = 16`, and 100 divan samples per case.
Input is FloatQuant-favorable: random `u64` bit patterns with the low
`k = 16` bits cleared via `f64::from_bits(rand_u64 & !0xFFFFu64)`, fixed
seed `42`. This forces the `secondary` child to be all zeros while
`primary` carries 48 bits of effective f64 precision per element. Only
`f64` is exercised in this phase; f32 is an open design question for a
later phase.

All "MB/s" numbers below are computed from the median per-iter time and
the raw buffer byte size (`N * size_of::<f64>() = 8 MB`).

## Hardware

```
Linux 6.18.5 x86_64
Model name: Intel(R) Xeon(R) Processor @ 2.80GHz
```

`lscpu` did not expose a `CPU MHz` field on this host; the nominal
frequency is the 2.80 GHz reported above.

## Throughput (MB/s)

| dtype | FloatQuant encode | FloatQuant decode | full Pco encode | full Pco decode |
| ----- | ----------------: | ----------------: | --------------: | --------------: |
| f64   |             1 321 |             3 805 |             810 |           2 292 |

## scalar_at (ns/op)

Each sample issues 1 000 random `execute_scalar` calls; per-op time is
`median / 1000`.

| dtype | FloatQuant (ns/op) | full Pco (ns/op) |
| ----- | -----------------: | ---------------: |
| f64   |                218 |          865 800 |

## Observations

- FloatQuant decode is ~1.7x faster than full-Pco decode on the same
  favorable input (3.81 GB/s vs 2.29 GB/s). The mode-only decode kernel
  is one tight pass over two contiguous `u64` buffers that issues a
  shift, an `or`, and a `f64::from_bits` per element; full-Pco pays for
  bin partition + tANS + page assembly on top of that. The structural
  gap is smaller than FloatMult's (~2.0x) because the FloatQuant kernel
  is purely bit manipulation — no float arithmetic — so it sits closer
  to memory bandwidth and the constant-factor wins from skipping pco
  bookkeeping are proportionally less dramatic.
- FloatQuant encode is ~1.6x faster than full-Pco encode (1 321 MB/s
  vs 810 MB/s). The encode does a `to_bits`, a shift, a mask, and two
  pushes; full-Pco additionally selects a mode, runs bin partitioning
  over a sample, and emits an entropy-coded page. Notably both encode
  paths are roughly 1.5x faster than their FloatMult equivalents at the
  same `N` — FloatQuant's encode is pure integer ops, while FloatMult
  has to divide, round, and multiply per element.
- scalar_at on a mode-only FloatQuant stack is ~218 ns per element: two
  child `execute_scalar` calls on `Primitive<u64>` plus the
  `(primary << k) | secondary` -> `f64::from_bits` recombination. Full
  PcoArray scalar_at materialises a full page per index and lands at
  ~866 µs per call — a ~4 000x gap, the same random-access cliff
  IntMult and FloatMult illustrate. The absolute scalar_at cost is
  within 1 ns of FloatMult's 220 ns, consistent with both modes paying
  the same two-child scalar_at overhead and a negligibly different
  recombination kernel.
- encode is ~2.9x slower than decode (6.1 ms vs 2.1 ms for 1 M f64).
  Decode is one tight pass that reads two u64 buffers and writes one
  f64 buffer; encode pays for two allocations (`primary` and
  `secondary`) plus the shift/mask split. The asymmetry is smaller than
  FloatMult's ~4.5x because the FloatQuant kernels on both sides are
  closer to memory-bandwidth bound — no per-element float math on
  either path.
