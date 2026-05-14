# FloatMultArray microbench (P2b)

Captured from `cargo bench -p vortex-float-mult --bench float_mult` with
`N = 1_000_000` `f64` elements, `base = 0.01`, and 100 divan samples per
case. Input is FloatMult-favorable: `x[i] = base * k_i` with `k_i` drawn
uniformly from `[0, 10_000_000)`, fixed seed `42`. Only `f64` is exercised
in this phase; f32 is an open design question for a later phase.

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

| dtype | FloatMult encode | FloatMult decode | full Pco encode | full Pco decode |
| ----- | ---------------: | ---------------: | --------------: | --------------: |
| f64   |              832 |            3 801 |             440 |           1 929 |

## scalar_at (ns/op)

Each sample issues 1 000 random `execute_scalar` calls; per-op time is
`median / 1000`.

| dtype | FloatMult (ns/op) | full Pco (ns/op) |
| ----- | ----------------: | ---------------: |
| f64   |               220 |        1 042 000 |

## Observations

- FloatMult decode is ~2.0x faster than full-Pco decode on the same
  favorable input (3.80 GB/s vs 1.93 GB/s). The mode-only decode kernel
  is a single `base * primary + ulp_offset` per element over two
  contiguous `i64` buffers producing one `f64` buffer; full-Pco pays for
  bin partition + tANS + page assembly on top of that. The structural
  speedup is the same kind of win we saw on IntMult (~2x at u64), with
  the float-bits arithmetic adding a slight overhead compared to a pure
  integer mul-add.
- FloatMult encode is ~1.9x faster than full-Pco encode (832 MB/s vs 440
  MB/s). The encode does a divide, round-to-i64, an `f64` mul for the
  approximation, and two pushes; full-Pco additionally selects a mode,
  runs bin partitioning over a sample, and emits an entropy-coded page.
- scalar_at on a mode-only FloatMult stack is ~220 ns per element: two
  child `execute_scalar` calls on `Primitive<i64>` plus the `(p, s, base)`
  -> `f64` recombination. Full PcoArray scalar_at materialises a full
  page per index and lands at ~1.04 ms per call — a ~4 700x gap. This is
  the same random-access cliff IntMult illustrated, and FloatMult clears
  it by a similar order of magnitude.
- encode is ~4.5x slower than decode (9.6 ms vs 2.1 ms for 1 M f64).
  Decode is one tight pass that reads two i64 buffers and writes one f64
  buffer; encode pays for `f64::round`, a saturating `as i64` cast, two
  allocations (`primary` and `secondary`), and the bit-level subtract.
  The asymmetry is larger than IntMult's ~6x because the FloatMult encode
  kernel is more arithmetic-heavy on a per-element basis, while the
  decode kernel is closer to memory-bandwidth bound.
