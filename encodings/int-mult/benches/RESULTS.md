# IntMultArray microbench (P2a)

Captured from `cargo bench -p vortex-int-mult --bench int_mult` with
`N = 1_000_000` elements per latent type, `base = 1000`, and 100 divan
samples per case. Input is IntMult-favorable: `latent[i] = base * k_i + r_i`
with `k_i` uniform in `[0, 1_000_000)` and `r_i` uniform in `[0, base)`,
fixed seed.

All "MB/s" numbers below are computed from the median per-iter time and the
raw buffer byte size (`N * size_of::<L>()`). The narrow widths `u8`/`u16`
are skipped: the signal is identical and the per-cell wall-time blows up
without telling us anything new.

## Hardware

```
Linux 6.18.5 x86_64
Model name: Intel(R) Xeon(R) Processor @ 2.80GHz
```

`lscpu` did not expose a `CPU MHz` field on this host; the nominal frequency
is the 2.80 GHz reported above.

## Throughput (MB/s) per L

| L   | IntMult encode | IntMult decode | full Pco encode | full Pco decode |
| --- | -------------: | -------------: | --------------: | --------------: |
| u32 |          1 072 |          7 444 |             763 |           2 006 |
| u64 |          1 123 |          7 091 |           1 260 |           3 633 |

## scalar_at (ns/op) per L

Each sample issues 1 000 random `execute_scalar` calls; per-op time is
`median / 1000`.

| L   | IntMult (ns/op) | full Pco (ns/op) |
| --- | --------------: | ---------------: |
| u32 |             204 |          498 500 |
| u64 |             218 |          534 900 |

## Observations

- IntMult decode keeps up with pco's full monolithic decode and then some:
  decode throughput is ~3.7x (u32) and ~2.0x (u64) higher than full-Pco
  decode. The IntMult kernel is a single `wrapping_mul + wrapping_add` per
  element over a contiguous primary buffer; the full-Pco decode pays for
  bin partition + tANS + page assembly on top of that. The structural
  speedup is what motivates the layered split.
- IntMult encode is slower than full-Pco encode at u64 (1.12 GB/s vs 1.26
  GB/s) but faster at u32 (1.07 GB/s vs 0.76 GB/s). The u32 win comes from
  pco picking a heavier entropy code budget on a tighter element width; at
  u64 our two-stream split costs a separate allocation that pco's
  one-buffer chunking avoids.
- scalar_at on a mode-only IntMult stack is ~200 ns per element regardless
  of latent width: two child `execute_scalar` calls (each O(1) on a
  PrimitiveArray) plus the `base * primary + secondary` combine. Full
  PcoArray must materialise an entire chunk per index, so its per-call
  cost is ~500 µs — a ~2500x gap. This is precisely the random-access
  cliff the Fast-RA profile is meant to close, and IntMult clears the
  expected order of magnitude with room to spare.
- u64 encode is ~7 ms for 1 M elements (~140 Mitem/s) and u64 decode is
  ~1.1 ms (~890 Mitem/s); the 6x asymmetry reflects that encode does two
  fresh allocations (`primary` and `secondary` buffers) while decode reuses
  the primary buffer in place and only reads from secondary.
