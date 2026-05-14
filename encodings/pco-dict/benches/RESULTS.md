# PcoDictArray microbench (P2d)

Captured from `cargo bench -p vortex-pco-dict --bench pco_dict` with
`N = 1_000_000` `i64` elements, dictionary size `256`, and 100 divan
samples per case. Input is PcoDict-favorable: 256 random `i64` values
are drawn once with seed `42`, then the `N` element stream is built by
sampling that pool uniformly with a separately seeded RNG. The result is
a buffer whose distinct cardinality is exactly 256 — the case where
pco's auto-mode would also pick Dict. Only `i64` is exercised in this
phase; smaller integer widths give the same shape with lower per-bench
times.

All "MB/s" numbers below are computed from the median per-iter time and
the raw buffer byte size (`N * size_of::<i64>() = 8 MB`).

## Hardware

```
Linux 6.18.5 x86_64
Model name: Intel(R) Xeon(R) Processor @ 2.80GHz
```

`lscpu` did not expose a `CPU MHz` field on this host; the nominal
frequency is the 2.80 GHz reported above.

## Throughput (MB/s)

| dtype | PcoDict encode | PcoDict decode | full Pco encode | full Pco decode |
| ----- | -------------: | -------------: | --------------: | --------------: |
| i64   |          1 130 |          3 473 |             638 |           2 734 |

## scalar_at (ns/op)

Each sample issues 1 000 random `execute_scalar` calls; per-op time is
`median / 1000`.

| dtype | PcoDict (ns/op) | full Pco (ns/op) |
| ----- | --------------: | ---------------: |
| i64   |             155 |          724 600 |

## Observations

- PcoDict decode is ~1.27x faster than full-Pco decode on the same
  dict-favorable input (3.47 GB/s vs 2.73 GB/s). The mode-only decode
  kernel is one tight pass that reads `u8` indices and gathers from a
  256-entry `i64` dictionary; the dict fits in two cache lines per page
  so the lookups are L1-resident. Full-Pco still pays its bin-partition
  + tANS + page-assembly overhead on top of the same logical work. The
  gap is smaller than FloatQuant's ~1.7x because the dict-gather kernel
  is gather-bound rather than the contiguous-bandwidth bit ops that
  FloatQuant decode runs, so the constant-factor wins from skipping pco
  bookkeeping don't compound as well.
- PcoDict encode is ~1.77x faster than full-Pco encode (1 130 MB/s vs
  638 MB/s). The encode does a hash-probe + insert into a small map
  (rarely growing past 256) and a second pass to emit `u8` indices;
  full-Pco runs mode-selection over a sample, runs bin partitioning on
  the chosen mode, and emits an entropy-coded page. Encode is still
  considerably slower than decode (~6.3x) because of the hash-table
  cost — at 256 unique entries the map fits in L1 but every input
  element pays for a probe.
- scalar_at on a mode-only PcoDict stack is ~155 ns per element: one
  child `execute_scalar` on `Primitive<u8>` plus a single
  dict-gather. Full PcoArray scalar_at materialises a full page per
  index and lands at ~725 µs per call — a ~4 700x gap, similar to the
  random-access cliff the other mode arrays illustrate. PcoDict's
  absolute scalar_at cost is ~70 ns lower than IntMult and FloatQuant
  because there is only one child to consult (vs two for `(primary,
  secondary)` pairs).
- This is the first mode array with a buffer (the dict) in addition to
  children, and the buffer pattern slotted in cleanly: `nbuffers` /
  `buffer` / `buffer_name` mirror `PcoArray`'s wiring and serialize /
  deserialize round-trip the dict bytes via
  `BufferHandle::new_host(...)` / `try_to_host_sync`. The dictionary
  itself is a `ByteBuffer` of `dict_len * size_of::<T>()` raw native
  bytes, which keeps `validate_parts` arithmetic trivial and lets
  decode reinterpret the bytes as `Buffer<T>` once without copying.
