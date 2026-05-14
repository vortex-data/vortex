# AnsArray Microbench Results (P5)

Microbenchmark results for the tANS entropy code layer over a `u8` symbol
stream. Two scenarios over `N = 1_000_000` symbols, both drawing from an
alphabet of 16. `ANS_SIZE_LOG = 12`.

## Hardware

```
Linux 6.18.5 x86_64
Intel(R) Xeon(R) Processor @ 2.10GHz
```

## Compression ratio

| Scenario | Raw | ANS | Ratio | Shannon limit |
|---|---:|---:|---:|---:|
| A — Zipf-skewed (alphabet=16, `p_k ∝ 1/(k+1)`) | 0.95 MiB | 0.41 MiB | 2.35x | ~2.36x |
| B — Uniform random (alphabet=16) | 0.95 MiB | 0.48 MiB | 2.00x | 2.00x |

Raw input is `N = 1_000_000` `u8` symbols (≈0.954 MiB).

The Shannon-limit columns are the theoretical ceiling: 8 bits/symbol over
the entropy H. For Zipf-16 with `p_k ∝ 1/(k+1)` (the truncated harmonic),
H ≈ 3.39 bits/symbol → 8/H ≈ 2.36x. For uniform-16, H = 4 bits/symbol →
8/H = 2.00x exactly.

## Throughput (MB/s)

### Scenario A — Zipf-skewed alphabet of 16

| Op | Fastest | Median | Mean | Slowest |
|---|---:|---:|---:|---:|
| encode | 75.7 MB/s | 74.0 MB/s | 71.3 MB/s | 54.3 MB/s |
| decode | 98.4 MB/s | 96.9 MB/s | 96.2 MB/s | 89.2 MB/s |

### Scenario B — Uniform random alphabet of 16

| Op | Fastest | Median | Mean | Slowest |
|---|---:|---:|---:|---:|
| encode | 108.5 MB/s | 104.6 MB/s | 101.9 MB/s | 70.3 MB/s |
| decode | 126.5 MB/s | 124.5 MB/s | 123.2 MB/s | 111.5 MB/s |

`scalar_at` is not benched: tANS decode is sequential by construction
(each call reconstructs the full stream from end to start), so
per-element random access is not meaningful at this layer. Batched
random access is a P6 concern.

## Observations

1. **Compression matches Shannon entropy closely.** On the Zipf-skewed
   scenario the observed ratio (2.35x) is within ~0.5% of the
   information-theoretic limit (~2.36x). On uniform random it lands
   exactly on the 2.00x ceiling. The quantized symbol-weight table at
   `size_log = 12` (table size 4096) is fine-grained enough that
   quantization loss is negligible at this alphabet size. This is the
   expected sign of a healthy tANS implementation.

2. **Decode is the fast direction, ~25-30% faster than encode.** This
   matches the algorithmic asymmetry: encode walks the symbol stream in
   reverse and pushes bits into a backward bit stream (an extra
   reversal step at the end), while decode is a tight forward loop of
   "consume `renorm_bits`, table-lookup, emit symbol".

3. **Decode throughput is markedly lower than upstream pco-stack
   layers.** Decode lands at ~100-125 MB/s, against ~3-8 GB/s for mode
   arrays and ~800 MB/s for BinPartition. The gap is structural rather
   than implementation-quality: this implementation is **single-state**
   (no four-way interleave) and is bit-serial — each output symbol
   carries a hard data dependency on the previous bit-shift and
   table-lookup. The single-state choice is intentional for P5;
   four-way SIMD interleave is documented as a P6 optimization. Note
   that the ratio (~100 MB/s of decoded output, but only ~40-50 MB/s
   of encoded *input*) means the bit-serial loop runs at ~50 MB/s of
   bit-stream throughput, comparable to other scalar bit-readers.

4. **Uniform is faster than Zipf in both directions.** Counter-
   intuitive, but consistent: the Zipf distribution dwells more in
   low-probability low-state-bit regions and triggers more
   renormalization steps per symbol on average. Uniform yields a
   stationary `renorm_bits` of exactly 4 per symbol (alphabet = 16,
   table size 4096), so the inner loop has fewer branches.

5. **Strategy: from-scratch reimplementation.** The tans module's
   doc-comment states the choice explicitly: pco's `ans` module is
   declared `mod ans;` (private) inside the pco crate, so it cannot
   be called downstream. The implementation here mirrors pco's
   algorithm (table layout, weight quantization, renormalization
   cutoff) and is therefore structurally compatible with pco's tANS
   bit stream, but does not re-emit pco's framing or depend on pco's
   private module. The single-state choice (vs. pco's four-way
   interleave) is purely a throughput optimization — compression
   ratio is identical.
