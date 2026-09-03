# Medium-row tail-load experiment

Recorded 2026-08-31 on CPU 2 with one fresh process at a time. OnPair is
single-threaded. Both binaries used `RUSTFLAGS='-C target-cpu=native'`; each
process used two warmups and seven timed iterations, reporting the internal
median. The baseline SHA-256 was
`f8d5ab76db0ef90224bac76e7ff1e956d498bfc0c76ddf3bd65ee530ef5a4bd2`; the
final candidate SHA-256 was
`6c9ea38a1dac68b9c6adac8d14ca4b6ccb240b03310bdc60341460b90147c6af`.
Every result reported `roundtrip_correct=true` and `packed_correct=true`, with
the same dictionary-token and code counts between variants.

The command shape was:

```bash
ONPAIR_WARMUPS=2 ONPAIR_ITERATIONS=7 taskset -c 2 \
  encode_bench CORPUS.onpair
```

## Balanced ClickBench repeats

AB/BA process order alternated on every pair. Times are milliseconds.

| Pair | Parse baseline | Parse gated | Full native baseline | Full native gated | Full fair baseline | Full fair gated |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 93.566850 | 92.370877 | 122.724188 | 121.455097 | 127.018777 | 125.159905 |
| 2 | 93.141029 | 92.387224 | 122.348889 | 121.484192 | 125.220037 | 125.850991 |
| 3 | 93.105387 | 91.832976 | 122.774452 | 121.071110 | 126.426804 | 125.227940 |
| 4 | 93.408150 | 92.052596 | 122.263562 | 121.476224 | 126.210662 | 125.147565 |
| 5 | 93.371806 | 92.314036 | 122.639421 | 121.427971 | 126.185652 | 125.310372 |
| 6 | 93.411198 | 91.988331 | 122.760356 | 121.523228 | 126.296140 | 125.572889 |
| 7 | 93.247509 | 92.492769 | 122.808083 | 121.654928 | 126.867994 | 125.053208 |
| 8 | 93.142913 | 91.972642 | 122.002890 | 121.474552 | 126.090226 | 124.791019 |
| **Median** | **93.309658** | **92.183316** | **122.681805** | **121.475388** | **126.253401** | **125.193923** |
| **Paired-median speedup** |  | **1.0128x (8/8 wins)** |  | **1.0097x (8/8 wins)** |  | **1.0090x (7/8 wins)** |

## Final 13-corpus screen

Dataset order alternated baseline-first and candidate-first. This is a breadth
screen with one fresh process per cell, not a replacement for the balanced
ClickBench repeats.

| Dataset | Parse ms baseline -> gated | Parse speedup | Full-native ms baseline -> gated | Full-native speedup | Full-fair speedup |
|---|---:|---:|---:|---:|---:|
| Amazon book titles | 245.793 -> 244.656 | 1.0046x | 305.068 -> 303.350 | 1.0057x | 0.9971x |
| Apache access | 100.535 -> 101.791 | 0.9877x | 124.504 -> 124.032 | 1.0038x | 0.9931x |
| ClickBench | 93.361 -> 91.512 | 1.0202x | 122.676 -> 120.133 | 1.0212x | 1.0157x |
| DBpedia abstracts | 190.657 -> 190.623 | 1.0002x | 235.471 -> 235.732 | 0.9989x | 1.0040x |
| FineWeb | 204.796 -> 208.088 | 0.9842x | 255.876 -> 257.489 | 0.9937x | 0.9849x |
| MS MARCO queries | 228.180 -> 223.106 | 1.0227x | 282.101 -> 278.723 | 1.0121x | 1.0045x |
| MS MARCO URLs | 222.389 -> 219.603 | 1.0127x | 282.702 -> 279.689 | 1.0108x | 1.0106x |
| OnPair titles | 128.454 -> 127.141 | 1.0103x | 155.158 -> 157.450 | 0.9854x | 0.9955x |
| Paper book reviews | 192.797 -> 194.591 | 0.9908x | 240.580 -> 243.208 | 0.9892x | 0.9958x |
| Paper news | 223.588 -> 219.843 | 1.0170x | 273.449 -> 273.254 | 1.0007x | 1.0002x |
| Paper tweets | 227.409 -> 227.180 | 1.0010x | 289.732 -> 288.577 | 1.0040x | 1.0047x |
| Stack v3 | 171.166 -> 171.635 | 0.9973x | 206.737 -> 207.936 | 0.9942x | 0.9949x |
| TPC-H comment | 141.052 -> 139.789 | 1.0090x | 173.437 -> 173.002 | 1.0025x | 1.0029x |
| **Median speedup** |  | **1.0046x** |  | **1.0025x** | **1.0002x** |

The fast path is selected from the actual parse input and is enabled only for
average row lengths in `(32, 256]` bytes. The final FineWeb, book-reviews,
Stack, and TPC-H cells therefore use the original loader; their small observed
differences reflect binary layout and run noise rather than the tail-load path.
