# NeaTS reference (C++) vs our Rust implementation

This directory contains the scripts and captured results from a head-to-head comparison
between the reference C++ NeaTS implementation (https://github.com/and-gue/NeaTS, ICDE 2025)
and our Rust port (`encodings/neats`).

## Setup

The reference is cloned and built separately (not vendored — it's GPLv3, we don't want it in
the Vortex tree):

```bash
git clone --depth 1 https://github.com/and-gue/NeaTS.git /tmp/neats-reference
cd /tmp/neats-reference && mkdir -p build && cd build
cmake .. && make DecompressorSIMD
```

Requirements: GCC ≥ 13, CMake ≥ 3.22. The reference compiles with `-march=native -mavx512f`
(the C++ uses `std::experimental::simd` and benefits significantly from AVX2/AVX-512).

## Running

```bash
# Datasets in benchmarks/real-data/ are loaded by our script and quantized to i64 before
# being handed to the reference (it expects raw binary i64 input).
python3 paper_vs_ours.py | tee results.txt
```

For each CSV column ≥ 1000 rows we:

1. Pick a quantization scale `10**k` based on the column's min non-zero abs value (so values
   like 0.785500 quantize to 785500 with k=6, then to i64).
2. Write the i64 array as raw binary and run `DecompressorSIMD <bin> <bpc>` where `bpc` is
   chosen as `ceil(log2(range)) - 4`, clamped to [8, 32]. This is the "max bits per
   correction" knob — the reference treats it as the residual budget.
3. Run our `neats-table` (in `vortex-bench/src/bin/neats_table.rs`) on the same re-quantized
   data via a single-column CSV.
4. Parse and print compressed bits side by side.

## Results summary

See `results.txt` for the full table. Headline numbers (lossless ratio = raw_bits / compressed_bits):

### Where the C++ reference wins (typically: noisy continuous signals)

| column | paper | ours (PCO) | paper win |
|---|---:|---:|---:|
| beijing-pollution/pm2.5 | 7.01× | 4.84× | 1.45× |
| beijing-pollution/TEMP | 12.09× | 4.50× | 2.69× |
| beijing-pollution/DEWP | 12.49× | 3.42× | 3.65× |
| co2-ppm-daily/value | 14.67× | 7.21× | 2.03× |
| exchange-rates-daily/Exchange rate | 18.41× | 7.21× | 2.55× |
| nab-ambient-temperature/value | 10.83× | 8.27× | 1.31× |
| btc-coinmetrics/PriceUSD | 3.19× | 1.49× | 2.14× |
| solar_AL_8cols/station_0 | 4.75× | 2.90× | 1.64× |
| global-temp-monthly/Mean | 2.10× | 1.26× | 1.67× |

### Where our Rust impl wins (typically: sparse / discrete / highly repetitive columns)

| column | paper | ours (PCO) | ours win |
|---|---:|---:|---:|
| beijing-pollution/No | 194× | 2164× | 11.15× |
| beijing-pollution/day | 25.48× | 172.20× | 6.76× |
| beijing-pollution/hour | 12.01× | 78.78× | 6.56× |
| exchange_rate_multivariate/0.211242 | 30.05× | 485.57× | 16.16× |
| exchange_rate_multivariate/0.006838 | 16.59× | 120.67× | 7.27× |
| btc-coinmetrics/PriceBTC | 28.10× | 1078.98× | 38.41× |
| sp500/PE10 | 5.38× | 20.64× | 3.84× |
| sp500/Consumer Price Index | 5.64× | 32.16× | 5.70× |
| vix-daily/LOW | 10.36× | 17.00× | 1.64× |

## Why the mixed picture

The two implementations are optimised differently:

- **Reference paper** is tight on **noisy continuous time-series** because its per-piece
  bit-pack stores residuals at exactly the piece's required width with no entropy-coding
  overhead — every bit is meaningful data. Its partitioner is DP-style (slow but optimal),
  so pieces are tightly sized.
- **Our Rust impl (PCO mode)** wins on **sparse / repetitive / heavily quantised columns**
  because PCO does FSE entropy coding on top of bit-pack, so all-zero residuals or
  small-alphabet residuals compress essentially for free. PCO also handles constant arrays
  exceptionally — our 47-byte output for `PriceBTC` is unbeatable on a column where most
  values are duplicates.

Neither is uniformly better. A future hybrid would do per-piece bit-pack (paper style) **with
FSE on top of the packed residuals** — that combination doesn't exist in either today.

## Speed (where the reference is publicly worse)

The reference's compress time is reported as 0.0–0.2 MB/s in `results.txt` — its
DP-style partitioning at the paper's quality setting is much slower than our bisection.
Decompress is 100–160 MB/s. Our compress runs at ~200-400 MB/s and decompress at
200-1500 MB/s depending on shape. We give up some ratio on noisy data but earn it back in
compress throughput by orders of magnitude.

## License notes

The reference NeaTS is **GPLv3**. We deliberately don't vendor it. The `run_compare.py` and
`paper_vs_ours.py` scripts in this directory are part of the Vortex repo (Apache-2.0); they
invoke the reference as a subprocess and parse its stdout — no source mixing.
