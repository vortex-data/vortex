# string-compress-bench

Compare string-compression algorithms on the synthetic data that ships with this
crate. Every backend exposes the same trait so the bench harness and the
report binary stay in lock-step.

## Backends

| name            | source                                              | pushdown                 |
| --------------- | --------------------------------------------------- | ------------------------ |
| `fsst-rs`       | [`fsst-rs`](https://crates.io/crates/fsst-rs)       | equality on compressed bytes |
| `fsst-cpp-8`    | vendored `cwida/fsst` (8-bit codes)                 | equality on compressed bytes |
| `fsst-cpp-12`   | vendored `cwida/fsst` (12-bit codes)                | equality on compressed bytes |
| `onpair`        | [`gargiulofrancesco/onpair_rs`](https://github.com/gargiulofrancesco/onpair_rs) | decompress + memcmp (no dictionary export) |
| `onpair16`      | same crate, 16-byte-token variant                   | decompress + memcmp                         |
| `onpair-cpp`    | vendored [`gargiulofrancesco/onpair_cpp`](https://github.com/gargiulofrancesco/onpair_cpp) | equality, substring (KMP), prefix — all on compressed tokens |

FSST equality is a true compressed-domain fast path: compress the needle once
through the same symbol table and `memcmp` the per-row codes. The Rust OnPair
port does not expose its LPM dictionary, so we fall back to per-row decompress
+ `memcmp`. The vendored C++ implementation under `vendor/onpair_cpp/` ships
KMP / EqAutomaton / PrefixAutomaton scanners, all of which run on the packed
token stream — this is what makes `onpair-cpp` the only backend that does
compressed-domain `LIKE '%needle%'` evaluation.

Building `onpair-cpp` requires Boost headers on the system include path
(`libboost-dev` on Debian/Ubuntu); the upstream library uses
`boost::unordered_flat_map` for the LPM hash table.

## Layout

```text
benchmarks/string-compress-bench/
├── Cargo.toml
├── build.rs                 # compiles vendored C++ FSST behind `fsst-cpp` feature
├── cpp/                     # renamed wrappers around the C++ libfsst + onpair sources
├── vendor/fsst_cpp/         # upstream cwida/fsst snapshot (MIT)
├── vendor/onpair_cpp/       # upstream gargiulofrancesco/onpair_cpp snapshot (MIT)
├── benches/string_compress.rs
└── src/
    ├── lib.rs
    ├── main.rs              # `string-compress-report` binary
    ├── datasets.rs          # synthetic corpora
    ├── harness.rs           # `run_backend` dispatch + timing
    └── backends/
        ├── mod.rs
        ├── fsst_rs_backend.rs
        ├── fsst_cpp_ffi.rs
        ├── fsst_cpp_8.rs
        ├── fsst_cpp_12.rs
        ├── onpair_backend.rs
        ├── onpair16_backend.rs
        ├── onpair_cpp_ffi.rs
        └── onpair_cpp_backend.rs
```

## Running

The Vortex workspace ships a dependency tree that requires AES + SSE2
intrinsics on x86-64 (transitively via `gxhash`). Most of the toolchain
already has this enabled, but a clean Cargo invocation needs the flag:

```bash
export RUSTFLAGS="-C target-feature=+aes,+sse2"

# Tabular report across all built-in backends + datasets
cargo run --release -p string-compress-bench --bin string-compress-report

# Pin to one backend / one dataset
cargo run --release -p string-compress-bench --bin string-compress-report -- \
    --only-backend fsst-cpp-12 --only-dataset urls --rows 8192

# Divan benches
cargo bench -p string-compress-bench
```

## Features

| feature       | default | effect                                                                          |
| ------------- | ------- | ------------------------------------------------------------------------------- |
| `fsst-cpp`    | on      | Compiles `vendor/fsst_cpp/` into the binary and enables `fsst-cpp-{8,12}`       |
| `onpair`      | on      | Pulls `onpair_rs` (git dep) and enables `onpair` + `onpair16`                    |
| `onpair-cpp`  | on      | Compiles `vendor/onpair_cpp/` (C++20 + Boost) and enables `onpair-cpp`           |

Turn off `fsst-cpp` if the host toolchain can not build C++17 with
`-fpermissive` (the FSST-12 source has a `unsigned long*` / `unsigned long long*`
type mismatch that needs the warning relaxed; the underlying byte sizes are
identical on every supported platform).

## Synthetic datasets

| name             | shape                                                              |
| ---------------- | ------------------------------------------------------------------ |
| `skewed_dict`    | 32-word vocab, Zipf-ish row mixture                                |
| `urls`           | `https://host/path?key=value` with a few hosts/paths               |
| `random_alnum`   | high-entropy alphanumeric bytes (worst case)                       |
| `long_prefix`    | every row shares a 60-byte prefix, then drifts                     |
| `natural_words`  | bag-of-words English-ish                                           |
| `json_like`      | small JSON snippets (punctuation-heavy)                             |
| `short_codes`    | `US-12345`-shaped fixed-length identifiers                          |
| `fsst12_high_card`| 512-token vocabulary of 3-5-byte enums, joined with `\|`. FSST-12 sweet spot — FSST-8's 255-symbol table is not big enough to hold the whole vocabulary |
| `log_templates`  | ~250-byte structured log lines with two short variable fields. OnPair sweet spot — its uncapped token can swallow the whole template; FSST symbols cap at 8 B and OnPair16 caps at 16 B |
| `adversarial_mix`| four interleaved anti-patterns (high-entropy session IDs, 9-byte periodic motifs, hex blobs, random ASCII) crafted so no backend can converge on a useful dictionary |

### Which backend is each dataset built for?

| dataset           | sweet spot for       | why                                                                    |
| ----------------- | -------------------- | ---------------------------------------------------------------------- |
| `skewed_dict`     | `fsst-cpp-8`         | small high-frequency vocab → tight 1-byte codes                        |
| `urls`            | `fsst-cpp-8`         | many recurring 2-8 byte fragments (`https://`, `.com`, `/v1/`)         |
| `natural_words`   | `fsst-cpp-8`         | classic FSST text workload                                             |
| `json_like`       | `fsst-cpp-8`         | repeating quoted keys + small value enums                              |
| `long_prefix`     | `fsst-cpp-8`         | 8-byte chunks of the shared prefix tile cleanly                        |
| `short_codes`     | `fsst-cpp-12`        | very low average length → table overhead dominates 8-bit codes         |
| `random_alnum`    | `fsst-cpp-12`        | 64-char alphabet exceeds FSST-8's effective coverage                   |
| `fsst12_high_card`| **`fsst-cpp-12`**    | 512 distinct enum values → only FSST-12's 4096-symbol table fits them  |
| `log_templates`   | **`onpair`**         | 250-byte shared template → uncapped OnPair token, vs 30+ FSST codes    |
| `adversarial_mix` | nobody — see below   | designed to defeat every algorithm                                     |

All seeds are pinned so the report is reproducible across runs.

## Knobs

The report binary exposes every meaningful tuning knob each algorithm has.
FSST is fully data-driven (no public knobs besides the training corpus); the
OnPair family has a few:

| flag                          | applies to       | default | description                                            |
| ----------------------------- | ---------------- | ------- | ------------------------------------------------------ |
| `--onpair-threshold`          | `onpair`, `onpair16` | 4   | Pair-merge frequency threshold (`OnPair::new`).        |
| `--onpair-cpp-bits`           | `onpair-cpp`     | 14      | Code width; `2^bits` dictionary slots, `bits/8` B per token. |
| `--onpair-cpp-seed`           | `onpair-cpp`     | 42      | Training-shuffle RNG seed.                             |
| `--onpair-cpp-fixed-threshold`| `onpair-cpp`     | 0       | Reserved for fixed-threshold mode; 0 = dynamic.        |
| `--rows`                      | corpus generator | 4096    | Rows per synthetic dataset.                            |
| `--iters`                     | timing           | 3       | Best-of-N iterations per phase.                        |

## Sample report

`rows=4096`, `iters=3`, `onpair_threshold=4`, `onpair_cpp_bits=14`,
`onpair_cpp_seed=42`, x86-64 with `RUSTFLAGS="-C target-feature=+aes,+sse2"`.

Sizes are in bytes; `ratio` = uncompressed / compressed-payload. `eq`,
`contains`, and `starts_with` are best-of-3 timings for the full corpus in
milliseconds. `PD` means the predicate ran in the compressed domain; `--`
means the backend decompressed each row first.

### `skewed_dict` — 51 397 B raw, 4 096 rows

| backend        | payload | ratio  |  compress | decompress | eq (PD?)      | contains (PD?)  | starts_with    |
| -------------- | ------: | -----: | --------: | ---------: | ------------: | --------------: | -------------: |
| `fsst-rs`      |  16 705 | 3.08×  |   1.19 ms |    0.13 ms |  0.008 ms PD  |   0.226 ms      |  0.143 ms      |
| `fsst-cpp-8`   |  16 760 | 3.07×  |   2.72 ms |    0.13 ms |  0.007 ms PD  |   0.237 ms      |  0.176 ms      |
| `fsst-cpp-12`  |  24 525 | 2.10×  |  27.40 ms |    0.22 ms |  0.006 ms PD  |   0.353 ms      |  0.280 ms      |
| `onpair`       |  25 383 | 2.02×  |   1.12 ms |    0.14 ms |  0.057 ms     |   0.142 ms      |  0.078 ms      |
| `onpair16`     |  25 431 | 2.02×  |   1.21 ms |    0.18 ms |  0.049 ms     |   0.148 ms      |  0.085 ms      |
| `onpair-cpp`   |  39 725 | 1.29×  |   0.71 ms |    0.15 ms |  0.014 ms PD  |   0.058 ms PD   |  0.030 ms      |

### `urls` — 179 571 B raw, 4 096 rows

| backend        | payload | ratio  |  compress | decompress | eq (PD?)      | contains (PD?)  | starts_with    |
| -------------- | ------: | -----: | --------: | ---------: | ------------: | --------------: | -------------: |
| `fsst-rs`      |  60 159 | 2.98×  |   1.48 ms |    0.22 ms |  0.001 ms PD  |   0.341 ms      |  0.184 ms      |
| `fsst-cpp-8`   |  51 221 | 3.51×  |   3.16 ms |    0.18 ms |  0.001 ms PD  |   0.462 ms      |  0.281 ms      |
| `fsst-cpp-12`  |  60 707 | 2.96×  |  47.58 ms |    0.29 ms |  0.002 ms PD  |   0.553 ms      |  0.396 ms      |
| `onpair`       |  61 835 | 2.90×  |   3.50 ms |    0.19 ms |  0.082 ms     |   0.257 ms      |  0.129 ms      |
| `onpair16`     |  74 390 | 2.41×  |   2.91 ms |    0.17 ms |  0.073 ms     |   0.296 ms      |  0.139 ms      |
| `onpair-cpp`   |  85 356 | 2.10×  |   1.99 ms |    0.14 ms |  0.004 ms PD  |   0.065 ms PD   |  0.024 ms      |

### `random_alnum` — 138 542 B raw, 4 096 rows (worst case)

| backend        | payload | ratio  |  compress | decompress | eq (PD?)      | contains (PD?)  | starts_with    |
| -------------- | ------: | -----: | --------: | ---------: | ------------: | --------------: | -------------: |
| `fsst-rs`      | 132 120 | 1.05×  |   2.19 ms |    0.33 ms |  0.001 ms PD  |   0.476 ms      |  0.206 ms      |
| `fsst-cpp-8`   | 131 861 | 1.05×  |   5.35 ms |    0.28 ms |  0.001 ms PD  |   0.651 ms      |  0.385 ms      |
| `fsst-cpp-12`  | 108 729 | 1.27×  | 183.68 ms |    0.29 ms |  0.001 ms PD  |   0.671 ms      |  0.412 ms      |
| `onpair`       | 169 592 | 0.82×  |   7.70 ms |    0.24 ms |  0.095 ms     |   0.332 ms      |  0.102 ms      |
| `onpair16`     | 169 614 | 0.82×  |   7.73 ms |    0.23 ms |  0.088 ms     |   0.359 ms      |  0.094 ms      |
| `onpair-cpp`   | 184 273 | 0.75×  |   6.38 ms |    0.21 ms |  0.004 ms PD  |   0.152 ms PD   |  0.023 ms      |

### `long_prefix` — 286 751 B raw, 4 096 rows

| backend        | payload | ratio  |  compress | decompress | eq (PD?)      | contains (PD?)  | starts_with    |
| -------------- | ------: | -----: | --------: | ---------: | ------------: | --------------: | -------------: |
| `fsst-rs`      |  68 856 | 4.16×  |   1.21 ms |    0.24 ms |  0.001 ms PD  |   0.166 ms      |  0.153 ms      |
| `fsst-cpp-8`   |  55 300 | 5.19×  |   2.04 ms |    0.18 ms |  0.002 ms PD  |   0.285 ms      |  0.284 ms      |
| `fsst-cpp-12`  |  72 920 | 3.93×  |  31.17 ms |    0.35 ms |  0.003 ms PD  |   0.365 ms      |  0.362 ms      |
| `onpair`       |  71 196 | 4.03×  |   3.83 ms |    0.15 ms |  0.056 ms     |   0.076 ms      |  0.075 ms      |
| `onpair16`     |  80 605 | 3.56×  |   2.76 ms |    0.17 ms |  0.064 ms     |   0.092 ms      |  0.081 ms      |
| `onpair-cpp`   |  92 254 | 3.11×  |   1.82 ms |    0.16 ms |  0.004 ms PD  |   0.120 ms PD   |  0.029 ms      |

### `natural_words` — 178 423 B raw, 4 096 rows

| backend        | payload | ratio  |  compress | decompress | eq (PD?)      | contains (PD?)  | starts_with    |
| -------------- | ------: | -----: | --------: | ---------: | ------------: | --------------: | -------------: |
| `fsst-rs`      |  40 166 | 4.44×  |   1.30 ms |    0.20 ms |  0.005 ms PD  |   0.444 ms      |  0.156 ms      |
| `fsst-cpp-8`   |  35 926 | 4.97×  |   2.80 ms |    0.19 ms |  0.005 ms PD  |   0.525 ms      |  0.235 ms      |
| `fsst-cpp-12`  |  54 157 | 3.29×  |  24.46 ms |    0.29 ms |  0.004 ms PD  |   0.637 ms      |  0.341 ms      |
| `onpair`       |  60 072 | 2.97×  |   3.69 ms |    0.21 ms |  0.081 ms     |   0.340 ms      |  0.093 ms      |
| `onpair16`     |  60 099 | 2.97×  |   2.35 ms |    0.18 ms |  0.059 ms     |   0.345 ms      |  0.069 ms      |
| `onpair-cpp`   |  65 100 | 2.74×  |   1.34 ms |    0.16 ms |  0.009 ms PD  |   0.059 ms PD   |  0.009 ms      |

### `json_like` — 183 994 B raw, 4 096 rows

| backend        | payload | ratio  |  compress | decompress | eq (PD?)      | contains (PD?)  | starts_with    |
| -------------- | ------: | -----: | --------: | ---------: | ------------: | --------------: | -------------: |
| `fsst-rs`      |  49 467 | 3.72×  |   1.35 ms |    0.17 ms |  0.001 ms PD  |   0.386 ms      |  0.125 ms      |
| `fsst-cpp-8`   |  46 763 | 3.93×  |   2.80 ms |    0.20 ms |  0.001 ms PD  |   0.459 ms      |  0.239 ms      |
| `fsst-cpp-12`  |  54 865 | 3.35×  |  39.19 ms |    0.24 ms |  0.005 ms PD  |   0.578 ms      |  0.320 ms      |
| `onpair`       |  71 420 | 2.58×  |   3.96 ms |    0.18 ms |  0.077 ms     |   0.316 ms      |  0.087 ms      |
| `onpair16`     |  69 655 | 2.64×  |   3.40 ms |    0.14 ms |  0.048 ms     |   0.304 ms      |  0.063 ms      |
| `onpair-cpp`   |  78 983 | 2.33×  |   1.91 ms |    0.14 ms |  0.008 ms PD  |   0.069 ms PD   |  0.008 ms      |

### `short_codes` — 32 768 B raw, 4 096 rows

| backend        | payload | ratio  |  compress | decompress | eq (PD?)      | contains (PD?)  | starts_with    |
| -------------- | ------: | -----: | --------: | ---------: | ------------: | --------------: | -------------: |
| `fsst-rs`      |  15 308 | 2.14×  |   1.25 ms |    0.16 ms |  0.001 ms PD  |   0.144 ms      |  0.108 ms      |
| `fsst-cpp-8`   |  15 249 | 2.15×  |   3.76 ms |    0.14 ms |  0.001 ms PD  |   0.221 ms      |  0.199 ms      |
| `fsst-cpp-12`  |  14 519 | 2.26×  |  34.22 ms |    0.20 ms |  0.010 ms PD  |   0.268 ms      |  0.243 ms      |
| `onpair`       |  30 714 | 1.07×  |   0.86 ms |    0.12 ms |  0.039 ms     |   0.091 ms      |  0.060 ms      |
| `onpair16`     |  30 767 | 1.07×  |   0.88 ms |    0.12 ms |  0.030 ms     |   0.091 ms      |  0.059 ms      |
| `onpair-cpp`   |  44 329 | 0.74×  |   0.73 ms |    0.14 ms |  0.004 ms PD  |   0.034 ms PD   |  0.011 ms      |

### `fsst12_high_card` — 149 054 B raw, 4 096 rows (FSST-12 sweet spot)

| backend        | payload | ratio  |  compress | decompress | eq (PD?)      | contains (PD?)  | starts_with    |
| -------------- | ------: | -----: | --------: | ---------: | ------------: | --------------: | -------------: |
| `fsst-rs`      |  90 427 | 1.65×  |   2.00 ms |    0.27 ms |  0.001 ms PD  |   0.404 ms      |  0.171 ms      |
| `fsst-cpp-8`   |  84 147 | 1.77×  |   4.85 ms |    0.22 ms |  0.001 ms PD  |   0.538 ms      |  0.303 ms      |
| **`fsst-cpp-12`** | **61 366** | **2.43×** | 72.17 ms | 0.24 ms |  0.001 ms PD  |   0.558 ms      |  0.340 ms      |
| `onpair`       |  76 638 | 1.94×  |   2.54 ms |    0.19 ms |  0.061 ms     |   0.312 ms      |  0.069 ms      |
| `onpair16`     |  77 013 | 1.94×  |   2.52 ms |    0.18 ms |  0.062 ms     |   0.280 ms      |  0.069 ms      |
| `onpair-cpp`   |  90 960 | 1.64×  |   2.05 ms |    0.15 ms |  0.004 ms PD  |   0.077 ms PD   |  0.020 ms      |

FSST-12 beats FSST-8 by ~37 % here (2.43× vs 1.77×) because the 512-entry
enum vocabulary overflows FSST-8's 255-symbol table — half the values get
demoted to byte-level codes. FSST-12 has room for all of them. The
compression-time cost is steep (≈15× slower than FSST-8) so the win only
pays off when you're size-bound, not throughput-bound.

### `log_templates` — 1 146 806 B raw, 4 096 rows (OnPair sweet spot)

| backend        | payload | ratio  |  compress | decompress | eq (PD?)      | contains (PD?)  | starts_with    |
| -------------- | ------: | -----: | --------: | ---------: | ------------: | --------------: | -------------: |
| `fsst-rs`      | 309 940 | 3.70×  |   2.63 ms |    0.54 ms |  0.001 ms PD  |   1.365 ms      |  0.243 ms      |
| `fsst-cpp-8`   | 243 594 | 4.71×  |   3.43 ms |    0.32 ms |  0.001 ms PD  |   1.477 ms      |  0.377 ms      |
| `fsst-cpp-12`  | 268 049 | 4.28×  |  22.65 ms |    0.60 ms |  0.001 ms PD  |   1.515 ms      |  0.396 ms      |
| **`onpair`**   | **171 176** | **6.70×** | 12.80 ms | 0.22 ms |  0.065 ms    |   1.375 ms      |  0.071 ms      |
| `onpair16`     | 275 533 | 4.16×  |   6.31 ms |    0.28 ms |  0.110 ms     |   1.228 ms      |  0.113 ms      |
| `onpair-cpp`   | 241 128 | 4.76×  |   3.89 ms |    0.28 ms |  0.004 ms PD  |   0.081 ms PD   |  0.009 ms      |

The 250-byte template fits into a single OnPair dictionary entry, so every
log line costs ~2 bytes for the template plus a handful for the two
variable fields. FSST has to chain ≈30 8-byte symbols to cover the same
template; OnPair16 needs ≈16 of its capped 16-byte tokens. `onpair-cpp` is
tuned for fast pushdown rather than ratio — its 14-bit code width and
dictionary layout cost some ratio relative to the no-cap Rust port but
still beat both FSST variants. (`onpair-cpp` keeps its huge pushdown lead
on `contains`: 0.08 ms vs 1.4 ms.)

### `adversarial_mix` — 126 085 B raw, 4 096 rows (designed to defeat every backend)

| backend        | payload | ratio  |  compress | decompress | eq (PD?)      | contains (PD?)  | starts_with    |
| -------------- | ------: | -----: | --------: | ---------: | ------------: | --------------: | -------------: |
| `fsst-rs`      | 112 645 | 1.12×  |   2.13 ms |    0.26 ms |  0.001 ms PD  |   0.347 ms      |  0.158 ms      |
| `fsst-cpp-8`   | 108 990 | 1.16×  |   5.23 ms |    0.27 ms |  0.001 ms PD  |   0.544 ms      |  0.315 ms      |
| `fsst-cpp-12`  |  93 717 | 1.35×  | 132.02 ms |    0.27 ms |  0.001 ms PD  |   0.564 ms      |  0.346 ms      |
| `onpair`       | 157 925 | 0.80×  |   7.14 ms |    0.22 ms |  0.077 ms     |   0.305 ms      |  0.085 ms      |
| `onpair16`     | 158 118 | 0.80×  |   7.20 ms |    0.21 ms |  0.074 ms     |   0.272 ms      |  0.081 ms      |
| `onpair-cpp`   | 178 870 | 0.70×  |   5.50 ms |    0.19 ms |  0.004 ms PD  |   0.165 ms PD   |  0.009 ms      |

Even the best FSST variant (`fsst-cpp-12`) only reaches 1.35×, and every
OnPair backend produces output *larger than the input* (every 16-bit token
costs 2 bytes for a payload of mostly-unique 1-byte symbols, plus the
dictionary header). The point of this dataset is to show that none of the
algorithms is magic — when the input is structured to defeat them, the
ratio collapses regardless of which backend you pick.

### Reading the table

- **Compression ratio**: `fsst-cpp-8` is the consistent winner on
  natural-text-shaped corpora at this scale. `fsst-cpp-12` only catches up
  on `random_alnum` where its larger alphabet pays off.
- **Compression speed**: `fsst-cpp-12` is the slowest by a wide margin
  (training does many more passes). `onpair-cpp` is roughly 2× faster than
  the Rust ports thanks to its tighter Boost-flat-map LPM.
- **Decompression speed**: roughly comparable across the board; FSST has a
  very tight inner loop and tiny advantage on long-codeword corpora.
- **Pushdown**: only `fsst-*` and `onpair-cpp` evaluate equality in the
  compressed domain. The Rust OnPair port pays the full decompress cost on
  every predicate. For `LIKE '%needle%'`, only `onpair-cpp` skips
  decompression entirely; FSST falls back to a per-row decode (the upstream
  `vortex-fsst` LIKE pushdown has a more specialised DFA for the common
  cases, which this bench does not exercise).
