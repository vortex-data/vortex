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

## Real-world datasets

Five small, license-clean corpora are vendored under `data/`. Each loader
reads the file at runtime; missing files are silently skipped, so a slim
checkout falls back to synthetic-only.

| name                  | source                                                | license     | strings | what it stresses                                          |
| --------------------- | ----------------------------------------------------- | ----------- | ------: | --------------------------------------------------------- |
| `pride_and_prejudice` | Project Gutenberg #1342, Jane Austen                  | Public dom. |   7 776 | Natural English prose, FSST-friendly short n-grams        |
| `english_words`       | dwyl/english-words `words_alpha.txt`                  | Unlicense   |  20 000 | High-cardinality short stems; FSST-12 sweet spot          |
| `gov_hostnames`       | cisagov/dotgov-data — US federal `.gov` domains       | CC0         |   6 695 | URL/hostname shape; recurring `.gov` + agency fragments   |
| `airport_records`     | datasets/airport-codes — pipe-delimited records       | ODC-PDDL    |   5 859 | Long records, repeated `iso_country` / `iso_region` tails |
| `world_cities`        | datasets/world-cities — `name, subcountry, country`   | CC-BY 3.0   |  15 707 | UTF-8 mixed scripts + recurring country-name suffixes     |

Attribution and exact source URLs live in `data/README.md`. The vendored
files are cleaned and truncated for size (≤500 KB each, ~1.8 MB total).

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
| `pride_and_prejudice` | `fsst-cpp-12`    | natural prose with broad short-stem vocabulary (real-world)            |
| `english_words`   | `fsst-cpp-12`        | 20 k distinct short words; FSST-8's 255-symbol cap leaves money on the table |
| `gov_hostnames`   | `fsst-cpp-12`        | many distinct agency-name fragments; shared `.gov` tail                |
| `airport_records` | `fsst-cpp-12`        | repeated pipe-delimited field values across 5 k records                |
| `world_cities`    | `fsst-cpp-12`        | UTF-8 mixed scripts + recurring country names                          |

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

### Real-world datasets

These five rows use the corpora vendored under `data/` (license + source
in `data/README.md`). FSST-12 wins ratio on every single one — real text
has the high-cardinality short-pattern shape its 4096-entry symbol table
is built for. `onpair-cpp` still owns the `contains` pushdown column.

#### `pride_and_prejudice` — 256 043 B raw, 4 096 rows of Austen prose

| backend        | payload | ratio  |  compress | decompress | eq (PD?)      | contains (PD?)  | starts_with    |
| -------------- | ------: | -----: | --------: | ---------: | ------------: | --------------: | -------------: |
| `fsst-rs`      | 129 826 | 1.97×  |   3.36 ms |    0.47 ms |  0.002 ms PD  |   0.756 ms      |  0.264 ms      |
| `fsst-cpp-8`   | 126 854 | 2.02×  |   7.38 ms |    0.37 ms |  0.002 ms PD  |   1.039 ms      |  0.549 ms      |
| **`fsst-cpp-12`** | **111 492** | **2.30×** | 139.19 ms | 0.43 ms | 0.003 ms PD  |   1.046 ms      |  0.547 ms      |
| `onpair`       | 151 220 | 1.69×  |   9.86 ms |    0.39 ms |  0.104 ms     |   0.630 ms      |  0.117 ms      |
| `onpair16`     | 150 786 | 1.70×  |   9.22 ms |    0.26 ms |  0.098 ms     |   0.577 ms      |  0.103 ms      |
| `onpair-cpp`   | 162 855 | 1.57×  |   6.41 ms |    0.28 ms |  0.008 ms PD  |   0.196 ms PD   |  0.019 ms      |

#### `english_words` — 37 752 B raw, 4 096 short words

| backend        | payload | ratio  |  compress | decompress | eq (PD?)      | contains (PD?)  | starts_with    |
| -------------- | ------: | -----: | --------: | ---------: | ------------: | --------------: | -------------: |
| `fsst-rs`      |  19 481 | 1.94×  |   2.21 ms |    0.21 ms |  0.002 ms PD  |   0.247 ms      |  0.164 ms      |
| `fsst-cpp-8`   |  19 313 | 1.95×  |   6.57 ms |    0.22 ms |  0.002 ms PD  |   0.343 ms      |  0.255 ms      |
| **`fsst-cpp-12`** | **15 833** | **2.38×** | 81.09 ms | 0.27 ms | 0.006 ms PD  |   0.408 ms      |  0.320 ms      |
| `onpair`       |  36 590 | 1.03×  |   1.87 ms |    0.23 ms |  0.058 ms     |   0.163 ms      |  0.072 ms      |
| `onpair16`     |  36 659 | 1.03×  |   1.88 ms |    0.21 ms |  0.054 ms     |   0.150 ms      |  0.063 ms      |
| `onpair-cpp`   |  51 557 | 0.73×  |   1.46 ms |    0.19 ms |  0.008 ms PD  |   0.095 ms PD   |  0.011 ms      |

#### `gov_hostnames` — 82 514 B raw, 4 096 hostnames

| backend        | payload | ratio  |  compress | decompress | eq (PD?)      | contains (PD?)  | starts_with    |
| -------------- | ------: | -----: | --------: | ---------: | ------------: | --------------: | -------------: |
| `fsst-rs`      |  27 816 | 2.97×  |   1.81 ms |    0.24 ms |  0.002 ms PD  |   0.356 ms      |  0.168 ms      |
| `fsst-cpp-8`   |  27 113 | 3.04×  |   5.15 ms |    0.23 ms |  0.002 ms PD  |   0.458 ms      |  0.275 ms      |
| **`fsst-cpp-12`** | **23 090** | **3.57×** | 66.35 ms | 0.35 ms | 0.004 ms PD  |   0.507 ms      |  0.327 ms      |
| `onpair`       |  44 079 | 1.87×  |   2.63 ms |    0.23 ms |  0.055 ms     |   0.270 ms      |  0.071 ms      |
| `onpair16`     |  44 352 | 1.86×  |   2.89 ms |    0.21 ms |  0.054 ms     |   0.255 ms      |  0.063 ms      |
| `onpair-cpp`   |  58 682 | 1.41×  |   2.11 ms |    0.24 ms |  0.008 ms PD  |   0.099 ms PD   |  0.014 ms      |

#### `airport_records` — 346 598 B raw, 4 096 pipe-delimited records

| backend        | payload | ratio  |  compress | decompress | eq (PD?)      | contains (PD?)  | starts_with    |
| -------------- | ------: | -----: | --------: | ---------: | ------------: | --------------: | -------------: |
| `fsst-rs`      | 167 846 | 2.06×  |   3.20 ms |    0.52 ms |  0.003 ms PD  |   0.713 ms      |  0.302 ms      |
| `fsst-cpp-8`   | 164 762 | 2.10×  |   6.88 ms |    0.47 ms |  0.002 ms PD  |   0.955 ms      |  0.572 ms      |
| **`fsst-cpp-12`** | **153 457** | **2.26×** | 138.29 ms | 0.50 ms | 0.003 ms PD  |   0.952 ms      |  0.552 ms      |
| `onpair`       | 191 335 | 1.81×  |  13.46 ms |    0.39 ms |  0.175 ms     |   0.583 ms      |  0.194 ms      |
| `onpair16`     | 191 514 | 1.81×  |  12.10 ms |    0.35 ms |  0.119 ms     |   0.499 ms      |  0.123 ms      |
| `onpair-cpp`   | 198 360 | 1.75×  |   7.98 ms |    0.33 ms |  0.007 ms PD  |   0.146 ms PD   |  0.011 ms      |

#### `world_cities` — 127 899 B raw, 4 096 city/country triples (mixed UTF-8)

| backend        | payload | ratio  |  compress | decompress | eq (PD?)      | contains (PD?)  | starts_with    |
| -------------- | ------: | -----: | --------: | ---------: | ------------: | --------------: | -------------: |
| `fsst-rs`      |  57 517 | 2.22×  |   2.15 ms |    0.28 ms |  0.002 ms PD  |   0.379 ms      |  0.212 ms      |
| `fsst-cpp-8`   |  53 795 | 2.38×  |   6.21 ms |    0.35 ms |  0.005 ms PD  |   0.572 ms      |  0.386 ms      |
| **`fsst-cpp-12`** | **42 276** | **3.03×** |  92.38 ms | 0.30 ms | 0.008 ms PD  |   0.619 ms      |  0.426 ms      |
| `onpair`       |  67 598 | 1.89×  |   5.05 ms |    0.26 ms |  0.100 ms     |   0.297 ms      |  0.116 ms      |
| `onpair16`     |  68 533 | 1.87×  |   4.32 ms |    0.25 ms |  0.069 ms     |   0.249 ms      |  0.074 ms      |
| `onpair-cpp`   |  83 115 | 1.54×  |   2.94 ms |    0.20 ms |  0.008 ms PD  |   0.140 ms PD   |  0.014 ms      |

The real-world result is unambiguous: when you measure against actual
human-readable strings instead of synthetic ones, **FSST-12 wins ratio in
every category** by 15-25 % over FSST-8 — its larger symbol table absorbs
the long tail of distinct short patterns that real text contains. The
trade-off is severe: training is 15-25× slower than FSST-8. For
write-once / read-many storage, that's an easy trade; for streaming
ingest, FSST-8 is still the practical pick.

### Summary chart — compression ratio per backend

Each `█` represents 0.5× ratio. `★` marks the winner for each dataset.
Higher is better.

```text
skewed_dict       (51 KB raw)  -- small high-frequency vocab, classic FSST shape
  fsst-rs       ██████▏      3.08×  ★
  fsst-cpp-8    ██████▏      3.07×
  fsst-cpp-12   ████▏        2.10×
  onpair        ████         2.02×
  onpair16      ████         2.02×
  onpair-cpp    ██▌          1.29×

urls              (180 KB raw)  -- recurring 2-8 byte fragments
  fsst-cpp-8    ███████      3.51×  ★
  fsst-rs       █████▉       2.98×
  fsst-cpp-12   █████▉       2.96×
  onpair        █████▉       2.90×
  onpair16      ████▉        2.41×
  onpair-cpp    ████▎        2.10×

random_alnum      (138 KB raw)  -- worst-case high-entropy bytes
  fsst-cpp-12   ██▌          1.27×  ★
  fsst-cpp-8    ██▏          1.05×
  fsst-rs       ██▏          1.05×
  onpair        █▋           0.82×
  onpair16      █▋           0.82×
  onpair-cpp    █▌           0.75×

long_prefix       (287 KB raw)  -- 60-byte shared prefix
  fsst-cpp-8    ██████████▍  5.19×  ★
  fsst-rs       ████████▎    4.16×
  onpair        ████         4.03×
  fsst-cpp-12   ███████▉     3.93×
  onpair16      ███████▏     3.56×
  onpair-cpp    ██████▎      3.11×

natural_words     (178 KB raw)  -- bag-of-words English-ish
  fsst-cpp-8    █████████▉   4.97×  ★
  fsst-rs       ████████▉    4.44×
  fsst-cpp-12   ██████▌      3.29×
  onpair        █████▉       2.97×
  onpair16      █████▉       2.97×
  onpair-cpp    █████▌       2.74×

json_like         (184 KB raw)  -- recurring quoted keys
  fsst-cpp-8    ███████▉     3.93×  ★
  fsst-rs       ███████▍     3.72×
  fsst-cpp-12   ██████▋      3.35×
  onpair16      █████▍       2.69×
  onpair        █████▎       2.62×
  onpair-cpp    ████▋        2.33×

short_codes       (33 KB raw)  -- 8-byte fixed-format identifiers
  fsst-cpp-12   ████▌        2.26×  ★
  fsst-cpp-8    ████▎        2.15×
  fsst-rs       ████▎        2.14×
  onpair        ██▏          1.08×
  onpair16      ██▏          1.06×
  onpair-cpp    █▌           0.74×

fsst12_high_card  (149 KB raw)  -- 512 distinct enum values
  fsst-cpp-12   ████▉        2.43×  ★  ← FSST-12 sweet spot (vs 1.77× for FSST-8)
  onpair        ███▉         1.94×
  onpair16      ███▉         1.94×
  fsst-cpp-8    ███▌         1.77×
  fsst-rs       ███▎         1.65×
  onpair-cpp    ███▎         1.64×

log_templates     (1147 KB raw)  -- 250-byte shared template
  onpair        █████████████▍  6.70×  ★  ← OnPair sweet spot
  onpair-cpp    █████████▌      4.76×
  fsst-cpp-8    █████████▍      4.71×
  fsst-cpp-12   ████████▌       4.28×
  onpair16      ████████▍       4.16×
  fsst-rs       ███████▍        3.70×

adversarial_mix   (126 KB raw)  -- designed to defeat every backend
  fsst-cpp-12   ██▊          1.35×  ← best of a bad bunch
  fsst-cpp-8    ██▍          1.16×
  fsst-rs       ██▎          1.12×
  onpair        █▋           0.80×  ← output LARGER than input
  onpair16      █▋           0.80×
  onpair-cpp    █▍           0.70×
```

### Summary chart — `LIKE '%needle%'` latency on `log_templates`

Each `█` represents 0.1 ms; full corpus 1.1 MB / 4 096 rows. Lower is
better. `★` marks the winner.

```text
  onpair-cpp    ▉                  0.08 ms  ★  compressed-domain KMP
  onpair16      ████████████▎      1.23 ms
  fsst-rs       █████████████▋     1.37 ms
  onpair        █████████████▊     1.38 ms
  fsst-cpp-8    ██████████████▊    1.48 ms
  fsst-cpp-12   ███████████████▏   1.52 ms
```

`onpair-cpp` is ~17× faster than the next-best backend on substring
search: its KMP automaton scans the packed token stream directly, while
every FSST variant and the Rust OnPair port has to decompress each row
first.

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
