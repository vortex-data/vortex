# OnPair + token-space block front-coding: empirical results

Reproduce: `cargo run --release -p onpair-bench -- all 1000000 2`.

## Methodology

All input columns are **lex-sorted** before encoding (the scenario under test). Every encoding's reported byte count includes the per-row 4-byte offset table so the comparisons are apples-to-apples for random row access.

- `raw (sorted)` — sum of sorted string lengths + offsets.
- `zstd-3 / zstd-9 monolithic` — one zstd of the concatenated bytes. Loses random access (best ratio, baseline for what's achievable).
- `zstd-3 block-1024` — zstd per 1024-row block; random access at block granularity.
- `fsst` — `fsst-rs` symbol table + per-row compressed payload + offsets.
- `byte front-code 256` — classical DELTA_BYTE_ARRAY style: anchor row per 256, others store `(shared_with_prev: u32, suffix_bytes)`.
- `onpair (12-bit)` — OnPair dict + bit-packed codes. No cross-row delta.
- `onpair + front-code N` — OnPair codes laid out as block front-coding in **token space**: per block of N, anchor row stores its full token sequence (bit-packed at OnPair's bit width), subsequent rows store `(shared_with_prev_tokens: u16, suffix_tokens)` with the suffix bit-packed at the same width. Random access cost: ≤N token prefix copies per row.

## tpch_l_comment (slice 0)

| Encoding | Bytes | MiB | Ratio vs raw | Bits/row | Encode ms |
|---|---:|---:|---:|---:|---:|
| raw (sorted) | 30511095 | 29.10 | 1.00× | 244.09 | 0 |
| zstd-3 monolithic | 9126617 | 8.70 | 3.34× | 73.01 | 138 |
| zstd-9 monolithic | 8280293 | 7.90 | 3.68× | 66.24 | 531 |
| zstd-3 block-1024 | 9403289 | 8.97 | 3.24× | 75.23 | 156 |
| fsst | 13451578 | 12.83 | 2.27× | 107.61 | 174 |
| byte front-code 256 | 17114829 | 16.32 | 1.78× | 136.92 | 52 |
| onpair (12-bit) | 9147399 | 8.72 | 3.34× | 73.18 | 478 |
| onpair + front-code 64 | 8627509 | 8.23 | 3.54× | 69.02 | 478 |
| onpair + front-code 256 | 8621318 | 8.22 | 3.54× | 68.97 | 478 |
| onpair + front-code 1024 | 8619807 | 8.22 | 3.54× | 68.96 | 478 |

## tpch_l_comment (slice 1)

| Encoding | Bytes | MiB | Ratio vs raw | Bits/row | Encode ms |
|---|---:|---:|---:|---:|---:|
| raw (sorted) | 30489453 | 29.08 | 1.00× | 243.92 | 0 |
| zstd-3 monolithic | 9119697 | 8.70 | 3.34× | 72.96 | 114 |
| zstd-9 monolithic | 8275012 | 7.89 | 3.68× | 66.20 | 531 |
| zstd-3 block-1024 | 9396156 | 8.96 | 3.24× | 75.17 | 152 |
| fsst | 13030562 | 12.43 | 2.34× | 104.24 | 156 |
| byte front-code 256 | 17100616 | 16.31 | 1.78× | 136.80 | 44 |
| onpair (12-bit) | 9138837 | 8.72 | 3.34× | 73.11 | 451 |
| onpair + front-code 64 | 8626234 | 8.23 | 3.53× | 69.01 | 451 |
| onpair + front-code 256 | 8620321 | 8.22 | 3.54× | 68.96 | 451 |
| onpair + front-code 1024 | 8618780 | 8.22 | 3.54× | 68.95 | 451 |

## clickbench_title (slice 0)

| Encoding | Bytes | MiB | Ratio vs raw | Bits/row | Encode ms |
|---|---:|---:|---:|---:|---:|
| raw (sorted) | 142409995 | 135.81 | 1.00× | 1139.28 | 0 |
| zstd-3 monolithic | 7037317 | 6.71 | 20.24× | 56.30 | 230 |
| zstd-9 monolithic | 6480156 | 6.18 | 21.98× | 51.84 | 479 |
| zstd-3 block-1024 | 7699198 | 7.34 | 18.50× | 61.59 | 150 |
| fsst | 73666346 | 70.25 | 1.93× | 589.33 | 521 |
| byte front-code 256 | 15562231 | 14.84 | 9.15× | 124.50 | 105 |
| onpair (12-bit) | 34375087 | 32.78 | 4.14× | 275.00 | 1524 |
| onpair + front-code 64 | 8566092 | 8.17 | 16.62× | 68.53 | 1524 |
| onpair + front-code 256 | 8259279 | 7.88 | 17.24× | 66.07 | 1524 |
| onpair + front-code 1024 | 8183143 | 7.80 | 17.40× | 65.47 | 1524 |

## clickbench_title (slice 1)

| Encoding | Bytes | MiB | Ratio vs raw | Bits/row | Encode ms |
|---|---:|---:|---:|---:|---:|
| raw (sorted) | 89712452 | 85.56 | 1.00× | 717.70 | 0 |
| zstd-3 monolithic | 7359227 | 7.02 | 12.19× | 58.87 | 176 |
| zstd-9 monolithic | 6748491 | 6.44 | 13.29× | 53.99 | 429 |
| zstd-3 block-1024 | 8155126 | 7.78 | 11.00× | 65.24 | 130 |
| fsst | 49914808 | 47.60 | 1.80× | 399.32 | 336 |
| byte front-code 256 | 16885422 | 16.10 | 5.31× | 135.08 | 65 |
| onpair (12-bit) | 24600516 | 23.46 | 3.65× | 196.80 | 1081 |
| onpair + front-code 64 | 8901110 | 8.49 | 10.08× | 71.21 | 1081 |
| onpair + front-code 256 | 8714389 | 8.31 | 10.29× | 69.72 | 1081 |
| onpair + front-code 1024 | 8668143 | 8.27 | 10.35× | 69.35 | 1081 |

## clickbench_url (slice 0)

| Encoding | Bytes | MiB | Ratio vs raw | Bits/row | Encode ms |
|---|---:|---:|---:|---:|---:|
| raw (sorted) | 92562192 | 88.27 | 1.00× | 740.50 | 0 |
| zstd-3 monolithic | 12339671 | 11.77 | 7.50× | 98.72 | 289 |
| zstd-9 monolithic | 10726221 | 10.23 | 8.63× | 85.81 | 834 |
| zstd-3 block-1024 | 14065840 | 13.41 | 6.58× | 112.53 | 201 |
| fsst | 55536381 | 52.96 | 1.67× | 444.29 | 384 |
| byte front-code 256 | 26632722 | 25.40 | 3.48× | 213.06 | 69 |
| onpair (12-bit) | 28438658 | 27.12 | 3.25× | 227.51 | 1911 |
| onpair + front-code 64 | 14559490 | 13.89 | 6.36× | 116.48 | 1911 |
| onpair + front-code 256 | 14393485 | 13.73 | 6.43× | 115.15 | 1911 |
| onpair + front-code 1024 | 14352125 | 13.69 | 6.45× | 114.82 | 1911 |

## clickbench_url (slice 1)

| Encoding | Bytes | MiB | Ratio vs raw | Bits/row | Encode ms |
|---|---:|---:|---:|---:|---:|
| raw (sorted) | 94726531 | 90.34 | 1.00× | 757.81 | 0 |
| zstd-3 monolithic | 15384380 | 14.67 | 6.16× | 123.08 | 388 |
| zstd-9 monolithic | 13018356 | 12.42 | 7.28× | 104.15 | 1100 |
| zstd-3 block-1024 | 18416371 | 17.56 | 5.14× | 147.33 | 269 |
| fsst | 56849499 | 54.22 | 1.67× | 454.80 | 445 |
| byte front-code 256 | 38781199 | 36.98 | 2.44× | 310.25 | 73 |
| onpair (12-bit) | 31578343 | 30.12 | 3.00× | 252.63 | 2329 |
| onpair + front-code 64 | 19280192 | 18.39 | 4.91× | 154.24 | 2329 |
| onpair + front-code 256 | 19135133 | 18.25 | 4.95× | 153.08 | 2329 |
| onpair + front-code 1024 | 19099175 | 18.21 | 4.96× | 152.79 | 2329 |

