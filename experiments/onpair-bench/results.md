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


# sort_bench: compare_fused vs decode-then-byte-compare

All three methods sort the same shuffled column and produce the same permutation (asserted in code). Method 1 sorts u16 token sequences via `compare_fused`. Method 2 sorts the pre-decoded `Vec<Vec<u8>>` directly (best case for the byte-compare baseline — decode cost is not charged). Method 3 decodes from the OnPair-encoded column and then sorts (realistic end-to-end cost when your storage form is encoded).

## tpch_l_comment

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused (tokens) | 265 | 47.7 | 530 |
| byte cmp (sort only, pre-decoded) | 139 | 90.7 | 279 |
| byte cmp (end-to-end: decode + sort) | 165 | 76.5 | 330 |

## clickbench_title

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused (tokens) | 277 | 383.5 | 554 |
| byte cmp (sort only, pre-decoded) | 196 | 541.5 | 392 |
| byte cmp (end-to-end: decode + sort) | 342 | 309.8 | 686 |

## clickbench_url

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused (tokens) | 297 | 122.7 | 594 |
| byte cmp (sort only, pre-decoded) | 168 | 216.8 | 336 |
| byte cmp (end-to-end: decode + sort) | 193 | 188.3 | 387 |


# sort_bench: compare_fused vs decode-then-byte-compare

All three methods sort the same shuffled column and produce the same permutation (asserted in code). Method 1 sorts u16 token sequences via `compare_fused`. Method 2 sorts the pre-decoded `Vec<Vec<u8>>` directly (best case for the byte-compare baseline — decode cost is not charged). Method 3 decodes from the OnPair-encoded column and then sorts (realistic end-to-end cost when your storage form is encoded).

## tpch_l_comment

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused (tokens) | 616 | 41.0 | 617 |
| byte cmp (sort only, pre-decoded) | 324 | 78.0 | 324 |
| byte cmp (end-to-end: decode + sort) | 364 | 69.4 | 364 |

## clickbench_title

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused (tokens) | 472 | 279.5 | 472 |
| byte cmp (sort only, pre-decoded) | 384 | 343.7 | 384 |
| byte cmp (end-to-end: decode + sort) | 579 | 227.9 | 579 |

## clickbench_url

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused (tokens) | 690 | 122.2 | 691 |
| byte cmp (sort only, pre-decoded) | 351 | 240.4 | 351 |
| byte cmp (end-to-end: decode + sort) | 524 | 161.0 | 525 |


# sort_bench: compare_fused vs decode-then-byte-compare

All three methods sort the same shuffled column and produce the same permutation (asserted in code). Method 1 sorts u16 token sequences via `compare_fused`. Method 2 sorts the pre-decoded `Vec<Vec<u8>>` directly (best case for the byte-compare baseline — decode cost is not charged). Method 3 decodes from the OnPair-encoded column and then sorts (realistic end-to-end cost when your storage form is encoded).

## tpch_l_comment

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused (tokens) | 962 | 26.3 | 963 |
| byte cmp (sort only, pre-decoded) | 330 | 76.6 | 330 |
| byte cmp (end-to-end: decode + sort) | 372 | 67.9 | 372 |

## clickbench_title

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused (tokens) | 601 | 219.4 | 602 |
| byte cmp (sort only, pre-decoded) | 385 | 342.5 | 385 |
| byte cmp (end-to-end: decode + sort) | 549 | 240.2 | 550 |

## clickbench_url

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused (tokens) | 931 | 90.7 | 931 |
| byte cmp (sort only, pre-decoded) | 384 | 219.7 | 384 |
| byte cmp (end-to-end: decode + sort) | 451 | 187.2 | 451 |


# sort_bench: compare_fused vs decode-then-byte-compare

All three methods sort the same shuffled column and produce the same permutation (asserted in code). Method 1 sorts u16 token sequences via `compare_fused`. Method 2 sorts the pre-decoded `Vec<Vec<u8>>` directly (best case for the byte-compare baseline — decode cost is not charged). Method 3 decodes from the OnPair-encoded column and then sorts (realistic end-to-end cost when your storage form is encoded).

## tpch_l_comment

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused (tokens) | 591 | 42.7 | 591 |
| byte cmp (sort only, pre-decoded) | 324 | 77.9 | 325 |
| byte cmp (end-to-end: decode + sort) | 349 | 72.3 | 350 |

## clickbench_title

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused (tokens) | 459 | 287.2 | 460 |
| byte cmp (sort only, pre-decoded) | 378 | 348.8 | 378 |
| byte cmp (end-to-end: decode + sort) | 570 | 231.2 | 571 |

## clickbench_url

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused (tokens) | 737 | 114.6 | 737 |
| byte cmp (sort only, pre-decoded) | 384 | 219.9 | 384 |
| byte cmp (end-to-end: decode + sort) | 446 | 189.3 | 446 |


# sort_bench: compare_fused vs decode-then-byte-compare

All three methods sort the same shuffled column and produce the same permutation (asserted in code). Method 1 sorts u16 token sequences via `compare_fused`. Method 2 sorts the pre-decoded `Vec<Vec<u8>>` directly (best case for the byte-compare baseline — decode cost is not charged). Method 3 decodes from the OnPair-encoded column and then sorts (realistic end-to-end cost when your storage form is encoded).

## tpch_l_comment

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused (flat tokens, unstable) | 459 | 55.1 | 459 |
| byte cmp (flat bytes, sort only, unstable) | 282 | 89.4 | 283 |
| byte cmp (decode + sort, end-to-end) | 304 | 83.1 | 304 |

## clickbench_title

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused (flat tokens, unstable) | 379 | 347.8 | 380 |
| byte cmp (flat bytes, sort only, unstable) | 322 | 409.0 | 323 |
| byte cmp (decode + sort, end-to-end) | 428 | 308.1 | 428 |

## clickbench_url

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused (flat tokens, unstable) | 612 | 137.8 | 613 |
| byte cmp (flat bytes, sort only, unstable) | 379 | 222.5 | 380 |
| byte cmp (decode + sort, end-to-end) | 439 | 192.1 | 440 |


# sort_bench: compare_fused vs decode-then-byte-compare

All three methods sort the same shuffled column and produce the same permutation (asserted in code). Method 1 sorts u16 token sequences via `compare_fused`. Method 2 sorts the pre-decoded `Vec<Vec<u8>>` directly (best case for the byte-compare baseline — decode cost is not charged). Method 3 decodes from the OnPair-encoded column and then sorts (realistic end-to-end cost when your storage form is encoded).

## tpch_l_comment

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 450 | 56.1 | 450 |
| compare_fused v2 (u64 prefix Phase 2) | 446 | 56.6 | 447 |
| byte cmp (flat bytes, sort only, unstable) | 282 | 89.6 | 282 |
| byte cmp (decode + sort, end-to-end) | 305 | 82.7 | 306 |

## clickbench_title

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 390 | 337.6 | 391 |
| compare_fused v2 (u64 prefix Phase 2) | 388 | 339.9 | 388 |
| byte cmp (flat bytes, sort only, unstable) | 347 | 379.3 | 348 |
| byte cmp (decode + sort, end-to-end) | 447 | 294.6 | 448 |

## clickbench_url

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 543 | 155.5 | 543 |
| compare_fused v2 (u64 prefix Phase 2) | 573 | 147.2 | 574 |
| byte cmp (flat bytes, sort only, unstable) | 362 | 233.1 | 362 |
| byte cmp (decode + sort, end-to-end) | 455 | 185.5 | 455 |


# sort_bench: compare_fused vs decode-then-byte-compare

All three methods sort the same shuffled column and produce the same permutation (asserted in code). Method 1 sorts u16 token sequences via `compare_fused`. Method 2 sorts the pre-decoded `Vec<Vec<u8>>` directly (best case for the byte-compare baseline — decode cost is not charged). Method 3 decodes from the OnPair-encoded column and then sorts (realistic end-to-end cost when your storage form is encoded).

## tpch_l_comment

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 482 | 52.4 | 482 |
| compare_fused v3 (row-prefix u64 fast path) | 405 | 62.4 | 405 |
| compare_fused v2 (u64 prefix Phase 2) | 439 | 57.5 | 440 |
| byte cmp (flat bytes, sort only, unstable) | 273 | 92.6 | 273 |
| byte cmp (decode + sort, end-to-end) | 296 | 85.3 | 296 |

## clickbench_title

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 401 | 328.6 | 402 |
| compare_fused v3 (row-prefix u64 fast path) | 429 | 307.1 | 430 |
| compare_fused v2 (u64 prefix Phase 2) | 386 | 341.4 | 387 |
| byte cmp (flat bytes, sort only, unstable) | 369 | 357.1 | 370 |
| byte cmp (decode + sort, end-to-end) | 457 | 288.5 | 458 |

## clickbench_url

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 530 | 159.1 | 531 |
| compare_fused v3 (row-prefix u64 fast path) | 579 | 145.6 | 580 |
| compare_fused v2 (u64 prefix Phase 2) | 547 | 154.2 | 548 |
| byte cmp (flat bytes, sort only, unstable) | 361 | 233.8 | 361 |
| byte cmp (decode + sort, end-to-end) | 403 | 209.2 | 404 |


# sort_bench: compare_fused vs decode-then-byte-compare

All three methods sort the same shuffled column and produce the same permutation (asserted in code). Method 1 sorts u16 token sequences via `compare_fused`. Method 2 sorts the pre-decoded `Vec<Vec<u8>>` directly (best case for the byte-compare baseline — decode cost is not charged). Method 3 decodes from the OnPair-encoded column and then sorts (realistic end-to-end cost when your storage form is encoded).

## clickbench_url

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 552 | 152.7 | 553 |
| compare_fused v3 (row-prefix u64 fast path) | 610 | 138.3 | 610 |
| compare_fused v2 (u64 prefix Phase 2) | 559 | 151.0 | 559 |
| byte cmp (flat bytes, sort only, unstable) | 371 | 227.4 | 371 |
| byte cmp (decode + sort, end-to-end) | 479 | 176.3 | 479 |

## clickbench_url almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 248 | 340.2 | 248 |
| compare_fused v3 (row-prefix u64 fast path) | 322 | 261.8 | 323 |
| compare_fused v2 (u64 prefix Phase 2) | 230 | 365.7 | 231 |
| byte cmp (flat bytes, sort only, unstable) | 162 | 518.6 | 163 |
| byte cmp (decode + sort, end-to-end) | 229 | 368.2 | 229 |


# sort_bench: compare_fused vs decode-then-byte-compare

All three methods sort the same shuffled column and produce the same permutation (asserted in code). Method 1 sorts u16 token sequences via `compare_fused`. Method 2 sorts the pre-decoded `Vec<Vec<u8>>` directly (best case for the byte-compare baseline — decode cost is not charged). Method 3 decodes from the OnPair-encoded column and then sorts (realistic end-to-end cost when your storage form is encoded).

## tpch_l_comment

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 464 | 54.4 | 465 |
| compare_fused v3 (row-prefix u64 fast path) | 448 | 56.3 | 449 |
| compare_fused v2 (u64 prefix Phase 2) | 456 | 55.4 | 456 |
| byte cmp (flat bytes, sort only, unstable) | 275 | 91.6 | 276 |
| byte cmp (decode + sort, end-to-end) | 309 | 81.8 | 309 |

## tpch_l_comment almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 258 | 98.0 | 258 |
| compare_fused v3 (row-prefix u64 fast path) | 228 | 110.7 | 228 |
| compare_fused v2 (u64 prefix Phase 2) | 246 | 102.5 | 247 |
| byte cmp (flat bytes, sort only, unstable) | 166 | 151.7 | 167 |
| byte cmp (decode + sort, end-to-end) | 183 | 137.7 | 184 |

## clickbench_title

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 394 | 334.8 | 394 |
| compare_fused v3 (row-prefix u64 fast path) | 436 | 302.3 | 437 |
| compare_fused v2 (u64 prefix Phase 2) | 368 | 358.2 | 369 |
| byte cmp (flat bytes, sort only, unstable) | 388 | 340.1 | 388 |
| byte cmp (decode + sort, end-to-end) | 514 | 256.6 | 514 |

## clickbench_title almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 162 | 810.6 | 163 |
| compare_fused v3 (row-prefix u64 fast path) | 165 | 799.0 | 165 |
| compare_fused v2 (u64 prefix Phase 2) | 149 | 880.6 | 150 |
| byte cmp (flat bytes, sort only, unstable) | 143 | 919.1 | 144 |
| byte cmp (decode + sort, end-to-end) | 231 | 570.2 | 232 |

## clickbench_url

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 545 | 155.0 | 545 |
| compare_fused v3 (row-prefix u64 fast path) | 591 | 142.8 | 591 |
| compare_fused v2 (u64 prefix Phase 2) | 549 | 153.7 | 550 |
| byte cmp (flat bytes, sort only, unstable) | 340 | 248.3 | 340 |
| byte cmp (decode + sort, end-to-end) | 409 | 206.5 | 409 |

## clickbench_url almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 233 | 361.8 | 233 |
| compare_fused v3 (row-prefix u64 fast path) | 258 | 327.3 | 258 |
| compare_fused v2 (u64 prefix Phase 2) | 230 | 366.2 | 231 |
| byte cmp (flat bytes, sort only, unstable) | 153 | 551.3 | 153 |
| byte cmp (decode + sort, end-to-end) | 222 | 380.0 | 222 |


# sort_bench: compare_fused vs decode-then-byte-compare

All three methods sort the same shuffled column and produce the same permutation (asserted in code). Method 1 sorts u16 token sequences via `compare_fused`. Method 2 sorts the pre-decoded `Vec<Vec<u8>>` directly (best case for the byte-compare baseline — decode cost is not charged). Method 3 decodes from the OnPair-encoded column and then sorts (realistic end-to-end cost when your storage form is encoded).

## tpch_l_comment

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 451 | 56.0 | 452 |
| compare_fused v3 (row-prefix u64 fast path) | 455 | 55.5 | 455 |
| compare_fused v2 (u64 prefix Phase 2) | 439 | 57.6 | 439 |
| byte cmp (flat bytes, sort only, unstable) | 274 | 92.2 | 274 |
| byte cmp (decode + sort, end-to-end) | 306 | 82.4 | 307 |

## tpch_l_comment almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 263 | 96.1 | 263 |
| compare_fused v3 (row-prefix u64 fast path) | 232 | 108.7 | 233 |
| compare_fused v2 (u64 prefix Phase 2) | 250 | 100.7 | 251 |
| byte cmp (flat bytes, sort only, unstable) | 171 | 147.1 | 172 |
| byte cmp (decode + sort, end-to-end) | 187 | 135.1 | 187 |

## clickbench_title

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 373 | 353.6 | 373 |
| compare_fused v3 (row-prefix u64 fast path) | 390 | 338.2 | 390 |
| compare_fused v2 (u64 prefix Phase 2) | 363 | 362.7 | 364 |
| byte cmp (flat bytes, sort only, unstable) | 328 | 401.5 | 329 |
| byte cmp (decode + sort, end-to-end) | 457 | 288.7 | 457 |

## clickbench_title almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 160 | 820.4 | 161 |
| compare_fused v3 (row-prefix u64 fast path) | 165 | 799.3 | 165 |
| compare_fused v2 (u64 prefix Phase 2) | 147 | 894.3 | 148 |
| byte cmp (flat bytes, sort only, unstable) | 142 | 928.0 | 142 |
| byte cmp (decode + sort, end-to-end) | 226 | 581.8 | 227 |

## clickbench_url

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 563 | 149.8 | 564 |
| compare_fused v3 (row-prefix u64 fast path) | 614 | 137.5 | 614 |
| compare_fused v2 (u64 prefix Phase 2) | 571 | 147.7 | 572 |
| byte cmp (flat bytes, sort only, unstable) | 340 | 248.0 | 341 |
| byte cmp (decode + sort, end-to-end) | 426 | 198.0 | 427 |

## clickbench_url almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 231 | 364.7 | 232 |
| compare_fused v3 (row-prefix u64 fast path) | 259 | 325.7 | 259 |
| compare_fused v2 (u64 prefix Phase 2) | 231 | 365.3 | 231 |
| byte cmp (flat bytes, sort only, unstable) | 157 | 537.5 | 157 |
| byte cmp (decode + sort, end-to-end) | 221 | 381.0 | 222 |

