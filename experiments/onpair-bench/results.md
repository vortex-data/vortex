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


# sort_bench: compare_fused vs decode-then-byte-compare

All three methods sort the same shuffled column and produce the same permutation (asserted in code). Method 1 sorts u16 token sequences via `compare_fused`. Method 2 sorts the pre-decoded `Vec<Vec<u8>>` directly (best case for the byte-compare baseline — decode cost is not charged). Method 3 decodes from the OnPair-encoded column and then sorts (realistic end-to-end cost when your storage form is encoded).

## tpch_l_comment

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 456 | 55.4 | 456 |
| two-pass: u128 key sort + refine ties | 145 | 173.9 | 145 |
| compare_fused v3 (row-prefix u64 fast path) | 408 | 61.9 | 408 |
| compare_fused v2 (u64 prefix Phase 2) | 441 | 57.3 | 441 |
| byte cmp (flat bytes, sort only, unstable) | 282 | 89.4 | 283 |
| byte cmp (decode + sort, end-to-end) | 301 | 83.7 | 302 |

## tpch_l_comment almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 259 | 97.6 | 259 |
| two-pass: u128 key sort + refine ties | 91 | 275.3 | 92 |
| compare_fused v3 (row-prefix u64 fast path) | 226 | 111.8 | 226 |
| compare_fused v2 (u64 prefix Phase 2) | 245 | 102.9 | 246 |
| byte cmp (flat bytes, sort only, unstable) | 190 | 133.0 | 190 |
| byte cmp (decode + sort, end-to-end) | 186 | 135.3 | 187 |

## clickbench_title

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 374 | 352.9 | 374 |
| two-pass: u128 key sort + refine ties | 266 | 495.7 | 266 |
| compare_fused v3 (row-prefix u64 fast path) | 382 | 345.4 | 382 |
| compare_fused v2 (u64 prefix Phase 2) | 382 | 344.7 | 383 |
| byte cmp (flat bytes, sort only, unstable) | 328 | 401.9 | 328 |
| byte cmp (decode + sort, end-to-end) | 430 | 306.7 | 430 |

## clickbench_title almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 171 | 770.7 | 171 |
| two-pass: u128 key sort + refine ties | 124 | 1057.2 | 125 |
| compare_fused v3 (row-prefix u64 fast path) | 166 | 791.0 | 167 |
| compare_fused v2 (u64 prefix Phase 2) | 175 | 750.4 | 176 |
| byte cmp (flat bytes, sort only, unstable) | 199 | 660.1 | 200 |
| byte cmp (decode + sort, end-to-end) | 229 | 574.5 | 230 |

## clickbench_url

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 548 | 153.9 | 549 |
| two-pass: u128 key sort + refine ties | 502 | 168.1 | 502 |
| compare_fused v3 (row-prefix u64 fast path) | 595 | 141.8 | 596 |
| compare_fused v2 (u64 prefix Phase 2) | 560 | 150.8 | 560 |
| byte cmp (flat bytes, sort only, unstable) | 339 | 248.9 | 339 |
| byte cmp (decode + sort, end-to-end) | 436 | 193.5 | 436 |

## clickbench_url almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 238 | 354.7 | 238 |
| two-pass: u128 key sort + refine ties | 204 | 412.6 | 205 |
| compare_fused v3 (row-prefix u64 fast path) | 257 | 327.6 | 258 |
| compare_fused v2 (u64 prefix Phase 2) | 228 | 369.8 | 228 |
| byte cmp (flat bytes, sort only, unstable) | 148 | 570.1 | 148 |
| byte cmp (decode + sort, end-to-end) | 221 | 381.7 | 221 |


# sort_bench: compare_fused vs decode-then-byte-compare

All three methods sort the same shuffled column and produce the same permutation (asserted in code). Method 1 sorts u16 token sequences via `compare_fused`. Method 2 sorts the pre-decoded `Vec<Vec<u8>>` directly (best case for the byte-compare baseline — decode cost is not charged). Method 3 decodes from the OnPair-encoded column and then sorts (realistic end-to-end cost when your storage form is encoded).

## tpch_l_comment

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 460 | 54.9 | 461 |
| two-pass: u128 key sort + refine ties | 143 | 176.6 | 143 |
| compare_fused v3 (row-prefix u64 fast path) | 418 | 60.4 | 419 |
| compare_fused v2 (u64 prefix Phase 2) | 449 | 56.2 | 450 |
| byte cmp (flat bytes, sort only, unstable) | 290 | 87.1 | 290 |
| byte cmp (decode + sort, end-to-end) | 305 | 82.9 | 305 |

## tpch_l_comment almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 258 | 97.6 | 259 |
| two-pass: u128 key sort + refine ties | 93 | 270.7 | 93 |
| compare_fused v3 (row-prefix u64 fast path) | 224 | 112.4 | 225 |
| compare_fused v2 (u64 prefix Phase 2) | 246 | 102.8 | 246 |
| byte cmp (flat bytes, sort only, unstable) | 165 | 153.0 | 165 |
| byte cmp (decode + sort, end-to-end) | 184 | 137.4 | 184 |

## clickbench_title

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 397 | 332.0 | 398 |
| two-pass: u128 key sort + refine ties | 283 | 465.1 | 284 |
| compare_fused v3 (row-prefix u64 fast path) | 396 | 333.3 | 396 |
| compare_fused v2 (u64 prefix Phase 2) | 382 | 345.3 | 382 |
| byte cmp (flat bytes, sort only, unstable) | 365 | 360.9 | 366 |
| byte cmp (decode + sort, end-to-end) | 466 | 282.8 | 467 |

## clickbench_title almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 171 | 768.6 | 172 |
| two-pass: u128 key sort + refine ties | 120 | 1095.6 | 120 |
| compare_fused v3 (row-prefix u64 fast path) | 157 | 838.8 | 157 |
| compare_fused v2 (u64 prefix Phase 2) | 178 | 738.9 | 179 |
| byte cmp (flat bytes, sort only, unstable) | 159 | 827.4 | 160 |
| byte cmp (decode + sort, end-to-end) | 296 | 445.3 | 296 |

## clickbench_url

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 564 | 149.7 | 564 |
| two-pass: u128 key sort + refine ties | 525 | 160.8 | 525 |
| compare_fused v3 (row-prefix u64 fast path) | 610 | 138.4 | 610 |
| compare_fused v2 (u64 prefix Phase 2) | 557 | 151.5 | 557 |
| byte cmp (flat bytes, sort only, unstable) | 368 | 229.4 | 368 |
| byte cmp (decode + sort, end-to-end) | 426 | 197.9 | 427 |

## clickbench_url almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 247 | 341.3 | 247 |
| two-pass: u128 key sort + refine ties | 211 | 398.7 | 212 |
| compare_fused v3 (row-prefix u64 fast path) | 261 | 322.6 | 262 |
| compare_fused v2 (u64 prefix Phase 2) | 232 | 363.8 | 232 |
| byte cmp (flat bytes, sort only, unstable) | 154 | 545.6 | 155 |
| byte cmp (decode + sort, end-to-end) | 221 | 381.9 | 221 |


# sort_bench: compare_fused vs decode-then-byte-compare

All three methods sort the same shuffled column and produce the same permutation (asserted in code). Method 1 sorts u16 token sequences via `compare_fused`. Method 2 sorts the pre-decoded `Vec<Vec<u8>>` directly (best case for the byte-compare baseline — decode cost is not charged). Method 3 decodes from the OnPair-encoded column and then sorts (realistic end-to-end cost when your storage form is encoded).

## tpch_l_comment

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 454 | 55.7 | 454 |
| two-pass: u128 key sort + refine ties | 146 | 172.6 | 146 |
| two-pass: 32B key sort + refine ties | 108 | 233.4 | 108 |
| compare_fused v3 (row-prefix u64 fast path) | 402 | 62.8 | 402 |
| compare_fused v2 (u64 prefix Phase 2) | 479 | 52.7 | 480 |
| byte cmp (flat bytes, sort only, unstable) | 290 | 87.0 | 291 |
| byte cmp (decode + sort, end-to-end) | 317 | 79.7 | 317 |

## tpch_l_comment almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 261 | 96.7 | 261 |
| two-pass: u128 key sort + refine ties | 92 | 274.8 | 92 |
| two-pass: 32B key sort + refine ties | 97 | 259.7 | 97 |
| compare_fused v3 (row-prefix u64 fast path) | 249 | 101.4 | 249 |
| compare_fused v2 (u64 prefix Phase 2) | 249 | 101.5 | 249 |
| byte cmp (flat bytes, sort only, unstable) | 186 | 135.3 | 187 |
| byte cmp (decode + sort, end-to-end) | 206 | 122.5 | 206 |

## clickbench_title

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 375 | 351.9 | 375 |
| two-pass: u128 key sort + refine ties | 295 | 446.0 | 296 |
| two-pass: 32B key sort + refine ties | 239 | 552.0 | 239 |
| compare_fused v3 (row-prefix u64 fast path) | 388 | 340.2 | 388 |
| compare_fused v2 (u64 prefix Phase 2) | 370 | 356.1 | 371 |
| byte cmp (flat bytes, sort only, unstable) | 363 | 363.3 | 363 |
| byte cmp (decode + sort, end-to-end) | 461 | 286.0 | 462 |

## clickbench_title almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 164 | 802.2 | 165 |
| two-pass: u128 key sort + refine ties | 117 | 1121.3 | 118 |
| two-pass: 32B key sort + refine ties | 106 | 1234.6 | 107 |
| compare_fused v3 (row-prefix u64 fast path) | 151 | 869.5 | 152 |
| compare_fused v2 (u64 prefix Phase 2) | 162 | 813.3 | 162 |
| byte cmp (flat bytes, sort only, unstable) | 148 | 889.0 | 148 |
| byte cmp (decode + sort, end-to-end) | 233 | 564.3 | 234 |

## clickbench_url

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 535 | 157.7 | 535 |
| two-pass: u128 key sort + refine ties | 486 | 173.5 | 487 |
| two-pass: 32B key sort + refine ties | 440 | 191.9 | 440 |
| compare_fused v3 (row-prefix u64 fast path) | 588 | 143.5 | 589 |
| compare_fused v2 (u64 prefix Phase 2) | 540 | 156.4 | 540 |
| byte cmp (flat bytes, sort only, unstable) | 370 | 228.0 | 370 |
| byte cmp (decode + sort, end-to-end) | 433 | 194.7 | 434 |

## clickbench_url almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 230 | 365.8 | 231 |
| two-pass: u128 key sort + refine ties | 207 | 407.5 | 207 |
| two-pass: 32B key sort + refine ties | 191 | 441.0 | 192 |
| compare_fused v3 (row-prefix u64 fast path) | 256 | 329.2 | 257 |
| compare_fused v2 (u64 prefix Phase 2) | 235 | 358.9 | 235 |
| byte cmp (flat bytes, sort only, unstable) | 173 | 485.9 | 174 |
| byte cmp (decode + sort, end-to-end) | 243 | 347.3 | 243 |


# sort_bench: compare_fused vs decode-then-byte-compare

All three methods sort the same shuffled column and produce the same permutation (asserted in code). Method 1 sorts u16 token sequences via `compare_fused`. Method 2 sorts the pre-decoded `Vec<Vec<u8>>` directly (best case for the byte-compare baseline — decode cost is not charged). Method 3 decodes from the OnPair-encoded column and then sorts (realistic end-to-end cost when your storage form is encoded).

## tpch_l_comment

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 463 | 54.6 | 463 |
| two-pass: u128 key sort + refine ties | 147 | 171.2 | 148 |
| two-pass: 32B key, sort u32 indices | 159 | 158.7 | 159 |
| two-pass: 32B key sort + refine ties | 111 | 226.9 | 111 |
| compare_fused v3 (row-prefix u64 fast path) | 409 | 61.7 | 410 |
| compare_fused v2 (u64 prefix Phase 2) | 437 | 57.9 | 437 |
| byte cmp (flat bytes, sort only, unstable) | 290 | 87.1 | 290 |
| byte cmp (decode + sort, end-to-end) | 287 | 87.9 | 288 |

## tpch_l_comment almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 258 | 97.9 | 258 |
| two-pass: u128 key sort + refine ties | 95 | 264.6 | 96 |
| two-pass: 32B key, sort u32 indices | 82 | 306.1 | 83 |
| two-pass: 32B key sort + refine ties | 73 | 342.7 | 74 |
| compare_fused v3 (row-prefix u64 fast path) | 226 | 111.8 | 226 |
| compare_fused v2 (u64 prefix Phase 2) | 246 | 102.7 | 246 |
| byte cmp (flat bytes, sort only, unstable) | 165 | 153.1 | 165 |
| byte cmp (decode + sort, end-to-end) | 183 | 138.1 | 183 |

## clickbench_title

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 389 | 338.6 | 390 |
| two-pass: u128 key sort + refine ties | 280 | 470.1 | 281 |
| two-pass: 32B key, sort u32 indices | 279 | 471.8 | 280 |
| two-pass: 32B key sort + refine ties | 250 | 526.4 | 251 |
| compare_fused v3 (row-prefix u64 fast path) | 413 | 318.9 | 414 |
| compare_fused v2 (u64 prefix Phase 2) | 402 | 327.8 | 403 |
| byte cmp (flat bytes, sort only, unstable) | 399 | 330.3 | 400 |
| byte cmp (decode + sort, end-to-end) | 482 | 273.6 | 482 |

## clickbench_title almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 169 | 781.0 | 169 |
| two-pass: u128 key sort + refine ties | 127 | 1032.8 | 128 |
| two-pass: 32B key, sort u32 indices | 119 | 1104.4 | 120 |
| two-pass: 32B key sort + refine ties | 119 | 1104.9 | 119 |
| compare_fused v3 (row-prefix u64 fast path) | 160 | 825.0 | 160 |
| compare_fused v2 (u64 prefix Phase 2) | 162 | 813.4 | 162 |
| byte cmp (flat bytes, sort only, unstable) | 166 | 791.2 | 167 |
| byte cmp (decode + sort, end-to-end) | 241 | 546.0 | 242 |

## clickbench_url

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 560 | 150.8 | 560 |
| two-pass: u128 key sort + refine ties | 537 | 157.2 | 537 |
| two-pass: 32B key, sort u32 indices | 498 | 169.4 | 499 |
| two-pass: 32B key sort + refine ties | 478 | 176.7 | 478 |
| compare_fused v3 (row-prefix u64 fast path) | 618 | 136.6 | 618 |
| compare_fused v2 (u64 prefix Phase 2) | 553 | 152.6 | 553 |
| byte cmp (flat bytes, sort only, unstable) | 393 | 214.6 | 394 |
| byte cmp (decode + sort, end-to-end) | 483 | 174.8 | 483 |

## clickbench_url almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 233 | 361.1 | 234 |
| two-pass: u128 key sort + refine ties | 208 | 404.8 | 209 |
| two-pass: 32B key, sort u32 indices | 209 | 403.0 | 210 |
| two-pass: 32B key sort + refine ties | 195 | 432.9 | 195 |
| compare_fused v3 (row-prefix u64 fast path) | 266 | 317.2 | 266 |
| compare_fused v2 (u64 prefix Phase 2) | 227 | 371.5 | 227 |
| byte cmp (flat bytes, sort only, unstable) | 168 | 502.4 | 168 |
| byte cmp (decode + sort, end-to-end) | 237 | 355.7 | 237 |


# sort_bench: compare_fused vs decode-then-byte-compare

All three methods sort the same shuffled column and produce the same permutation (asserted in code). Method 1 sorts u16 token sequences via `compare_fused`. Method 2 sorts the pre-decoded `Vec<Vec<u8>>` directly (best case for the byte-compare baseline — decode cost is not charged). Method 3 decodes from the OnPair-encoded column and then sorts (realistic end-to-end cost when your storage form is encoded).

## clickbench_url

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 540 | 156.3 | 540 |
| two-pass: u128 key sort + refine ties | 508 | 166.0 | 509 |
| two-pass: 32B key + byte cmp refine | 329 | 256.6 | 329 |
| two-pass: 32B key, sort u32 indices | 483 | 174.7 | 483 |
| two-pass: 32B key sort + refine ties | 455 | 185.3 | 456 |
| compare_fused v3 (row-prefix u64 fast path) | 583 | 144.8 | 583 |
| compare_fused v2 (u64 prefix Phase 2) | 538 | 156.8 | 539 |
| byte cmp (flat bytes, sort only, unstable) | 368 | 229.5 | 368 |
| byte cmp (decode + sort, end-to-end) | 425 | 198.4 | 426 |

## clickbench_url almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 228 | 369.4 | 229 |
| two-pass: u128 key sort + refine ties | 209 | 404.0 | 209 |
| two-pass: 32B key + byte cmp refine | 131 | 640.0 | 132 |
| two-pass: 32B key, sort u32 indices | 203 | 415.7 | 203 |
| two-pass: 32B key sort + refine ties | 203 | 415.1 | 203 |
| compare_fused v3 (row-prefix u64 fast path) | 244 | 345.3 | 245 |
| compare_fused v2 (u64 prefix Phase 2) | 227 | 372.0 | 227 |
| byte cmp (flat bytes, sort only, unstable) | 156 | 538.9 | 157 |
| byte cmp (decode + sort, end-to-end) | 222 | 378.8 | 223 |


# sort_bench: compare_fused vs decode-then-byte-compare

All three methods sort the same shuffled column and produce the same permutation (asserted in code). Method 1 sorts u16 token sequences via `compare_fused`. Method 2 sorts the pre-decoded `Vec<Vec<u8>>` directly (best case for the byte-compare baseline — decode cost is not charged). Method 3 decodes from the OnPair-encoded column and then sorts (realistic end-to-end cost when your storage form is encoded).

## tpch_l_comment

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 453 | 55.7 | 454 |
| two-pass: u128 key sort + refine ties | 150 | 167.6 | 151 |
| two-pass: 32B key + byte cmp refine | 118 | 214.1 | 118 |
| two-pass: 32B key, sort u32 indices | 156 | 161.8 | 156 |
| two-pass: 32B key sort + refine ties | 110 | 228.0 | 111 |
| compare_fused v3 (row-prefix u64 fast path) | 401 | 63.0 | 401 |
| compare_fused v2 (u64 prefix Phase 2) | 441 | 57.3 | 441 |
| byte cmp (flat bytes, sort only, unstable) | 275 | 91.7 | 276 |
| byte cmp (decode + sort, end-to-end) | 297 | 84.9 | 298 |

## tpch_l_comment almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 262 | 96.4 | 262 |
| two-pass: u128 key sort + refine ties | 96 | 263.2 | 96 |
| two-pass: 32B key + byte cmp refine | 84 | 299.2 | 84 |
| two-pass: 32B key, sort u32 indices | 87 | 290.2 | 87 |
| two-pass: 32B key sort + refine ties | 77 | 328.2 | 77 |
| compare_fused v3 (row-prefix u64 fast path) | 230 | 109.5 | 231 |
| compare_fused v2 (u64 prefix Phase 2) | 251 | 100.4 | 252 |
| byte cmp (flat bytes, sort only, unstable) | 166 | 151.5 | 167 |
| byte cmp (decode + sort, end-to-end) | 192 | 131.2 | 193 |

## clickbench_title

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 373 | 353.3 | 374 |
| two-pass: u128 key sort + refine ties | 267 | 492.9 | 268 |
| two-pass: 32B key + byte cmp refine | 207 | 635.6 | 208 |
| two-pass: 32B key, sort u32 indices | 253 | 520.7 | 254 |
| two-pass: 32B key sort + refine ties | 227 | 581.2 | 227 |
| compare_fused v3 (row-prefix u64 fast path) | 379 | 347.4 | 380 |
| compare_fused v2 (u64 prefix Phase 2) | 368 | 357.8 | 369 |
| byte cmp (flat bytes, sort only, unstable) | 339 | 388.8 | 340 |
| byte cmp (decode + sort, end-to-end) | 420 | 314.0 | 420 |

## clickbench_title almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 165 | 798.9 | 165 |
| two-pass: u128 key sort + refine ties | 120 | 1099.0 | 120 |
| two-pass: 32B key + byte cmp refine | 91 | 1437.5 | 92 |
| two-pass: 32B key, sort u32 indices | 114 | 1154.1 | 114 |
| two-pass: 32B key sort + refine ties | 109 | 1205.5 | 109 |
| compare_fused v3 (row-prefix u64 fast path) | 150 | 879.1 | 150 |
| compare_fused v2 (u64 prefix Phase 2) | 153 | 859.0 | 154 |
| byte cmp (flat bytes, sort only, unstable) | 148 | 886.1 | 149 |
| byte cmp (decode + sort, end-to-end) | 249 | 528.5 | 250 |

## clickbench_url

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 539 | 156.6 | 539 |
| two-pass: u128 key sort + refine ties | 495 | 170.3 | 496 |
| two-pass: 32B key + byte cmp refine | 295 | 286.3 | 295 |
| two-pass: 32B key, sort u32 indices | 461 | 183.1 | 461 |
| two-pass: 32B key sort + refine ties | 440 | 191.9 | 440 |
| compare_fused v3 (row-prefix u64 fast path) | 585 | 144.2 | 586 |
| compare_fused v2 (u64 prefix Phase 2) | 542 | 155.7 | 543 |
| byte cmp (flat bytes, sort only, unstable) | 331 | 254.5 | 332 |
| byte cmp (decode + sort, end-to-end) | 407 | 207.3 | 407 |

## clickbench_url almost-sorted

| Method | Time (ms) | MB/s (raw) | ns/row |
|---|---:|---:|---:|
| compare_fused v1 (slice cmp Phase 2) | 229 | 367.5 | 230 |
| two-pass: u128 key sort + refine ties | 207 | 407.2 | 207 |
| two-pass: 32B key + byte cmp refine | 126 | 669.6 | 126 |
| two-pass: 32B key, sort u32 indices | 202 | 417.4 | 202 |
| two-pass: 32B key sort + refine ties | 187 | 449.7 | 188 |
| compare_fused v3 (row-prefix u64 fast path) | 241 | 350.3 | 241 |
| compare_fused v2 (u64 prefix Phase 2) | 221 | 380.5 | 222 |
| byte cmp (flat bytes, sort only, unstable) | 150 | 562.9 | 150 |
| byte cmp (decode + sort, end-to-end) | 213 | 395.8 | 213 |

