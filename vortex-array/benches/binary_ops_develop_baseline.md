<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright the Vortex contributors -->

# Binary Ops Develop Baseline

Baseline captured from local `origin/develop` at `9444d20ae`.

Command:

```bash
target/release/deps/binary_ops-450fea57c1778552 --bench --color never --sample-count 60 --min-time 1
```

Timer precision: `41 ns`

| bench | fastest | median | mean |
|---|---:|---:|---:|
| `add_i64_nonnull` | `36.62 us` | `37.66 us` | `38.58 us` |
| `add_i64_nullable` | `35.58 us` | `37.74 us` | `38.49 us` |
| `and_bool_nullable` | `2.124 us` | `2.582 us` | `2.621 us` |
| `div_i64_nonnull` | `37.29 us` | `37.79 us` | `38.64 us` |
| `eq_i64_constant` | `7.207 us` | `7.457 us` | `7.606 us` |
| `lt_i64_nullable` | `9.54 us` | `9.832 us` | `9.995 us` |
| `mul_i8_nonnull` | `32.04 us` | `32.41 us` | `32.87 us` |
| `mul_i16_nonnull` | `32.79 us` | `33.12 us` | `33.66 us` |
| `mul_i32_constant` | `27.54 us` | `28.12 us` | `28.96 us` |
| `mul_i32_nonnull` | `34.24 us` | `34.83 us` | `36.37 us` |
| `mul_i32_nullable` | `33.08 us` | `36.83 us` | `38.21 us` |
| `mul_i64_nonnull` | `37.16 us` | `37.58 us` | `38.3 us` |
| `mul_u8_nonnull` | `32.12 us` | `32.45 us` | `33.01 us` |
| `mul_u16_nonnull` | `32.79 us` | `33.16 us` | `33.58 us` |
| `mul_u32_nonnull` | `34.2 us` | `34.58 us` | `35.01 us` |
| `or_bool_constant` | `1.499 us` | `1.665 us` | `1.671 us` |
| `sub_i64_constant` | `30.41 us` | `30.91 us` | `32.27 us` |
