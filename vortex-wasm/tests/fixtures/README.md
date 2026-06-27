<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: Copyright the Vortex contributors
-->

# WASM kernel test fixtures

These `.wasm` files are decoder kernels compiled from the examples in
`vortex-wasm-guest/examples/`. They are committed so `tests/kernel_roundtrip.rs` can exercise the
full host/guest pipeline with real kernels (via `include_bytes!`) without building a
`wasm32-unknown-unknown` toolchain at test time.

| Fixture | Source example |
| --- | --- |
| `identity_kernel.wasm` | `examples/identity-kernel` |
| `for_kernel.wasm` | `examples/for-kernel` |
| `for_bitpack_kernel.wasm` | `examples/for-bitpack-kernel` |

## Rebuilding

After changing the guest SDK or an example kernel, rebuild and copy the fixtures:

```bash
cd vortex-wasm-guest/examples
for k in identity-kernel for-kernel for-bitpack-kernel; do
  (cd "$k" && cargo build --target wasm32-unknown-unknown --release)
done
cp identity-kernel/target/wasm32-unknown-unknown/release/identity_kernel.wasm \
   ../../vortex-wasm/tests/fixtures/
cp for-kernel/target/wasm32-unknown-unknown/release/for_kernel.wasm \
   ../../vortex-wasm/tests/fixtures/
cp for-bitpack-kernel/target/wasm32-unknown-unknown/release/for_bitpack_kernel.wasm \
   ../../vortex-wasm/tests/fixtures/
```
