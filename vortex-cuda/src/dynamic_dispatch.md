# Dynamic Dispatch

## Overview

Dynamic dispatch executes a sequence of transformation operations in a single fused kernel launch. Instead of running separate kernels for each step (which requires writing intermediate results to global memory), dynamic dispatch chains all operations together and applies them in one pass. Note that this does not imply the compiler fusing together different operations at compile time.

## Motivation

Separate kernels for each decoding step require intermediate results written to global memory:

```
Kernel 1: Bitunpack → global memory
Kernel 2: FoR decode → global memory
Kernel 3: ALP decode → global memory
```

```
Dynamic Dispatch Kernel: Bitunpack → FoR → ALP → output
```

## Operations

### Source Operations

Decompress the source data:

- Bitunpack: Decompress variable bit-width encoded data (1-64 bits per element)

### Scalar Operations

Apply scalar operations on each element:

- Frame-of-Reference (FoR): Add a reference value to each element
- Zigzag: Decode zigzag-encoded signed integers
- ALP: Apply floating-point decode with factors `f` and `e`

It is possible to chain up to 8 scalar operations in sequence.

## Usage

```rust
use vortex_cuda::dynamic_dispatch::{DynamicDispatchPlan, SourceOp, ScalarOp};

// Define a decoding plan that will be executed in a single GPU kernel.
let plan = DynamicDispatchPlan::new(
    SourceOp::bitunpack(6),              // Unpack 6-bit encoded data
    &[
        ScalarOp::frame_of_ref(100),     // Add offset of 100
        ScalarOp::alp(10.0, 0.01),       // Apply ALP decode
    ],
);
```
