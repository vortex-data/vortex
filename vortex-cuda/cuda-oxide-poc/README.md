# cuda-oxide POC for Vortex CUDA kernels

This is an intentionally isolated prototype for trying [`cuda-oxide`](https://github.com/NVlabs/cuda-oxide) without changing the production `vortex-cuda` build.

The POC includes:

1. a tiny Vortex-style frame-of-reference kernel in Rust, and
2. a first tagged-payload dispatch prototype for a synthetic `For(BitPacked<u32>)` case.

The FoR kernel is:

```rust
#[kernel]
pub fn for_in_place_i32(reference: i32, mut values: DisjointSlice<i32>) {
    let idx = thread::index_1d();

    if let Some(value) = values.get_mut(idx) {
        *value += reference;
    }
}
```

That corresponds to the simple `values[i] += reference` behavior in `../kernels/src/for.cu`.

## Why this is excluded from the workspace

The main Vortex workspace is pinned to stable Rust in `../../rust-toolchain.toml`. `cuda-oxide` currently requires a nightly toolchain with `rustc-dev` and `llvm-tools`, so this package is excluded from the root workspace and has its own local `rust-toolchain.toml`.

Normal Vortex builds should remain unaffected.

## Prerequisites

Follow the upstream `cuda-oxide` setup instructions. At minimum:

```sh
rustup component add rust-src rustc-dev llvm-tools --toolchain nightly-2026-04-03
cargo +nightly-2026-04-03 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
cargo +nightly-2026-04-03 oxide doctor
```

You also need a working CUDA toolkit and NVIDIA driver.

## Run

From this directory:

```sh
cargo +nightly-2026-04-03 oxide run
```

Expected output includes validation and CUDA-event microbenchmarks:

```text
cuda-oxide FoR POC passed for 4096 i32 values
cuda-oxide FoR benchmark: len=16777216 launches=200 total=11.681 ms avg=58.404 us throughput=2140.26 GiB/s
cuda-oxide tagged-payload dispatch POC passed for 4096 u32 values
cuda-oxide tagged-payload dispatch benchmark: case=for_bitpacked_u32 len=16777216 launches=200 total=5.721 ms avg=28.607 us throughput=2594.39 GiB/s
```

The exact benchmark numbers vary by GPU, clocks, target architecture, and driver state.

## What this proves

This validates the smallest useful loop:

1. kernel authored in Rust,
2. compiled to CUDA PTX by `cuda-oxide`,
3. generated PTX loaded from `vortex_cuda_oxide_poc.ptx`,
4. launched from Rust,
5. result checked against CPU output.

It also validates a compact dispatch-plan shape:

```rust
#[repr(u32)]
enum SimpleDispatchKind {
    ForBitPackedU32 = 1,
    // ...
}

#[repr(C)]
struct SimpleDispatchPlan {
    kind: u32,
    _reserved: u32,
    payload0: u64,
    payload1: u64,
}
```

`payload0`/`payload1` are variant payload words. For `ForBitPackedU32`, `payload0` is the packed-buffer device pointer and `payload1` packs `bit_width` and `reference`.

The current dispatch kernel specializes the benchmark case (`bit_width = 6`) and decodes four output values per CUDA thread. It now models the two logical steps explicitly inside one global kernel:

```text
SourceOp::BitPackedU32 -> ScalarOp::FoRU32 -> store
```

The source step is factored into a Rust helper function. cuda-oxide emits that helper as a PTX `.func`, while `simple_dispatch_u32` remains the global `.entry`; the scalar helper is simple enough to inline.

This made it comparable to the focused production `.cu` dynamic-dispatch comparison bench:

```text
cu dynamic_dispatch benchmark: case=for_bitpacked_u32 len=16777216 launches=200 total=6.571 ms avg=32.854 us throughput=2259.05 GiB/s
```

The upstream typed embedded-module loader was attempted first, but in this nested standalone package the binary did not expose a discoverable embedded artifact bundle even though `cargo oxide` generated the PTX and embed object. Direct PTX loading keeps this POC useful while we investigate artifact embedding separately.

The benchmark is intentionally a microbenchmark: it times only repeated kernel launches with CUDA events after warmup. It does not include Vortex decode planning, buffer movement, async dispatch overhead, or Criterion sampling. The FoR throughput counts one read and one write of `i32` per element. The dispatch throughput approximates packed-input reads plus output writes.

A literal Rust `union` payload was attempted first, but current cuda-oxide lowered the union field access incorrectly in PTX for this kernel shape. The working prototype uses explicit payload words instead, which is still close to the production packed-plan idea and avoids a large by-value plan argument.

## What this does not prove yet

This does not yet answer the production integration questions:

- why embedded artifact discovery failed for this nested standalone package,
- whether `cuda-oxide` device artifacts can be loaded through Vortex's existing `cudarc`-based `KernelLoader`,
- whether `cuda-core` buffers/streams can interoperate cleanly with existing `cudarc` buffers/streams,
- how generated PTX quality compares with `nvcc` for current kernels,
- whether cuda-oxide can support Rust `union` payloads correctly for plan parsing,
- how to port complex kernels such as `dynamic_dispatch.cu`,
- whether CUB/nvCOMP/Arrow C Device ABI bridges can or should change.

## Suggested next experiments

1. Add more tagged-payload dispatch variants matching `dynamic_dispatch_cuda.rs` cases.
2. Compare PTX/SASS register count and memory instructions against `dynamic_dispatch.cu`.
3. Add a second FoR kernel with separate input/output buffers, matching `for_in_out_*` in `for.cu`.
4. Benchmark against the existing `vortex-cuda` FoR and dynamic-dispatch benchmarks.
5. Try sharing raw CUDA stream/device pointers with the current `cudarc` path.
6. If runtime interop is viable, add an opt-in `cuda-oxide` launch path behind an experimental feature.
