# CUDA Support

Vortex provides GPU-accelerated decompression and compute through the `vortex-cuda` crate. CUDA
kernels integrate with Vortex's deferred execution model -- expression trees can be shipped to
the GPU in bulk and fused into efficient kernel launches.

## CudaSession

The `CudaSession` maintains CUDA context, a registry of compiled kernels, and a pool of streams
for concurrent execution. It follows the same vtable registry pattern as other Vortex components:
encodings register their GPU kernels at session creation, and the session resolves them by
encoding ID at execution time.

```rust
let session = CudaSession::try_new()?;
let ctx = session.create_execution_ctx();
```

The session lazily loads PTX modules and caches compiled kernels. Multiple execution contexts
can share a single session.

## Execution Context

`CudaExecutionCtx` manages kernel launches and tracks CUDA events for profiling. It acquires
streams from the session's pool and records events around kernel launches to measure execution
time.

The execution context integrates with Vortex's incremental execution model. When an array is
executed on the GPU, the same child-first optimization strategy applies -- each encoding can
provide an optimized GPU kernel via `execute_parent`, or fall back to incremental execution.

## Device Buffers

`CudaDeviceBuffer` wraps GPU memory allocations with alignment guarantees matching Vortex's
buffer conventions. Device buffers can be created by:

- Allocating directly on the GPU.
- Transferring from a host buffer (the allocator handles alignment and pinned memory).
- Memory-mapping a file segment directly to the GPU (with appropriate hardware support).

The buffer handle tracks the CUDA context and ensures proper cleanup on drop.

## Kernel Architecture

CUDA kernels are written in C++ and compiled to PTX (NVIDIA's intermediate representation)
at build time. The build script invokes `nvcc` to produce PTX files that are embedded in the
binary and loaded at runtime.

Kernels typically use a fixed launch configuration optimized for Vortex's common array sizes:

- 64 threads per block (2 warps).
- 32 elements per thread.
- Grid dimensions computed from array length: `(len / 2048, 1, 1)`.

The `launch_cuda_kernel!` macro handles grid/block configuration, argument marshaling, and
event recording:

```rust
launch_cuda_kernel!(ctx, "dict_lookup_u32", len, |builder| {
    builder
        .arg(&codes_buffer)
        .arg(&values_buffer)
        .arg(&output_buffer)
});
```

## Stream Pool

The session maintains a pool of CUDA streams (default 4) for concurrent kernel execution.
Streams are allocated round-robin to execution contexts, allowing multiple kernels to execute
in parallel when they have no data dependencies.

Stream synchronization is handled automatically by the execution context -- callers can treat
execution as synchronous while the runtime pipelines independent operations.

## Supported Encodings

The following encodings have GPU-accelerated kernels:

| Encoding           | Kernel                                   |
|--------------------|------------------------------------------|
| ALP                | Floating-point decompression             |
| BitPacked          | Bit unpacking (8/16/32/64-bit variants)  |
| Dictionary         | Dictionary lookup                        |
| DecimalByteParts   | Decimal reconstruction                   |
| Frame of Reference | FoR decompression                        |
| Sequence           | Sequence expansion                       |
| ZigZag             | ZigZag decoding                          |
| ZSTD               | GPU-accelerated decompression via nvCOMP |

Additional kernels exist for filter, slice, and patch operations.

## External Libraries

Vortex integrates with NVIDIA libraries for operations that benefit from highly optimized
implementations:

**nvCOMP** -- NVIDIA's compression library provides GPU-accelerated ZSTD decompression. The
`vortex-cuda/nvcomp` crate provides Rust bindings that dynamically load `libnvcomp.so` at
runtime. Currently Linux-only (x86_64 and ARM64).

**CUB** -- NVIDIA's CUDA Unbound library provides GPU primitives. Vortex uses `DeviceSelect`
for GPU-side filtering operations. The `vortex-cuda/cub` crate compiles a thin wrapper that
is loaded at runtime.

## Build Requirements

Building with CUDA support requires:

- NVIDIA CUDA Toolkit with `nvcc` compiler.
- CUDA 12.0 or later (builds target CUDA 12.0.80 for compatibility).
- Linux (nvCOMP bindings are Linux-only).

If `nvcc` is not available at build time, the crate compiles without PTX generation and GPU
operations will fail at runtime with an appropriate error.

## Integration with Deferred Execution

GPU execution integrates with Vortex's deferred execution model described in
[Execution](execution). When a `ScalarFnArray` tree is executed on the GPU:

1. The tree is traversed to identify operations with GPU kernels.
2. Compatible subtrees are batched into a single GPU execution plan.
3. The plan is executed with kernel fusion where possible, reducing memory traffic.
4. Results are returned as device buffers that can remain on GPU for further computation.

This batching is why deferral matters for GPU performance -- eager execution would launch
many small kernels with host-device synchronization between each, while deferred execution
can fuse the entire expression tree into fewer, larger kernel launches.

## Interoperability

Vortex arrays with on-device buffer handles can be converted to Apache Arrow `DeviceArray`
format for interoperability with other GPU libraries. An `ArrayStream` of Vortex arrays
that have been executed on the GPU can be exported as a stream of Arrow DeviceArrays without
copying data back to the host.

Future work includes direct conversion to cuDF DataFrames, enabling zero-copy handoff to
RAPIDS libraries for GPU-accelerated analytics.
