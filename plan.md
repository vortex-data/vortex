# vortex-metal Implementation Plan

This document outlines the implementation plan for `vortex-metal`, a crate analogous to `vortex-cuda` that enables GPU-accelerated array execution on Apple Silicon using the Metal framework.

## Overview

The `vortex-metal` crate will mirror the architecture of `vortex-cuda`, providing:
- A `MetalDeviceBuffer` that implements the `DeviceBuffer` trait
- Session and execution context types for managing Metal resources
- Kernel executors for Vortex array encodings
- Metal shader equivalents to the CUDA kernels

Primary Rust binding: [`objc2-metal`](https://docs.rs/objc2-metal/0.3.2/objc2_metal/) (v0.3.2+)

---

## 1. DeviceBuffer Implementation

### Question: Do we need a new `DeviceBuffer` variant?

**Yes.** The existing `DeviceBuffer` trait (defined in `vortex-array/src/buffer.rs`) is designed to be backend-agnostic:

```rust
pub trait DeviceBuffer: 'static + Send + Sync + Debug + DynEq + DynHash {
    fn as_any(&self) -> &dyn Any;
    fn len(&self) -> usize;
    fn alignment(&self) -> Alignment;
    fn copy_to_host_sync(&self, alignment: Alignment) -> VortexResult<ByteBuffer>;
    fn copy_to_host(&self, alignment: Alignment) -> VortexResult<BoxFuture<'static, VortexResult<ByteBuffer>>>;
    fn slice(&self, range: Range<usize>) -> Arc<dyn DeviceBuffer>;
    fn aligned(self: Arc<Self>, alignment: Alignment) -> VortexResult<Arc<dyn DeviceBuffer>>;
}
```

We need `MetalDeviceBuffer` to wrap Metal's `MTLBuffer` type (from objc2-metal: `Retained<ProtocolObject<dyn MTLBuffer>>`).

### MetalDeviceBuffer Design

```rust
/// A DeviceBuffer wrapping a Metal GPU allocation.
#[derive(Clone)]
pub struct MetalDeviceBuffer {
    /// The underlying Metal buffer (reference-counted by objc2)
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Offset in bytes from the start of the allocation
    offset: usize,
    /// Length in bytes
    len: usize,
    /// Minimum required alignment
    alignment: Alignment,
    /// Reference to the command queue for scheduling copies
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}
```

### Key Implementation Details

| Aspect | CUDA (cudarc) | Metal (objc2-metal) |
|--------|---------------|---------------------|
| **Buffer Type** | `CudaSlice<T>` | `Retained<ProtocolObject<dyn MTLBuffer>>` |
| **Device Pointer** | `CUdeviceptr` (u64) | `buffer.contents()` returns `*mut c_void` |
| **Memory Mode** | Dedicated GPU memory | Shared memory (unified on Apple Silicon) |
| **Async Copy** | `cuMemcpyDtoHAsync_v2` | Blit command encoder + completion handler |
| **Synchronization** | Stream callbacks | Command buffer completion handlers |

### Metal Buffer Storage Modes

Metal on Apple Silicon uses **unified memory**, so buffers can be:
- `MTLStorageModeShared` - CPU and GPU can both access (default, zero-copy possible)
- `MTLStorageModePrivate` - GPU-only, requires explicit copies
- `MTLStorageModeManaged` - Explicit sync required (macOS only, not on iOS)

**Recommendation**: Start with `MTLStorageModeShared` for simplicity. This allows zero-copy access from both CPU and GPU. If performance profiling shows issues with cache coherency, consider `MTLStorageModePrivate` with explicit blits.

### Slice Implementation

Metal buffers don't support native slicing like CUDA views. Options:
1. **Track offset/length** (like `CudaDeviceBuffer` does) - buffers share the underlying allocation
2. **Create new buffer with `newBufferWithBytesNoCopy`** - would require careful lifetime management

**Recommendation**: Follow CUDA's approach with offset/length tracking in `MetalDeviceBuffer`.

---

## 2. New Types Mirroring vortex-cuda

### Type Mapping

| vortex-cuda Type | vortex-metal Equivalent | Purpose |
|------------------|-------------------------|---------|
| `CudaSession` | `MetalSession` | Holds device, command queue, kernel registry |
| `CudaExecutionCtx` | `MetalExecutionCtx` | Per-execution context with command buffer |
| `CudaExecute` | `MetalExecute` | Trait for GPU-accelerated array execution |
| `VortexCudaStream` | `MetalCommandBuffer` | Work submission unit |
| `VortexCudaStreamPool` | `MetalCommandQueuePool` | Reusable command queues |
| `KernelLoader` | `MetalLibraryLoader` | Loads/caches compiled Metal libraries |
| `CudaKernelEvents` | `MetalKernelEvents` | Timing information |
| `LaunchStrategy` | `MetalLaunchStrategy` | Kernel launch configuration |

### MetalSession

```rust
pub struct MetalSession {
    /// The Metal device (typically system default)
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Command queue for work submission
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Registry of kernel implementations
    kernels: Arc<DashMap<ArrayId, &'static dyn MetalExecute>>,
    /// Library loader with caching
    library_loader: Arc<MetalLibraryLoader>,
}
```

### MetalExecutionCtx

```rust
pub struct MetalExecutionCtx {
    /// Current command buffer for this execution
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    /// CPU execution context for fallback
    ctx: ExecutionCtx,
    /// Metal session reference
    metal_session: MetalSession,
    /// Launch strategy
    strategy: Arc<dyn MetalLaunchStrategy>,
}
```

### MetalExecute Trait

```rust
#[async_trait]
pub trait MetalExecute: 'static + Send + Sync + Debug {
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut MetalExecutionCtx,
    ) -> VortexResult<Canonical>;
}
```

### MetalLibraryLoader

Unlike CUDA which loads PTX files, Metal compiles shader source at runtime or uses pre-compiled metallib files.

```rust
pub struct MetalLibraryLoader {
    /// Cache of compiled Metal libraries
    libraries: DashMap<String, Retained<ProtocolObject<dyn MTLLibrary>>>,
    /// Cache of pipeline states
    pipelines: DashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}
```

### Shader Compilation Strategy

Options:
1. **Runtime compilation** - Ship `.metal` source files, compile with `newLibraryWithSource:options:error:`
2. **Ahead-of-time compilation** - Use `xcrun metal` and `xcrun metallib` in build.rs, ship `.metallib`
3. **Hybrid** - Ship metallib with fallback to runtime compilation

**Recommendation**: Start with runtime compilation for development flexibility. Add AOT compilation in build.rs for release builds.

---

## 3. Implementation Plan

### Phase 1: Core Infrastructure

**Goal**: Establish the foundational types and a working end-to-end test.

#### 3.1.1 Create Crate Structure

```
vortex-metal/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── device_buffer.rs      # MetalDeviceBuffer
│   ├── session.rs            # MetalSession, MetalSessionExt
│   ├── executor.rs           # MetalExecutionCtx, MetalExecute trait
│   ├── command_buffer.rs     # Wrapper around MTLCommandBuffer
│   ├── library_loader.rs     # MetalLibraryLoader
│   └── kernel/
│       └── mod.rs
└── shaders/
    ├── common.metal          # Shared types/utilities
    └── for.metal             # First kernel: Frame-of-Reference
```

#### 3.1.2 Cargo.toml Dependencies

```toml
[package]
name = "vortex-metal"
# ...

[dependencies]
objc2 = "0.6"
objc2-foundation = { version = "0.3", features = ["NSError", "NSString"] }
objc2-metal = { version = "0.3", features = ["all"] }
objc2-quartz-core = { version = "0.3", features = ["CAMetalLayer"] }
block2 = "0.6"  # For completion handlers
async-trait = { workspace = true }
futures = { workspace = true }
vortex = { workspace = true }
vortex-array = { workspace = true }
# ... other common deps
```

#### 3.1.3 MetalDeviceBuffer Implementation

Implement the `DeviceBuffer` trait for Metal buffers with:
- Proper handling of shared memory semantics
- Slice tracking via offset/length
- Async copy using blit encoder with completion handler
- Hash/Eq based on buffer pointer + offset + length

### Phase 2: Simple Encoding - Frame-of-Reference (FoR)

**Goal**: Implement a complete kernel to validate the architecture.

#### Why FoR First?

1. **Simple operation**: Just adds a scalar reference to each element
2. **In-place execution**: Doesn't require separate output buffer
3. **Type templating**: Tests our approach for generating multiple type variants
4. **Matches CUDA pattern**: Direct port from `for.cu`

#### 3.2.1 Metal Shader: for.metal

```metal
#include <metal_stdlib>
using namespace metal;

// Kernel configuration - must match Rust constants
constant uint ELEMENTS_PER_THREAD = 32;

template <typename T>
kernel void for_kernel(
    device T* values [[buffer(0)]],
    constant T& reference [[buffer(1)]],
    constant uint64_t& array_len [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint base_idx = gid * ELEMENTS_PER_THREAD;

    for (uint i = 0; i < ELEMENTS_PER_THREAD && (base_idx + i) < array_len; ++i) {
        values[base_idx + i] = values[base_idx + i] + reference;
    }
}

// Explicit instantiations for each type
// (Metal doesn't support extern "C" templates like CUDA)
kernel void for_u8(device uint8_t* v [[buffer(0)]], constant uint8_t& r [[buffer(1)]],
                   constant uint64_t& len [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    for_kernel(v, r, len, gid);
}

kernel void for_u16(device uint16_t* v [[buffer(0)]], constant uint16_t& r [[buffer(1)]],
                    constant uint64_t& len [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    for_kernel(v, r, len, gid);
}

kernel void for_u32(device uint32_t* v [[buffer(0)]], constant uint32_t& r [[buffer(1)]],
                    constant uint64_t& len [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    for_kernel(v, r, len, gid);
}

kernel void for_u64(device uint64_t* v [[buffer(0)]], constant uint64_t& r [[buffer(1)]],
                    constant uint64_t& len [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    for_kernel(v, r, len, gid);
}

kernel void for_i8(device int8_t* v [[buffer(0)]], constant int8_t& r [[buffer(1)]],
                   constant uint64_t& len [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    for_kernel(v, r, len, gid);
}

kernel void for_i16(device int16_t* v [[buffer(0)]], constant int16_t& r [[buffer(1)]],
                    constant uint64_t& len [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    for_kernel(v, r, len, gid);
}

kernel void for_i32(device int32_t* v [[buffer(0)]], constant int32_t& r [[buffer(1)]],
                    constant uint64_t& len [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    for_kernel(v, r, len, gid);
}

kernel void for_i64(device int64_t* v [[buffer(0)]], constant int64_t& r [[buffer(1)]],
                    constant uint64_t& len [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    for_kernel(v, r, len, gid);
}
```

#### 3.2.2 FoRExecutor Implementation

```rust
#[derive(Debug)]
pub(crate) struct FoRExecutor;

#[async_trait]
impl MetalExecute for FoRExecutor {
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut MetalExecutionCtx,
    ) -> VortexResult<Canonical> {
        let for_array: FoRArray = array.try_into()?;

        match_each_native_simd_ptype!(for_array.ptype(), |P| {
            decode_for::<P>(for_array, ctx).await
        })
    }
}

async fn decode_for<P: NativePType>(
    array: FoRArray,
    ctx: &mut MetalExecutionCtx,
) -> VortexResult<Canonical> {
    let array_len = array.encoded().len();
    let reference: P = array.reference_scalar().try_into()?;

    // Execute child and ensure on device
    let canonical = array.encoded().clone().execute_metal(ctx).await?;
    let primitive = canonical.into_primitive();
    let PrimitiveArrayParts { buffer, validity, .. } = primitive.into_parts();

    let device_buffer = ctx.ensure_on_device(buffer).await?;

    // Load kernel function
    let kernel_name = format!("for_{}", P::PTYPE.to_string().to_lowercase());
    let pipeline = ctx.load_pipeline("for", &kernel_name)?;

    // Create compute command encoder
    let encoder = ctx.command_buffer().computeCommandEncoder()?;
    encoder.setComputePipelineState(&pipeline);
    encoder.setBuffer_offset_atIndex(device_buffer.metal_buffer(), 0, 0);

    // Set reference as constant buffer
    let ref_bytes = reference.to_le_bytes();
    encoder.setBytes_length_atIndex(ref_bytes.as_ptr().cast(), ref_bytes.len(), 1);

    // Set array length
    let len_bytes = (array_len as u64).to_le_bytes();
    encoder.setBytes_length_atIndex(len_bytes.as_ptr().cast(), 8, 2);

    // Calculate grid and threadgroup sizes
    let thread_execution_width = pipeline.threadExecutionWidth();
    let threads_per_group = MTLSize { width: thread_execution_width, height: 1, depth: 1 };
    let num_threadgroups = (array_len as u64).div_ceil(thread_execution_width * 32);
    let grid_size = MTLSize { width: num_threadgroups, height: 1, depth: 1 };

    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threads_per_group);
    encoder.endEncoding();

    // Wait for completion
    ctx.wait_for_completion().await?;

    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        device_buffer.into_buffer_handle(),
        P::PTYPE,
        validity,
    )))
}
```

### Phase 3: Test Cases

#### 3.3.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    use rstest::rstest;

    fn make_for_array<T: NativePType>(input: Vec<T>, reference: T) -> FoRArray { ... }

    #[rstest]
    #[case::u8(make_for_array((0..2050).map(|i| (i % 246) as u8).collect(), 10u8))]
    #[case::u16(make_for_array((0..2050).map(|i| (i % 2050) as u16).collect(), 1000u16))]
    #[case::u32(make_for_array((0..2050).map(|i| (i % 2050) as u32).collect(), 100000u32))]
    #[case::u64(make_for_array((0..2050).map(|i| (i % 2050) as u64).collect(), 1000000u64))]
    #[tokio::test]
    async fn test_metal_for_decompression(#[case] for_array: FoRArray) -> VortexResult<()> {
        let session = MetalSession::default();
        let mut ctx = session.create_execution_ctx()?;

        let cpu_result = for_array.to_canonical()?;
        let gpu_result = FoRExecutor
            .execute(for_array.to_array(), &mut ctx)
            .await?
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);
        Ok(())
    }
}
```

#### 3.3.2 Integration Tests

- Test buffer allocation and deallocation
- Test host-to-device and device-to-host copies
- Test slicing behavior
- Test multiple sequential kernel dispatches
- Test concurrent command buffers

### Phase 4: Additional Encodings

Once Phase 2 is validated, implement additional encodings in priority order:

1. **ZigZag** - Simple bit manipulation, good test of signed/unsigned handling
2. **Dict** - Tests gather pattern and multi-buffer kernels
3. **BitPacked** - Tests more complex unpacking logic
4. **RunEnd** - Tests scan/prefix-sum patterns
5. **Constant** - Trivial kernel, tests broadcast pattern

### Phase 5: Advanced Features

1. **Command buffer pooling** - Reuse command buffers for throughput
2. **Triple buffering** - Pipeline CPU/GPU work
3. **Shared events** - Cross-queue synchronization if needed
4. **Performance counters** - GPU timing via Metal's counter sampling
5. **AOT shader compilation** - build.rs integration for metallib generation

---

## 4. Key Differences from CUDA

| Aspect | CUDA | Metal |
|--------|------|-------|
| **Memory Model** | Discrete GPU memory, explicit copies | Unified memory (Apple Silicon) |
| **Shader Language** | CUDA C++ → PTX | Metal Shading Language |
| **Compilation** | nvcc at build time | Runtime or xcrun at build time |
| **Streams** | CUDA streams (ordered queues) | Command buffers (committed units) |
| **Synchronization** | Stream callbacks, events | Completion handlers, shared events |
| **Kernel Launch** | Grid/block dimensions | Threadgroups/threads per threadgroup |
| **Types** | C++ templates with `extern "C"` | No external linkage for templates |

### Memory Considerations

Apple Silicon's unified memory means:
- **No explicit H2D/D2H copies needed** for `MTLStorageModeShared` buffers
- `buffer.contents()` returns a CPU-accessible pointer
- GPU may need `synchronize()` calls for cache coherency
- This is fundamentally different from CUDA's copy-based model

However, we should still model the API similarly to CUDA for:
- Future support for discrete AMD GPUs on Mac (external GPUs)
- Consistency with vortex-cuda API
- Potential optimizations with `MTLStorageModePrivate`

---

## 5. Open Questions

1. **Unified memory optimization**: Should `copy_to_host` on Apple Silicon be a no-op that just returns a view?
2. **Shader source distribution**: Ship .metal files or embed as string literals?
3. **Feature gating**: Should this be `#[cfg(target_os = "macos")]` or also support iOS?
4. **Half-precision**: Metal has excellent f16 support - worth prioritizing?
5. **Command buffer granularity**: One per execution or batch multiple kernels?

---

## 6. Success Criteria

Phase 1 is complete when:
- [x] `MetalDeviceBuffer` passes all `DeviceBuffer` trait requirements
- [x] `MetalSession` can initialize and detect the default Metal device
- [x] Basic buffer allocation and copy roundtrip works

Phase 2 is complete when:
- [x] FoR kernel compiles and loads successfully
- [x] All integer types (i8-i64, u8-u64) pass correctness tests
- [x] Performance is within 2x of CPU execution (sanity check)

Full implementation is complete when:
- [ ] All encodings from vortex-cuda have Metal equivalents
- [ ] Integration tests match vortex-cuda test coverage
- [ ] Benchmarks show meaningful speedup for large arrays

---

## 7. Implementation Results

### Phase 1 & 2: Completed ✓

The core infrastructure and FoR kernel have been implemented successfully.

#### Files Created

```
vortex-metal/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── device_buffer.rs      # MetalDeviceBuffer implementation
│   ├── session.rs            # MetalSession, MetalSessionExt
│   ├── executor.rs           # MetalExecutionCtx, MetalExecute trait
│   ├── library_loader.rs     # MetalLibraryLoader for shader compilation
│   └── kernel/
│       ├── mod.rs
│       └── for_.rs           # FoRExecutor implementation
├── shaders/
│   └── for.metal             # FoR decompression kernel
└── benches/
    └── for_metal.rs          # Benchmarks comparing Metal vs CPU
```

#### Key Design Decisions

1. **Synchronous execution model**: Unlike vortex-cuda's async design, Metal execution is synchronous because:
   - Apple Silicon uses unified memory (no copy overhead)
   - `MTLCommandBuffer` is not `Send`, making async-trait complex
   - For FoR's simple operations, overhead of async would dominate

2. **Shared storage mode**: Using `MTLStorageModeShared` for all buffers, allowing zero-copy access from both CPU and GPU.

3. **Simplified architecture**: No separate command buffer wrapper or stream pool - single `command_buffer` field in `MetalExecutionCtx` suffices for now.

4. **Offset/length slicing**: Following CUDA's pattern of tracking offset/length rather than creating new Metal buffer views.

#### Test Results

All 8 FoR tests pass:
```
test kernel::for_::tests::test_metal_for_i16 ... ok
test kernel::for_::tests::test_metal_for_i32 ... ok
test kernel::for_::tests::test_metal_for_i64 ... ok
test kernel::for_::tests::test_metal_for_i8 ... ok
test kernel::for_::tests::test_metal_for_u16 ... ok
test kernel::for_::tests::test_metal_for_u32 ... ok
test kernel::for_::tests::test_metal_for_u64 ... ok
test kernel::for_::tests::test_metal_for_u8 ... ok
```

#### Benchmark Results

FoR decompression benchmarks (M3 Max, Apple Silicon):

| Size | Type | Metal | CPU | Notes |
|------|------|-------|-----|-------|
| 100K | u32 | 12.4 µs (30 GiB/s) | 11.9 µs (31 GiB/s) | CPU slightly faster |
| 1M | u32 | 111 µs (33.5 GiB/s) | 111 µs (33.5 GiB/s) | Parity |
| 10M | u32 | 1.17 ms (31.8 GiB/s) | 1.17 ms (31.9 GiB/s) | Parity |
| 100K | u64 | 23.1 µs (32.3 GiB/s) | 22.7 µs (32.8 GiB/s) | CPU slightly faster |
| 1M | u64 | 219 µs (34.0 GiB/s) | 215 µs (34.7 GiB/s) | CPU slightly faster |
| 10M | u64 | 2.30 ms (32.4 GiB/s) | 2.30 ms (32.4 GiB/s) | Parity |

**Analysis**: FoR decoding is a simple `value + reference` operation that is entirely memory-bound. Both Metal and CPU achieve ~30-34 GiB/s throughput, which is near memory bandwidth limits. This validates that:
1. Metal kernel launches have minimal overhead
2. Unified memory eliminates copy costs
3. For compute-bound kernels (BitPacked, Dict), Metal should show advantages

#### Implementation Notes

1. **objc2-metal API quirks**:
   - Some methods require `unsafe` blocks (e.g., `setBuffer_offset_atIndex`)
   - `NonNull` pointers required for `setBytes_length_atIndex`
   - No `features = ["all"]` - just use default features

2. **No async_trait**: Metal objects aren't `Send`, so we use synchronous execution with `commit()` + `waitUntilCompleted()`.

3. **Shader compilation**: Runtime compilation via `newLibraryWithSource_options_error` with caching in `MetalLibraryLoader`.

#### Next Steps

1. Implement ZigZag kernel (simple bit manipulation)
2. Implement Dict kernel (gather pattern)
3. Implement BitPacked kernel (compute-intensive, should show GPU advantage)
4. Add AOT shader compilation in build.rs for release builds
