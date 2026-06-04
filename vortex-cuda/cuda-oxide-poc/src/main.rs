//! Minimal cuda-oxide prototype for a Vortex-style frame-of-reference decode.
//!
//! This proves that a CUDA kernel can be written in Rust, compiled to PTX by
//! cuda-oxide, launched from Rust, validated against CPU output, and timed with
//! CUDA events.
//!
//! Note on dynamic-dispatch prototyping: avoid passing a large by-value "mega
//! plan" to kernels. That shape can inflate register pressure. A closer shape to
//! Vortex's production dynamic dispatch is a compact tag plus variant payloads
//! loaded from device memory, or separate specialized prototype kernels per case.

use std::ffi::c_void;

use cuda_core::CudaContext;
use cuda_core::DeviceBuffer;
use cuda_core::LaunchConfig;
use cuda_device::DisjointSlice;
use cuda_device::cuda_module;
use cuda_device::kernel;
use cuda_device::thread;

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
enum SimpleSourceKind {
    BitPackedU32 = 1,
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
enum SimpleScalarKind {
    None = 0,
    FoRU32 = 1,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SimpleDispatchPlan {
    // Store tags as raw u32 for device ABI stability. cuda-oxide currently
    // lowers `#[repr(u32)] enum` struct fields as byte-sized in this path.
    source_kind: u32,
    scalar0_kind: u32,

    // SourceOp::BitPackedU32 payload:
    //   source_payload0 = packed_ptr
    //   source_payload1 = bit_width
    source_payload0: u64,
    source_payload1: u64,

    // ScalarOp::FoRU32 payload:
    //   scalar0_payload0 = reference
    scalar0_payload0: u64,
}

#[derive(Clone, Copy, Debug)]
struct BitPackedU32Payload {
    packed_ptr: u64,
    bit_width: u32,
}

#[derive(Clone, Copy, Debug)]
struct FoRU32Payload {
    reference: u32,
}

// SAFETY: `SimpleDispatchPlan` is POD: raw tags plus plain integer/device-pointer payloads.
unsafe impl cuda_core::DeviceCopy for SimpleDispatchPlan {}

#[cuda_module]
mod kernels {
    use super::*;

    /// Rust version of the simple integer FoR CUDA kernel in
    /// `vortex-cuda/kernels/src/for.cu`.
    ///
    /// Existing CUDA shape:
    /// `values[i] = values[i] + reference`
    #[kernel]
    pub fn for_in_place_i32(reference: i32, mut values: DisjointSlice<i32>) {
        let idx = thread::index_1d();

        if let Some(value) = values.get_mut(idx) {
            *value += reference;
        }
    }

    #[kernel]
    pub fn simple_dispatch_u32(plan: *const SimpleDispatchPlan, mut output: DisjointSlice<u32>) {
        let group_idx = thread::index_1d().get() as u64;
        let base = group_idx * 4;
        if base >= output.len() as u64 {
            return;
        }

        let plan = unsafe { &*plan };
        if plan.source_kind != SimpleSourceKind::BitPackedU32 as u32 {
            return;
        }

        let values = source_bitpacked_6bw_u32(base, plan.source_payload0, plan.source_payload1);
        let values = if plan.scalar0_kind == SimpleScalarKind::FoRU32 as u32 {
            scalar_for_u32x4(values, plan.scalar0_payload0 as u32)
        } else {
            values
        };

        store_group_value(&mut output, base, 0, values.v0);
        store_group_value(&mut output, base, 1, values.v1);
        store_group_value(&mut output, base, 2, values.v2);
        store_group_value(&mut output, base, 3, values.v3);
    }

    #[derive(Clone, Copy)]
    struct U32x4 {
        v0: u32,
        v1: u32,
        v2: u32,
        v3: u32,
    }

    fn source_bitpacked_6bw_u32(base: u64, packed_ptr: u64, bit_width: u64) -> U32x4 {
        if bit_width != 6 {
            return U32x4 {
                v0: 0,
                v1: 0,
                v2: 0,
                v3: 0,
            };
        }

        // SourceOp::BitPackedU32: decode four residuals into registers.
        // Four 6-bit values occupy 24 bits. For any group alignment, two packed
        // u32 words contain all four values.
        let bit_pos = base * 6;
        let word_idx = bit_pos >> 5;
        let bit_off = (bit_pos & 31) as u32;
        let bits = ((load_u32(packed_ptr, word_idx) as u64)
            | ((load_u32(packed_ptr, word_idx + 1) as u64) << 32))
            >> bit_off;

        U32x4 {
            v0: (bits as u32) & 0x3f,
            v1: ((bits >> 6) as u32) & 0x3f,
            v2: ((bits >> 12) as u32) & 0x3f,
            v3: ((bits >> 18) as u32) & 0x3f,
        }
    }

    fn scalar_for_u32x4(values: U32x4, reference: u32) -> U32x4 {
        U32x4 {
            v0: values.v0 + reference,
            v1: values.v1 + reference,
            v2: values.v2 + reference,
            v3: values.v3 + reference,
        }
    }

    fn load_u32(ptr: u64, idx: u64) -> u32 {
        unsafe { *((ptr as *const u32).add(idx as usize)) }
    }

    fn store_group_value(output: &mut DisjointSlice<u32>, base: u64, lane: u32, value: u32) {
        let out_idx = base + lane as u64;
        if out_idx < output.len() as u64 {
            unsafe {
                *output.get_unchecked_mut(out_idx as usize) = value;
            }
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.default_stream();

    // `cargo oxide` currently generates this PTX next to the package manifest.
    // Loading it directly keeps the POC runnable even when embedded artifact discovery
    // fails for this standalone nested package.
    let module = ctx
        .load_module_from_file("vortex_cuda_oxide_poc.ptx")
        .expect("load generated cuda-oxide PTX");
    let function = module
        .load_function("for_in_place_i32")
        .expect("load for_in_place_i32 kernel");
    let dispatch_function = module
        .load_function("simple_dispatch_u32")
        .expect("load simple_dispatch_u32 kernel");

    let target = std::env::var("VORTEX_CUDA_OXIDE_POC").unwrap_or_else(|_| "all".to_string());

    match target.as_str() {
        "for" => {
            validate_for_kernel(&stream, &function);
            benchmark_for_kernel(&ctx, &stream, &function);
        }
        "dispatch" => {
            validate_simple_dispatch_for_bitpacked_u32(&stream, &dispatch_function);
            benchmark_simple_dispatch_for_bitpacked_u32(&ctx, &stream, &dispatch_function);
        }
        "all" => {
            validate_for_kernel(&stream, &function);
            benchmark_for_kernel(&ctx, &stream, &function);
            validate_simple_dispatch_for_bitpacked_u32(&stream, &dispatch_function);
            benchmark_simple_dispatch_for_bitpacked_u32(&ctx, &stream, &dispatch_function);
        }
        other => panic!("unknown target '{other}', expected one of: for, dispatch, all"),
    }
}

fn validate_for_kernel(stream: &cuda_core::CudaStream, function: &cuda_core::CudaFunction) {
    let reference = 1_000_i32;
    let input: Vec<i32> = (0..4096).map(|i| i % 251).collect();
    let expected: Vec<i32> = input.iter().map(|value| value + reference).collect();
    let values = DeviceBuffer::from_host(stream, &input).expect("copy host input to device");

    launch_for_kernel(stream, function, reference, &values);

    let actual = values
        .to_host_vec(stream)
        .expect("copy output back to host");
    assert_eq!(actual, expected);

    println!("cuda-oxide FoR POC passed for {} i32 values", input.len());
}

fn benchmark_for_kernel(
    ctx: &std::sync::Arc<CudaContext>,
    stream: &cuda_core::CudaStream,
    function: &cuda_core::CudaFunction,
) {
    const BENCH_LEN: usize = 16 * 1024 * 1024;
    const WARMUP_LAUNCHES: usize = 20;
    const TIMED_LAUNCHES: usize = 200;

    let reference = 1_i32;
    let input = vec![0_i32; BENCH_LEN];
    let values = DeviceBuffer::from_host(stream, &input).expect("copy benchmark input to device");

    for _ in 0..WARMUP_LAUNCHES {
        launch_for_kernel(stream, function, reference, &values);
    }
    stream.synchronize().expect("synchronize warmup launches");

    let start = ctx
        .new_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
        .expect("create start event");
    let stop = ctx
        .new_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
        .expect("create stop event");

    start.record(stream).expect("record benchmark start");
    for _ in 0..TIMED_LAUNCHES {
        launch_for_kernel(stream, function, reference, &values);
    }
    stop.record(stream).expect("record benchmark stop");

    let total_ms = start.elapsed_ms(&stop).expect("measure benchmark time");
    let avg_us = total_ms * 1_000.0 / TIMED_LAUNCHES as f32;

    // In-place add reads and writes one i32 per element.
    let bytes_per_launch = BENCH_LEN * std::mem::size_of::<i32>() * 2;
    let total_gib = bytes_per_launch as f64 * TIMED_LAUNCHES as f64 / 1024.0_f64.powi(3);
    let gib_per_s = total_gib / (total_ms as f64 / 1_000.0);

    println!(
        "cuda-oxide FoR benchmark: len={} launches={} total={:.3} ms avg={:.3} us throughput={:.2} GiB/s",
        BENCH_LEN, TIMED_LAUNCHES, total_ms, avg_us, gib_per_s
    );
}

fn validate_simple_dispatch_for_bitpacked_u32(
    stream: &cuda_core::CudaStream,
    function: &cuda_core::CudaFunction,
) {
    let len = 4096;
    let bit_width = 6;
    let reference = 100_000_u32;
    let residuals: Vec<u32> = (0..len).map(|i| (i % 64) as u32).collect();
    let expected: Vec<u32> = residuals.iter().map(|value| value + reference).collect();

    let packed = pack_lsb_u32(&residuals, bit_width);
    let packed = DeviceBuffer::from_host(stream, &packed).expect("copy packed residuals");
    let output = DeviceBuffer::<u32>::zeroed(stream, len).expect("alloc dispatch output");
    let source = BitPackedU32Payload {
        packed_ptr: packed.cu_deviceptr(),
        bit_width,
    };
    let scalar = FoRU32Payload { reference };
    let plan = SimpleDispatchPlan {
        source_kind: SimpleSourceKind::BitPackedU32 as u32,
        scalar0_kind: SimpleScalarKind::FoRU32 as u32,
        source_payload0: source.packed_ptr,
        source_payload1: source.bit_width as u64,
        scalar0_payload0: scalar.reference as u64,
    };
    let plan = DeviceBuffer::from_host(stream, &[plan]).expect("copy dispatch plan");

    launch_simple_dispatch_u32(stream, function, &plan, &output);

    let actual = output.to_host_vec(stream).expect("copy dispatch output");
    if actual != expected {
        let first = actual
            .iter()
            .zip(expected.iter())
            .position(|(actual, expected)| actual != expected)
            .unwrap();
        panic!(
            "dispatch mismatch at {first}: actual={} expected={} actual_window={:?} expected_window={:?}",
            actual[first],
            expected[first],
            &actual[first..(first + 16).min(actual.len())],
            &expected[first..(first + 16).min(expected.len())]
        );
    }

    println!("cuda-oxide tagged-payload dispatch POC passed for {len} u32 values");
}

fn benchmark_simple_dispatch_for_bitpacked_u32(
    ctx: &std::sync::Arc<CudaContext>,
    stream: &cuda_core::CudaStream,
    function: &cuda_core::CudaFunction,
) {
    const BENCH_LEN: usize = 16 * 1024 * 1024;
    const BIT_WIDTH: u32 = 6;
    const REFERENCE: u32 = 100_000;
    const WARMUP_LAUNCHES: usize = 20;
    const TIMED_LAUNCHES: usize = 200;

    let residuals: Vec<u32> = (0..BENCH_LEN).map(|i| (i % 64) as u32).collect();
    let packed = pack_lsb_u32(&residuals, BIT_WIDTH);
    let packed = DeviceBuffer::from_host(stream, &packed).expect("copy benchmark packed residuals");
    let output = DeviceBuffer::<u32>::zeroed(stream, BENCH_LEN).expect("alloc benchmark output");
    let source = BitPackedU32Payload {
        packed_ptr: packed.cu_deviceptr(),
        bit_width: BIT_WIDTH,
    };
    let scalar = FoRU32Payload {
        reference: REFERENCE,
    };
    let plan = SimpleDispatchPlan {
        source_kind: SimpleSourceKind::BitPackedU32 as u32,
        scalar0_kind: SimpleScalarKind::FoRU32 as u32,
        source_payload0: source.packed_ptr,
        source_payload1: source.bit_width as u64,
        scalar0_payload0: scalar.reference as u64,
    };
    let plan = DeviceBuffer::from_host(stream, &[plan]).expect("copy benchmark dispatch plan");

    for _ in 0..WARMUP_LAUNCHES {
        launch_simple_dispatch_u32(stream, function, &plan, &output);
    }
    stream.synchronize().expect("synchronize dispatch warmup");

    let start = ctx
        .new_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
        .expect("create dispatch start event");
    let stop = ctx
        .new_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))
        .expect("create dispatch stop event");

    start
        .record(stream)
        .expect("record dispatch benchmark start");
    for _ in 0..TIMED_LAUNCHES {
        launch_simple_dispatch_u32(stream, function, &plan, &output);
    }
    stop.record(stream).expect("record dispatch benchmark stop");

    let total_ms = start.elapsed_ms(&stop).expect("measure dispatch benchmark");
    let avg_us = total_ms * 1_000.0 / TIMED_LAUNCHES as f32;

    // Approximate bytes: packed input read plus one u32 output write.
    let packed_bytes_per_launch = BENCH_LEN * BIT_WIDTH as usize / 8;
    let output_bytes_per_launch = BENCH_LEN * std::mem::size_of::<u32>();
    let bytes_per_launch = packed_bytes_per_launch + output_bytes_per_launch;
    let total_gib = bytes_per_launch as f64 * TIMED_LAUNCHES as f64 / 1024.0_f64.powi(3);
    let gib_per_s = total_gib / (total_ms as f64 / 1_000.0);

    println!(
        "cuda-oxide tagged-payload dispatch benchmark: case=for_bitpacked_u32 len={} launches={} total={:.3} ms avg={:.3} us throughput={:.2} GiB/s",
        BENCH_LEN, TIMED_LAUNCHES, total_ms, avg_us, gib_per_s
    );
}

fn launch_for_kernel(
    stream: &cuda_core::CudaStream,
    function: &cuda_core::CudaFunction,
    reference: i32,
    values: &DeviceBuffer<i32>,
) {
    let config = LaunchConfig::for_num_elems(values.len() as u32);
    let mut reference_arg = reference;
    let mut values_ptr = values.cu_deviceptr();
    let mut values_len = values.len() as u64;
    let mut kernel_params = [
        &mut reference_arg as *mut i32 as *mut c_void,
        &mut values_ptr as *mut _ as *mut c_void,
        &mut values_len as *mut u64 as *mut c_void,
    ];

    // SAFETY: `function` is loaded from the PTX emitted for `for_in_place_i32`, and
    // the argument list matches the generated PTX signature: `(i32, ptr, u64)`.
    unsafe {
        cuda_core::launch_kernel_on_stream(
            function,
            config.grid_dim,
            config.block_dim,
            config.shared_mem_bytes,
            stream,
            &mut kernel_params,
        )
        .expect("launch for_in_place_i32");
    }
}

fn launch_simple_dispatch_u32(
    stream: &cuda_core::CudaStream,
    function: &cuda_core::CudaFunction,
    plan: &DeviceBuffer<SimpleDispatchPlan>,
    output: &DeviceBuffer<u32>,
) {
    let config = LaunchConfig::for_num_elems(output.len().div_ceil(4) as u32);
    let mut plan_ptr = plan.cu_deviceptr();
    let mut output_ptr = output.cu_deviceptr();
    let mut output_len = output.len() as u64;
    let mut kernel_params = [
        &mut plan_ptr as *mut _ as *mut c_void,
        &mut output_ptr as *mut _ as *mut c_void,
        &mut output_len as *mut u64 as *mut c_void,
    ];

    // SAFETY: `function` is loaded from the PTX emitted for `simple_dispatch_u32`, and
    // the argument list matches the generated PTX signature: `(plan_ptr, output_ptr, len)`.
    unsafe {
        cuda_core::launch_kernel_on_stream(
            function,
            config.grid_dim,
            config.block_dim,
            config.shared_mem_bytes,
            stream,
            &mut kernel_params,
        )
        .expect("launch simple_dispatch_u32");
    }
}

fn pack_lsb_u32(values: &[u32], bit_width: u32) -> Vec<u32> {
    assert!((1..=32).contains(&bit_width));
    let total_bits = values.len() * bit_width as usize;
    // One guard word lets the device unpacker read `word_idx + 1` at boundary crossings.
    let mut packed = vec![0_u32; total_bits.div_ceil(32) + 1];
    let mask = if bit_width == 32 {
        u32::MAX
    } else {
        (1_u32 << bit_width) - 1
    };

    for (idx, value) in values.iter().copied().enumerate() {
        let value = value & mask;
        let bit_pos = idx * bit_width as usize;
        let word_idx = bit_pos / 32;
        let bit_off = (bit_pos % 32) as u32;
        packed[word_idx] |= value << bit_off;
        let bits_in_lo = 32 - bit_off;
        if bits_in_lo < bit_width {
            packed[word_idx + 1] |= value >> bits_in_lo;
        }
    }

    packed
}
