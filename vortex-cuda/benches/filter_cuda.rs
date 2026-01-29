// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for GPU filtering using CUB DeviceSelect::Flagged.

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use std::ffi::c_void;
use std::mem::size_of;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaView;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::sys::CUevent_flags;
use futures::executor::block_on;
use vortex_cub::filter::CubFilterable;
use vortex_cub::filter::cudaStream_t;
use vortex_cuda::CudaDeviceBuffer;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

const BENCH_SIZES: &[(usize, &str)] = &[(1_000_000, "1M"), (10_000_000, "10M")];
const SELECTIVITIES: &[(f64, &str)] = &[(0.1, "10pct"), (0.5, "50pct"), (0.9, "90pct")];

/// Creates input data of the given length.
fn make_input_data<T: From<u8> + Clone>(len: usize) -> Vec<T> {
    (0..len).map(|i| T::from((i % 256) as u8)).collect()
}

/// Creates a packed bitmask with the given selectivity.
fn make_bitmask(len: usize, selectivity: f64) -> (Vec<u8>, usize) {
    let num_bytes = len.div_ceil(8);
    let mut packed = vec![0u8; num_bytes];
    let mut true_count = 0;

    // Calculate stride to spread selections evenly across the array.
    let stride = ((1.0 / selectivity).round() as usize).max(1);

    for idx in 0..len {
        // Select every stride-th element to spread work across threads.
        if idx % stride == 0 {
            let byte_idx = idx / 8;
            let bit_idx = idx % 8;
            packed[byte_idx] |= 1 << bit_idx;
            true_count += 1;
        }
    }

    (packed, true_count)
}

/// Runs the CUB filter kernel and returns elapsed GPU time.
#[expect(clippy::too_many_arguments)]
async fn run_filter_timed<T: CubFilterable + cudarc::driver::DeviceRepr>(
    d_input: CudaView<'_, T>,
    d_bitmask: CudaView<'_, u8>,
    d_output: &mut CudaSlice<T>,
    d_temp: &mut CudaSlice<u8>,
    d_num_selected: &mut CudaSlice<i64>,
    num_items: i64,
    temp_bytes: usize,
    cuda_ctx: &mut CudaExecutionCtx,
) -> VortexResult<Duration> {
    let stream = cuda_ctx.stream();
    let ctx = stream.context();

    let start_event = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .map_err(|e| vortex_err!("Failed to create start event: {:?}", e))?;
    start_event
        .record(stream)
        .map_err(|e| vortex_err!("Failed to record start event: {:?}", e))?;

    // Get raw pointers
    let stream_ptr = stream.cu_stream() as cudaStream_t;
    let d_input_ptr = d_input.device_ptr(stream).0 as *const T;
    let d_bitmask_ptr = d_bitmask.device_ptr(stream).0 as *const u8;
    let d_output_ptr = d_output.device_ptr_mut(stream).0 as *mut T;
    let d_temp_ptr = d_temp.device_ptr_mut(stream).0 as *mut c_void;
    let d_num_selected_ptr = d_num_selected.device_ptr_mut(stream).0 as *mut i64;

    unsafe {
        T::filter_bitmask(
            d_temp_ptr,
            temp_bytes,
            d_input_ptr,
            d_bitmask_ptr,
            0, // bit_offset
            d_output_ptr,
            d_num_selected_ptr,
            num_items,
            stream_ptr,
        )
        .map_err(|e| vortex_err!("Filter kernel execution failed: {}", e))?;
    }

    let end_event = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .map_err(|e| vortex_err!("Failed to create end event: {:?}", e))?;

    end_event
        .record(stream)
        .map_err(|e| vortex_err!("Failed to record end event: {:?}", e))?;

    let elapsed_ms = start_event
        .elapsed_ms(&end_event)
        .map_err(|e| vortex_err!("Failed to get elapsed time: {:?}", e))?;

    Ok(Duration::from_secs_f32(elapsed_ms / 1000.0))
}

/// Benchmark filter for a specific type.
fn benchmark_filter_type<T>(c: &mut Criterion, type_name: &str)
where
    T: CubFilterable + cudarc::driver::DeviceRepr + From<u8> + Clone + Send + Sync + 'static,
{
    let mut group = c.benchmark_group(format!("Filter_cuda_{type_name}"));
    group.sample_size(10);

    for (len, len_label) in BENCH_SIZES {
        for (selectivity, sel_label) in SELECTIVITIES {
            let input_data = make_input_data::<T>(*len);
            let (bitmask, true_count) = make_bitmask(*len, *selectivity);

            // Throughput based on input size (bytes read)
            group.throughput(Throughput::Bytes((len * size_of::<T>()) as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("{len_label}_{sel_label}"), true_count),
                &(input_data, bitmask, true_count),
                |b, (input_data, bitmask, true_count)| {
                    b.iter_custom(|iters| {
                        let mut cuda_ctx =
                            CudaSession::create_execution_ctx(&VortexSession::empty())
                                .vortex_expect("failed to create execution context");

                        let num_items = input_data.len() as i64;
                        #[expect(clippy::expect_used)]
                        let temp_bytes =
                            T::get_temp_size(num_items).expect("failed to get temp size");

                        let mut total_time = Duration::ZERO;

                        for _ in 0..iters {
                            // Copy input to device
                            let d_input_handle =
                                block_on(cuda_ctx.copy_to_device(input_data.clone()).unwrap())
                                    .vortex_expect("failed to copy input to device");
                            let d_input = d_input_handle
                                .as_device()
                                .as_any()
                                .downcast_ref::<CudaDeviceBuffer<T>>()
                                .unwrap();

                            // Copy bitmask to device
                            let d_bitmask_handle =
                                block_on(cuda_ctx.copy_to_device(bitmask.clone()).unwrap())
                                    .vortex_expect("failed to copy bitmask to device");
                            let d_bitmask = d_bitmask_handle
                                .as_device()
                                .as_any()
                                .downcast_ref::<CudaDeviceBuffer<u8>>()
                                .unwrap();

                            // Allocate output and temp buffers
                            let mut d_output: CudaSlice<T> = cuda_ctx
                                .device_alloc(*true_count)
                                .vortex_expect("failed to allocate output");
                            let mut d_temp: CudaSlice<u8> = cuda_ctx
                                .device_alloc(temp_bytes.max(1))
                                .vortex_expect("failed to allocate temp");
                            let mut d_num_selected: CudaSlice<i64> = cuda_ctx
                                .device_alloc(1)
                                .vortex_expect("failed to allocate num_selected");

                            let kernel_time = block_on(run_filter_timed(
                                d_input.as_view(),
                                d_bitmask.as_view(),
                                &mut d_output,
                                &mut d_temp,
                                &mut d_num_selected,
                                num_items,
                                temp_bytes,
                                &mut cuda_ctx,
                            ))
                            .vortex_expect("kernel execution failed");

                            total_time += kernel_time;
                        }

                        total_time
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark i32 filter
fn benchmark_filter_i32(c: &mut Criterion) {
    benchmark_filter_type::<i32>(c, "i32");
}

/// Benchmark i64 filter
fn benchmark_filter_i64(c: &mut Criterion) {
    benchmark_filter_type::<i64>(c, "i64");
}

/// Benchmark f64 filter
fn benchmark_filter_f64(c: &mut Criterion) {
    benchmark_filter_type::<f64>(c, "f64");
}

pub fn benchmark_filter_cuda(c: &mut Criterion) {
    benchmark_filter_i32(c);
    benchmark_filter_i64(c);
    benchmark_filter_f64(c);
}

criterion::criterion_group!(benches, benchmark_filter_cuda);

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
