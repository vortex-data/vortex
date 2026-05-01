// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for GPU filtering using CUB DeviceSelect::Flagged.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

mod bench_config;

use std::ffi::c_void;
use std::fmt::Debug;
use std::mem::size_of;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaView;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::DeviceRepr;
use cudarc::driver::sys::CUevent_flags;
use futures::executor::block_on;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::session::VortexSession;
use vortex_cub::filter::CubFilterable;
use vortex_cub::filter::cudaStream_t;
use vortex_cuda::CudaDeviceBuffer;
use vortex_cuda::CudaExecutionCtx;
use vortex_cuda::CudaSession;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

use crate::bench_config::BENCH_SIZES;
const SELECTIVITIES: &[(f64, &str)] = &[(0.1, "10%"), (0.5, "50%"), (0.9, "90%")];

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
async fn run_filter_timed<T: CubFilterable + DeviceRepr>(
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
    let (d_input_ptr, record_d_input) = d_input.device_ptr(stream);
    let (d_bitmask_ptr, record_d_bitmask) = d_bitmask.device_ptr(stream);
    let (d_output_ptr, record_d_output) = d_output.device_ptr_mut(stream);
    let (d_temp_ptr, record_d_temp) = d_temp.device_ptr_mut(stream);
    let (d_num_selected_ptr, record_d_num_selected) = d_num_selected.device_ptr_mut(stream);

    unsafe {
        T::filter_bitmask(
            d_temp_ptr as *mut c_void,
            temp_bytes,
            d_input_ptr as *const T,
            d_bitmask_ptr as *const u8,
            0, // bit_offset
            d_output_ptr as *mut T,
            d_num_selected_ptr as *mut i64,
            num_items,
            stream_ptr,
        )
        .map_err(|e| vortex_err!("Filter kernel execution failed: {}", e))?;
    }
    drop((
        record_d_input,
        record_d_bitmask,
        record_d_output,
        record_d_temp,
        record_d_num_selected,
    ));

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
    T: CubFilterable + DeviceRepr + From<u8> + Debug + Clone + Send + Sync + 'static,
{
    let mut group = c.benchmark_group(format!("cuda/filter_{type_name}"));

    for (len, len_label) in BENCH_SIZES {
        for (selectivity, sel_label) in SELECTIVITIES {
            let input_data = make_input_data::<T>(*len);
            let (bitmask, true_count) = make_bitmask(*len, *selectivity);

            // Throughput based on input size (bytes read)
            group.throughput(Throughput::Bytes((len * size_of::<T>()) as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("select/{sel_label}"), len_label),
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
                                .downcast_ref::<CudaDeviceBuffer>()
                                .unwrap();

                            // Copy bitmask to device
                            let d_bitmask_handle =
                                block_on(cuda_ctx.copy_to_device(bitmask.clone()).unwrap())
                                    .vortex_expect("failed to copy bitmask to device");
                            let d_bitmask = d_bitmask_handle
                                .as_device()
                                .as_any()
                                .downcast_ref::<CudaDeviceBuffer>()
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

/// Benchmark filter for all types.
fn benchmark_filter(c: &mut Criterion) {
    benchmark_filter_type::<i32>(c, "i32");
    benchmark_filter_type::<i64>(c, "i64");
}

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config();
    targets = benchmark_filter
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
