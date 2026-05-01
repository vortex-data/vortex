// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for DateTimeParts decoding.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

mod bench_config;
mod timed_launch_strategy;

use std::mem::size_of;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use futures::executor::block_on;
use vortex::array::IntoArray;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::encodings::datetime_parts::DateTimeParts;
use vortex::encodings::datetime_parts::DateTimePartsArray;
use vortex::error::VortexExpect;
use vortex::extension::datetime::TimeUnit;
use vortex::extension::datetime::Timestamp;
use vortex::session::VortexSession;
use vortex_cuda::CudaDispatchMode;
use vortex_cuda::CudaSession;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

use crate::timed_launch_strategy::TimedLaunchStrategy;

fn make_datetimeparts_array(len: usize, time_unit: TimeUnit) -> DateTimePartsArray {
    let days: Vec<i16> = (0..len).map(|i| (i / 1000) as i16).collect();
    let days_arr = PrimitiveArray::new(Buffer::from(days), Validity::NonNullable).into_array();
    let seconds_arr = ConstantArray::new(0i8, len).into_array();
    let subseconds_arr = ConstantArray::new(0i8, len).into_array();

    let dtype = DType::Extension(Timestamp::new(time_unit, Nullability::NonNullable).erased());

    DateTimeParts::try_new(dtype, days_arr, seconds_arr, subseconds_arr)
        .vortex_expect("Failed to create DateTimePartsArray")
}

fn benchmark_datetimeparts(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda/datetimeparts");

    for &(len, len_str) in bench_config::BENCH_SIZES {
        group.throughput(Throughput::Bytes((len * size_of::<i64>()) as u64));

        let (time_unit, unit_str) = (TimeUnit::Milliseconds, "ms");
        let dtp_array = make_datetimeparts_array(len, time_unit);

        group.bench_with_input(
            BenchmarkId::new(unit_str, len_str),
            &dtp_array,
            |b, dtp_array| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();

                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context")
                        .with_dispatch_mode(CudaDispatchMode::StandaloneOnly)
                        .with_launch_strategy(Arc::new(timed));

                    for _ in 0..iters {
                        // block on immediately here
                        block_on(dtp_array.clone().into_array().execute_cuda(&mut cuda_ctx))
                            .unwrap();
                    }

                    Duration::from_nanos(timer.load(Ordering::Relaxed))
                });
            },
        );
    }

    group.finish();
}

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config();
    targets = benchmark_datetimeparts
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
