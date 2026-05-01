// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for FoR decompression.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

mod bench_config;
mod timed_launch_strategy;

use std::mem::size_of;
use std::ops::Add;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::DeviceRepr;
use futures::executor::block_on;
use vortex::array::IntoArray;
use vortex::array::LEGACY_SESSION;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::dtype::NativePType;
use vortex::dtype::PType;
use vortex::encodings::fastlanes::BitPackedData;
use vortex::encodings::fastlanes::FoR;
use vortex::encodings::fastlanes::FoRArray;
use vortex::error::VortexExpect;
use vortex::scalar::Scalar;
use vortex::session::VortexSession;
use vortex_cuda::CudaDispatchMode;
use vortex_cuda::CudaSession;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

use crate::bench_config::BENCH_SIZES;
use crate::timed_launch_strategy::TimedLaunchStrategy;
const REFERENCE_VALUE: u8 = 10;

/// Creates a FoR array with the specified type and length.
fn make_for_array_typed<T>(len: usize, bp: bool) -> FoRArray
where
    T: NativePType + From<u8> + Add<Output = T>,
    Scalar: From<T>,
{
    let reference = <T as From<u8>>::from(REFERENCE_VALUE);
    let data: Vec<T> = (0..len)
        .map(|i| <T as From<u8>>::from((i % 256) as u8))
        .collect();

    let primitive_array =
        PrimitiveArray::new(Buffer::from(data), Validity::NonNullable).into_array();

    if bp && T::PTYPE != PType::U8 {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let child =
            BitPackedData::encode(&primitive_array, 8, &mut ctx).vortex_expect("failed to bitpack");
        FoR::try_new(child.into_array(), reference.into())
            .vortex_expect("failed to create FoR array")
    } else {
        FoR::try_new(primitive_array, reference.into()).vortex_expect("failed to create FoR array")
    }
}

/// Benchmark FoR decompression for a specific type.
fn benchmark_for_typed<T>(c: &mut Criterion, type_name: &str)
where
    T: NativePType + DeviceRepr + From<u8> + Add<Output = T>,
    Scalar: From<T>,
{
    let mut group = c.benchmark_group("cuda/for");

    for &(len, len_str) in BENCH_SIZES {
        group.throughput(Throughput::Bytes((len * size_of::<T>()) as u64));

        let for_array = make_for_array_typed::<T>(len, false);

        group.bench_with_input(
            BenchmarkId::new(type_name, len_str),
            &for_array,
            |b, for_array| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();

                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context")
                        .with_dispatch_mode(CudaDispatchMode::StandaloneOnly)
                        .with_launch_strategy(Arc::new(timed));

                    for _ in 0..iters {
                        block_on(for_array.clone().into_array().execute_cuda(&mut cuda_ctx))
                            .unwrap();
                    }

                    Duration::from_nanos(timer.load(Ordering::Relaxed))
                });
            },
        );
    }

    group.finish();
}

fn benchmark_ffor_typed<T>(c: &mut Criterion, type_name: &str)
where
    T: NativePType + DeviceRepr + From<u8> + Add<Output = T>,
    Scalar: From<T>,
{
    let mut group = c.benchmark_group("cuda/ffor");

    for &(len, len_str) in BENCH_SIZES {
        group.throughput(Throughput::Bytes((len * size_of::<T>()) as u64));

        let for_array = make_for_array_typed::<T>(len, true);

        group.bench_with_input(
            BenchmarkId::new(type_name, len_str),
            &for_array,
            |b, for_array| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();

                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context")
                        .with_dispatch_mode(CudaDispatchMode::StandaloneOnly)
                        .with_launch_strategy(Arc::new(timed));

                    for _ in 0..iters {
                        block_on(for_array.clone().into_array().execute_cuda(&mut cuda_ctx))
                            .unwrap();
                    }

                    Duration::from_nanos(timer.load(Ordering::Relaxed))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark FoR decompression for all types.
fn benchmark_for(c: &mut Criterion) {
    benchmark_for_typed::<u8>(c, "u8");
    benchmark_for_typed::<u16>(c, "u16");
    benchmark_for_typed::<u32>(c, "u32");
    benchmark_for_typed::<u64>(c, "u64");
}

/// Benchmark FOR+BP decompression for all types.
fn benchmark_ffor(c: &mut Criterion) {
    benchmark_ffor_typed::<u8>(c, "u8");
    benchmark_ffor_typed::<u16>(c, "u16");
    benchmark_ffor_typed::<u32>(c, "u32");
    benchmark_ffor_typed::<u64>(c, "u64");
}

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config();
    targets = benchmark_for, benchmark_ffor
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
