// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for bit unpacking.

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

mod common;

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
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity::NonNullable;
use vortex_buffer::Buffer;
use vortex_cuda::CudaSession;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;
use vortex_dtype::NativePType;
use vortex_error::VortexExpect;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::unpack_iter::BitPacked;
use vortex_session::VortexSession;

use crate::common::TimedLaunchStrategy;

const N_ROWS: usize = 100_000_000;

/// Create a bit-packed array with the given bit width
fn make_bitpacked_array<T>(bit_width: u8, len: usize) -> BitPackedArray
where
    T: NativePType + Add<Output = T> + From<u8>,
{
    let max_val = (1u64 << bit_width).saturating_sub(1);

    let values: Vec<T> = (0..len)
        .map(|i| {
            let val = ((i as u64 % 256) & max_val) as u8;
            <T as From<u8>>::from(val)
        })
        .collect();

    let primitive_array = PrimitiveArray::new(Buffer::from(values), NonNullable);
    BitPackedArray::encode(primitive_array.as_ref(), bit_width)
        .vortex_expect("failed to create BitPacked array")
}

/// Generic benchmark function for a specific type and bit width
fn benchmark_bitunpack_typed<T>(c: &mut Criterion, bit_width: u8, type_name: &str)
where
    T: BitPacked + NativePType + DeviceRepr + Add<Output = T> + From<u8>,
    T::Physical: DeviceRepr,
{
    let mut group = c.benchmark_group(format!("bitunpack_cuda_{}", type_name));
    group.sample_size(10);

    let array = make_bitpacked_array::<T>(bit_width, N_ROWS);
    let nbytes = N_ROWS * size_of::<T>();

    group.throughput(Throughput::Bytes(nbytes as u64));

    group.bench_with_input(
        BenchmarkId::new("bitunpack", format!("{}bw", bit_width)),
        &array,
        |b, array| {
            b.iter_custom(|iters| {
                let timed = TimedLaunchStrategy::default();
                let timer = Arc::clone(&timed.total_time_ns);

                let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                    .vortex_expect("failed to create execution context")
                    .with_launch_strategy(Arc::new(timed));

                for _ in 0..iters {
                    block_on(array.to_array().execute_cuda(&mut cuda_ctx)).unwrap();
                }

                Duration::from_nanos(timer.load(Ordering::Relaxed))
            });
        },
    );

    group.finish();
}

fn benchmark_bitunpack(c: &mut Criterion) {
    benchmark_bitunpack_typed::<u8>(c, 3, "u8");
    benchmark_bitunpack_typed::<u16>(c, 5, "u16");
    benchmark_bitunpack_typed::<u32>(c, 6, "u32");
    benchmark_bitunpack_typed::<u64>(c, 8, "u64");
}

criterion::criterion_group!(benches, benchmark_bitunpack);

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
