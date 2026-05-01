// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for bit unpacking.

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
use vortex::array::validity::Validity::NonNullable;
use vortex::buffer::Buffer;
use vortex::dtype::NativePType;
use vortex::encodings::fastlanes::BitPackedArray;
use vortex::encodings::fastlanes::BitPackedData;
use vortex::encodings::fastlanes::unpack_iter::BitPacked;
use vortex::error::VortexExpect;
use vortex::session::VortexSession;
use vortex_cuda::CudaDispatchMode;
use vortex_cuda::CudaSession;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

use crate::timed_launch_strategy::TimedLaunchStrategy;

/// Patch frequencies to benchmark (as fractions)
const PATCH_FREQUENCIES: &[(f64, &str)] = &[(0.01, "1%"), (0.10, "10%")];

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
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    BitPackedData::encode(&primitive_array.into_array(), bit_width, &mut ctx)
        .vortex_expect("failed to create BitPacked array")
}

/// Create a bit-packed array with the given bit width and patch frequency.
///
/// `patch_frequency` is a fraction (0.0 to 1.0) indicating what proportion of values
/// should exceed the bit width and become patches.
///
/// This function uses bit_width=6 internally since patch values need to exceed
/// the bit width but still fit in u8 for the From<u8> trait bound.
fn make_bitpacked_array_with_patches<T>(len: usize, patch_frequency: f64) -> BitPackedArray
where
    T: NativePType + Add<Output = T> + From<u8>,
{
    // Use bit_width=6 so max packed value is 63, and patch values (64-255) fit in u8
    let bit_width: u8 = 6;
    let max_packed_val = (1u64 << bit_width) - 1; // 63

    // Deterministic patch placement: place patches at regular intervals
    let patch_interval = if patch_frequency > 0.0 {
        (1.0 / patch_frequency) as usize
    } else {
        usize::MAX
    };

    let values: Vec<T> = (0..len)
        .map(|i| {
            if patch_interval > 0 && i % patch_interval == 0 {
                // Patch value: 128 exceeds 6-bit max (63)
                <T as From<u8>>::from(128)
            } else {
                // Normal value that fits within 6 bits (0-63)
                let val = (i as u64 & max_packed_val) as u8;
                <T as From<u8>>::from(val)
            }
        })
        .collect();

    let primitive_array = PrimitiveArray::new(Buffer::from(values), NonNullable).into_array();
    let mut ctx = LEGACY_SESSION.create_execution_ctx();
    BitPackedData::encode(&primitive_array, bit_width, &mut ctx)
        .vortex_expect("failed to create BitPacked array with patches")
}

/// Generic benchmark function for a specific type and bit width
fn benchmark_bitunpack_typed<T>(c: &mut Criterion, bit_width: u8, type_name: &str)
where
    T: BitPacked + NativePType + DeviceRepr + Add<Output = T> + From<u8>,
    T::Physical: DeviceRepr,
{
    let mut group = c.benchmark_group(format!("cuda/bitpacked_{}", type_name));

    for &(n_rows, size_str) in bench_config::BENCH_SIZES {
        let array = make_bitpacked_array::<T>(bit_width, n_rows);
        let nbytes = n_rows * size_of::<T>();

        group.throughput(Throughput::Bytes(nbytes as u64));

        group.bench_with_input(
            BenchmarkId::new(format!("unpack/{}bw", bit_width), size_str),
            &array,
            |b, array| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();

                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context")
                        .with_dispatch_mode(CudaDispatchMode::StandaloneOnly)
                        .with_launch_strategy(Arc::new(timed));

                    for _ in 0..iters {
                        block_on(array.clone().into_array().execute_cuda(&mut cuda_ctx)).unwrap();
                    }

                    Duration::from_nanos(timer.load(Ordering::Relaxed))
                });
            },
        );
    }

    group.finish();
}

fn benchmark_bitunpack(c: &mut Criterion) {
    benchmark_bitunpack_typed::<u8>(c, 3, "u8");
    benchmark_bitunpack_typed::<u16>(c, 5, "u16");
    benchmark_bitunpack_typed::<u32>(c, 6, "u32");
    benchmark_bitunpack_typed::<u64>(c, 8, "u64");
}

/// Benchmark function for unpacking with patches at various frequencies
fn benchmark_bitunpack_with_patches_typed<T>(c: &mut Criterion, type_name: &str)
where
    T: BitPacked + NativePType + DeviceRepr + Add<Output = T> + From<u8>,
    T::Physical: DeviceRepr,
{
    let mut group = c.benchmark_group(format!("cuda/bitpacked_patched_{}", type_name));

    for &(n_rows, size_str) in bench_config::BENCH_SIZES {
        let nbytes = n_rows * size_of::<T>();
        group.throughput(Throughput::Bytes(nbytes as u64));

        for &(patch_freq, patch_label) in PATCH_FREQUENCIES {
            let array = make_bitpacked_array_with_patches::<T>(n_rows, patch_freq);

            group.bench_with_input(
                BenchmarkId::new(format!("unpack/{}", patch_label), size_str),
                &array,
                |b, array| {
                    b.iter_custom(|iters| {
                        let timed = TimedLaunchStrategy::default();
                        let timer = timed.timer();

                        let mut cuda_ctx =
                            CudaSession::create_execution_ctx(&VortexSession::empty())
                                .vortex_expect("failed to create execution context")
                                .with_dispatch_mode(CudaDispatchMode::StandaloneOnly)
                                .with_launch_strategy(Arc::new(timed));

                        for _ in 0..iters {
                            block_on(array.clone().into_array().execute_cuda(&mut cuda_ctx))
                                .unwrap();
                        }

                        Duration::from_nanos(timer.load(Ordering::Relaxed))
                    });
                },
            );
        }
    }

    group.finish();
}

fn benchmark_bitunpack_with_patches(c: &mut Criterion) {
    benchmark_bitunpack_with_patches_typed::<u8>(c, "u8");
    benchmark_bitunpack_with_patches_typed::<u16>(c, "u16");
    benchmark_bitunpack_with_patches_typed::<u32>(c, "u32");
    benchmark_bitunpack_with_patches_typed::<u64>(c, "u64");
}

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config();
    targets = benchmark_bitunpack, benchmark_bitunpack_with_patches
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
