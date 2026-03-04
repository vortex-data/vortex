// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Metal benchmarks for FoR decompression.

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::cast_possible_truncation)]

use std::mem::size_of;
use std::ops::Add;
use std::time::Instant;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::validity::Validity;
use vortex::buffer::Buffer;
use vortex::dtype::NativePType;
use vortex::encodings::fastlanes::FoRArray;
use vortex::error::VortexExpect;
use vortex::scalar::Scalar;
use vortex::session::VortexSession;
use vortex_metal::CanonicalMetalExt;
use vortex_metal::MetalArrayExt;
use vortex_metal::MetalSession;
use vortex_metal::metal_available;

const BENCH_ARGS: &[(usize, &str)] = &[(100_000, "100K"), (1_000_000, "1M"), (10_000_000, "10M")];
const REFERENCE_VALUE: u8 = 10;

/// Creates a FoR array with the specified type and length.
fn make_for_array_typed<T>(len: usize) -> FoRArray
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

    FoRArray::try_new(primitive_array, reference.into()).vortex_expect("failed to create FoR array")
}

/// Benchmark FoR decompression on Metal for a specific type.
fn benchmark_for_metal_typed<T>(c: &mut Criterion, type_name: &str)
where
    T: NativePType + From<u8> + Add<Output = T>,
    Scalar: From<T>,
{
    let mut group = c.benchmark_group("for_metal");
    group.sample_size(20);

    let session = MetalSession::new().expect("Failed to create Metal session");

    for &(len, len_str) in BENCH_ARGS {
        group.throughput(Throughput::Bytes((len * size_of::<T>()) as u64));

        let for_array = make_for_array_typed::<T>(len);

        group.bench_with_input(
            BenchmarkId::new("metal", format!("{len_str}_{type_name}")),
            &for_array,
            |b, for_array| {
                b.iter_custom(|iters| {
                    let start = Instant::now();

                    for _ in 0..iters {
                        let mut ctx = session
                            .create_execution_ctx(&VortexSession::empty())
                            .expect("failed to create context");

                        let result = for_array
                            .to_array()
                            .execute_metal(&mut ctx)
                            .expect("Metal execution failed")
                            .into_host()
                            .expect("Failed to copy to host");

                        // Prevent optimization from eliding the work
                        std::hint::black_box(result);
                    }

                    start.elapsed()
                });
            },
        );

        // Also benchmark CPU for comparison
        group.bench_with_input(
            BenchmarkId::new("cpu", format!("{len_str}_{type_name}")),
            &for_array,
            |b, for_array| {
                b.iter(|| {
                    let result = for_array.to_canonical().expect("CPU execution failed");
                    std::hint::black_box(result);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark FoR decompression for all types.
fn benchmark_for(c: &mut Criterion) {
    benchmark_for_metal_typed::<u32>(c, "u32");
    benchmark_for_metal_typed::<u64>(c, "u64");
}

criterion::criterion_group!(benches, benchmark_for);

fn main() {
    if metal_available() {
        Criterion::default().configure_from_args().final_summary();
        benches();
    } else {
        eprintln!("Metal is not available on this system");
    }
}
