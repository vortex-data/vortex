// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for dictionary decoding.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

mod bench_config;
mod timed_launch_strategy;

use std::fmt::Debug;
use std::mem::size_of;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use cudarc::driver::DeviceRepr;
use futures::executor::block_on;
use vortex::array::IntoArray;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::validity::Validity::NonNullable;
use vortex::buffer::Buffer;
use vortex::dtype::NativePType;
use vortex::error::VortexExpect;
use vortex::session::VortexSession;
use vortex_cuda::CudaDispatchMode;
use vortex_cuda::CudaSession;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

use crate::timed_launch_strategy::TimedLaunchStrategy;

const BENCH_ARGS: &[(usize, &str)] = &[(10_000_000, "10M")];

/// Configuration for a dictionary benchmark specifying value and code types along with dictionary size.
struct DictBenchConfig {
    dict_size: usize,
    value_type_name: &'static str,
    code_type_name: &'static str,
}

/// Creates a Dict array with parameterized value type V and code type C.
fn make_dict_array_typed<V, C>(len: usize, dict_size: usize) -> DictArray
where
    V: NativePType + From<u32>,
    C: NativePType + TryFrom<usize>,
    <C as TryFrom<usize>>::Error: Debug,
{
    // Dictionary values
    let values: Vec<V> = (0..dict_size)
        .map(|i| <V as From<u32>>::from((i * 1000) as u32))
        .collect();
    let values_array = PrimitiveArray::new(Buffer::from(values), NonNullable);

    // Codes cycling through all dictionary values
    let codes: Vec<C> = (0..len)
        .map(|i| C::try_from(i % dict_size).unwrap())
        .collect();
    let codes_array = PrimitiveArray::new(Buffer::from(codes), NonNullable);

    DictArray::try_new(codes_array.into_array(), values_array.into_array())
        .vortex_expect("failed to create Dict array")
}

/// Benchmark Dict decompression for specific value and code types.
fn benchmark_dict_typed<V, C>(c: &mut Criterion, config: &DictBenchConfig)
where
    V: NativePType + DeviceRepr + From<u32>,
    C: NativePType + DeviceRepr + TryFrom<usize>,
    <C as TryFrom<usize>>::Error: Debug,
{
    let mut group = c.benchmark_group("dict_cuda");

    for (len, len_str) in BENCH_ARGS {
        // Throughput is based on output size (values read from dictionary)
        group.throughput(Throughput::Bytes((len * size_of::<V>()) as u64));

        let dict_array = make_dict_array_typed::<V, C>(*len, config.dict_size);

        group.bench_with_input(
            BenchmarkId::new(
                "dict",
                format!(
                    "{len_str}_{}_values_{}_codes",
                    config.value_type_name, config.code_type_name
                ),
            ),
            &dict_array,
            |b, dict_array| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();

                    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
                        .vortex_expect("failed to create execution context")
                        .with_dispatch_mode(CudaDispatchMode::StandaloneOnly)
                        .with_launch_strategy(Arc::new(timed));

                    for _ in 0..iters {
                        block_on(dict_array.clone().into_array().execute_cuda(&mut cuda_ctx))
                            .vortex_expect("execute");
                    }

                    Duration::from_nanos(timer.load(Ordering::Relaxed))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark Dict decompression for all type combinations.
fn benchmark_dict(c: &mut Criterion) {
    // u32 values with u8 codes
    benchmark_dict_typed::<u32, u8>(
        c,
        &DictBenchConfig {
            dict_size: 256,
            value_type_name: "u32",
            code_type_name: "u8",
        },
    );

    // u32 values with u16 codes
    benchmark_dict_typed::<u32, u16>(
        c,
        &DictBenchConfig {
            dict_size: 4096,
            value_type_name: "u32",
            code_type_name: "u16",
        },
    );

    // u64 values with u8 codes
    benchmark_dict_typed::<u64, u8>(
        c,
        &DictBenchConfig {
            dict_size: 256,
            value_type_name: "u64",
            code_type_name: "u8",
        },
    );

    // u64 values with u32 codes
    benchmark_dict_typed::<u64, u32>(
        c,
        &DictBenchConfig {
            dict_size: 65536,
            value_type_name: "u64",
            code_type_name: "u32",
        },
    );
}

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config();
    targets = benchmark_dict
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
