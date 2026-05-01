// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for ALP decompression.

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
use cudarc::driver::DeviceRepr;
use futures::executor::block_on;
use vortex::array::IntoArray;
use vortex::array::LEGACY_SESSION;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::validity::Validity::NonNullable;
use vortex::buffer::Buffer;
use vortex::dtype::NativePType;
use vortex::encodings::alp::ALPArray;
use vortex::encodings::alp::ALPArrayExt;
use vortex::encodings::alp::ALPFloat;
use vortex::encodings::alp::alp_encode;
use vortex::error::VortexExpect;
use vortex::session::VortexSession;
use vortex_cuda::CudaDispatchMode;
use vortex_cuda::CudaSession;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

use crate::bench_config::BENCH_SIZES;
use crate::timed_launch_strategy::TimedLaunchStrategy;

/// Patch frequencies to benchmark (as fractions).
const PATCH_FREQUENCIES: &[(f64, &str)] = &[(0.0, "0%"), (0.01, "1%"), (0.10, "10%")];

/// Create an ALP-encoded array of `len` floats with the requested patch frequency.
///
/// Base values are integer-valued floats which encode cleanly. When `patch_frequency > 0`,
/// PI is placed at regular intervals; PI cannot round-trip through ALP at the chosen
/// exponents, so each PI becomes a patch.
fn make_alp_array<T>(len: usize, patch_frequency: f64) -> ALPArray
where
    T: ALPFloat + NativePType,
{
    let patch_interval = if patch_frequency > 0.0 {
        (1.0 / patch_frequency) as usize
    } else {
        usize::MAX
    };
    let outlier = T::from(std::f64::consts::PI).unwrap();

    let values: Buffer<T> = (0..len)
        .map(|i| {
            if patch_interval != usize::MAX && i % patch_interval == 0 {
                outlier
            } else {
                T::from((i % 256) as u32).unwrap()
            }
        })
        .collect();

    let primitive_array = PrimitiveArray::new(values, NonNullable);
    let encoded = alp_encode(
        primitive_array.as_view(),
        None,
        &mut LEGACY_SESSION.create_execution_ctx(),
    )
    .vortex_expect("failed to ALP-encode array");

    if patch_frequency > 0.0 {
        assert!(
            encoded.patches().is_some(),
            "expected patches for patch_frequency={patch_frequency}",
        );
    }

    encoded
}

fn benchmark_alp_decode_typed<T>(c: &mut Criterion, type_name: &str)
where
    T: ALPFloat + NativePType + DeviceRepr,
{
    let mut group = c.benchmark_group(format!("cuda/alp_{}", type_name));

    for &(len, len_str) in BENCH_SIZES {
        group.throughput(Throughput::Bytes((len * size_of::<T>()) as u64));

        for &(patch_freq, patch_label) in PATCH_FREQUENCIES {
            let array = make_alp_array::<T>(len, patch_freq);

            group.bench_with_input(
                BenchmarkId::new(patch_label, len_str),
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

fn benchmark_alp_decode(c: &mut Criterion) {
    benchmark_alp_decode_typed::<f32>(c, "f32");
    benchmark_alp_decode_typed::<f64>(c, "f64");
}

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config();
    targets = benchmark_alp_decode
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
