// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for ALP-RD decompression.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

mod bench_config;
mod timed_launch_strategy;

use std::f64::consts::PI;
use std::mem::size_of;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::Ordering;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use futures::executor::block_on;
use mimalloc::MiMalloc;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::validity::Validity::NonNullable;
use vortex::buffer::Buffer;
use vortex::dtype::NativePType;
use vortex::encodings::alp::ALPFloat;
use vortex::encodings::alp::ALPRD;
use vortex::encodings::alp::ALPRDArray;
use vortex::encodings::alp::ALPRDArrayExt;
use vortex::encodings::alp::ALPRDArraySlotsExt;
use vortex::encodings::alp::ALPRDFloat;
use vortex::encodings::alp::RDEncoder;
use vortex::encodings::alp::RDEncoderExt;
use vortex::error::VortexExpect;
use vortex::session::VortexSession;
use vortex_array::test_harness::mark_optimized;
use vortex_cuda::CudaDispatchMode;
use vortex_cuda::CudaSession;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda_driver::DeviceRepr;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

use crate::bench_config::BENCH_SIZES;
use crate::timed_launch_strategy::TimedLaunchStrategy;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_cuda::cuda_session);

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Patch frequencies to benchmark (as fractions).
const PATCH_FREQUENCIES: &[(f64, &str)] = &[(0.0, "0%"), (0.01, "1%"), (0.10, "10%")];

/// The value at `row` before outliers: full-precision floats in `[1, 4)`, which ALP can't encode
/// but whose few distinct left parts fit ALP-RD's dictionary.
fn base_value<T: ALPFloat>(row: usize) -> T {
    T::from(1.0 + (row % 1000) as f64 * PI * 1e-3).unwrap()
}

/// Create an ALP-RD-encoded array of `len` floats with the requested patch frequency.
///
/// The dictionary is sampled from outlier-free values, so each outlier, a value far outside the
/// base range, has a left part outside the dictionary and becomes a patch. Like the ALP benchmark,
/// the children are left uncompressed, so only the ALP-RD kernel is timed.
fn make_alp_rd_array<T>(len: usize, patch_frequency: f64) -> ALPRDArray
where
    T: ALPFloat + ALPRDFloat + NativePType,
{
    let patch_interval = if patch_frequency > 0.0 {
        (1.0 / patch_frequency) as usize
    } else {
        usize::MAX
    };
    let outlier = T::from(3.0e30).unwrap();

    let values: Buffer<T> = (0..len)
        .map(|row| {
            if patch_interval != usize::MAX && row % patch_interval == 0 {
                outlier
            } else {
                base_value(row)
            }
        })
        .collect();
    let sample: Vec<T> = (0..1024).map(base_value).collect();

    let primitive_array = PrimitiveArray::new(values, NonNullable);
    let encoded = RDEncoder::new(&sample).encode(primitive_array.as_view());
    let mut ctx = SESSION.create_execution_ctx();
    let left_parts = encoded
        .left_parts()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let right_parts = encoded
        .right_parts()
        .clone()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let encoded = ALPRD::try_new(
        encoded.dtype().clone(),
        left_parts.into_array(),
        encoded.left_parts_dictionary().clone(),
        right_parts.into_array(),
        encoded.right_bit_width(),
        encoded.left_parts_patches(),
    )
    .vortex_expect("failed to rebuild ALP-RD with uncompressed children");

    if patch_frequency > 0.0 {
        assert!(
            encoded.left_parts_patches().is_some(),
            "expected patches for patch_frequency={patch_frequency}",
        );
    }

    encoded
}

fn benchmark_alp_rd_decode_typed<T>(c: &mut Criterion, type_name: &str)
where
    T: ALPFloat + ALPRDFloat + NativePType + DeviceRepr,
{
    let mut group = c.benchmark_group("cuda");

    for &(len, len_str) in BENCH_SIZES {
        group.throughput(Throughput::Bytes((len * size_of::<T>()) as u64));

        for &(patch_freq, patch_label) in PATCH_FREQUENCIES {
            let array = make_alp_rd_array::<T>(len, patch_freq);
            mark_optimized(array.as_array(), &vortex_cuda::cuda_session()).unwrap();

            group.bench_with_input(
                BenchmarkId::new(format!("cuda/alprd_{}/{}", type_name, patch_label), len_str),
                &array,
                |b, array| {
                    b.iter_custom(|iters| {
                        let timed = TimedLaunchStrategy::default();
                        let timer = timed.timer();

                        let mut cuda_ctx =
                            CudaSession::create_execution_ctx(&vortex_cuda::cuda_session())
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

fn benchmark_alp_rd_decode(c: &mut Criterion) {
    benchmark_alp_rd_decode_typed::<f32>(c, "f32");
    benchmark_alp_rd_decode_typed::<f64>(c, "f64");
}

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config();
    targets = benchmark_alp_rd_decode
}

#[cuda_available]
fn main() {
    benches();
    Criterion::default().configure_from_args().final_summary();
}

#[cuda_not_available]
fn main() {}
