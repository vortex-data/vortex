// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for hybrid dispatch: measures the full `execute_cuda` pipeline
//! including plan building, allocation, fused decompression, and CUB filtering.

#![allow(clippy::unwrap_used)]
#![allow(clippy::expect_used)]

use std::mem::size_of;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use tokio::runtime::Builder;
use vortex::array::DynArray;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::scalar::Scalar;
use vortex::array::validity::Validity::NonNullable;
use vortex::buffer::Buffer;
use vortex::encodings::fastlanes::BitPackedArray;
use vortex::encodings::fastlanes::FoRArray;
use vortex::error::VortexExpect;
use vortex::mask::Mask;
use vortex::session::VortexSession;
use vortex_cuda::CudaSession;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

const BENCH_ARGS: &[(usize, &str)] = &[
    (1_000_000, "1M"),
    (10_000_000, "10M"),
    (100_000_000, "100M"),
];

fn make_for_bitpacked(len: usize, bit_width: u8, reference: u32) -> vortex::array::ArrayRef {
    let max_val = (1u64 << bit_width).saturating_sub(1);
    let residuals: Vec<u32> = (0..len)
        .map(|i| (i as u64 % (max_val + 1)) as u32)
        .collect();
    let prim = PrimitiveArray::new(Buffer::from(residuals), NonNullable);
    let bp = BitPackedArray::encode(&prim.into_array(), bit_width).vortex_expect("bitpack");
    FoRArray::try_new(bp.into_array(), Scalar::from(reference))
        .vortex_expect("for")
        .into_array()
}

// ---------------------------------------------------------------------------
// FoR(BitPacked) via execute_cuda → try_dyn_dispatch → fully fused
// ---------------------------------------------------------------------------
fn bench_hybrid_for_bitpacked(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_for_bitpacked_6bw");
    group.sample_size(10);

    for (len, len_str) in BENCH_ARGS {
        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));

        let array = make_for_bitpacked(*len, 6, 100_000);

        group.bench_with_input(BenchmarkId::new("execute_cuda", len_str), len, |b, _| {
            let rt = Builder::new_current_thread().enable_all().build().unwrap();
            let mut ctx =
                CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");

            b.iter(|| {
                let result = rt.block_on(array.clone().execute_cuda(&mut ctx)).unwrap();
                std::hint::black_box(result);
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Filter(FoR(BitPacked), 50% mask) via execute_cuda
//   → try_dyn_dispatch fallback → FilterExecutor → child fuses → CUB filter
// ---------------------------------------------------------------------------
fn bench_hybrid_filter_for_bitpacked(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_filter_for_bitpacked_6bw");
    group.sample_size(10);

    for (len, len_str) in BENCH_ARGS {
        group.throughput(Throughput::Bytes((len * size_of::<u32>()) as u64));

        let array = make_for_bitpacked(*len, 6, 100_000);
        let mask = Mask::from_iter((0..*len).map(|i| i % 2 == 0));
        let filtered = array.filter(mask).vortex_expect("filter");

        group.bench_with_input(BenchmarkId::new("execute_cuda", len_str), len, |b, _| {
            let rt = Builder::new_current_thread().enable_all().build().unwrap();
            let mut ctx =
                CudaSession::create_execution_ctx(&VortexSession::empty()).vortex_expect("ctx");

            b.iter(|| {
                let result = rt
                    .block_on(filtered.clone().execute_cuda(&mut ctx))
                    .unwrap();
                std::hint::black_box(result);
            });
        });
    }

    group.finish();
}

fn benchmark_hybrid_dispatch(c: &mut Criterion) {
    bench_hybrid_for_bitpacked(c);
    bench_hybrid_filter_for_bitpacked(c);
}

criterion::criterion_group!(benches, benchmark_hybrid_dispatch);

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
