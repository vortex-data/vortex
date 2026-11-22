// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for decoding narrow integers to a wider type: `Widen` arrays and
//! `DecimalByteParts` arrays, whose most significant part is a `Widen` of a narrow integer.
//!
//! The timer sums the time of every kernel the decode launches.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

mod bench_config;
mod timed_launch_strategy;

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
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::DictArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::WidenArray;
use vortex::array::validity::Validity::NonNullable;
use vortex::buffer::Buffer;
use vortex::dtype::DType;
use vortex::dtype::DecimalDType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::encodings::decimal_byte_parts::DecimalByteParts;
use vortex::encodings::fastlanes::BitPackedData;
use vortex::encodings::fastlanes::FoR;
use vortex::error::VortexExpect;
use vortex::scalar::Scalar;
use vortex::session::VortexSession;
use vortex_array::test_harness::mark_optimized;
use vortex_cuda::CudaSession;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;

use crate::bench_config::BENCH_SIZES;
use crate::timed_launch_strategy::TimedLaunchStrategy;

static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_cuda::cuda_session);

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn bitpacked_i32(values: impl Iterator<Item = i32>, bit_width: u8) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::new(Buffer::from_iter(values), NonNullable).into_array();
    BitPackedData::encode(&values, bit_width, &mut ctx)
        .vortex_expect("bitpack")
        .into_array()
}

fn widen_i64(array: ArrayRef) -> ArrayRef {
    WidenArray::try_new(
        array,
        DType::Primitive(PType::I64, Nullability::NonNullable),
    )
    .vortex_expect("widen")
    .into_array()
}

/// `Widen<i64>(FoR<i32>(BitPacked<i32>))` with negative values.
fn widen_for_bitpacked(len: usize) -> ArrayRef {
    let encoded = bitpacked_i32((0..len).map(|i| (i % 4096) as i32), 12);
    let for_array = FoR::try_new(encoded, Scalar::from(-1_000_000i32)).vortex_expect("for");
    widen_i64(for_array.into_array())
}

/// `DecimalByteParts` with a `Widen<i64>(BitPacked<i32>)` part, like TPC-H prices.
fn decimal_bitpacked_i32(len: usize) -> ArrayRef {
    let msp = widen_i64(bitpacked_i32((0..len).map(|i| (i % 1_000_000) as i32), 20));
    DecimalByteParts::try_new(msp, DecimalDType::new(15, 2))
        .vortex_expect("decimal byte parts")
        .into_array()
}

/// `DecimalByteParts` with a `Widen<i64>(BitPacked<i8>)` part, like TPC-H discounts.
fn decimal_bitpacked_i8(len: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let values = PrimitiveArray::new(
        Buffer::from_iter((0..len).map(|i| (i % 11) as i8)),
        NonNullable,
    );
    let msp = BitPackedData::encode(&values.into_array(), 4, &mut ctx).vortex_expect("bitpack");
    DecimalByteParts::try_new(widen_i64(msp.into_array()), DecimalDType::new(15, 2))
        .vortex_expect("decimal byte parts")
        .into_array()
}

/// `DecimalByteParts` with a `Widen<i64>(Dict<i16>)` part, like TPC-H quantities.
fn decimal_dict_i16(len: usize) -> ArrayRef {
    let mut ctx = SESSION.create_execution_ctx();
    let codes = PrimitiveArray::new(
        Buffer::from_iter((0..len).map(|i| (i % 50) as u8)),
        NonNullable,
    );
    let codes = BitPackedData::encode(&codes.into_array(), 6, &mut ctx).vortex_expect("bitpack");
    let values = PrimitiveArray::new(
        Buffer::from_iter((0..50i16).map(|i| (i - 25) * 100)),
        NonNullable,
    );
    let dict = DictArray::try_new(codes.into_array(), values.into_array()).vortex_expect("dict");
    DecimalByteParts::try_new(widen_i64(dict.into_array()), DecimalDType::new(15, 2))
        .vortex_expect("decimal byte parts")
        .into_array()
}

/// Builds the benchmarked array with the given length.
type MakeArray = fn(usize) -> ArrayRef;

fn benchmark_widen(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");

    let cases: [(&str, MakeArray); 4] = [
        ("cuda/widen/i32_for_bitpacked_to_i64", widen_for_bitpacked),
        (
            "cuda/decimal_byte_parts/i32_bitpacked_p15",
            decimal_bitpacked_i32,
        ),
        (
            "cuda/decimal_byte_parts/i8_bitpacked_p15",
            decimal_bitpacked_i8,
        ),
        ("cuda/decimal_byte_parts/i16_dict_p15", decimal_dict_i16),
    ];
    for (name, make_array) in cases {
        for &(len, len_str) in BENCH_SIZES {
            group.throughput(Throughput::Bytes((len * size_of::<i64>()) as u64));

            let array = make_array(len);
            mark_optimized(&array, &vortex_cuda::cuda_session()).unwrap();

            group.bench_with_input(BenchmarkId::new(name, len_str), &array, |b, array| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();

                    let mut cuda_ctx =
                        CudaSession::create_execution_ctx(&vortex_cuda::cuda_session())
                            .vortex_expect("failed to create execution context")
                            .with_launch_strategy(Arc::new(timed));

                    for _ in 0..iters {
                        block_on(array.clone().execute_cuda(&mut cuda_ctx)).unwrap();
                    }

                    Duration::from_nanos(timer.load(Ordering::Relaxed))
                });
            });
        }
    }

    group.finish();
}

criterion::criterion_group! {
    name = benches;
    config = bench_config::cuda_bench_config();
    targets = benchmark_widen
}

#[cuda_available]
fn main() {
    LazyLock::force(&SESSION);
    benches();
    Criterion::default().configure_from_args().final_summary();
}

#[cuda_not_available]
fn main() {}
