// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA benchmarks for FSST decompression.

#![expect(clippy::unwrap_used)]

#[allow(dead_code)]
mod bench_config;
mod timed_launch_strategy;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::Throughput;
use futures::executor::block_on;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::match_each_integer_ptype;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::encodings::fsst::FSST;
use vortex::encodings::fsst::FSSTArrayExt;
use vortex::encodings::fsst::FSSTArraySlotsExt;
use vortex::error::VortexExpect;
use vortex_cuda::CudaDispatchMode;
use vortex_cuda::CudaSession;
use vortex_cuda::VarBinExportLayout;
use vortex_cuda::arrow::DeviceArrayExt;
use vortex_cuda::arrow::release_device_array;
use vortex_cuda::executor::CudaArrayExt;
use vortex_cuda_macros::cuda_available;
use vortex_cuda_macros::cuda_not_available;
use vortex_fsst::FSSTSymbolTable;
use vortex_fsst::test_utils::make_fsst_clickbench_urls;

use crate::timed_launch_strategy::TimedLaunchStrategy;

// Bench-local size instead of the workspace 100M default: each input is a
// clickbench URL, much heavier per-element than the fixed-width primitives
// other kernels benchmark.
const BENCH_SIZES: &[(usize, &str)] = &[(10_000_000, "10M")];

fn session_with_varbin_layout(layout: VarBinExportLayout) -> vortex::session::VortexSession {
    vortex::array::array_session().with_some(
        CudaSession::try_default()
            .vortex_expect("failed to create CUDA session")
            .with_varbin_export_layout(layout),
    )
}

struct FSSTBenchFixture {
    utf8: ArrayRef,
    binary: ArrayRef,
    uncompressed_size: u64,
}

fn make_fixture(n: usize) -> FSSTBenchFixture {
    let mut setup_ctx = CudaSession::create_execution_ctx(&vortex_cuda::cuda_session())
        .vortex_expect("failed to create execution context");
    let fsst = make_fsst_clickbench_urls(n, setup_ctx.execution_ctx());

    let lens = fsst
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(setup_ctx.execution_ctx())
        .vortex_expect("canonicalize uncompressed_lengths");
    #[allow(clippy::unnecessary_cast)]
    let uncompressed_size = match_each_integer_ptype!(lens.ptype(), |P| {
        lens.as_slice::<P>().iter().map(|x| *x as u64).sum()
    });

    let binary = FSST::try_new_with_symbol_table(
        DType::Binary(Nullability::NonNullable),
        Arc::new(
            FSSTSymbolTable::new_padded(
                fsst.padded_symbols().clone(),
                fsst.padded_symbol_lengths().clone(),
                fsst.n_symbols(),
            )
            .vortex_expect("construction"),
        ),
        fsst.codes(),
        fsst.uncompressed_lengths().clone(),
        setup_ctx.execution_ctx(),
    )
    .vortex_expect("rebuild FSST fixture with Binary dtype")
    .into_array();

    FSSTBenchFixture {
        utf8: fsst.into_array(),
        binary,
        uncompressed_size,
    }
}

fn benchmark_fsst_cuda_decompress(c: &mut Criterion) {
    let mut group = c.benchmark_group("cuda");

    for &(n, len_str) in BENCH_SIZES {
        let fixture = make_fixture(n);

        group.throughput(Throughput::Bytes(fixture.uncompressed_size));
        group.bench_with_input(
            BenchmarkId::new("cuda/fsst/decompress_to_varbinview", len_str),
            &fixture.utf8,
            |b, fsst_array| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();

                    let mut cuda_ctx =
                        CudaSession::create_execution_ctx(&vortex_cuda::cuda_session())
                            .vortex_expect("failed to create execution context")
                            .with_dispatch_mode(CudaDispatchMode::StandaloneOnly)
                            .with_launch_strategy(Arc::new(timed));

                    for _ in 0..iters {
                        block_on(fsst_array.clone().execute_cuda(&mut cuda_ctx)).unwrap();
                    }
                    Duration::from_nanos(timer.load(Ordering::Relaxed))
                });
            },
        );

        for (name, array, layout) in [
            (
                "cuda/fsst/decompress_to_varbin",
                &fixture.utf8,
                VarBinExportLayout::VarBin,
            ),
            (
                "cuda/fsst/export_binary",
                &fixture.binary,
                VarBinExportLayout::VarBin,
            ),
            (
                "cuda/fsst/export_utf8_view",
                &fixture.utf8,
                VarBinExportLayout::VarBinView,
            ),
            (
                "cuda/fsst/export_binary_view",
                &fixture.binary,
                VarBinExportLayout::VarBinView,
            ),
        ] {
            group.bench_with_input(BenchmarkId::new(name, len_str), array, |b, array| {
                b.iter_custom(|iters| {
                    let timed = TimedLaunchStrategy::default();
                    let timer = timed.timer();

                    let session = session_with_varbin_layout(layout);
                    let mut cuda_ctx = CudaSession::create_execution_ctx(&session)
                        .vortex_expect("failed to create execution context")
                        .with_dispatch_mode(CudaDispatchMode::StandaloneOnly)
                        .with_launch_strategy(Arc::new(timed));

                    for _ in 0..iters {
                        let mut exported =
                            block_on((*array).clone().export_device_array(&mut cuda_ctx))
                                .vortex_expect("export FSST device array");
                        release_device_array(&mut exported);
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
    targets = benchmark_fsst_cuda_decompress
}

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
