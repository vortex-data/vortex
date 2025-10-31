// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use cudarc::driver::CudaContext;
use futures::TryStreamExt;
use rand::prelude::IteratorRandom;
use rand::{Rng, rng};
use tokio::runtime::Runtime;
use vortex_array::arrays::StructArray;
use vortex_array::{ArrayRef, IntoArray};
use vortex_buffer::Buffer;
use vortex_error::VortexUnwrap;
use vortex_file::{FileGpuSegmentSource, VortexOpenOptions, VortexWriteOptions};

// Data sizes: 1GB, 2.5GB, 5GB, 10GB
// These are approximate sizes in bytes, accounting for bit-packing compression
const DATA_SIZES: &[(usize, &str)] = &[
    (268_435_456, "1GB"), // ~1GB when unpacked (268M * 4 bytes)
];

#[allow(clippy::cast_possible_truncation)]
fn make_test_array(len: usize) -> ArrayRef {
    let mut rng = rng();
    let numbers = (0..len)
        .map(|i| rng.random_range(0..64))
        .collect::<Buffer<u32>>()
        .into_array();
    let float_value = (0..64).map(|v| v as f32 / 10f32).collect::<Vec<_>>();
    let floats = (0..len)
        .map(|i| float_value.iter().choose_stable(&mut rng).unwrap())
        .collect::<Buffer<f32>>()
        .into_array();

    StructArray::from_fields(&[("numbers", numbers), ("floats", floats)])
        .vortex_unwrap()
        .into_array()
}

fn benchmark_gpu_scan(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    let mut group = c.benchmark_group("gpu_scan");

    group.sample_size(10);
    let bench_file_name = "/tmp/test-vx/bench_out.vortex";

    for (len, label) in DATA_SIZES {
        let len = len.next_multiple_of(1024);
        let array = make_test_array(len);

        runtime.block_on(async {
            VortexWriteOptions::default()
                .write(
                    tokio::fs::File::create(bench_file_name).await.unwrap(),
                    array.to_array_stream(),
                )
                .await
                .unwrap();
        });

        let cuda_ctx = CudaContext::new(0).unwrap();
        cuda_ctx.set_blocking_synchronize().unwrap();
        group.throughput(Throughput::Bytes((len * size_of::<u32>() * 2) as u64));
        group.bench_function(*label, |b| {
            b.to_async(&runtime).iter_with_large_drop(async || {
                let file = std::fs::File::open(bench_file_name).unwrap();
                let vx_file = VortexOpenOptions::new()
                    .open(bench_file_name)
                    .await
                    .vortex_unwrap();
                let stream = vx_file
                    .gpu_scan(
                        cuda_ctx.clone(),
                        Arc::new(FileGpuSegmentSource::new(
                            vx_file.footer.segment_map().clone(),
                            cuda_ctx.default_stream(),
                            file,
                        )),
                    )
                    .vortex_unwrap()
                    .into_array_stream()
                    .vortex_unwrap()
                    .try_collect::<Vec<_>>()
                    .await
                    .vortex_unwrap();
                stream

                // VortexOpenOptions::new()
                //     .open(bench_file_name)
                //     .await
                //     .vortex_unwrap()
                //     .gpu_scan(ctx.clone())
                //     .vortex_unwrap()
                //     .into_array_stream()
                //     .vortex_unwrap()
                //     .try_collect::<Vec<_>>()
                //     .await
                //     .vortex_unwrap()
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_gpu_scan);

criterion_main!(benches);
