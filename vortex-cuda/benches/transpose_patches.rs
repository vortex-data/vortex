// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used)]
#![allow(clippy::cast_possible_truncation)]

use criterion::{BenchmarkId, Criterion};
use futures::executor::block_on;
use vortex::buffer::{Buffer, buffer};
use vortex::session::VortexSession;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::PType;
use vortex_array::patches::Patches;
use vortex_array::validity::Validity;
use vortex_cuda::{CudaSession, transpose_patches};
use vortex_cuda_macros::{cuda_available, cuda_not_available};
use vortex_error::VortexExpect;

fn benchmark_transpose(c: &mut Criterion) {
    let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
        .vortex_expect("failed to create execution context");

    let patches = block_on(async {
        // Assume that we have 64k values, and we have 1024 patches evenly disbursed across
        // the range.
        let indices = (0..1024).map(|x| x * 64).collect::<Buffer<u32>>();

        let values = buffer![-1.0f32; 1024];

        let device_indices = cuda_ctx.copy_to_device(&indices)?.await?;
        let device_values = cuda_ctx.copy_to_device(&values)?.await?;

        Patches::new(
            64 * 1024,
            0,
            PrimitiveArray::from_buffer_handle(device_indices, PType::U32, Validity::NonNullable)
                .into_array(),
            PrimitiveArray::from_buffer_handle(device_values, PType::F32, Validity::NonNullable)
                .into_array(),
            None,
        )
    })
    .unwrap();

    let mut group = c.benchmark_group("transpose");
    group.bench_with_input(
        BenchmarkId::new("transpose_patches", 0),
        &patches,
        |b, patches| {
            b.iter(|| block_on(async { transpose_patches(patches, &mut cuda_ctx).await.unwrap() }))
        },
    );

    block_on(async move {});
}

criterion::criterion_group!(benches, benchmark_transpose);

#[cuda_available]
criterion::criterion_main!(benches);

#[cuda_not_available]
fn main() {}
