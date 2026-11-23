// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks a take of random rows of a chunked fixed-size list array of ALP elements with
//! patches, such as rows of embeddings.
//!
//! A slice of an array with patches does its work at execution, so a take of runs of rows can
//! cost more for each run than for bit-packed elements.

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]
use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng as _;
use rand::rngs::StdRng;
use vortex_alp::alp_encode;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;
fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}
static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty()
        .with::<ArraySession>()
        .with::<KernelSession>()
        .with::<MemorySession>();
    vortex_fastlanes::initialize(&session);
    vortex_alp::initialize(&session);
    session
});
/// Rows taken in one batch, few enough to keep the CodSpeed simulation under 1 ms.
const ROWS: &[usize] = &[2, 4];
/// A chunked array of 16 chunks of 65536 `f32` values each, in rows of `list_size` values. About
/// one value in twelve is an outlier that becomes a patch.
fn embeddings(list_size: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0);
    let mut ctx = SESSION.create_execution_ctx();
    let chunk_rows = 65536 / list_size;
    let chunks = (0..16)
        .map(|_| {
            let v: Buffer<f32> = (0..chunk_rows * list_size)
                .map(|_| {
                    if rng.random_range(0..12) == 0 {
                        rng.random_range(-1.0e6f32..1.0e6)
                    } else {
                        f32::from(rng.random_range(0u16..1000)) / 100.0
                    }
                })
                .collect();
            let alp = alp_encode(
                PrimitiveArray::new(v, Validity::NonNullable).as_view(),
                None,
                &mut ctx,
            )
            .unwrap()
            .into_array();
            FixedSizeListArray::new(alp, list_size as u32, Validity::NonNullable, chunk_rows)
                .into_array()
        })
        .collect::<Vec<_>>();
    let dtype = chunks[0].dtype().clone();
    ChunkedArray::try_new(chunks, dtype).unwrap().into_array()
}
/// Takes `taken` random rows in random order.
fn bench(bencher: Bencher, list_size: usize, taken: usize) {
    let array = embeddings(list_size);
    let n = array.len() as u32;
    let mut rng = StdRng::seed_from_u64(1);
    let indices: Buffer<u32> = (0..taken).map(|_| rng.random_range(0..n)).collect();
    let indices = indices.into_array();
    bencher
        .with_inputs(|| (&array, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(a, i, ctx)| {
            a.take((*i).clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap()
        });
}
#[divan::bench(args = ROWS)]
fn alp_rows_768(b: Bencher, taken: usize) {
    bench(b, 768, taken)
}
