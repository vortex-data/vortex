// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks take from a chunked array of compressed fixed-size list chunks.

#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use divan::counter::BytesCount;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::FixedSizeList;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

fn main() {
    LazyLock::force(&SESSION);
    divan::main();
}

static SESSION: LazyLock<VortexSession> = LazyLock::new(array_session);

const NUM_LISTS: usize = 500;
const NUM_CHUNKS: usize = 8;
const NUM_INDICES: &[usize] = &[64];
const LIST_SIZES: &[usize] = &[8, 16, 32];

fn create_random_indices(num_indices: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(42);
    let indices: Buffer<u64> = (0..num_indices)
        .map(|_| rng.random_range(0..NUM_LISTS) as u64)
        .collect();
    indices.into_array()
}

fn create_chunked_compressed_fsl(list_size: usize) -> ArrayRef {
    let compressor = BtrBlocksCompressor::default();
    let lists_per_chunk = NUM_LISTS.div_ceil(NUM_CHUNKS);
    let mut chunks = Vec::new();
    let mut start = 0usize;
    let mut ctx = SESSION.create_execution_ctx();

    while start < NUM_LISTS {
        let end = (start + lists_per_chunk).min(NUM_LISTS);
        let elements: Buffer<i64> =
            ((start * list_size) as i64..(end * list_size) as i64).collect();
        let fsl = FixedSizeListArray::new(
            elements.into_array(),
            u32::try_from(list_size).unwrap(),
            Validity::NonNullable,
            end - start,
        )
        .into_array();
        let compressed = compressor.compress(&fsl, &mut ctx).unwrap();

        let compressed_fsl = compressed.as_::<FixedSizeList>();
        assert!(
            !compressed_fsl.elements().is::<Primitive>(),
            "expected compressed FSL elements"
        );
        chunks.push(compressed);
        start = end;
    }

    let dtype = chunks[0].dtype().clone();
    ChunkedArray::try_new(chunks, dtype).unwrap().into_array()
}

fn bench_take<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    let array = create_chunked_compressed_fsl(LIST_SIZE);
    let indices = create_random_indices(num_indices);

    bencher
        .counter(BytesCount::of_many::<i64>(num_indices * LIST_SIZE))
        .with_inputs(|| (&array, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, ctx)| {
            array
                .take((*indices).clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap()
        });
}

#[divan::bench(args = NUM_INDICES, consts = LIST_SIZES)]
fn take_chunked_compressed_fsl<const LIST_SIZE: usize>(bencher: Bencher, num_indices: usize) {
    bench_take::<LIST_SIZE>(bencher, num_indices);
}
