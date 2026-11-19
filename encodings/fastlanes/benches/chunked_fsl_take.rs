// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks a take of random rows of a chunked fixed-size list array with bit-packed elements.
//!
//! A batch of token sequences is a take of random rows of a file. The rows are not in chunk
//! order, so the take gathers long runs of elements out of chunk order.

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray as _;
use vortex_array::RecursiveCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::session::AggregateFnSession;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::memory::MemorySession;
use vortex_array::optimizer::kernels::KernelSession;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_fastlanes::bitpack_compress::bitpack_to_best_bit_width;
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
        .with::<AggregateFnSession>()
        .with::<MemorySession>();
    vortex_fastlanes::initialize(&session);
    session
});

/// Tokens in each row.
const LIST_SIZE: usize = 2048;

/// Rows in each chunk.
const CHUNK_ROWS: usize = 32;

/// Chunks of the array.
const NUM_CHUNKS: usize = 16;

/// Rows taken in one batch, few enough to keep the CodSpeed simulation under 1 ms.
const TAKEN_ROWS: &[usize] = &[4, 8];

/// A chunked array of rows of tokens, with bit-packed elements of 17 bits.
fn chunked_tokens() -> ArrayRef {
    chunked_lists(LIST_SIZE, CHUNK_ROWS)
}

/// A chunked array of rows of `list_size` bit-packed elements, with `chunk_rows` rows in each
/// chunk.
fn chunked_lists(list_size: usize, chunk_rows: usize) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(0);
    let mut ctx = SESSION.create_execution_ctx();

    let chunks = (0..NUM_CHUNKS)
        .map(|_| {
            let tokens: Buffer<u32> = (0..chunk_rows * list_size)
                .map(|_| rng.random_range(0..100_000u32))
                .collect();
            let tokens = PrimitiveArray::new(tokens, Validity::NonNullable);
            let packed = bitpack_to_best_bit_width(&tokens, &mut ctx).unwrap();

            FixedSizeListArray::new(
                packed.into_array(),
                list_size as u32,
                Validity::NonNullable,
                chunk_rows,
            )
            .into_array()
        })
        .collect::<Vec<_>>();

    let dtype = chunks[0].dtype().clone();
    ChunkedArray::try_new(chunks, dtype).unwrap().into_array()
}

/// Takes `taken` random rows in random order.
#[divan::bench(args = TAKEN_ROWS)]
fn take_random_rows(bencher: Bencher, taken: usize) {
    let array = chunked_tokens();

    let mut rng = StdRng::seed_from_u64(1);
    let indices: Buffer<u32> = (0..taken)
        .map(|_| rng.random_range(0..(NUM_CHUNKS * CHUNK_ROWS) as u32))
        .collect();
    let indices = indices.into_array();

    bencher
        .with_inputs(|| (&array, &indices, SESSION.create_execution_ctx()))
        .bench_refs(|(array, indices, ctx)| {
            array
                .take((*indices).clone())
                .unwrap()
                .execute::<RecursiveCanonical>(ctx)
                .unwrap()
        });
}
