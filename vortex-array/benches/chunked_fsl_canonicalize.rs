// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks canonicalizing a [`ChunkedArray`] of [`FixedSizeListArray`] chunks.
//!
//! Parameterized over:
//! - Number of chunks
//! - Fixed size list length (elements per list)

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

/// Number of lists in each chunk.
const LISTS_PER_CHUNK: usize = 1_000;

/// Number of chunks in the source array.
const NUM_CHUNKS: &[usize] = &[2, 8, 32];

/// Fixed size list lengths (elements per list).
const LIST_SIZES: &[usize] = &[16, 256, 1024];

/// Creates a `FixedSizeListArray` with the given list size and number of lists.
fn create_fsl(list_size: usize, num_lists: usize) -> FixedSizeListArray {
    let total_elements = list_size * num_lists;
    let elements: Buffer<i64> = (0..total_elements as i64).collect();
    FixedSizeListArray::new(
        elements.into_array(),
        list_size as u32,
        Validity::NonNullable,
        num_lists,
    )
}

/// Builds a `ChunkedArray` of `FixedSizeListArray` chunks.
fn create_chunked_fsl(list_size: usize, num_chunks: usize) -> ChunkedArray {
    let chunk = create_fsl(list_size, LISTS_PER_CHUNK);
    let dtype = chunk.dtype().clone();
    let chunks = (0..num_chunks)
        .map(|_| chunk.clone().into_array())
        .collect();
    ChunkedArray::try_new(chunks, dtype).unwrap()
}

#[divan::bench(args = NUM_CHUNKS, consts = LIST_SIZES)]
fn canonicalize<const LIST_SIZE: usize>(bencher: Bencher, num_chunks: usize) {
    let chunked = create_chunked_fsl(LIST_SIZE, num_chunks).into_array();

    bencher
        .with_inputs(|| (&chunked, SESSION.create_execution_ctx()))
        .bench_refs(|(array, execution_ctx)| {
            array.clone().execute::<Canonical>(execution_ctx).unwrap()
        });
}
