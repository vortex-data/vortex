// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks canonicalizing a [`ChunkedArray`] whose chunks each take a few rows of an ALP-RD
//! encoded segment.
//!
//! `array.take(indices)` is represented as a lazy `Dict` over the encoded values. Canonicalizing
//! the chunked array appends each such `Dict` to a builder, so these benchmarks measure whether
//! that append runs the ALP-RD take kernel on the selected rows or decodes each segment in full.

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use std::sync::LazyLock;

use divan::Bencher;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng as _;
use rand::rngs::StdRng;
use vortex_alp::RDEncoder;
use vortex_alp::RDEncoderExt;
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
use vortex_array::test_harness::mark_optimized;
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

/// Number of encoded segments, and so of chunks in the canonicalized array.
const SEGMENTS: usize = if cfg!(codspeed) { 2 } else { 4 };

/// Elements per vector.
const LIST_SIZE: usize = 64;

/// Rows (vectors) per segment.
const ROWS_PER_SEGMENT: usize = if cfg!(codspeed) { 512 } else { 4096 };

/// Rows taken from each segment.
const TAKEN_ROWS: &[usize] = if cfg!(codspeed) {
    &[1, 16]
} else {
    &[1, 16, 256, 1024, 2048]
};

/// Encodes `len` random `f32` values in `[-1, 1)` with ALP-RD.
fn alp_rd_segment(len: usize, seed: u64) -> ArrayRef {
    let mut rng = StdRng::seed_from_u64(seed);
    let values: Buffer<f32> = (0..len).map(|_| rng.random_range(-1.0f32..1.0)).collect();
    let primitive = PrimitiveArray::new(values, Validity::NonNullable);
    RDEncoder::new(primitive.as_slice::<f32>())
        .encode(primitive.as_view())
        .into_array()
}

/// Evenly spaced row indices of a segment of `rows` rows.
fn spread_indices(rows: usize, taken: usize) -> ArrayRef {
    let stride = rows / taken;
    (0..taken)
        .map(|i| (i * stride) as u32)
        .collect::<Buffer<u32>>()
        .into_array()
}

/// A chunked array of `taken` rows taken lazily from each of [`SEGMENTS`] segments.
fn chunked_takes(taken: usize, segment: impl Fn(u64) -> ArrayRef) -> ArrayRef {
    let chunks = (0..SEGMENTS)
        .map(|i| {
            let segment = segment(i as u64);
            let indices = spread_indices(segment.len(), taken);
            segment.take(indices).unwrap()
        })
        .collect::<Vec<_>>();
    let dtype = chunks[0].dtype().clone();
    let chunked = ChunkedArray::try_new(chunks, dtype).unwrap().into_array();
    mark_optimized(&chunked, &SESSION).unwrap();
    chunked
}

/// Canonicalizes `array` and every nested child, as reading the values out would.
fn bench_canonicalize(bencher: Bencher, array: &ArrayRef) {
    bencher
        .with_inputs(|| (array, SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| (*array).clone().execute::<RecursiveCanonical>(ctx).unwrap());
}

/// Takes rows of an ALP-RD `f32` segment, as a primitive column.
#[divan::bench(args = TAKEN_ROWS)]
fn primitive(bencher: Bencher, taken: usize) {
    let array = chunked_takes(taken, |seed| {
        alp_rd_segment(ROWS_PER_SEGMENT * LIST_SIZE, seed)
    });
    bench_canonicalize(bencher, &array);
}

/// Takes vectors of a `FixedSizeList<f32, 64>` segment whose elements are ALP-RD encoded.
#[divan::bench(args = TAKEN_ROWS)]
fn fixed_size_list(bencher: Bencher, taken: usize) {
    let array = chunked_takes(taken, |seed| {
        FixedSizeListArray::new(
            alp_rd_segment(ROWS_PER_SEGMENT * LIST_SIZE, seed),
            LIST_SIZE as u32,
            Validity::NonNullable,
            ROWS_PER_SEGMENT,
        )
        .into_array()
    });
    bench_canonicalize(bencher, &array);
}
