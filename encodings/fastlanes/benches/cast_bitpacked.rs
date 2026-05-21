// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks the cost of widening a bit-packed narrow integer column to a wider integer type on
//! decompression (e.g. `u16 -> u32`).
//!
//! Three strategies are compared:
//!
//! - `canonical_cast`: the current path. `cast(bit_packed)` has no pushdown, so it canonicalizes to
//!   a full-length `u16` `PrimitiveArray` and then casts that to `u32`. Two full-length buffers are
//!   allocated and the `u16` intermediate is written to RAM and read back.
//! - `chunked_canonical_cast`: the same, but over a `ChunkedArray` of bit-packed chunks, i.e.
//!   `chunked(cast(bit_packed))`. This is the shape produced by a scan over a chunked column.
//! - `pushdown`: unpacks each 1024-element FastLanes chunk into a cache-resident scratch buffer and
//!   cast-copies straight into the `u32` output. Only the `u32` output is allocated/touched in RAM.

#![expect(clippy::unwrap_used)]
#![expect(clippy::cast_possible_truncation)]

use std::sync::LazyLock;

use divan::Bencher;
use rand::RngExt;
use rand::SeedableRng;
use rand::prelude::StdRng;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::builders::PrimitiveBuilder;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::session::ArraySession;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_fastlanes::BitPackedArray;
use vortex_fastlanes::BitPackedData;
use vortex_fastlanes::bitpack_decompress::unpack_and_cast_into_builder;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

const U32: DType = DType::Primitive(PType::U32, Nullability::NonNullable);

// (chunk_len, chunk_count, fraction_patched)
const ARGS: &[(usize, usize, f64)] = &[
    (65_536, 1, 0.00),
    (65_536, 1, 0.01),
    (65_536, 16, 0.00),
    (65_536, 16, 0.01),
    (1_048_576, 1, 0.00),
    (1_048_576, 1, 0.01),
];

/// Build a single bit-packed `u16` chunk. Most values fit in `bit_width` bits; `fraction_patched`
/// of them are large enough to require patches.
fn make_chunk(rng: &mut StdRng, len: usize, fraction_patched: f64) -> BitPackedArray {
    let bit_width = 9u8;
    let cap = 1u16 << bit_width;
    let values = (0..len)
        .map(|_| {
            if rng.random_bool(fraction_patched) {
                rng.random_range(cap..u16::MAX)
            } else {
                rng.random_range(0..cap)
            }
        })
        .collect::<BufferMut<u16>>();
    let array = PrimitiveArray::new(values, Validity::NonNullable);
    BitPackedData::encode(
        &array.into_array(),
        bit_width,
        &mut SESSION.create_execution_ctx(),
    )
    .vortex_expect("encode")
}

fn make_chunks(len: usize, count: usize, fraction_patched: f64) -> Vec<BitPackedArray> {
    let mut rng = StdRng::seed_from_u64(0);
    (0..count)
        .map(|_| make_chunk(&mut rng, len, fraction_patched))
        .collect()
}

fn single(chunks: &[BitPackedArray]) -> ArrayRef {
    if chunks.len() == 1 {
        chunks[0].clone().into_array()
    } else {
        ChunkedArray::from_iter(chunks.iter().map(|c| c.clone().into_array())).into_array()
    }
}

#[cfg(not(codspeed))]
#[divan::bench(args = ARGS)]
fn canonical_cast(bencher: Bencher, (chunk_len, chunk_count, frac): (usize, usize, f64)) {
    let chunks = make_chunks(chunk_len, chunk_count, frac);
    bencher
        .with_inputs(|| (single(&chunks), SESSION.create_execution_ctx()))
        .bench_refs(|(array, ctx)| {
            array
                .clone()
                .cast(U32)
                .unwrap()
                .execute::<PrimitiveArray>(ctx)
                .unwrap()
        });
}

#[cfg(not(codspeed))]
#[divan::bench(args = ARGS)]
fn pushdown(bencher: Bencher, (chunk_len, chunk_count, frac): (usize, usize, f64)) {
    let chunks = make_chunks(chunk_len, chunk_count, frac);
    let total = chunk_len * chunk_count;
    bencher
        .with_inputs(|| {
            (
                chunks.clone(),
                PrimitiveBuilder::<u32>::with_capacity(Nullability::NonNullable, total),
                SESSION.create_execution_ctx(),
            )
        })
        .bench_refs(|(chunks, builder, ctx)| {
            for chunk in chunks.iter() {
                unpack_and_cast_into_builder::<u16, u32>(chunk.as_view(), builder, ctx).unwrap();
            }
            builder.finish_into_primitive()
        });
}
