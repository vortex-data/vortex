// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Micro-benchmark for ChunkedArray<VarBinView> → canonical.
//!
//! Simulates the hot path from Arrow conversion:
//!   ChunkedArray of Slice(VarBinViewArray) → _canonicalize
//!
//! Each chunk is a zero-copy slice of a source VarBinViewArray that has K
//! backing buffers. Without buffer deduplication the canonical output holds
//! K × chunk_count buffer references; with deduplication it holds K.
//!
//! Vary `k_buffers` and `chunk_count` to expose the K×N blowup.

use divan::Bencher;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::builders::BufferGrowthStrategy;
use vortex_array::builders::VarBinViewBuilder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexExpect;

fn main() {
    divan::main();
}

/// (k_buffers, chunk_count, rows_per_chunk)
///
/// k_buffers controls how many backing buffers the source array has.
/// Without dedup, canonicalization accumulates k_buffers × chunk_count buffer
/// references in the output.
const ARGS: &[(usize, usize, usize)] = &[
    (1, 100, 10),
    (8, 100, 10),
    (64, 100, 10),
    (1, 1_000, 3),
    (8, 1_000, 3),
    (64, 1_000, 3),
];

/// Build a `VarBinViewArray` with exactly `k_buffers` backing buffers by
/// using a fixed small buffer size so the builder flushes on every `k`-th row.
fn make_source(k_buffers: usize, total_rows: usize) -> VarBinViewArray {
    // Each outlined string is 20 bytes. Choose buffer_bytes so that exactly
    // ceil(total_rows / k_buffers) strings fit per buffer.
    const STR_LEN: usize = 20;
    let rows_per_buf = total_rows.div_ceil(k_buffers);
    let buffer_bytes =
        u32::try_from(rows_per_buf * STR_LEN).vortex_expect("buffer size fits in u32");

    let mut builder = VarBinViewBuilder::new(
        DType::Binary(Nullability::NonNullable),
        total_rows,
        Default::default(),
        BufferGrowthStrategy::fixed(buffer_bytes),
        0.0,
    );

    for i in 0..total_rows {
        // 20-byte string — always outlined (> 12 bytes).
        builder.append_value(format!("row{i:016}").as_bytes());
    }

    builder.finish_into_varbinview()
}

/// Build N equal-sized slices of the same source `VarBinViewArray` as a
/// `ChunkedArray`. Every chunk shares all K backing buffers of the source.
fn make_chunks(k_buffers: usize, chunk_count: usize, rows_per_chunk: usize) -> ChunkedArray {
    let source = make_source(k_buffers, chunk_count * rows_per_chunk);
    (0..chunk_count)
        .map(|i| {
            source
                .slice(i * rows_per_chunk..(i + 1) * rows_per_chunk)
                .vortex_expect("slice within bounds")
        })
        .collect::<ChunkedArray>()
}

#[divan::bench(args = ARGS)]
fn chunked_varbinview_canonicalize(
    bencher: Bencher,
    (k_buffers, chunk_count, rows_per_chunk): (usize, usize, usize),
) {
    let chunks = make_chunks(k_buffers, chunk_count, rows_per_chunk).into_array();

    bencher
        .with_inputs(|| &chunks)
        .bench_refs(|chunks| chunks.to_canonical())
}
