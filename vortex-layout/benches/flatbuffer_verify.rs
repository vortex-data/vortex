// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks the cost of FlatBuffers verification for Layout and Array messages.
//!
//! Compares three modes for each shape:
//! - `root::<T>()` — full verification (current default for footer/layout/array decode).
//! - `root_with_opts::<T>()` — verification with the Vortex Layout `VerifierOptions`.
//! - `root_unchecked::<T>() + first field touch` — the unsafe lower bound.
//!
//! The shapes mirror what Vortex actually serializes:
//! - chunked-of-flat (deep, narrow): models row groups.
//! - struct-of-flat (wide): models a wide schema.
//! - chunked-of-struct (both): models a wide schema with row groups.

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use std::env;
use std::sync::LazyLock;

use divan::Bencher;
use flatbuffers::FlatBufferBuilder;
use flatbuffers::VerifierOptions;
use flatbuffers::root;
use flatbuffers::root_unchecked;
use flatbuffers::root_with_opts;
use vortex_flatbuffers::layout as fbl;

fn main() {
    divan::main();
}

static LAYOUT_VERIFIER: LazyLock<VerifierOptions> = LazyLock::new(|| VerifierOptions {
    max_tables: env::var("VORTEX_MAX_LAYOUT_TABLES")
        .ok()
        .and_then(|lmt| lmt.parse::<usize>().ok())
        .unwrap_or(1_000_000),
    max_depth: env::var("VORTEX_MAX_LAYOUT_DEPTH")
        .ok()
        .and_then(|lmt| lmt.parse::<usize>().ok())
        .unwrap_or(64),
    max_apparent_size: 1 << 31,
    ignore_missing_null_terminator: false,
});

// ----- Layout flatbuffer builders -----

/// Build a flat leaf layout: one segment, no children, small metadata.
fn build_flat_leaf<'a>(fbb: &mut FlatBufferBuilder<'a>) -> flatbuffers::WIPOffset<fbl::Layout<'a>> {
    let segments = fbb.create_vector(&[0u32]);
    fbl::Layout::create(
        fbb,
        &fbl::LayoutArgs {
            encoding: 1,
            row_count: 1024,
            metadata: None,
            children: None,
            segments: Some(segments),
        },
    )
}

/// Build a struct layout with `n_fields` flat children.
fn build_struct<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    n_fields: usize,
) -> flatbuffers::WIPOffset<fbl::Layout<'a>> {
    let children: Vec<_> = (0..n_fields).map(|_| build_flat_leaf(fbb)).collect();
    let children = fbb.create_vector(&children);
    fbl::Layout::create(
        fbb,
        &fbl::LayoutArgs {
            encoding: 3, // Columnar
            row_count: 1024,
            metadata: None,
            children: Some(children),
            segments: None,
        },
    )
}

/// Build `n_chunks` chunks of (struct of `n_fields` flat).
fn build_chunked_of_struct(n_chunks: usize, n_fields: usize) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(1 << 16);
    let chunks: Vec<_> = (0..n_chunks)
        .map(|_| build_struct(&mut fbb, n_fields))
        .collect();
    let children = fbb.create_vector(&chunks);
    let root = fbl::Layout::create(
        &mut fbb,
        &fbl::LayoutArgs {
            encoding: 2, // Chunked
            row_count: 1024 * n_chunks as u64,
            metadata: None,
            children: Some(children),
            segments: None,
        },
    );
    fbb.finish_minimal(root);
    fbb.finished_data().to_vec()
}

// ----- Array flatbuffer builders -----

use vortex_flatbuffers::array as fba;

fn build_array_node<'a>(
    fbb: &mut FlatBufferBuilder<'a>,
    n_buffers: usize,
    children: Vec<flatbuffers::WIPOffset<fba::ArrayNode<'a>>>,
) -> flatbuffers::WIPOffset<fba::ArrayNode<'a>> {
    let buffers: Vec<u16> = (0..n_buffers as u16).collect();
    let buffers = fbb.create_vector(&buffers);
    let children = if children.is_empty() {
        None
    } else {
        Some(fbb.create_vector(&children))
    };
    fba::ArrayNode::create(
        fbb,
        &fba::ArrayNodeArgs {
            encoding: 1,
            metadata: None,
            children,
            buffers: Some(buffers),
            stats: None,
        },
    )
}

/// Build an Array tree: top-level struct with `n_fields` flat primitive children.
fn build_struct_array(n_fields: usize) -> Vec<u8> {
    let mut fbb = FlatBufferBuilder::with_capacity(1 << 16);
    let leaves: Vec<_> = (0..n_fields)
        .map(|_| build_array_node(&mut fbb, 1, vec![]))
        .collect();
    let root_node = build_array_node(&mut fbb, 0, leaves);
    // Build a vector of buffer descriptors, one per child.
    let bufs: Vec<_> = (0..n_fields)
        .map(|_| fba::Buffer::new(0, 0, fba::Compression::None, 1024))
        .collect();
    let buffers = fbb.create_vector(&bufs);
    let array = fba::Array::create(
        &mut fbb,
        &fba::ArrayArgs {
            root: Some(root_node),
            buffers: Some(buffers),
        },
    );
    fbb.finish_minimal(array);
    fbb.finished_data().to_vec()
}

// ----- Benchmarks: Layout -----

/// Tuples are (n_chunks, n_fields). Picked to cover small/medium/large/very-wide.
const LAYOUT_SHAPES: &[(usize, usize)] = &[
    (1, 8),     // single chunk, narrow struct           — small footer-like
    (1, 100),   // single chunk, wide struct
    (16, 32),   // 16 chunks of 32-field struct          — medium
    (128, 32),  // 128 chunks of 32-field struct         — large
    (1024, 32), // 1024 chunks                           — very large
    (1, 1000),  // single chunk, very wide               — wide-only
];

#[divan::bench(args = LAYOUT_SHAPES)]
fn layout_root_checked(bencher: Bencher, shape: &(usize, usize)) {
    let bytes = build_chunked_of_struct(shape.0, shape.1);
    bencher.bench(|| {
        let layout = root::<fbl::Layout>(divan::black_box(&bytes)).unwrap();
        divan::black_box(layout.row_count());
    });
}

#[divan::bench(args = LAYOUT_SHAPES)]
fn layout_root_with_opts(bencher: Bencher, shape: &(usize, usize)) {
    let bytes = build_chunked_of_struct(shape.0, shape.1);
    bencher.bench(|| {
        let layout =
            root_with_opts::<fbl::Layout>(&LAYOUT_VERIFIER, divan::black_box(&bytes)).unwrap();
        divan::black_box(layout.row_count());
    });
}

#[divan::bench(args = LAYOUT_SHAPES)]
fn layout_root_unchecked(bencher: Bencher, shape: &(usize, usize)) {
    let bytes = build_chunked_of_struct(shape.0, shape.1);
    bencher.bench(|| {
        // SAFETY: bytes were produced by our own builder above.
        let layout = unsafe { root_unchecked::<fbl::Layout>(divan::black_box(&bytes)) };
        divan::black_box(layout.row_count());
    });
}

/// Report buffer size for context.
#[divan::bench(args = LAYOUT_SHAPES)]
fn layout_buffer_size(shape: &(usize, usize)) -> usize {
    build_chunked_of_struct(shape.0, shape.1).len()
}

// ----- Benchmarks: Array -----

const ARRAY_FIELDS: &[usize] = &[1, 8, 32, 100, 1000];

#[divan::bench(args = ARRAY_FIELDS)]
fn array_root_checked(bencher: Bencher, n_fields: usize) {
    let bytes = build_struct_array(n_fields);
    bencher.bench(|| {
        let array = root::<fba::Array>(divan::black_box(&bytes)).unwrap();
        divan::black_box(array.buffers().map(|b| b.len()));
    });
}

#[divan::bench(args = ARRAY_FIELDS)]
fn array_root_unchecked(bencher: Bencher, n_fields: usize) {
    let bytes = build_struct_array(n_fields);
    bencher.bench(|| {
        // SAFETY: bytes were produced by our own builder above.
        let array = unsafe { root_unchecked::<fba::Array>(divan::black_box(&bytes)) };
        divan::black_box(array.buffers().map(|b| b.len()));
    });
}

#[divan::bench(args = ARRAY_FIELDS)]
fn array_buffer_size(n_fields: usize) -> usize {
    build_struct_array(n_fields).len()
}
