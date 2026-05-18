// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks `SerializedArray::buffer_lengths()` against the old `root::<Array>`-based path.
//!
//! Motivation: `buffer_lengths()` is called per `SerializedArray` (and called multiple times in
//! the display path). The previous implementation re-ran the full FlatBuffer verifier on every
//! call, even though the buffer was already validated at construction time. This bench measures
//! the actual saving.

#![expect(clippy::cast_possible_truncation)]
#![expect(clippy::unwrap_used)]

use divan::Bencher;
use flatbuffers::FlatBufferBuilder;
use flatbuffers::root;
use vortex_array::serde::SerializedArray;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_flatbuffers::array as fba;

fn main() {
    divan::main();
}

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

/// Build an Array tree: struct of `n_fields` flat leaves, each owning one buffer.
fn build_struct_array_bytes(n_fields: usize) -> ByteBuffer {
    let mut fbb = FlatBufferBuilder::with_capacity(1 << 16);
    let leaves: Vec<_> = (0..n_fields)
        .map(|_| build_array_node(&mut fbb, 1, vec![]))
        .collect();
    let root_node = build_array_node(&mut fbb, 0, leaves);
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
    ByteBuffer::from(fbb.finished_data().to_vec()).aligned(Alignment::none())
}

const ARRAY_FIELDS: &[usize] = &[1, 8, 32, 100, 1000];

/// Post-fix implementation: uses `root_as_array_unchecked` on an already-validated buffer.
#[divan::bench(args = ARRAY_FIELDS)]
fn buffer_lengths_fixed(bencher: Bencher, n_fields: usize) {
    let bytes = build_struct_array_bytes(n_fields);
    let sa = SerializedArray::from_array_tree(bytes).unwrap();
    bencher.bench_local(|| {
        let lengths = sa.buffer_lengths();
        divan::black_box(lengths);
    });
}

/// Legacy implementation: re-runs the FlatBuffer verifier on every call.
/// Replicates the pre-fix `buffer_lengths()` body byte-for-byte against the same payload so
/// we can compare apples-to-apples.
#[divan::bench(args = ARRAY_FIELDS)]
fn buffer_lengths_legacy_root(bencher: Bencher, n_fields: usize) {
    let bytes = build_struct_array_bytes(n_fields);
    bencher.bench_local(|| {
        let fb_array = root::<fba::Array>(bytes.as_ref()).unwrap();
        let lengths: Vec<usize> = fb_array
            .buffers()
            .map(|buffers| buffers.iter().map(|b| b.length() as usize).collect())
            .unwrap_or_default();
        divan::black_box(lengths);
    });
}
