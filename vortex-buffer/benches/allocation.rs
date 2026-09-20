// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Allocate-and-drop cost of Vortex buffers next to their `bytes` and Arrow equivalents.
//!
//! A single allocate-and-drop pair takes tens of nanoseconds. CodSpeed's simulation runs each
//! benchmark once and adds a fixed harness cost of roughly half a microsecond of reported time,
//! so a pair on its own measured that cost and moved by more than 10% on pull requests that did
//! not touch Rust code. Every iteration therefore repeats the operation [`BATCH`] times.

use allocator_api2::alloc::Global;
use arrow_buffer::MutableBuffer;
use bytes::BytesMut;
use divan::Bencher;
use mimalloc::MiMalloc;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;

/// Sizes start at 64 bytes: none of these types allocate for a zero-byte request.
const SIZES: &[usize] = &[64, 256, 1024, 16_384, 65_536];

/// Allocate-and-drop pairs per timed iteration.
const BATCH: usize = 256;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

/// Allocates and drops [`BATCH`] values, black-boxing each one so the pair cannot be elided.
fn allocate_drop_batch<T>(mut allocate: impl FnMut() -> T) {
    for _ in 0..BATCH {
        drop(divan::black_box(allocate()));
    }
}

/// One batch of zero-filled vectors, built outside the timed region.
///
/// Converting a `Vec` copies it when the allocation is not aligned for the buffer, so the batch
/// shrinks with the size to keep each iteration at about a mebibyte of copying.
fn vec_batch(size: usize) -> Vec<Vec<u8>> {
    let len = ((1 << 20) / size).clamp(16, BATCH);
    (0..len).map(|_| vec![0u8; size]).collect()
}

#[divan::bench(args = SIZES)]
fn allocate_drop_vortex(bencher: Bencher, size: usize) {
    bencher.bench(|| allocate_drop_batch(|| BufferMut::<u8>::with_capacity(size)));
}

#[divan::bench(args = SIZES)]
fn allocate_drop_vortex_custom(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| BufferAllocatorRef::new(Global))
        .bench_refs(|allocator| allocate_drop_batch(|| allocator.with_capacity::<u8>(size)));
}

#[divan::bench(args = SIZES)]
fn allocate_drop_vortex_minimal_alignment(bencher: Bencher, size: usize) {
    bencher.bench(|| {
        allocate_drop_batch(|| {
            BufferMut::<u8>::with_capacity_preferred_aligned(size, Alignment::of::<u8>(), None)
        })
    });
}

#[divan::bench(args = SIZES)]
fn allocate_drop_bytes(bencher: Bencher, size: usize) {
    bencher.bench(|| allocate_drop_batch(|| BytesMut::with_capacity(size)));
}

#[divan::bench(args = SIZES)]
fn allocate_drop_arrow(bencher: Bencher, size: usize) {
    bencher.bench(|| allocate_drop_batch(|| MutableBuffer::with_capacity(size)));
}

#[divan::bench(args = SIZES)]
fn allocate_freeze_drop_vortex(bencher: Bencher, size: usize) {
    bencher.bench(|| allocate_drop_batch(|| BufferMut::<u8>::with_capacity(size).freeze()));
}

#[divan::bench(args = SIZES)]
fn allocate_freeze_drop_vortex_custom(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| BufferAllocatorRef::new(Global))
        .bench_refs(|allocator| {
            allocate_drop_batch(|| allocator.with_capacity::<u8>(size).freeze())
        });
}

#[divan::bench(args = SIZES)]
fn allocate_freeze_drop_vortex_minimal_alignment(bencher: Bencher, size: usize) {
    bencher.bench(|| {
        allocate_drop_batch(|| {
            BufferMut::<u8>::with_capacity_preferred_aligned(size, Alignment::of::<u8>(), None)
                .freeze()
        })
    });
}

#[divan::bench(args = SIZES)]
fn allocate_freeze_drop_bytes(bencher: Bencher, size: usize) {
    bencher.bench(|| allocate_drop_batch(|| BytesMut::with_capacity(size).freeze()));
}

#[divan::bench(args = SIZES)]
fn allocate_freeze_drop_arrow(bencher: Bencher, size: usize) {
    bencher.bench(|| {
        allocate_drop_batch(|| arrow_buffer::Buffer::from(MutableBuffer::with_capacity(size)))
    });
}

#[divan::bench(args = SIZES)]
fn from_vec_drop_vortex(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| vec_batch(size))
        .bench_values(|vecs| {
            for values in vecs {
                drop(divan::black_box(Buffer::from(values)));
            }
        });
}

#[divan::bench(args = SIZES)]
fn from_vec_drop_bytes(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| vec_batch(size))
        .bench_values(|vecs| {
            for values in vecs {
                drop(divan::black_box(bytes::Bytes::from(values)));
            }
        });
}

#[divan::bench(args = SIZES)]
fn from_vec_drop_arrow(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| vec_batch(size))
        .bench_values(|vecs| {
            for values in vecs {
                drop(divan::black_box(arrow_buffer::Buffer::from_vec(values)));
            }
        });
}
