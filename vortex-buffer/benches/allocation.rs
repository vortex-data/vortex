// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use allocator_api2::alloc::Global;
use arrow_buffer::MutableBuffer;
use bytes::BytesMut;
use divan::Bencher;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;

const SIZES: &[usize] = &[0, 64, 256, 1024, 16_384, 65_536];

fn main() {
    divan::main();
}

#[divan::bench(args = SIZES)]
fn allocate_drop_vortex(bencher: Bencher, size: usize) {
    bencher.bench(|| drop(BufferMut::<u8>::with_capacity(size)));
}

#[divan::bench(args = SIZES)]
fn allocate_drop_vortex_custom(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| BufferAllocatorRef::new(Global))
        .bench_refs(|allocator| drop(allocator.with_capacity::<u8>(size)));
}

#[divan::bench(args = SIZES)]
fn allocate_drop_vortex_minimal_alignment(bencher: Bencher, size: usize) {
    bencher.bench(|| {
        drop(BufferMut::<u8>::with_capacity_preferred_aligned(
            size,
            Alignment::of::<u8>(),
            None,
        ))
    });
}

#[divan::bench(args = SIZES)]
fn allocate_drop_bytes(bencher: Bencher, size: usize) {
    bencher.bench(|| drop(BytesMut::with_capacity(size)));
}

#[divan::bench(args = SIZES)]
fn allocate_drop_arrow(bencher: Bencher, size: usize) {
    bencher.bench(|| drop(MutableBuffer::with_capacity(size)));
}

#[divan::bench(args = SIZES)]
fn allocate_freeze_drop_vortex(bencher: Bencher, size: usize) {
    bencher.bench(|| drop(BufferMut::<u8>::with_capacity(size).freeze()));
}

#[divan::bench(args = SIZES)]
fn allocate_freeze_drop_vortex_custom(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| BufferAllocatorRef::new(Global))
        .bench_refs(|allocator| drop(allocator.with_capacity::<u8>(size).freeze()));
}

#[divan::bench(args = SIZES)]
fn allocate_freeze_drop_vortex_minimal_alignment(bencher: Bencher, size: usize) {
    bencher.bench(|| {
        drop(
            BufferMut::<u8>::with_capacity_preferred_aligned(size, Alignment::of::<u8>(), None)
                .freeze(),
        )
    });
}

#[divan::bench(args = SIZES)]
fn allocate_freeze_drop_bytes(bencher: Bencher, size: usize) {
    bencher.bench(|| drop(BytesMut::with_capacity(size).freeze()));
}

#[divan::bench(args = SIZES)]
fn allocate_freeze_drop_arrow(bencher: Bencher, size: usize) {
    bencher.bench(|| {
        let buffer: arrow_buffer::Buffer = MutableBuffer::with_capacity(size).into();
        drop(buffer)
    });
}

#[divan::bench(args = SIZES)]
fn from_vec_drop_vortex(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| vec![0u8; size])
        .bench_values(|values| drop(Buffer::from(values)));
}

#[divan::bench(args = SIZES)]
fn from_vec_drop_bytes(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| vec![0u8; size])
        .bench_values(|values| drop(bytes::Bytes::from(values)));
}

#[divan::bench(args = SIZES)]
fn from_vec_drop_arrow(bencher: Bencher, size: usize) {
    bencher
        .with_inputs(|| vec![0u8; size])
        .bench_values(|values| drop(arrow_buffer::Buffer::from_vec(values)));
}
