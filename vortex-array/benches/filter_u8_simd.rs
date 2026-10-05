// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Direct comparison of the production compression kernels, including in-place compaction.

#![allow(clippy::unwrap_used)]

use std::fmt::Debug;

use divan::Bencher;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_mask::Mask;

// Compile the production sources into this benchmark so no public benchmark-only API is needed.
#[allow(dead_code)]
#[path = "../src/arrays/filter/execute/simd_compress/mod.rs"]
mod simd_compress;
#[allow(dead_code)]
#[path = "../src/arrays/filter/execute/slice.rs"]
mod slice;

fn main() {
    validate::<u8>();
    divan::main();
}

fn values<T: From<u8>>(len: usize) -> Vec<T> {
    (0..len)
        .map(|i| T::from(u8::try_from(i % 251).unwrap()))
        .collect()
}

fn mask(len: usize, offset: usize, density: f64) -> Mask {
    let mut state = 0x1234_5678_9abc_def0u64;
    let bits = BitBuffer::from_iter((0..len + offset).map(|_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f64 / u64::MAX as f64) < density
    }));
    Mask::from_buffer(bits.slice(offset..len + offset))
}

fn validate<T: From<u8> + Copy + PartialEq + Debug>() {
    let allocator = BufferAllocatorRef::statically_allocated();
    for offset in 0..8 {
        for len in [64, 65, 127, 128, 151, 513] {
            let bits = BitBuffer::collect_bool(len + offset, |i| i % 5 < 3);
            let mask = Mask::from_buffer(bits.slice(offset..len + offset));
            let mask = mask.values().unwrap();
            let input = values::<T>(len);
            let expected = slice::filter_slice_by_bitmap(&input, mask, &allocator);
            let actual = simd_compress::filter_slice_by_bitmap(&input, mask, &allocator).unwrap();
            assert_eq!(actual.as_slice(), expected.as_slice());
            let mut compacted = input;
            let written = simd_compress::filter_slice_mut_by_bitmap(&mut compacted, mask).unwrap();
            assert_eq!(&compacted[..written], expected.as_slice());
        }
    }
}

#[divan::bench(types = [u8], args = [0.6, 0.75])]
fn allocated<T: From<u8> + Copy + Send + Sync>(bencher: Bencher, density: f64) {
    let values = values::<T>(65_536 / size_of::<T>());
    let mask = mask(values.len(), 0, density);
    let mask = mask.values().unwrap();
    let allocator = BufferAllocatorRef::statically_allocated();
    bencher.bench(|| {
        simd_compress::filter_slice_by_bitmap(divan::black_box(&values), mask, &allocator).unwrap()
    });
}

#[divan::bench(types = [u8], args = [0.6, 0.75])]
fn in_place<T: From<u8> + Copy + Send + Sync>(bencher: Bencher, density: f64) {
    let values = values::<T>(65_536 / size_of::<T>());
    let mask = mask(values.len(), 0, density);
    let mask = mask.values().unwrap();
    bencher.with_inputs(|| values.clone()).bench_refs(|values| {
        simd_compress::filter_slice_mut_by_bitmap(divan::black_box(values), mask).unwrap()
    });
}
