// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::cast_possible_truncation)]

use arrow_buffer::BooleanBuffer;
use divan::{Bencher, black_box};
use vortex_array::pipeline::bits::{
    BitAlignedChunkedIterator, MaskSliceIterator, TrueSliceIterator,
};
use vortex_array::pipeline::{N, N_WORDS};

fn create_test_data(len: usize, pattern: fn(usize) -> bool) -> Vec<u8> {
    // Ensure data is 64-bit (8-byte) aligned
    let byte_len = len.div_ceil(8);
    let aligned_byte_len = (byte_len + 7) & !7; // Round up to nearest multiple of 8

    // Create aligned vector by using u64 allocation and converting to bytes
    let u64_len = aligned_byte_len / 8;
    let mut u64_data = vec![0u64; u64_len];

    for i in 0..len {
        if pattern(i) {
            let u64_idx = i / 64;
            let bit_idx = i % 64;
            u64_data[u64_idx] |= 1u64 << bit_idx;
        }
    }

    // Convert to bytes while maintaining alignment
    let bytes =
        unsafe { std::slice::from_raw_parts(u64_data.as_ptr() as *const u8, aligned_byte_len) }
            .to_vec();

    bytes
}

#[divan::bench(args = [0, 1, 7, 13])]
fn boolean_buffer_method(bencher: Bencher, bit_offset: usize) {
    let total_bits = 100 * N; // 100 chunks worth of data (already 64-bit aligned since N=1024)
    let data = create_test_data(total_bits + bit_offset, |i| i % 3 == 0);

    bencher
        .with_inputs(|| BooleanBuffer::new(data.clone().into(), bit_offset, total_bits))
        .bench_values(|buffer| {
            let bit_chunks = buffer.bit_chunks();
            let mut chunk_iter = bit_chunks.iter();
            let remainder = bit_chunks.remainder_bits();
            let mut done = false;

            // Manually collect 16 u64 chunks to form one [usize; N_WORDS] array
            while !done {
                let mut chunk_array = [0usize; N_WORDS];

                for i in 0..N_WORDS {
                    if let Some(u64_chunk) = chunk_iter.next() {
                        chunk_array[i] = black_box(u64_chunk as usize);
                    } else {
                        done = true;
                        break;
                    }
                    black_box(chunk_array.as_slice());
                }
            }

            if remainder != 0 {
                let mut chunk_array = [0usize; N_WORDS];
                chunk_array[0] = remainder as usize;
                black_box(chunk_array.as_slice());
            }
        })
}

#[divan::bench(args = [0, 1, 7, 13])]
fn bit_aligned_iterator_method(bencher: Bencher, bit_offset: usize) {
    let total_bits = 100 * N; // 100 chunks worth of data (already 64-bit aligned since N=1024)
    let data = create_test_data(total_bits + bit_offset, |i| i % 3 == 0);
    let buffer = BooleanBuffer::new(data.into(), bit_offset, total_bits);
    let true_count = buffer.count_set_bits();

    bencher.with_inputs(|| (&buffer, true_count)).bench_values(|(buffer, true_count)| {
        let mut iter = BitAlignedChunkedIterator::new(buffer, true_count);

        while let Some(chunk) = black_box(iter.next_chunk()) {
            // Just test getting the slice, don't push to vector
            black_box(chunk);
        }
    })
}

fn main() {
    divan::main();
}
