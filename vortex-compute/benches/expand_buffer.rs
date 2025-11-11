// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expand benchmarks for `Buffer`.

use divan::Bencher;
use vortex_buffer::{Buffer, BufferMut};
use vortex_compute::expand::Expand;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

const BUFFER_SIZE: usize = 1024;

const SELECTIVITIES: &[f64] = &[
    0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99,
];

fn create_test_buffer<T>(size: usize) -> Buffer<T>
where
    T: Copy + Default + From<u8> + Send + 'static,
{
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        #[expect(clippy::cast_possible_truncation)]
        data.push(T::from((i % 256) as u8));
    }
    Buffer::from(data)
}

fn create_test_buffer_mut<T>(size: usize) -> BufferMut<T>
where
    T: Copy + Default + From<u8> + Send + 'static,
{
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        #[expect(clippy::cast_possible_truncation)]
        data.push(T::from((i % 256) as u8));
    }
    Buffer::from(data).into_mut()
}

fn generate_mask(len: usize, selectivity: f64) -> Mask {
    let mut selection = vec![false; len];
    let mut indices: Vec<usize> = (0..len).collect();

    // Shuffle indices deterministically.
    const SHUFFLE_MULTIPLIER: usize = 13;
    for idx in (1..len).rev() {
        indices.swap(idx, (idx * SHUFFLE_MULTIPLIER) % (idx + 1));
    }

    #[expect(clippy::cast_possible_truncation)]
    let num_selected = ((len as f64) * selectivity).round() as usize;
    for i in 0..num_selected {
        selection[indices[i]] = true;
    }

    Mask::from_iter(selection)
}

#[divan::bench(types = [u8, u32, u64], args = SELECTIVITIES, sample_count = 1000)]
fn expand_copy<T: Copy + Default + From<u8> + Send + 'static>(bencher: Bencher, selectivity: f64) {
    bencher
        .with_inputs(|| {
            let mask = generate_mask(BUFFER_SIZE, selectivity);
            let true_count = mask.true_count();
            let buffer = create_test_buffer::<T>(true_count);
            (buffer, mask)
        })
        .bench_values(|(buffer, mask)| {
            let result = buffer.expand(&mask);
            divan::black_box(result);
        });
}

#[divan::bench(types = [u8, u32, u64], args = SELECTIVITIES, sample_count = 1000)]
fn expand_inplace<T: Copy + Default + From<u8> + Send + 'static>(
    bencher: Bencher,
    selectivity: f64,
) {
    bencher
        .with_inputs(|| {
            let mask = generate_mask(BUFFER_SIZE, selectivity);
            let true_count = mask.true_count();
            let buffer = create_test_buffer_mut::<T>(true_count);
            (buffer, mask)
        })
        .bench_values(|(mut buffer, mask)| {
            let result = buffer.expand(&mask);
            divan::black_box(result);
        });
}
