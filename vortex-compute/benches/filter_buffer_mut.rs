// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! In-place filter benchmarks for `BufferMut`.

use divan::Bencher;
use vortex_buffer::BufferMut;
use vortex_compute::filter::Filter;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

const BUFFER_SIZE: usize = 1024;

const SELECTIVITIES: &[f64] = &[
    0.01, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.99,
];

fn create_test_buffer<T>(size: usize) -> BufferMut<T>
where
    T: Copy + Default + From<u8>,
{
    let mut buffer = BufferMut::with_capacity(size);
    for i in 0..size {
        #[expect(clippy::cast_possible_truncation)]
        buffer.push(T::from((i % 256) as u8));
    }
    buffer
}

fn generate_mask(len: usize, selectivity: f64) -> Mask {
    #[expect(clippy::cast_possible_truncation)]
    #[expect(clippy::cast_sign_loss)]
    let num_selected = ((len as f64) * selectivity).round() as usize;

    let mut selection = vec![false; len];
    let mut indices: Vec<usize> = (0..len).collect();

    // Simple deterministic shuffle.
    for i in (1..len).rev() {
        let j = (i * 7 + 13) % (i + 1);
        indices.swap(i, j);
    }

    for i in 0..num_selected.min(len) {
        selection[indices[i]] = true;
    }

    Mask::from_iter(selection)
}

#[derive(Copy, Clone, Default)]
#[allow(dead_code)]
struct LargeElement([u8; 32]);

impl From<u8> for LargeElement {
    fn from(value: u8) -> Self {
        LargeElement([value; 32])
    }
}

#[divan::bench(types = [u8, u32, u64, LargeElement], args = SELECTIVITIES, sample_count = 1000)]
fn filter_selectivity<T: Copy + Default + From<u8>>(bencher: Bencher, selectivity: f64) {
    let mask = generate_mask(BUFFER_SIZE, selectivity);
    bencher
        .with_inputs(|| {
            let buffer = create_test_buffer::<T>(BUFFER_SIZE);
            (buffer, mask.clone())
        })
        .bench_values(|(mut buffer, mask)| {
            buffer.filter(&mask);
            divan::black_box(buffer);
        });
}
