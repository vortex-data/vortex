// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expand benchmarks for `Buffer`.

use vortex_buffer::Buffer;
use vortex_compute::expand::Expand;
use vortex_mask::Mask;

// buffer size, selectivity
const PARAMETERS: &[(usize, f64)] = &[
    (256, 0.1),
    (256, 0.5),
    (256, 0.9),
    (1024, 0.1),
    (1024, 0.5),
    (1024, 0.9),
    (4096, 0.1),
    (4096, 0.5),
    (4096, 0.9),
    (16384, 0.1),
    (16384, 0.5),
    (16384, 0.9),
];

fn main() {
    divan::main();
}

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

#[divan::bench(types = [u8, u32, u64], args = PARAMETERS, sample_count = 1000)]
fn expand_buffer<T: Copy + Default + From<u8> + Send + 'static>(
    bencher: divan::Bencher,
    (buffer_size, selectivity): (usize, f64),
) {
    bencher
        .with_inputs(|| {
            let mask = generate_mask(buffer_size, selectivity);
            let true_count = mask.true_count();
            let buffer = create_test_buffer::<T>(true_count);
            (buffer, mask)
        })
        .bench_refs(|(buffer, mask)| {
            let result = buffer.expand(mask);
            divan::black_box(result);
        });
}
