// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! In-place filter benchmarks for `BufferMut`.

use std::fmt;

use divan::Bencher;
use vortex_buffer::BufferMut;
use vortex_compute::filter::Filter;
use vortex_mask::Mask;

fn main() {
    divan::main();
}

const BUFFER_SIZE: usize = 1024;

// Full selectivity spectrum with extra detail around the 80% threshold.
const SELECTIVITIES: &[f64] = &[
    0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.75, 0.78, 0.79, 0.80, 0.81, 0.82, 0.85, 0.99,
];

#[derive(Copy, Clone, Debug)]
enum Pattern {
    Random,
    Contiguous,
    Alternating,
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Pattern::Random => write!(f, "random"),
            Pattern::Contiguous => write!(f, "contiguous"),
            Pattern::Alternating => write!(f, "alternating"),
        }
    }
}

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

fn generate_mask(len: usize, selectivity: f64, pattern: Pattern) -> Mask {
    #[expect(clippy::cast_possible_truncation)]
    #[expect(clippy::cast_sign_loss)]
    let num_selected = ((len as f64) * selectivity).round() as usize;

    let selection = match pattern {
        Pattern::Random => {
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
            selection
        }
        Pattern::Contiguous => {
            let mut selection = vec![false; len];
            let start = (len.saturating_sub(num_selected)) / 2;
            for i in start..(start + num_selected).min(len) {
                selection[i] = true;
            }
            selection
        }
        Pattern::Alternating => {
            let mut selection = vec![false; len];
            if num_selected > 0 {
                let step = len.max(1) / num_selected.max(1);
                let step = step.max(1);
                for i in (0..len).step_by(step).take(num_selected) {
                    selection[i] = true;
                }
            }
            selection
        }
    };

    Mask::from_iter(selection)
}

#[divan::bench(types = [u8, u32, u64], args = SELECTIVITIES, sample_count = 1000)]
fn filter_selectivity<T: Copy + Default + From<u8>>(bencher: Bencher, selectivity: f64) {
    let mask = generate_mask(BUFFER_SIZE, selectivity, Pattern::Random);
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

#[divan::bench_group(sample_count = 1000)]
mod patterns_at_threshold {
    use super::*;

    const SELECTIVITY: f64 = 0.50;

    #[divan::bench]
    fn random(bencher: Bencher) {
        let mask = generate_mask(BUFFER_SIZE, SELECTIVITY, Pattern::Random);
        bencher
            .with_inputs(|| {
                let buffer = create_test_buffer::<u32>(BUFFER_SIZE);
                (buffer, mask.clone())
            })
            .bench_values(|(mut buffer, mask)| {
                buffer.filter(&mask);
                divan::black_box(buffer);
            });
    }

    #[divan::bench]
    fn contiguous(bencher: Bencher) {
        let mask = generate_mask(BUFFER_SIZE, SELECTIVITY, Pattern::Contiguous);
        bencher
            .with_inputs(|| {
                let buffer = create_test_buffer::<u32>(BUFFER_SIZE);
                (buffer, mask.clone())
            })
            .bench_values(|(mut buffer, mask)| {
                buffer.filter(&mask);
                divan::black_box(buffer);
            });
    }

    #[divan::bench]
    fn alternating(bencher: Bencher) {
        let mask = generate_mask(BUFFER_SIZE, SELECTIVITY, Pattern::Alternating);
        bencher
            .with_inputs(|| {
                let buffer = create_test_buffer::<u32>(BUFFER_SIZE);
                (buffer, mask.clone())
            })
            .bench_values(|(mut buffer, mask)| {
                buffer.filter(&mask);
                divan::black_box(buffer);
            });
    }
}

#[derive(Copy, Clone, Default)]
#[allow(dead_code)]
struct LargeElement([u8; 32]);

impl From<u8> for LargeElement {
    fn from(value: u8) -> Self {
        LargeElement([value; 32])
    }
}

#[divan::bench(args = SELECTIVITIES, sample_count = 1000)]
fn filter_large_element(bencher: Bencher, selectivity: f64) {
    let mask = generate_mask(BUFFER_SIZE, selectivity, Pattern::Random);
    bencher
        .with_inputs(|| {
            let buffer = create_test_buffer::<LargeElement>(BUFFER_SIZE);
            (buffer, mask.clone())
        })
        .bench_values(|(mut buffer, mask)| {
            buffer.filter(&mask);
            divan::black_box(buffer);
        });
}
