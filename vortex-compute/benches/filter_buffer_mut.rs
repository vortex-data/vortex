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

// Buffer size to test - focusing on 1024 for now
const BUFFER_SIZE: usize = 1024;

// Pattern types for testing.
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

/// Creates a test buffer filled with sequential values.
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

/// Generates a mask with the specified selectivity and pattern.
fn generate_mask(len: usize, selectivity: f64, pattern: Pattern) -> Mask {
    #[expect(clippy::cast_possible_truncation)]
    #[expect(clippy::cast_sign_loss)]
    let num_selected = ((len as f64) * selectivity).round() as usize;

    let selection = match pattern {
        Pattern::Random => {
            // Random selection - distribute selected elements randomly.
            // Use a deterministic pattern for reproducibility.
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
            // One contiguous block in the middle.
            let mut selection = vec![false; len];
            let start = (len.saturating_sub(num_selected)) / 2;
            for i in start..(start + num_selected).min(len) {
                selection[i] = true;
            }
            selection
        }
        Pattern::Alternating => {
            // Select every nth element to achieve desired selectivity.
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

// ===== PRIMARY BENCHMARK: Full Selectivity Spectrum =====
// This shows performance across the entire selectivity range
// with extra detail around the 80% threshold.

// Macro to generate a type/size benchmark module with all selectivity benchmarks.
macro_rules! type_size_bench_group {
    ($mod_name:ident, $type:ty) => {
        #[divan::bench_group]
        mod $mod_name {
            use super::*;
            type T = $type;
            const SIZE: usize = BUFFER_SIZE;

            // Inner macro for generating individual selectivity benchmarks.
            macro_rules! selectivity_bench {
                ($name: ident,$selectivity: expr) => {
                    #[divan::bench(sample_count = 1000)]
                    fn $name(bencher: Bencher) {
                        bencher
                            .with_inputs(|| {
                                let buffer = create_test_buffer::<T>(SIZE);
                                let mask = generate_mask(SIZE, $selectivity, Pattern::Random);
                                (buffer, mask)
                            })
                            .bench_values(|(mut buffer, mask)| {
                                buffer.filter(&mask);
                                divan::black_box(buffer);
                            });
                    }
                };
            }

            // Generate benchmarks for each selectivity level.
            selectivity_bench!(sel_01_percent, 0.01);
            selectivity_bench!(sel_25_percent, 0.25);
            selectivity_bench!(sel_50_percent, 0.50);
            selectivity_bench!(sel_75_percent, 0.75);
            selectivity_bench!(sel_78_percent, 0.78);
            selectivity_bench!(sel_79_percent, 0.79);
            selectivity_bench!(sel_80_percent, 0.80);
            selectivity_bench!(sel_81_percent, 0.81);
            selectivity_bench!(sel_82_percent, 0.82);
            selectivity_bench!(sel_85_percent, 0.85);
            selectivity_bench!(sel_99_percent, 0.99);
        }
    };
}

// Generate benchmark modules for each type.
type_size_bench_group!(u8_1024, u8);
type_size_bench_group!(u32_1024, u32);
type_size_bench_group!(u64_1024, u64);

// ===== PATTERN COMPARISON AT THRESHOLD =====
// Test different patterns but ONLY at the 80% threshold where the algorithm choice matters most.
// This tests whether certain patterns perform better with the index-based vs slice-based approach.

#[divan::bench_group]
mod u32_1024_patterns {
    use super::*;
    type T = u32;
    const SIZE: usize = BUFFER_SIZE;
    const SELECTIVITY: f64 = 0.80;

    #[divan::bench(sample_count = 1000)]
    fn random(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let buffer = create_test_buffer::<T>(SIZE);
                let mask = generate_mask(SIZE, SELECTIVITY, Pattern::Random);
                (buffer, mask)
            })
            .bench_values(|(mut buffer, mask)| {
                buffer.filter(&mask);
                divan::black_box(buffer);
            });
    }

    #[divan::bench(sample_count = 1000)]
    fn contiguous(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let buffer = create_test_buffer::<T>(SIZE);
                let mask = generate_mask(SIZE, SELECTIVITY, Pattern::Contiguous);
                (buffer, mask)
            })
            .bench_values(|(mut buffer, mask)| {
                buffer.filter(&mask);
                divan::black_box(buffer);
            });
    }

    #[divan::bench(sample_count = 1000)]
    fn alternating(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let buffer = create_test_buffer::<T>(SIZE);
                let mask = generate_mask(SIZE, SELECTIVITY, Pattern::Alternating);
                (buffer, mask)
            })
            .bench_values(|(mut buffer, mask)| {
                buffer.filter(&mask);
                divan::black_box(buffer);
            });
    }
}

// ===== LARGE ELEMENT BENCHMARKS =====
// Test with larger element sizes at the critical threshold range to understand
// how memcpy performance affects the algorithms.

#[derive(Copy, Clone, Default)]
#[allow(dead_code)]
struct LargeElement([u8; 32]);

impl From<u8> for LargeElement {
    fn from(value: u8) -> Self {
        LargeElement([value; 32])
    }
}

#[divan::bench_group]
mod large_elem_1024 {
    use super::*;
    type T = LargeElement;
    const SIZE: usize = BUFFER_SIZE;

    #[divan::bench(sample_count = 1000)]
    fn sel_50_percent(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let buffer = create_test_buffer::<T>(SIZE);
                let mask = generate_mask(SIZE, 0.50, Pattern::Random);
                (buffer, mask)
            })
            .bench_values(|(mut buffer, mask)| {
                buffer.filter(&mask);
                divan::black_box(buffer);
            });
    }

    #[divan::bench(sample_count = 1000)]
    fn sel_75_percent(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let buffer = create_test_buffer::<T>(SIZE);
                let mask = generate_mask(SIZE, 0.75, Pattern::Random);
                (buffer, mask)
            })
            .bench_values(|(mut buffer, mask)| {
                buffer.filter(&mask);
                divan::black_box(buffer);
            });
    }

    #[divan::bench(sample_count = 1000)]
    fn sel_79_percent(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let buffer = create_test_buffer::<T>(SIZE);
                let mask = generate_mask(SIZE, 0.79, Pattern::Random);
                (buffer, mask)
            })
            .bench_values(|(mut buffer, mask)| {
                buffer.filter(&mask);
                divan::black_box(buffer);
            });
    }

    #[divan::bench(sample_count = 1000)]
    fn sel_80_percent(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let buffer = create_test_buffer::<T>(SIZE);
                let mask = generate_mask(SIZE, 0.80, Pattern::Random);
                (buffer, mask)
            })
            .bench_values(|(mut buffer, mask)| {
                buffer.filter(&mask);
                divan::black_box(buffer);
            });
    }

    #[divan::bench(sample_count = 1000)]
    fn sel_81_percent(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let buffer = create_test_buffer::<T>(SIZE);
                let mask = generate_mask(SIZE, 0.81, Pattern::Random);
                (buffer, mask)
            })
            .bench_values(|(mut buffer, mask)| {
                buffer.filter(&mask);
                divan::black_box(buffer);
            });
    }

    #[divan::bench]
    fn sel_85_percent(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let buffer = create_test_buffer::<T>(SIZE);
                let mask = generate_mask(SIZE, 0.85, Pattern::Random);
                (buffer, mask)
            })
            .bench_values(|(mut buffer, mask)| {
                buffer.filter(&mask);
                divan::black_box(buffer);
            });
    }

    #[divan::bench]
    fn sel_90_percent(bencher: Bencher) {
        bencher
            .with_inputs(|| {
                let buffer = create_test_buffer::<T>(SIZE);
                let mask = generate_mask(SIZE, 0.90, Pattern::Random);
                (buffer, mask)
            })
            .bench_values(|(mut buffer, mask)| {
                buffer.filter(&mask);
                divan::black_box(buffer);
            });
    }
}
