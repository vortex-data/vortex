// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Run expansion kernels on identical inputs, without array construction or validity:
//!
//! - `baseline`: one `push_n` per run, the loop before segmented decoding.
//! - `segmented`: `decode_runs` from `src/fill.rs`, compiled for the build's target features.
//! - `fearless`: the same algorithm on fearless_simd vectors, dispatched at runtime.

#![expect(clippy::cast_possible_truncation)]

use std::fmt;

use divan::Bencher;
use mimalloc::MiMalloc;
use num_traits::NumCast;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_buffer::BufferMut;

#[path = "../../src/fill.rs"]
#[expect(dead_code)]
mod fill;
mod simd;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() {
    divan::main();
}

const LENGTH: usize = 100_000;

/// Run lengths whose mean misleads the per-segment kernel choice, or that the old per-run loop
/// predicts perfectly.
#[derive(Clone, Copy, Debug)]
enum Pattern {
    Fixed(usize),
    /// 1, 31, 1, 31, ...: mean 16, like `Fixed(16)`, but unpredictable per run lengths.
    Alternating1And31,
    /// 1,023 single-element runs, then one of 65,536 rows.
    OneLongPer1024,
    /// Uniform in `1..=16`.
    Random16,
    /// Geometric with mean 4.
    Geometric4,
    /// Half single-element runs, the rest uniform in `2..=200`.
    ClickbenchLike,
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Pattern::Fixed(n) => write!(f, "fixed_{n}"),
            Pattern::Alternating1And31 => write!(f, "alternating_1_31"),
            Pattern::OneLongPer1024 => write!(f, "one_long_per_1024"),
            Pattern::Random16 => write!(f, "random_16"),
            Pattern::Geometric4 => write!(f, "geometric_4"),
            Pattern::ClickbenchLike => write!(f, "clickbench_like"),
        }
    }
}

const PATTERNS: &[Pattern] = &[
    Pattern::Fixed(1),
    Pattern::Fixed(4),
    Pattern::Fixed(16),
    Pattern::Fixed(1024),
    Pattern::Alternating1And31,
    Pattern::OneLongPer1024,
    Pattern::Random16,
    Pattern::Geometric4,
    Pattern::ClickbenchLike,
];

fn ends(pattern: Pattern) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(0);
    let mut ends = Vec::new();
    let mut pos = 0;
    while pos < LENGTH {
        let n = match pattern {
            Pattern::Fixed(n) => n,
            Pattern::Alternating1And31 => {
                if ends.len().is_multiple_of(2) {
                    1
                } else {
                    31
                }
            }
            Pattern::OneLongPer1024 => {
                if ends.len() % 1024 == 1023 {
                    65536
                } else {
                    1
                }
            }
            Pattern::Random16 => rng.random_range(1..=16),
            Pattern::Geometric4 => {
                let mut n = 1;
                while rng.random_bool(0.75) {
                    n += 1;
                }
                n
            }
            Pattern::ClickbenchLike => {
                if rng.random_bool(0.5) {
                    1
                } else {
                    rng.random_range(2..=200)
                }
            }
        };
        pos = (pos + n).min(LENGTH);
        ends.push(pos);
    }
    ends
}

trait Value: simd::Lane + NumCast + PartialEq + fmt::Debug + Sync {}

impl<T: simd::Lane + NumCast + PartialEq + fmt::Debug + Sync> Value for T {}

type Kernel<T> = fn(&[usize], &[T]) -> BufferMut<T>;

fn baseline<T: Copy>(ends: &[usize], values: &[T]) -> BufferMut<T> {
    let mut decoded = BufferMut::with_capacity(LENGTH);
    for (&end, &value) in ends.iter().zip(values) {
        assert!(end >= decoded.len() && end <= LENGTH);
        // SAFETY: the assertion establishes enough capacity for this run.
        unsafe { decoded.push_n_unchecked(value, end - decoded.len()) };
    }
    decoded
}

fn segmented<T: Copy>(ends: &[usize], values: &[T]) -> BufferMut<T> {
    fill::decode_runs(ends.iter().copied().zip(values.iter().copied()), LENGTH)
}

fn fearless<T: simd::Lane>(ends: &[usize], values: &[T]) -> BufferMut<T> {
    simd::decode_runs(ends.iter().copied().zip(values.iter().copied()), LENGTH)
}

fn bench_kernel<T: Value>(
    bencher: Bencher,
    pattern: Pattern,
    kernel: Kernel<T>,
) {
    let ends = ends(pattern);
    // Adjacent runs differ, and every value fits the narrowest type.
    let values: Vec<T> = (0..ends.len())
        .map(|i| T::from(i % 251).expect("fits u8"))
        .collect();
    assert_eq!(
        kernel(&ends, &values).as_slice(),
        baseline(&ends, &values).as_slice()
    );
    bencher.bench(|| kernel(divan::black_box(&ends), divan::black_box(&values)));
}

#[divan::bench(types = [u8, u16, u32, u64], args = PATTERNS)]
fn decode_baseline<T: Value>(
    bencher: Bencher,
    pattern: Pattern,
) {
    bench_kernel::<T>(bencher, pattern, baseline);
}

#[divan::bench(types = [u8, u16, u32, u64], args = PATTERNS)]
fn decode_segmented<T: Value>(
    bencher: Bencher,
    pattern: Pattern,
) {
    bench_kernel::<T>(bencher, pattern, segmented);
}

#[divan::bench(types = [u8, u16, u32, u64], args = PATTERNS)]
fn decode_fearless<T: Value>(
    bencher: Bencher,
    pattern: Pattern,
) {
    bench_kernel::<T>(bencher, pattern, fearless);
}
