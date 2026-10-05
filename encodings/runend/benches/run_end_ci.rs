// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Paired native comparison of the original loop, PR #10296, and forced kernels.
//!
//! On AArch64, pass `--cardinality` to compare u32 dictionaries of 4, 8, 16 and 32 values.

#![allow(dead_code, clippy::cast_possible_truncation, clippy::print_stdout)]

use std::fmt::Debug;
use std::hint::black_box;
use std::time::Duration;
use std::time::Instant;

use itertools::Itertools;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_buffer::BufferMut;

#[path = "../src/fill.rs"]
mod candidate;
#[cfg(target_arch = "aarch64")]
mod ci_cardinality;
// Written from the pinned PR commit by the CI workflow before compilation.
#[path = "ci_support/forced_exact.rs"]
#[allow(unused_variables)]
mod forced_exact;
#[path = "ci_support/forced_head.rs"]
#[allow(unused_variables)]
mod forced_head;
#[path = "ci_support/pr.rs"]
mod pr;
#[path = "ci_support/staged_direct.rs"]
#[allow(unused_variables)]
mod staged_direct;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const ROUNDS: usize = 12;
const BATCH: usize = 64;

#[inline(never)]
fn baseline<T: Copy>(ends: &[u64], values: &[T], length: usize) -> BufferMut<T> {
    let mut decoded = BufferMut::with_capacity(length);
    for (end, &value) in ends
        .iter()
        .map(|&end| (end as usize).min(length))
        .zip_eq(values)
    {
        assert!(end >= decoded.len());
        assert!(end <= length);
        // SAFETY: the assertions establish enough capacity for this run.
        unsafe { decoded.push_n_unchecked(value, end - decoded.len()) };
    }
    decoded
}

#[inline(never)]
fn current<T: Copy>(ends: &[u64], values: &[T], length: usize) -> BufferMut<T> {
    pr::decode_runs(
        ends.iter()
            .map(|&end| (end as usize).min(length))
            .zip_eq(values.iter().copied()),
        length,
    )
}

#[inline(never)]
fn proposed<T: Copy>(ends: &[u64], values: &[T], length: usize) -> BufferMut<T> {
    candidate::decode_runs(
        ends.iter()
            .map(|&end| (end as usize).min(length))
            .zip_eq(values.iter().copied()),
        length,
    )
}

macro_rules! forced_wrapper {
    ($name:ident, $module:ident) => {
        #[inline(never)]
        fn $name<T: Copy>(ends: &[u64], values: &[T], length: usize) -> BufferMut<T> {
            $module::decode_runs(
                ends.iter()
                    .map(|&end| (end as usize).min(length))
                    .zip_eq(values.iter().copied()),
                length,
            )
        }
    };
}
forced_wrapper!(head_only, forced_head);
forced_wrapper!(exact_only, forced_exact);
forced_wrapper!(staged_old, staged_direct);

fn describe<T>(name: &str, length: usize, pattern: &str, ends: &[u64]) {
    let (head, threshold) = match size_of::<T>() {
        1 => (32, 24),
        2 => (32, 40),
        4 => (16, 60),
        8 => (8, 25),
        _ => (0, 0),
    };
    let mut start = 0;
    for (segment, chunk) in ends.chunks(1024).enumerate() {
        let mut prev = start;
        let mut min = usize::MAX;
        let mut max = 0;
        let mut singles = 0;
        let mut tails = 0;
        let mut head_excess = 0;
        for &end in chunk {
            let n = end as usize - prev;
            min = min.min(n);
            max = max.max(n);
            singles += usize::from(n == 1);
            tails += usize::from(n > head && head > 0);
            head_excess += head.saturating_sub(n) * size_of::<T>();
            prev = end as usize;
        }
        let rows = prev - start;
        let selected = if head == 0 {
            "wide_exact"
        } else if rows < threshold * chunk.len() {
            "head_tail"
        } else {
            "exact"
        };
        eprintln!(
            "{name},{length},{pattern},{segment},{},{rows},{:.3},{min},{max},{singles},{tails},{head_excess},{selected}",
            chunk.len(),
            rows as f64 / chunk.len() as f64
        );
        start = prev;
    }
}

fn run<T: Copy + PartialEq + Debug>(name: &str, make: impl Fn(usize) -> T) {
    for length in [1000, 10_000, 100_000] {
        for pattern in [
            "fixed1",
            "fixed2",
            "fixed3",
            "fixed4",
            "fixed8",
            "fixed16",
            "fixed64",
            "fixed256",
            "fixed1024",
            "fixed4096",
            "fixed16384",
            "fixed65536",
            "alternating1_31",
            "mixed1_511",
            "one_long_per1024",
            "random16",
            "random128",
            "geometric4",
            "clickbench_like",
        ] {
            let mut rng = StdRng::seed_from_u64(0);
            let mut ends = Vec::new();
            let mut pos = 0;
            while pos < length {
                let n = match pattern {
                    "random16" => rng.random_range(1..=16),
                    "random128" => rng.random_range(1..=128),
                    "geometric4" => {
                        let mut n = 1;
                        while rng.random_bool(0.75) {
                            n += 1;
                        }
                        n
                    }
                    "clickbench_like" => {
                        if rng.random_bool(0.5) {
                            1
                        } else {
                            rng.random_range(2..=200)
                        }
                    }
                    "alternating1_31" => {
                        if ends.len() % 2 == 0 {
                            1
                        } else {
                            31
                        }
                    }
                    "mixed1_511" => {
                        if rng.random_bool(0.5) {
                            1
                        } else {
                            511
                        }
                    }
                    "one_long_per1024" => {
                        if ends.len() % 1024 == 1023 {
                            65536
                        } else {
                            1
                        }
                    }
                    "fixed1" => 1,
                    "fixed2" => 2,
                    "fixed3" => 3,
                    "fixed4" => 4,
                    "fixed8" => 8,
                    "fixed16" => 16,
                    "fixed64" => 64,
                    "fixed1024" => 1024,
                    "fixed4096" => 4096,
                    "fixed16384" => 16384,
                    "fixed65536" => 65536,
                    _ => 256,
                };
                pos = (pos + n).min(length);
                ends.push(pos as u64);
            }
            describe::<T>(name, length, pattern, &ends);
            let values: Vec<T> = (0..ends.len()).map(&make).collect();
            let mut expected = Vec::with_capacity(length);
            let mut prev = 0;
            for (&end, &value) in ends.iter().zip(&values) {
                expected.extend(std::iter::repeat_n(value, end as usize - prev));
                prev = end as usize;
            }
            type Kernel<T> = fn(&[u64], &[T], usize) -> BufferMut<T>;
            let variants: [(&str, Kernel<T>); 6] = [
                ("baseline", baseline::<T>),
                ("pr", current::<T>),
                ("candidate", proposed::<T>),
                ("forced_head", head_only::<T>),
                ("forced_exact", exact_only::<T>),
                ("staged_direct", staged_old::<T>),
            ];
            for (_, f) in variants {
                assert_eq!(f(&ends, &values, length).as_slice(), expected.as_slice());
                for _ in 0..100 {
                    black_box(f(black_box(&ends), black_box(&values), black_box(length)));
                }
            }
            for round in 0..ROUNDS {
                // Each implementation appears equally often in each position.
                for offset in 0..variants.len() {
                    let (variant, f) = variants[(round + offset) % variants.len()];
                    let start = Instant::now();
                    let mut count = 0;
                    while start.elapsed() < Duration::from_millis(8) {
                        for _ in 0..BATCH {
                            black_box(f(black_box(&ends), black_box(&values), black_box(length)));
                        }
                        count += BATCH;
                    }
                    println!(
                        "{name},{length},{pattern},{variant},{round},{:.3}",
                        start.elapsed().as_nanos() as f64 / count as f64
                    );
                }
            }
        }
    }
}

fn main() {
    #[cfg(target_arch = "aarch64")]
    if std::env::args().any(|arg| arg == "--cardinality") {
        ci_cardinality::run();
        return;
    }
    println!("type,length,pattern,variant,round,ns");
    eprintln!(
        "type,length,pattern,segment,runs,rows,mean,min,max,singletons,tails,head_excess_bytes,selected"
    );
    run("u8", |i| i as u8);
    run("u32", |i| i as u32);
    run("u64", |i| i as u64);
    run("i128", |i| i as i128);
    run("BinaryView", |i| BinaryView::from((i as u128) << 32));
    run("wide32", |i| [i as u64; 4]);
}
