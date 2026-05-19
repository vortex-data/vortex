// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Throughput comparison harness for the AVX2 vs AVX-512 take kernels.
//!
//! Run with:
//! ```text
//! cargo test --release -p vortex-array \
//!     arrays::primitive::compute::take::perf_compare::run_perf_compare \
//!     -- --ignored --nocapture
//! ```

#![cfg(test)]
#![cfg(target_arch = "x86_64")]

use std::hint::black_box;
use std::time::Instant;

use rand::SeedableRng;
use rand::distr::Uniform;
use rand::prelude::*;
use rand::rngs::StdRng;

use super::avx2::take_avx2;
use super::avx512::take_avx512;
use super::take_primitive_scalar;

#[derive(Copy, Clone)]
struct Shape {
    label: &'static str,
    values_len: usize,
    indices_len: usize,
}

const SHAPES: &[Shape] = &[
    Shape { label: "tiny  (V=256,    N=1k)", values_len: 256, indices_len: 1_000 },
    Shape { label: "small (V=4k,     N=10k)", values_len: 4_096, indices_len: 10_000 },
    Shape { label: "mid   (V=64k,    N=100k)", values_len: 65_536, indices_len: 100_000 },
    Shape { label: "large (V=1M,     N=1M)", values_len: 1 << 20, indices_len: 1 << 20 },
];

fn avx512_available() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512dq")
        && is_x86_feature_detected!("avx512vl")
}

fn time_run<F: FnMut()>(mut f: F, iters: u32) -> f64 {
    // Warm-up.
    for _ in 0..3 {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_secs_f64() / iters as f64
}

fn pick_iters(indices_len: usize) -> u32 {
    // Aim for a few hundred ms of total work per cell.
    let base: u64 = 50_000_000;
    let est = (base / (indices_len as u64).max(1)).max(20).min(20_000);
    est as u32
}

fn bench_one<V, I>(
    label: &str,
    val_type: &str,
    idx_type: &str,
    values: &[V],
    indices: &[I],
) where
    V: vortex_array::dtype::NativePType,
    I: vortex_array::dtype::UnsignedPType,
{
    let iters = pick_iters(indices.len());

    let scalar = time_run(
        || {
            let out = take_primitive_scalar(values, indices);
            black_box(out);
        },
        iters,
    );

    let avx2 = if is_x86_feature_detected!("avx2") {
        Some(time_run(
            || {
                // SAFETY: runtime-checked.
                let out = unsafe { take_avx2(values, indices) };
                black_box(out);
            },
            iters,
        ))
    } else {
        None
    };

    let avx512 = if avx512_available() {
        Some(time_run(
            || {
                // SAFETY: runtime-checked.
                let out = unsafe { take_avx512(values, indices) };
                black_box(out);
            },
            iters,
        ))
    } else {
        None
    };

    let bytes_per_call = indices.len() * size_of::<V>();
    let gbs = |secs: f64| (bytes_per_call as f64) / secs / 1e9;
    let melems = |secs: f64| (indices.len() as f64) / secs / 1e6;

    let fmt = |t: Option<f64>| match t {
        Some(s) => format!("{:>8.2} M/s ({:>5.2} GB/s)", melems(s), gbs(s)),
        None => "       n/a              ".to_string(),
    };

    let speedup_vs_avx2 = match (avx2, avx512) {
        (Some(a), Some(b)) => format!("  {:.2}x", a / b),
        _ => "        ".to_string(),
    };

    println!(
        "  {:<30} idx={:<3} val={:<3} | scalar {} | avx2 {} | avx512 {} |{}",
        label,
        idx_type,
        val_type,
        fmt(Some(scalar)),
        fmt(avx2),
        fmt(avx512),
        speedup_vs_avx2,
    );
}

fn make_random_indices<I>(rng: &mut StdRng, values_len: usize, indices_len: usize) -> Vec<I>
where
    I: vortex_array::dtype::UnsignedPType + num_traits::NumCast,
{
    let dist = Uniform::new(0u64, values_len as u64).unwrap();
    (0..indices_len)
        .map(|_| I::from(rng.sample(dist)).unwrap())
        .collect()
}

#[test]
#[ignore]
fn run_perf_compare() {
    println!();
    println!(
        "avx2 available: {}, avx512 available: {}",
        is_x86_feature_detected!("avx2"),
        avx512_available()
    );
    println!();

    for &shape in SHAPES {
        let mut rng = StdRng::seed_from_u64(0xC0FFEE_u64 ^ shape.indices_len as u64);

        // i32 values, u32 indices — the bread-and-butter case (16 lanes on AVX-512).
        {
            let values: Vec<i32> = (0..shape.values_len as i32).collect();
            let indices: Vec<u32> = make_random_indices(&mut rng, shape.values_len, shape.indices_len);
            bench_one(shape.label, "i32", "u32", &values, &indices);
        }

        // i64 values, u32 indices — 64-bit values (8 lanes on AVX-512 vs 4 on AVX2).
        {
            let values: Vec<i64> = (0..shape.values_len as i64).collect();
            let indices: Vec<u32> = make_random_indices(&mut rng, shape.values_len, shape.indices_len);
            bench_one(shape.label, "i64", "u32", &values, &indices);
        }

        // i32 values, u64 indices — u64 idx → 32-bit val (8 lanes on AVX-512 vs 4 on AVX2).
        {
            let values: Vec<i32> = (0..shape.values_len as i32).collect();
            let indices: Vec<u64> = make_random_indices(&mut rng, shape.values_len, shape.indices_len);
            bench_one(shape.label, "i32", "u64", &values, &indices);
        }

        // i32 values, u8 indices — narrow idx, fits 16 in AVX-512.
        if shape.values_len <= 256 {
            let values: Vec<i32> = (0..shape.values_len as i32).collect();
            let indices: Vec<u8> = make_random_indices(&mut rng, shape.values_len, shape.indices_len);
            bench_one(shape.label, "i32", "u8", &values, &indices);
        }

        println!();
    }
}
