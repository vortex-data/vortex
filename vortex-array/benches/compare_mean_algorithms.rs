// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark and numerical error comparison of two mean computation algorithms:
//!
//! 1. **Sum/Count**: `mean = sum(x_i) / n`
//! 2. **Sub-mean merge (Welford-style)**: `mean_{k+1} = mean_k + (x_{k+1} - mean_k) / (k+1)`
//!    and chunk-level: `mean = mean_A + (n_B / (n_A + n_B)) * (mean_B - mean_A)`
//!
//! Tests both f64 and i128 decimal (fixed-point) representations.

#![allow(clippy::unwrap_used, clippy::cast_precision_loss)]

use std::hint::black_box;

use divan::Bencher;

fn main() {
    // Run the numerical error analysis first, then divan benchmarks.
    println!("=== Numerical Error Analysis ===\n");
    error_analysis_f64();
    error_analysis_decimal();
    println!("\n=== Throughput Benchmarks ===\n");
    divan::main();
}

// ---------------------------------------------------------------------------
// Algorithm implementations
// ---------------------------------------------------------------------------

/// Naive sum/count mean for f64.
fn mean_sum_count_f64(values: &[f64]) -> f64 {
    let mut sum: f64 = 0.0;
    let mut count: u64 = 0;
    for &v in values {
        sum += v;
        count += 1;
    }
    sum / count as f64
}

/// Welford online mean for f64 (element-wise).
fn mean_welford_f64(values: &[f64]) -> f64 {
    let mut mean: f64 = 0.0;
    let mut count: u64 = 0;
    for &v in values {
        count += 1;
        mean += (v - mean) / count as f64;
    }
    mean
}

/// Kahan-compensated sum then divide for f64.
fn mean_kahan_f64(values: &[f64]) -> f64 {
    let mut sum: f64 = 0.0;
    let mut comp: f64 = 0.0; // compensation
    let mut count: u64 = 0;
    for &v in values {
        let y = v - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        count += 1;
    }
    sum / count as f64
}

/// Chunk-level sub-mean merge for f64. Simulates chunked array processing.
fn mean_chunk_merge_f64(values: &[f64], chunk_size: usize) -> f64 {
    let mut global_mean: f64 = 0.0;
    let mut global_count: u64 = 0;

    for chunk in values.chunks(chunk_size) {
        // Compute chunk mean using Welford (stable within chunk).
        let chunk_mean = mean_welford_f64(chunk);
        let chunk_count = chunk.len() as u64;

        // Merge: mean = mean_A + (n_B / (n_A + n_B)) * (mean_B - mean_A)
        let new_count = global_count + chunk_count;
        if new_count == chunk_count {
            global_mean = chunk_mean;
        } else {
            global_mean +=
                (chunk_count as f64 / new_count as f64) * (chunk_mean - global_mean);
        }
        global_count = new_count;
    }
    global_mean
}

/// Naive sum/count mean for decimal (i128 fixed-point with given scale).
/// Returns the mean as f64 for error comparison.
fn mean_sum_count_decimal(values: &[i128], scale: u32) -> Option<f64> {
    let mut sum: i128 = 0;
    let mut count: u64 = 0;
    for &v in values {
        sum = sum.checked_add(v)?;
        count += 1;
    }
    let scale_factor = 10f64.powi(scale as i32);
    Some(sum as f64 / count as f64 / scale_factor)
}

/// Sub-mean merge for decimal (i128 fixed-point).
/// Each chunk: sum the chunk (checked), compute chunk_sum and chunk_count.
/// Merge: combined_sum = sum_A + sum_B (can overflow!), so instead we use
/// the weighted mean formula on f64-converted means.
/// Returns the mean as f64 for error comparison.
fn mean_chunk_merge_decimal(values: &[i128], scale: u32, chunk_size: usize) -> f64 {
    let scale_factor = 10f64.powi(scale as i32);
    let mut global_mean: f64 = 0.0;
    let mut global_count: u64 = 0;

    for chunk in values.chunks(chunk_size) {
        // Chunk sum in integer domain (exact within chunk if no overflow).
        let mut chunk_sum: i128 = 0;
        let mut chunk_count: u64 = 0;
        for &v in chunk {
            // Use wrapping here to show the algorithm; in production we'd checked_add.
            chunk_sum = chunk_sum.wrapping_add(v);
            chunk_count += 1;
        }
        let chunk_mean = (chunk_sum as f64) / (chunk_count as f64) / scale_factor;

        let new_count = global_count + chunk_count;
        if new_count == chunk_count {
            global_mean = chunk_mean;
        } else {
            global_mean +=
                (chunk_count as f64 / new_count as f64) * (chunk_mean - global_mean);
        }
        global_count = new_count;
    }
    global_mean
}

// ---------------------------------------------------------------------------
// Numerical error analysis
// ---------------------------------------------------------------------------

fn error_analysis_f64() {
    println!("--- f64 ---");
    println!(
        "{:<40} {:>15} {:>15} {:>15}",
        "Scenario", "sum/count err", "welford err", "chunk-merge err"
    );
    println!("{}", "-".repeat(88));

    // Scenario 1: Large offset, small variance (catastrophic cancellation)
    {
        let n = 1_000_000usize;
        let offset = 1e15_f64;
        let values: Vec<f64> = (0..n).map(|i| offset + (i as f64) * 1e-3).collect();
        let truth = offset + ((n - 1) as f64) * 1e-3 / 2.0;

        let sc = mean_sum_count_f64(&values);
        let wf = mean_welford_f64(&values);
        let cm = mean_chunk_merge_f64(&values, 1024);

        println!(
            "{:<40} {:>15.6e} {:>15.6e} {:>15.6e}",
            "large offset + small var (1M)",
            (sc - truth).abs() / truth.abs(),
            (wf - truth).abs() / truth.abs(),
            (cm - truth).abs() / truth.abs(),
        );
    }

    // Scenario 2: Alternating large positive/negative (cancellation)
    {
        let n = 1_000_000usize;
        let big = 1e15_f64;
        let values: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { big } else { -big + 1.0 })
            .collect();
        // True mean = (n/2 * big + n/2 * (-big + 1)) / n = 0.5
        let truth = 0.5;

        let sc = mean_sum_count_f64(&values);
        let wf = mean_welford_f64(&values);
        let cm = mean_chunk_merge_f64(&values, 1024);

        println!(
            "{:<40} {:>15.6e} {:>15.6e} {:>15.6e}",
            "alternating ±1e15 (1M)",
            (sc - truth).abs() / truth.abs(),
            (wf - truth).abs() / truth.abs(),
            (cm - truth).abs() / truth.abs(),
        );
    }

    // Scenario 3: Near overflow
    {
        let n = 10_000usize;
        let big = f64::MAX / 100.0; // close to overflow for sum
        let values: Vec<f64> = vec![big; n];
        let truth = big;

        let sc = mean_sum_count_f64(&values);
        let wf = mean_welford_f64(&values);
        let cm = mean_chunk_merge_f64(&values, 1024);

        let sc_err = if sc.is_infinite() {
            f64::INFINITY
        } else {
            (sc - truth).abs() / truth.abs()
        };

        println!(
            "{:<40} {:>15.6e} {:>15.6e} {:>15.6e}",
            "near overflow (10K × MAX/100)",
            sc_err,
            (wf - truth).abs() / truth.abs(),
            (cm - truth).abs() / truth.abs(),
        );
    }

    // Scenario 4: Uniform small values (baseline, both should be fine)
    {
        let n = 1_000_000usize;
        let values: Vec<f64> = (0..n).map(|i| (i as f64) * 0.001).collect();
        let truth = ((n - 1) as f64) * 0.001 / 2.0;

        let sc = mean_sum_count_f64(&values);
        let wf = mean_welford_f64(&values);
        let cm = mean_chunk_merge_f64(&values, 1024);

        println!(
            "{:<40} {:>15.6e} {:>15.6e} {:>15.6e}",
            "uniform small (1M, baseline)",
            (sc - truth).abs() / truth.abs(),
            (wf - truth).abs() / truth.abs(),
            (cm - truth).abs() / truth.abs(),
        );
    }

    // Scenario 5: Extreme scale difference
    {
        let n = 100_000usize;
        let values: Vec<f64> = (0..n)
            .map(|i| if i == 0 { 1e18 } else { 1.0 })
            .collect();
        let truth = (1e18 + (n as f64 - 1.0)) / n as f64;

        let sc = mean_sum_count_f64(&values);
        let wf = mean_welford_f64(&values);
        let cm = mean_chunk_merge_f64(&values, 1024);

        println!(
            "{:<40} {:>15.6e} {:>15.6e} {:>15.6e}",
            "one 1e18 + rest 1.0 (100K)",
            (sc - truth).abs() / truth.abs(),
            (wf - truth).abs() / truth.abs(),
            (cm - truth).abs() / truth.abs(),
        );
    }
}

fn error_analysis_decimal() {
    println!("\n--- Decimal (i128, scale=2) ---");
    println!(
        "{:<40} {:>18} {:>18}",
        "Scenario", "sum/count", "chunk-merge"
    );
    println!("{}", "-".repeat(78));

    let scale = 2u32;
    let scale_factor = 10f64.powi(scale as i32);

    // Scenario 1: Normal values (both exact)
    {
        let n = 100_000usize;
        let values: Vec<i128> = (0..n).map(|i| (i as i128) * 100 + 150).collect();
        let truth = values.iter().sum::<i128>() as f64 / n as f64 / scale_factor;

        let sc = mean_sum_count_decimal(&values, scale);
        let cm = mean_chunk_merge_decimal(&values, scale, 1024);

        println!(
            "{:<40} {:>18.6e} {:>18.6e}",
            "normal range (100K)",
            sc.map(|v| (v - truth).abs() / truth.abs())
                .unwrap_or(f64::INFINITY),
            (cm - truth).abs() / truth.abs(),
        );
    }

    // Scenario 2: Near i128 overflow — sum/count fails, chunk-merge survives
    {
        let n = 1000usize;
        let big = i128::MAX / 500; // sum of 1000 of these overflows i128
        let values: Vec<i128> = vec![big; n];
        let truth = big as f64 / scale_factor;

        let sc = mean_sum_count_decimal(&values, scale);
        let cm = mean_chunk_merge_decimal(&values, scale, 100);

        println!(
            "{:<40} {:>18} {:>18.6e}",
            "near i128 overflow (1K × MAX/500)",
            sc.map(|v| format!("{:.6e}", (v - truth).abs() / truth.abs()))
                .unwrap_or_else(|| "OVERFLOW (null)".to_string()),
            (cm - truth).abs() / truth.abs(),
        );
    }

    // Scenario 3: Large count, moderate values
    {
        let n = 10_000_000usize;
        let values: Vec<i128> = (0..n).map(|i| (i as i128 % 100_000) * 100).collect();
        let truth = values.iter().sum::<i128>() as f64 / n as f64 / scale_factor;

        let sc = mean_sum_count_decimal(&values, scale);
        let cm = mean_chunk_merge_decimal(&values, scale, 4096);

        println!(
            "{:<40} {:>18.6e} {:>18.6e}",
            "moderate values (10M)",
            sc.map(|v| (v - truth).abs() / truth.abs())
                .unwrap_or(f64::INFINITY),
            (cm - truth).abs() / truth.abs(),
        );
    }

    // Scenario 4: Mixed positive/negative near limits
    {
        let n = 2000usize;
        let big = i128::MAX / 1000;
        let values: Vec<i128> = (0..n)
            .map(|i| if i % 2 == 0 { big } else { -big + 1 })
            .collect();
        // True mean ≈ 0.5 / scale_factor in decimal terms
        let truth_unscaled = 0.5_f64;
        let truth = truth_unscaled / scale_factor;

        let sc = mean_sum_count_decimal(&values, scale);
        let cm = mean_chunk_merge_decimal(&values, scale, 100);

        println!(
            "{:<40} {:>18} {:>18.6e}",
            "alternating ±big (2K)",
            sc.map(|v| format!("{:.6e}", (v - truth).abs() / truth.abs()))
                .unwrap_or_else(|| "OVERFLOW (null)".to_string()),
            (cm - truth).abs() / truth.abs(),
        );
    }
}

// ---------------------------------------------------------------------------
// Throughput benchmarks (divan)
// ---------------------------------------------------------------------------

const BENCH_SIZE: usize = 1_000_000;
const CHUNK_SIZE: usize = 1024;

fn make_f64_data() -> Vec<f64> {
    // Deterministic pseudo-random-ish data with large offset to stress precision.
    (0..BENCH_SIZE)
        .map(|i| 1e12 + (i as f64) * 0.1 + ((i * 7 + 13) % 1000) as f64 * 0.01)
        .collect()
}

fn make_decimal_data() -> Vec<i128> {
    (0..BENCH_SIZE)
        .map(|i| (i as i128) * 100 + ((i * 7 + 13) % 1000) as i128)
        .collect()
}

#[divan::bench]
fn f64_sum_count(bencher: Bencher) {
    let data = make_f64_data();
    bencher.bench_local(|| black_box(mean_sum_count_f64(black_box(&data))));
}

#[divan::bench]
fn f64_welford(bencher: Bencher) {
    let data = make_f64_data();
    bencher.bench_local(|| black_box(mean_welford_f64(black_box(&data))));
}

#[divan::bench]
fn f64_kahan_sum(bencher: Bencher) {
    let data = make_f64_data();
    bencher.bench_local(|| black_box(mean_kahan_f64(black_box(&data))));
}

#[divan::bench]
fn f64_chunk_merge(bencher: Bencher) {
    let data = make_f64_data();
    bencher.bench_local(|| black_box(mean_chunk_merge_f64(black_box(&data), CHUNK_SIZE)));
}

#[divan::bench]
fn decimal_sum_count(bencher: Bencher) {
    let data = make_decimal_data();
    bencher.bench_local(|| black_box(mean_sum_count_decimal(black_box(&data), 2)));
}

#[divan::bench]
fn decimal_chunk_merge(bencher: Bencher) {
    let data = make_decimal_data();
    bencher.bench_local(|| black_box(mean_chunk_merge_decimal(black_box(&data), 2, CHUNK_SIZE)));
}
