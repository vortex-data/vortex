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

/// Sub-mean merge for decimal — PURE INTEGER arithmetic, no f64 intermediate.
///
/// Each chunk computes `chunk_mean = chunk_sum / chunk_count` (integer truncation).
/// Chunks are merged via:
///   `mean = mean_A + (n_B * (mean_B - mean_A)) / (n_A + n_B)`
///
/// All operations are i128. Two truncation points per merge:
///   1. `chunk_sum / chunk_count` (intra-chunk mean)
///   2. `(n_B * delta) / total_count` (merge step)
///
/// Returns the final integer mean (unscaled). Caller divides by scale_factor.
fn mean_chunk_merge_decimal_integer(values: &[i128], chunk_size: usize) -> i128 {
    let mut global_mean: i128 = 0;
    let mut global_count: i128 = 0;

    for chunk in values.chunks(chunk_size) {
        // Chunk sum — small enough chunk that this won't overflow.
        let mut chunk_sum: i128 = 0;
        let chunk_count = chunk.len() as i128;
        for &v in chunk {
            chunk_sum = chunk_sum.wrapping_add(v);
        }
        // TRUNCATION POINT 1: integer division for chunk mean.
        let chunk_mean = chunk_sum / chunk_count;

        let new_count = global_count + chunk_count;
        if global_count == 0 {
            global_mean = chunk_mean;
        } else {
            // TRUNCATION POINT 2: integer division for merge.
            // mean = mean_A + (n_B * (mean_B - mean_A)) / (n_A + n_B)
            let delta = chunk_mean - global_mean;
            global_mean += (chunk_count * delta) / new_count;
        }
        global_count = new_count;
    }
    global_mean
}

/// Wrapper that also does the f64-via-intermediate approach for comparison.
/// Each chunk sums in integer, but merges chunk means via f64 weighted average.
fn mean_chunk_merge_decimal_via_f64(values: &[i128], scale: u32, chunk_size: usize) -> f64 {
    let scale_factor = 10f64.powi(scale as i32);
    let mut global_mean: f64 = 0.0;
    let mut global_count: u64 = 0;

    for chunk in values.chunks(chunk_size) {
        let mut chunk_sum: i128 = 0;
        let chunk_count = chunk.len() as u64;
        for &v in chunk {
            chunk_sum = chunk_sum.wrapping_add(v);
        }
        let chunk_mean = (chunk_sum as f64) / (chunk_count as f64) / scale_factor;

        let new_count = global_count + chunk_count;
        if global_count == 0 {
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
        "{:<40} {:>18} {:>18} {:>18}",
        "Scenario", "sum/count", "int-merge", "f64-merge"
    );
    println!("{}", "-".repeat(96));

    let scale = 2u32;
    let scale_factor = 10f64.powi(scale as i32);

    // Helper: compute relative error, or "OVERFLOW" if None.
    let fmt_err = |result: Option<f64>, truth: f64| -> String {
        match result {
            Some(v) if truth.abs() > 0.0 => format!("{:.6e}", (v - truth).abs() / truth.abs()),
            Some(_) => format!("{:.6e}", 0.0),
            None => "OVERFLOW (null)".to_string(),
        }
    };

    // Scenario 1: Normal values
    {
        let n = 100_000usize;
        let values: Vec<i128> = (0..n).map(|i| (i as i128) * 100 + 150).collect();
        // Ground truth: exact sum (fits i128) / n / scale.
        let exact_sum: i128 = values.iter().sum();
        let truth = exact_sum as f64 / n as f64 / scale_factor;

        let sc = mean_sum_count_decimal(&values, scale);
        let int_merge = mean_chunk_merge_decimal_integer(&values, 1024);
        let int_merge_f64 = int_merge as f64 / scale_factor;
        let f64_merge = mean_chunk_merge_decimal_via_f64(&values, scale, 1024);

        // Also show the raw integer truncation: exact_sum/n vs int_merge.
        let exact_int_mean = exact_sum / n as i128;
        println!(
            "{:<40} {:>18} {:>18} {:>18}",
            "normal range (100K)",
            fmt_err(sc, truth),
            fmt_err(Some(int_merge_f64), truth),
            fmt_err(Some(f64_merge), truth),
        );
        println!(
            "  exact_sum/n={}, int_merge={}, diff={}",
            exact_int_mean, int_merge, int_merge - exact_int_mean,
        );
    }

    // Scenario 2: Near i128 overflow — sum/count fails
    {
        let n = 1000usize;
        let big = i128::MAX / 500;
        let values: Vec<i128> = vec![big; n];
        let truth = big as f64 / scale_factor;

        let sc = mean_sum_count_decimal(&values, scale);
        let int_merge = mean_chunk_merge_decimal_integer(&values, 100);
        let int_merge_f64 = int_merge as f64 / scale_factor;
        let f64_merge = mean_chunk_merge_decimal_via_f64(&values, scale, 100);

        println!(
            "{:<40} {:>18} {:>18} {:>18}",
            "near i128 overflow (1K × MAX/500)",
            fmt_err(sc, truth),
            fmt_err(Some(int_merge_f64), truth),
            fmt_err(Some(f64_merge), truth),
        );
        println!("  expected={}, int_merge={}, diff={}", big, int_merge, int_merge - big);
    }

    // Scenario 3: Large count, moderate values — shows truncation accumulation.
    {
        let n = 10_000_000usize;
        let values: Vec<i128> = (0..n).map(|i| (i as i128 % 100_000) * 100).collect();
        let exact_sum: i128 = values.iter().sum();
        let truth = exact_sum as f64 / n as f64 / scale_factor;

        let sc = mean_sum_count_decimal(&values, scale);
        let int_merge = mean_chunk_merge_decimal_integer(&values, 4096);
        let int_merge_f64 = int_merge as f64 / scale_factor;
        let f64_merge = mean_chunk_merge_decimal_via_f64(&values, scale, 4096);

        let exact_int_mean = exact_sum / n as i128;
        println!(
            "{:<40} {:>18} {:>18} {:>18}",
            "moderate values (10M)",
            fmt_err(sc, truth),
            fmt_err(Some(int_merge_f64), truth),
            fmt_err(Some(f64_merge), truth),
        );
        println!(
            "  exact_sum/n={}, int_merge={}, diff={}",
            exact_int_mean, int_merge, int_merge - exact_int_mean,
        );
    }

    // Scenario 4: Mixed positive/negative near limits
    {
        let n = 2000usize;
        let big = i128::MAX / 1000;
        let values: Vec<i128> = (0..n)
            .map(|i| if i % 2 == 0 { big } else { -big + 1 })
            .collect();
        // Exact: n/2 * big + n/2 * (-big+1) = n/2 * 1 = 1000
        // Mean = 1000 / 2000 = 0 (integer truncation!) but true mean = 0.5
        let truth_unscaled = 0.5_f64;
        let truth = truth_unscaled / scale_factor;

        let sc = mean_sum_count_decimal(&values, scale);
        let int_merge = mean_chunk_merge_decimal_integer(&values, 100);
        let int_merge_f64 = int_merge as f64 / scale_factor;
        let f64_merge = mean_chunk_merge_decimal_via_f64(&values, scale, 100);

        println!(
            "{:<40} {:>18} {:>18} {:>18}",
            "alternating ±big (2K)",
            fmt_err(sc, truth),
            fmt_err(Some(int_merge_f64), truth),
            fmt_err(Some(f64_merge), truth),
        );
        println!("  int_merge={} (should be 0 due to truncation, true=0.5)", int_merge);
    }

    // Scenario 5: Chunk size sensitivity — same data, varying chunk sizes.
    {
        let n = 1_000_000usize;
        let values: Vec<i128> = (0..n).map(|i| (i as i128) * 7 + 3).collect();
        let exact_sum: i128 = values.iter().sum();
        let exact_int_mean = exact_sum / n as i128;
        println!("\n  Chunk size sensitivity (1M values, exact_int_mean={}):", exact_int_mean);
        for &cs in &[64, 256, 1024, 4096, 65536] {
            let int_merge = mean_chunk_merge_decimal_integer(&values, cs);
            println!(
                "    chunk_size={:>6}: int_merge={}, diff={}",
                cs, int_merge, int_merge - exact_int_mean,
            );
        }
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
fn decimal_int_merge(bencher: Bencher) {
    let data = make_decimal_data();
    bencher.bench_local(|| black_box(mean_chunk_merge_decimal_integer(black_box(&data), CHUNK_SIZE)));
}

#[divan::bench]
fn decimal_f64_merge(bencher: Bencher) {
    let data = make_decimal_data();
    bencher.bench_local(|| {
        black_box(mean_chunk_merge_decimal_via_f64(
            black_box(&data),
            2,
            CHUNK_SIZE,
        ))
    });
}
