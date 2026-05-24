// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Report driver for the decimal split-layout experiment.
//!
//! Prints two Markdown tables:
//!   1. Compression: interleaved (Arrow) vs split (per-limb) for synthetic
//!      magnitude classes, i128 + i256, and real TPC-H `lineitem` columns.
//!   2. Arithmetic throughput: Arrow / AoS-scalar / SoA-scalar / SoA-AVX512.
//!
//! Usage: `cargo run --release -p decimal-split-experiment --bin decimal_split_analyze`

use std::hint::black_box;
use std::time::Duration;
use std::time::Instant;

use arrow_buffer::i256;
use decimal_split_experiment::arrow_ref;
use decimal_split_experiment::compress;
use decimal_split_experiment::data;
use decimal_split_experiment::data::Magnitude;
use decimal_split_experiment::layout::SplitI128;
use decimal_split_experiment::layout::SplitI256;
use decimal_split_experiment::scalar;
use decimal_split_experiment::simd;

const ZSTD_LEVEL: i32 = 3;
const COMPRESS_N: usize = 1 << 20;
const ARITH_N: usize = 1 << 20;

fn main() {
    println!("# Decimal split-layout experiment\n");
    println!(
        "AVX-512 path active: **{}**  (cores reported by std: {})\n",
        simd::avx512_active(),
        std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
    );

    compression_report();
    arithmetic_report();
}

fn compression_report() {
    println!("## Compression: interleaved (Arrow) vs split limbs\n");
    println!("zstd level {ZSTD_LEVEL}, {COMPRESS_N} values per column.\n");

    println!("ffor bits = bits/value after frame-of-reference, per limb. A 0 means");
    println!("the limb is constant and effectively free under bit-packing.\n");
    println!("bitpack ratio = raw interleaved bytes / FFoR-bit-packed split bytes.");
    println!("Bit-packing the split limbs is something Arrow / a 128-bit value cannot");
    println!("do directly (FastLanes has no 128/256-bit lane), so it is unique to the split.\n");

    println!("### i128");
    println!(
        "| column | ffor bits (lo,hi) | bitpack ratio | zstd interleaved (B) | zstd split (B) | zstd ratio |"
    );
    println!("|---|---|---:|---:|---:|---:|");
    let emit_i128 = |label: String, r: &compress::I128Report| {
        println!(
            "| {} | ({}, {}) | {:.1}x | {} | {} | {:.2}x |",
            label,
            r.lo_ffor_bits,
            r.hi_ffor_bits,
            r.bitpack_ratio(),
            r.aos_zstd,
            r.split_zstd(),
            r.zstd_ratio(),
        );
    };
    for mag in [Magnitude::Small, Magnitude::Medium, Magnitude::Large] {
        let values = data::gen_i128(COMPRESS_N, mag, 7);
        let r = compress::analyze_i128(&values, ZSTD_LEVEL);
        emit_i128(format!("synthetic {}", mag.label()), &r);
    }
    // Real data (best effort).
    match std::panic::catch_unwind(|| data::tpch_lineitem_decimals(0.05)) {
        Ok(columns) => {
            for col in columns {
                let r = compress::analyze_i128(&col.values, ZSTD_LEVEL);
                emit_i128(
                    format!("tpch {} (p{} s{})", col.name, col.precision, col.scale),
                    &r,
                );
            }
        }
        Err(_) => println!("| tpch lineitem | _generation failed (skipped)_ | | | | |"),
    }

    println!("\n### i256");
    println!(
        "| column | ffor bits (l0..l3) | bitpack ratio | zstd interleaved (B) | zstd split (B) | zstd ratio |"
    );
    println!("|---|---|---:|---:|---:|---:|");
    for mag in [Magnitude::Small, Magnitude::Medium, Magnitude::Large] {
        let values = data::gen_i256(COMPRESS_N, mag, 9);
        let r = compress::analyze_i256(&values, ZSTD_LEVEL);
        println!(
            "| synthetic {} | {:?} | {:.1}x | {} | {} | {:.2}x |",
            mag.label(),
            r.limb_ffor_bits,
            r.bitpack_ratio(),
            r.aos_zstd,
            r.split_zstd(),
            r.zstd_ratio(),
        );
    }
    println!();
}

/// Time `f` for at least `min_dur`, return nanoseconds per call.
fn time_per_call(min_dur: Duration, mut f: impl FnMut()) -> f64 {
    // Warmup.
    f();
    let mut iters: u64 = 0;
    let start = Instant::now();
    loop {
        f();
        iters += 1;
        if start.elapsed() >= min_dur {
            break;
        }
    }
    start.elapsed().as_nanos() as f64 / iters as f64
}

fn throughput(ns_per_call: f64, n: usize) -> f64 {
    // Million items per second.
    (n as f64) / (ns_per_call / 1e9) / 1e6
}

fn arithmetic_report() {
    let dur = Duration::from_millis(300);
    println!("## Arithmetic throughput (add), {ARITH_N} values\n");
    println!("Throughput in **M items/s** (higher is better). Speedup = AVX-512 split vs Arrow.\n");

    println!("### i128 add");
    println!(
        "| magnitude | Arrow | AoS scalar | SoA scalar | SoA AVX-512 | lo-only AVX-512* | speedup |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|");
    for mag in [Magnitude::Small, Magnitude::Medium, Magnitude::Large] {
        let a_vec = data::gen_i128(ARITH_N, mag, 1);
        let b_vec = data::gen_i128(ARITH_N, mag, 2);

        let a_arrow = arrow_ref::decimal128(&a_vec, 38, 0);
        let b_arrow = arrow_ref::decimal128(&b_vec, 38, 0);
        let arrow = throughput(
            time_per_call(dur, || {
                black_box(arrow_ref::add_decimal128(
                    black_box(&a_arrow),
                    black_box(&b_arrow),
                ));
            }),
            ARITH_N,
        );

        let mut out_aos = vec![0i128; ARITH_N];
        let aos = throughput(
            time_per_call(dur, || {
                scalar::add_i128_aos(
                    black_box(&a_vec),
                    black_box(&b_vec),
                    black_box(&mut out_aos),
                );
            }),
            ARITH_N,
        );

        let a_soa = SplitI128::from_aos(&a_vec);
        let b_soa = SplitI128::from_aos(&b_vec);
        let mut out_soa = a_soa.zeroed_like();
        let soa_scalar = throughput(
            time_per_call(dur, || {
                scalar::add_i128_soa(
                    black_box(&a_soa),
                    black_box(&b_soa),
                    black_box(&mut out_soa),
                );
            }),
            ARITH_N,
        );
        let soa_simd = throughput(
            time_per_call(dur, || {
                simd::add_i128(
                    black_box(&a_soa),
                    black_box(&b_soa),
                    black_box(&mut out_soa),
                );
            }),
            ARITH_N,
        );
        let lo_only = throughput(
            time_per_call(dur, || {
                scalar::add_i128_lo_only(
                    black_box(&a_soa),
                    black_box(&b_soa),
                    black_box(&mut out_soa),
                );
            }),
            ARITH_N,
        );

        println!(
            "| {} | {:.0} | {:.0} | {:.0} | {:.0} | {:.0} | {:.2}x |",
            mag.label(),
            arrow,
            aos,
            soa_scalar,
            soa_simd,
            lo_only,
            soa_simd / arrow,
        );
    }
    println!(
        "\n*lo-only assumes the high limb is a known constant (the small-decimal case); shown for all magnitudes for reference only.\n"
    );

    println!("### i256 add");
    println!("| magnitude | Arrow | SoA scalar | SoA AVX-512 | speedup |");
    println!("|---|---:|---:|---:|---:|");
    for mag in [Magnitude::Small, Magnitude::Medium, Magnitude::Large] {
        let a_vec: Vec<i256> = data::gen_i256(ARITH_N, mag, 1);
        let b_vec: Vec<i256> = data::gen_i256(ARITH_N, mag, 2);

        let a_arrow = arrow_ref::decimal256(&a_vec, 76, 0);
        let b_arrow = arrow_ref::decimal256(&b_vec, 76, 0);
        let arrow = throughput(
            time_per_call(dur, || {
                black_box(arrow_ref::add_decimal256(
                    black_box(&a_arrow),
                    black_box(&b_arrow),
                ));
            }),
            ARITH_N,
        );

        let a_soa = SplitI256::from_aos(&a_vec);
        let b_soa = SplitI256::from_aos(&b_vec);
        let mut out_soa = a_soa.zeroed_like();
        let soa_scalar = throughput(
            time_per_call(dur, || {
                scalar::add_i256_soa(
                    black_box(&a_soa),
                    black_box(&b_soa),
                    black_box(&mut out_soa),
                );
            }),
            ARITH_N,
        );
        let soa_simd = throughput(
            time_per_call(dur, || {
                simd::add_i256(
                    black_box(&a_soa),
                    black_box(&b_soa),
                    black_box(&mut out_soa),
                );
            }),
            ARITH_N,
        );

        println!(
            "| {} | {:.0} | {:.0} | {:.0} | {:.2}x |",
            mag.label(),
            arrow,
            soa_scalar,
            soa_simd,
            soa_simd / arrow,
        );
    }
    println!();
}
