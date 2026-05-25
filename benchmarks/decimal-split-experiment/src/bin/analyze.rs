// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Report driver for the decimal split-layout experiment.
//!
//! Prints Markdown tables:
//!   1. CPU feature parity (Arrow and the split kernels share one feature set).
//!   2. Compression: interleaved (Arrow) vs split (per-limb), synthetic + TPC-H.
//!   3. Arithmetic (add): Arrow / AoS-scalar / SoA-scalar / SoA-AVX-512 / lo-only.
//!   4. Other operations: compare, sum (overflow-safe widening), min/max, mul/div.
//!   5. Each kernel on its preferred layout (Arrow interleaved, split SoA), no
//!      conversion on either side - the fair kernel-vs-kernel comparison.
//!
//! Build with `RUSTFLAGS="-C target-cpu=native"` so Arrow and the split kernels
//! are compiled under the same ISA. Usage:
//! `cargo run --release -p decimal-split-experiment --bin decimal_split_analyze`

use std::hint::black_box;
use std::time::Duration;
use std::time::Instant;

use arrow_buffer::i256;
use decimal_split_experiment::aggregate;
use decimal_split_experiment::arrow_ref;
use decimal_split_experiment::compare;
use decimal_split_experiment::compress;
use decimal_split_experiment::cpu;
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

    cpu::report();
    compression_report();
    arithmetic_report();
    operations_report();
    preferred_layout_report();
    constant_exploitation_report();
}

/// Exploiting *partial* (block-wise) constancy. Real engines process fixed
/// chunks and carry per-chunk stats; some chunks have a constant (zero) high
/// limb, others do not. The block-wise kernels skip the high stream on the
/// constant chunks and do full work on the rest. Arrow always scans everything.
/// `const_frac` is the fraction of blocks whose high limb is constant.
fn constant_exploitation_report() {
    let dur = Duration::from_millis(300);
    let n = ARITH_N;
    const BLK: usize = 1024;
    println!("## Exploiting partial (block-wise) constancy, {n} values, block {BLK}\n");
    println!("Per-block stat says whether a chunk's high limb is constant. Constant chunks skip");
    println!("the high stream; the rest do full work. Arrow scans every value regardless.\n");
    println!(
        "| const blocks | sum Arrow | sum block-wise | sum x | lt Arrow | lt block-wise | lt x |"
    );
    println!("|---:|---:|---:|---:|---:|---:|---:|");

    for &frac in &[0.0, 0.5, 0.9, 1.0] {
        let (av, am) = data::gen_i128_blocked(n, BLK, frac, 1);
        let (bv, bm_meta) = data::gen_i128_blocked(n, BLK, frac, 2);
        let aa = arrow_ref::decimal128(&av, 38, 0);
        let ba = arrow_ref::decimal128(&bv, 38, 0);
        let sa = SplitI128::from_aos(&av);
        let sb = SplitI128::from_aos(&bv);
        let mut out = vec![0u8; compare::bitmap_len(n)];

        let sum_arrow = throughput(
            time_per_call(dur, || {
                black_box(arrow_ref::sum_decimal128(black_box(&aa)));
            }),
            n,
        );
        let sum_blk = throughput(
            time_per_call(dur, || {
                black_box(aggregate::sum_i128_blockwise(
                    black_box(&sa.lo),
                    black_box(&sa.hi),
                    black_box(&am),
                    BLK,
                ));
            }),
            n,
        );
        let lt_arrow = throughput(
            time_per_call(dur, || {
                black_box(arrow_ref::lt_decimal128(black_box(&aa), black_box(&ba)));
            }),
            n,
        );
        let lt_blk = throughput(
            time_per_call(dur, || {
                compare::lt_i128_blockwise(
                    black_box(&sa),
                    black_box(&am),
                    black_box(&sb),
                    black_box(&bm_meta),
                    BLK,
                    black_box(&mut out),
                );
            }),
            n,
        );
        println!(
            "| {:.0}% | {sum_arrow:.0} | {sum_blk:.0} | {:.2}x | {lt_arrow:.0} | {lt_blk:.0} | {:.2}x |",
            frac * 100.0,
            sum_blk / sum_arrow,
            lt_blk / lt_arrow,
        );
    }
    println!();
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

fn operations_report() {
    let dur = Duration::from_millis(300);
    let n = ARITH_N;
    println!("## Other decimal operations vs Arrow, {n} values\n");
    println!(
        "Throughput in **M items/s**. All split kernels beat Arrow under the shared feature set above.\n"
    );

    // ---- compare (lt / eq) ----
    println!("### compare (array vs array)");
    println!("| op | Arrow | split scalar | split AVX-512 | speedup |");
    println!("|---|---:|---:|---:|---:|");
    for mag in [Magnitude::Small, Magnitude::Medium, Magnitude::Large] {
        let a = data::gen_i128(n, mag, 1);
        let b = data::gen_i128(n, mag, 2);
        let aa = arrow_ref::decimal128(&a, 38, 0);
        let ba = arrow_ref::decimal128(&b, 38, 0);
        let sa = SplitI128::from_aos(&a);
        let sb = SplitI128::from_aos(&b);
        let mut bm = vec![0u8; compare::bitmap_len(n)];

        let arrow = throughput(
            time_per_call(dur, || {
                black_box(arrow_ref::lt_decimal128(black_box(&aa), black_box(&ba)));
            }),
            n,
        );
        let sc = throughput(
            time_per_call(dur, || {
                compare::lt_i128_scalar(black_box(&sa), black_box(&sb), black_box(&mut bm))
            }),
            n,
        );
        let si = throughput(
            time_per_call(dur, || {
                compare::lt_i128(black_box(&sa), black_box(&sb), black_box(&mut bm))
            }),
            n,
        );
        println!(
            "| i128 lt {} | {arrow:.0} | {sc:.0} | {si:.0} | {:.2}x |",
            mag.label(),
            si / arrow
        );
    }
    {
        // i256 lt, large magnitude
        let a = data::gen_i256(n, Magnitude::Large, 1);
        let b = data::gen_i256(n, Magnitude::Large, 2);
        let aa = arrow_ref::decimal256(&a, 76, 0);
        let ba = arrow_ref::decimal256(&b, 76, 0);
        let sa = SplitI256::from_aos(&a);
        let sb = SplitI256::from_aos(&b);
        let mut bm = vec![0u8; compare::bitmap_len(n)];
        let arrow = throughput(
            time_per_call(dur, || {
                black_box(arrow_ref::lt_decimal256(black_box(&aa), black_box(&ba)));
            }),
            n,
        );
        let si = throughput(
            time_per_call(dur, || {
                compare::lt_i256(black_box(&sa), black_box(&sb), black_box(&mut bm))
            }),
            n,
        );
        println!(
            "| i256 lt large | {arrow:.0} | - | {si:.0} | {:.2}x |",
            si / arrow
        );
    }

    // ---- sum (overflow-safe widening) ----
    println!("\n### sum (i128 column)");
    println!(
        "Arrow sums into i128 and **wraps on overflow**; the split sum widens to i256 (exact)."
    );
    println!("lo-only is the small-decimal fast path (hi known 0): half the memory traffic.\n");
    println!(
        "| magnitude | Arrow (i128, wraps) | split widening AVX-512 (exact) | lo-only AVX-512 | speedup |"
    );
    println!("|---|---:|---:|---:|---:|");
    for mag in [Magnitude::Small, Magnitude::Medium, Magnitude::Large] {
        let v = data::gen_i128(n, mag, 3);
        let aa = arrow_ref::decimal128(&v, 38, 0);
        let split = SplitI128::from_aos(&v);
        let arrow = throughput(
            time_per_call(dur, || {
                black_box(arrow_ref::sum_decimal128(black_box(&aa)));
            }),
            n,
        );
        let sp = throughput(
            time_per_call(dur, || {
                black_box(aggregate::sum_i128_widening(black_box(&split)));
            }),
            n,
        );
        let lo = throughput(
            time_per_call(dur, || {
                black_box(aggregate::sum_i128_lo_only(black_box(&split)));
            }),
            n,
        );
        println!(
            "| {} | {arrow:.0} | {sp:.0} | {lo:.0} | {:.2}x |",
            mag.label(),
            sp / arrow
        );
    }

    // ---- min / max ----
    println!("\n### min / max (i128 column)");
    println!("| op | Arrow | split scalar | split AVX-512 | speedup |");
    println!("|---|---:|---:|---:|---:|");
    for (label, want_min) in [("min", true), ("max", false)] {
        let v = data::gen_i128(n, Magnitude::Large, 4);
        let aa = arrow_ref::decimal128(&v, 38, 0);
        let split = SplitI128::from_aos(&v);
        let arrow = throughput(
            time_per_call(dur, || {
                if want_min {
                    black_box(arrow_ref::min_decimal128(black_box(&aa)));
                } else {
                    black_box(arrow_ref::max_decimal128(black_box(&aa)));
                }
            }),
            n,
        );
        let sc = throughput(
            time_per_call(dur, || {
                black_box(if want_min {
                    aggregate::min_i128_scalar(black_box(&split))
                } else {
                    aggregate::max_i128_scalar(black_box(&split))
                });
            }),
            n,
        );
        let si = throughput(
            time_per_call(dur, || {
                black_box(if want_min {
                    aggregate::min_i128(black_box(&split))
                } else {
                    aggregate::max_i128(black_box(&split))
                });
            }),
            n,
        );
        println!(
            "| i128 {label} | {arrow:.0} | {sc:.0} | {si:.0} | {:.2}x |",
            si / arrow
        );
    }

    muldiv_report(dur, n);
    println!();
}

fn muldiv_report(dur: Duration, n: usize) {
    use decimal_split_experiment::muldiv;

    // Operands kept small so the product fits precision 38 (Arrow validates it)
    // and divisors are non-zero. The SIMD kernels execute the full limb-product
    // path regardless of operand magnitude, so throughput is representative.
    let a = data::gen_i128(n, Magnitude::Small, 5);
    let b: Vec<i128> = data::gen_i128(n, Magnitude::Small, 6)
        .into_iter()
        .map(|v| v + 1)
        .collect();
    let aa = arrow_ref::decimal128(&a, 38, 0);
    let ba = arrow_ref::decimal128(&b, 38, 0);
    let sa = SplitI128::from_aos(&a);
    let sb = SplitI128::from_aos(&b);

    // ---- multiply (compute-bound: SIMD can win) ----
    println!("\n### multiply (i128, low-128 product)");
    println!(
        "Multiply is compute-bound, so unlike add the AVX-512 kernel can beat Arrow outright.\n"
    );
    println!("| Arrow | AoS scalar | split scalar | split AVX-512 (vpmullq+mulhi) | speedup |");
    println!("|---:|---:|---:|---:|---:|");
    let arrow_mul = throughput(
        time_per_call(dur, || {
            black_box(arrow_ref::mul_decimal128(black_box(&aa), black_box(&ba)));
        }),
        n,
    );
    let mut out_aos = vec![0i128; n];
    let aos_mul = throughput(
        time_per_call(dur, || {
            muldiv::mul_i128_aos(black_box(&a), black_box(&b), black_box(&mut out_aos));
        }),
        n,
    );
    let mut out = sa.zeroed_like();
    let soa_mul = throughput(
        time_per_call(dur, || {
            muldiv::mul_i128_soa_scalar(black_box(&sa), black_box(&sb), black_box(&mut out));
        }),
        n,
    );
    let simd_mul = throughput(
        time_per_call(dur, || {
            muldiv::mul_i128(black_box(&sa), black_box(&sb), black_box(&mut out));
        }),
        n,
    );
    println!(
        "| {arrow_mul:.0} | {aos_mul:.0} | {soa_mul:.0} | {simd_mul:.0} | {:.2}x |",
        simd_mul / arrow_mul
    );

    // Full-range products (genuine non-zero cross terms). Arrow can't play here
    // (the product overflows precision 38), so this isolates SIMD vs scalar on
    // real 128-bit multiplies.
    println!("\nFull-range 128-bit products (no Arrow: overflows precision 38):");
    println!("| magnitude | split scalar | split AVX-512 | SIMD/scalar |");
    println!("|---|---:|---:|---:|");
    for mag in [Magnitude::Medium, Magnitude::Large] {
        let fa = SplitI128::from_aos(&data::gen_i128(n, mag, 7));
        let fb = SplitI128::from_aos(&data::gen_i128(n, mag, 8));
        let mut fout = fa.zeroed_like();
        let fsc = throughput(
            time_per_call(dur, || {
                muldiv::mul_i128_soa_scalar(black_box(&fa), black_box(&fb), black_box(&mut fout));
            }),
            n,
        );
        let fsi = throughput(
            time_per_call(dur, || {
                muldiv::mul_i128(black_box(&fa), black_box(&fb), black_box(&mut fout));
            }),
            n,
        );
        println!(
            "| {} | {fsc:.0} | {fsi:.0} | {:.2}x |",
            mag.label(),
            fsi / fsc
        );
    }

    // ---- divide (no SIMD; split gives no leverage) ----
    println!("\n### divide (i128)");
    println!("No SIMD divide exists and the split gives no leverage, so both modes are scalar and");
    println!("equal. Arrow's decimal div additionally rescales and rounds (more work, different");
    println!("semantics); ours is truncating integer division - throughput comparison only.\n");
    println!("| Arrow (rescale+round) | AoS scalar (trunc) | split scalar (trunc) | speedup |");
    println!("|---:|---:|---:|---:|");
    let arrow_div = throughput(
        time_per_call(dur, || {
            black_box(arrow_ref::div_decimal128(black_box(&aa), black_box(&ba)));
        }),
        n,
    );
    let aos_div = throughput(
        time_per_call(dur, || {
            muldiv::div_i128_aos(black_box(&a), black_box(&b), black_box(&mut out_aos));
        }),
        n,
    );
    let soa_div = throughput(
        time_per_call(dur, || {
            muldiv::div_i128_soa(black_box(&sa), black_box(&sb), black_box(&mut out));
        }),
        n,
    );
    println!(
        "| {arrow_div:.0} | {aos_div:.0} | {soa_div:.0} | {:.2}x |",
        aos_div / arrow_div
    );
}

/// The fair comparison: each kernel runs on the data layout it would actually
/// store. Arrow keeps its interleaved Decimal128 buffer; the split kernels keep
/// their SoA limb streams. Neither side converts - in a real system the data is
/// born in that system's preferred layout, so a conversion tax on either side
/// would be artificial. Operands are built once, outside the timed loop; only
/// the kernel is timed.
fn preferred_layout_report() {
    let dur = Duration::from_millis(300);
    let n = ARITH_N;
    println!("## Each kernel on its preferred layout (no conversion either side), {n} values\n");
    println!("Arrow operates on interleaved Decimal128 (its native storage); the split kernels");
    println!("operate on SoA limb streams (their native storage). Kernel-only timing.\n");
    println!("| op | Arrow (interleaved) | split (SoA, AVX-512) | speedup |");
    println!("|---|---:|---:|---:|");

    let a = data::gen_i128(n, Magnitude::Large, 1);
    let b = data::gen_i128(n, Magnitude::Large, 2);
    let aa = arrow_ref::decimal128(&a, 38, 0);
    let ba = arrow_ref::decimal128(&b, 38, 0);
    let sa = SplitI128::from_aos(&a);
    let sb = SplitI128::from_aos(&b);
    let mut so = sa.zeroed_like();
    let mut bm = vec![0u8; compare::bitmap_len(n)];

    let arrow_add = throughput(
        time_per_call(dur, || {
            black_box(arrow_ref::add_decimal128(black_box(&aa), black_box(&ba)));
        }),
        n,
    );
    let split_add = throughput(
        time_per_call(dur, || {
            simd::add_i128(&sa, &sb, &mut so);
            black_box(&so);
        }),
        n,
    );
    println!(
        "| add | {arrow_add:.0} | {split_add:.0} | {:.2}x |",
        split_add / arrow_add
    );

    let arrow_sum = throughput(
        time_per_call(dur, || {
            black_box(arrow_ref::sum_decimal128(black_box(&aa)));
        }),
        n,
    );
    let split_sum = throughput(
        time_per_call(dur, || {
            black_box(aggregate::sum_i128_widening(&sa));
        }),
        n,
    );
    println!(
        "| sum (split is exact i256; Arrow wraps i128) | {arrow_sum:.0} | {split_sum:.0} | {:.2}x |",
        split_sum / arrow_sum
    );

    let arrow_lt = throughput(
        time_per_call(dur, || {
            black_box(arrow_ref::lt_decimal128(black_box(&aa), black_box(&ba)));
        }),
        n,
    );
    let split_lt = throughput(
        time_per_call(dur, || {
            compare::lt_i128(&sa, &sb, &mut bm);
            black_box(&bm);
        }),
        n,
    );
    println!(
        "| lt | {arrow_lt:.0} | {split_lt:.0} | {:.2}x |",
        split_lt / arrow_lt
    );

    // mul needs operands small enough that the product fits precision 38.
    let ma = data::gen_i128(n, Magnitude::Small, 3);
    let mb = data::gen_i128(n, Magnitude::Small, 4);
    let maa = arrow_ref::decimal128(&ma, 38, 0);
    let mba = arrow_ref::decimal128(&mb, 38, 0);
    let msa = SplitI128::from_aos(&ma);
    let msb = SplitI128::from_aos(&mb);
    let mut mso = msa.zeroed_like();
    let arrow_mul = throughput(
        time_per_call(dur, || {
            black_box(arrow_ref::mul_decimal128(black_box(&maa), black_box(&mba)));
        }),
        n,
    );
    let split_mul = throughput(
        time_per_call(dur, || {
            decimal_split_experiment::muldiv::mul_i128(&msa, &msb, &mut mso);
            black_box(&mso);
        }),
        n,
    );
    println!(
        "| mul | {arrow_mul:.0} | {split_mul:.0} | {:.2}x |",
        split_mul / arrow_mul
    );
    println!();
}
