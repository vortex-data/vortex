// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Report driver for the decimal split-layout experiment.
//!
//! Prints Markdown tables:
//!   1. CPU feature parity (Arrow and the split kernels share one feature set).
//!   2. Compression: interleaved (Arrow) vs split (per-limb), synthetic + TPC-H.
//!   3. Arithmetic (add): Arrow / AoS-scalar / SoA-scalar / SoA-AVX-512 / lo-only.
//!   4. Other operations: compare, sum (overflow-safe widening), min/max, mul/div.
//!   5. Apples-to-apples from a common storage layout: both regimes measured -
//!      stored-interleaved (split pays the transpose) and stored-split (Arrow
//!      pays the gather).
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
    endtoend_report();
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

/// Apples-to-apples from a *common* starting layout.
///
/// Every table above pre-splits operands outside the timed loop, which is the
/// honest comparison only when decimals are *stored* split (the compression
/// encoding). This section instead assumes data is stored interleaved (Arrow's
/// native layout) and charges the split kernels for the AoS->SoA transpose on
/// input, plus the SoA->AoS merge on output for elementwise ops. It exposes when
/// the split's algorithmic win survives the transpose tax and when it does not.
fn endtoend_report() {
    let dur = Duration::from_millis(300);
    let n = ARITH_N;
    println!("## Apples-to-apples from a common layout, {n} values\n");

    // Transpose cost alone, for reference.
    let v = data::gen_i128(n, Magnitude::Large, 1);
    let mut split = SplitI128::from_aos(&v);
    let mut merged = vec![0i128; n];
    let t_split = throughput(
        time_per_call(dur, || {
            SplitI128::split_into(black_box(&v), black_box(&mut split));
        }),
        n,
    );
    let t_merge = throughput(
        time_per_call(dur, || {
            split.merge_into(black_box(&mut merged));
        }),
        n,
    );
    println!(
        "Transpose throughput: AoS->SoA **{t_split:.0}**, SoA->AoS **{t_merge:.0}** M items/s.\n"
    );

    println!("### Regime A: data stored INTERLEAVED (split pays the transpose)");
    println!("Split column = full pipeline (split inputs [+ merge output]) + kernel.\n");
    println!("| op | Arrow | split end-to-end | speedup |");
    println!("|---|---:|---:|---:|");

    // add: 2 input transposes + kernel + 1 output merge.
    let a = data::gen_i128(n, Magnitude::Large, 1);
    let b = data::gen_i128(n, Magnitude::Large, 2);
    let aa = arrow_ref::decimal128(&a, 38, 0);
    let ba = arrow_ref::decimal128(&b, 38, 0);
    let mut sa = SplitI128::from_aos(&a);
    let mut sb = SplitI128::from_aos(&b);
    let mut so = sa.zeroed_like();
    let mut out_aos = vec![0i128; n];

    let arrow_add = throughput(
        time_per_call(dur, || {
            black_box(arrow_ref::add_decimal128(black_box(&aa), black_box(&ba)));
        }),
        n,
    );
    let split_add = throughput(
        time_per_call(dur, || {
            SplitI128::split_into(black_box(&a), &mut sa);
            SplitI128::split_into(black_box(&b), &mut sb);
            simd::add_i128(&sa, &sb, &mut so);
            so.merge_into(&mut out_aos);
            black_box(&out_aos);
        }),
        n,
    );
    println!(
        "| add (2 split + merge) | {arrow_add:.0} | {split_add:.0} | {:.2}x |",
        split_add / arrow_add
    );

    // sum: 1 input transpose + reduction, no merge.
    let sum_arrow = throughput(
        time_per_call(dur, || {
            black_box(arrow_ref::sum_decimal128(black_box(&aa)));
        }),
        n,
    );
    let sum_split = throughput(
        time_per_call(dur, || {
            SplitI128::split_into(black_box(&a), &mut sa);
            black_box(aggregate::sum_i128_widening(&sa));
        }),
        n,
    );
    println!(
        "| sum (1 split) | {sum_arrow:.0} | {sum_split:.0} | {:.2}x |",
        sum_split / sum_arrow
    );

    // compare lt: 2 input transposes + kernel -> bitmap, no merge.
    let mut bm = vec![0u8; compare::bitmap_len(n)];
    let cmp_arrow = throughput(
        time_per_call(dur, || {
            black_box(arrow_ref::lt_decimal128(black_box(&aa), black_box(&ba)));
        }),
        n,
    );
    let cmp_split = throughput(
        time_per_call(dur, || {
            SplitI128::split_into(black_box(&a), &mut sa);
            SplitI128::split_into(black_box(&b), &mut sb);
            compare::lt_i128(&sa, &sb, &mut bm);
            black_box(&bm);
        }),
        n,
    );
    println!(
        "| lt (2 split) | {cmp_arrow:.0} | {cmp_split:.0} | {:.2}x |",
        cmp_split / cmp_arrow
    );

    println!("\n### Regime B: data stored SPLIT (Arrow pays the gather)");
    println!("This is what the turn-1 compression encoding gives: operands already live as limb");
    println!(
        "streams. The split kernels run directly; Arrow must first gather them into interleaved"
    );
    println!("Decimal128 (single pass) before its kernel. Arrow column = gather + kernel.\n");
    println!("| op | Arrow end-to-end | split kernel | speedup |");
    println!("|---|---:|---:|---:|");

    let arrowb_add = throughput(
        time_per_call(dur, || {
            let ag = arrow_ref::decimal128_from_split(&sa, 38, 0);
            let bg = arrow_ref::decimal128_from_split(&sb, 38, 0);
            black_box(arrow_ref::add_decimal128(&ag, &bg));
        }),
        n,
    );
    let splitb_add = throughput(
        time_per_call(dur, || {
            simd::add_i128(&sa, &sb, &mut so);
            black_box(&so);
        }),
        n,
    );
    println!(
        "| add | {arrowb_add:.0} | {splitb_add:.0} | {:.2}x |",
        splitb_add / arrowb_add
    );

    let arrowb_sum = throughput(
        time_per_call(dur, || {
            let ag = arrow_ref::decimal128_from_split(&sa, 38, 0);
            black_box(arrow_ref::sum_decimal128(&ag));
        }),
        n,
    );
    let splitb_sum = throughput(
        time_per_call(dur, || {
            black_box(aggregate::sum_i128_widening(&sa));
        }),
        n,
    );
    println!(
        "| sum | {arrowb_sum:.0} | {splitb_sum:.0} | {:.2}x |",
        splitb_sum / arrowb_sum
    );

    let arrowb_lt = throughput(
        time_per_call(dur, || {
            let ag = arrow_ref::decimal128_from_split(&sa, 38, 0);
            let bg = arrow_ref::decimal128_from_split(&sb, 38, 0);
            black_box(arrow_ref::lt_decimal128(&ag, &bg));
        }),
        n,
    );
    let splitb_lt = throughput(
        time_per_call(dur, || {
            compare::lt_i128(&sa, &sb, &mut bm);
            black_box(&bm);
        }),
        n,
    );
    println!(
        "| lt | {arrowb_lt:.0} | {splitb_lt:.0} | {:.2}x |",
        splitb_lt / arrowb_lt
    );
    println!();
}
