// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Micro-benchmark for measuring cycle counts using rdtsc.
//!
//! Run with: ./target/release/examples/perf_transpose [baseline|scalar|avx2|...]

use std::hint::black_box;

use vortex_fastlanes::transpose;

const WARMUP_ITERATIONS: usize = 100_000;
const MEASURE_ITERATIONS: usize = 1_000_000;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn rdtsc() -> u64 {
    unsafe {
        let lo: u32;
        let hi: u32;
        core::arch::asm!(
            "rdtsc",
            out("eax") lo,
            out("edx") hi,
            options(nomem, nostack)
        );
        ((hi as u64) << 32) | (lo as u64)
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn rdtsc() -> u64 {
    0
}

fn main() {
    let input = [0x55u8; 128]; // Alternating pattern
    let mut output = [0u8; 128];

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        // Run all benchmarks
        run_all_benchmarks(&input, &mut output);
        return;
    }

    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("scalar");
    run_benchmark(mode, &input, &mut output);
}

fn run_all_benchmarks(input: &[u8; 128], output: &mut [u8; 128]) {
    println!("FastLanes 1024-bit Transpose - Cycle Measurements");
    println!("==================================================");
    println!("Iterations: {}", MEASURE_ITERATIONS);
    println!();

    let modes = [
        "baseline",
        "scalar",
        "scalar_fast",
        "bmi2",
        "avx2",
        "avx2_gfni",
        "avx512_gfni",
        "avx512_vbmi",
        "vbmi_dual",
        "vbmi_quad",
    ];

    for mode in &modes {
        run_benchmark(mode, input, output);
    }
}

fn run_benchmark(mode: &str, input: &[u8; 128], output: &mut [u8; 128]) {
    // Warmup
    for _ in 0..WARMUP_ITERATIONS {
        match mode {
            "baseline" => {
                transpose::transpose_1024_baseline(black_box(input), black_box(output));
            }
            "scalar" => {
                transpose::transpose_1024_scalar(black_box(input), black_box(output));
            }
            "scalar_fast" => {
                transpose::transpose_1024_scalar_fast(black_box(input), black_box(output));
            }
            #[cfg(target_arch = "x86_64")]
            "bmi2" => {
                use vortex_fastlanes::transpose::x86;
                if x86::has_bmi2() {
                    unsafe {
                        x86::transpose_1024_bmi2(black_box(input), black_box(output));
                    }
                }
            }
            #[cfg(target_arch = "x86_64")]
            "avx2" => {
                use vortex_fastlanes::transpose::x86;
                if x86::has_avx2() {
                    unsafe {
                        x86::transpose_1024_avx2(black_box(input), black_box(output));
                    }
                }
            }
            #[cfg(target_arch = "x86_64")]
            "avx2_gfni" => {
                use vortex_fastlanes::transpose::x86;
                if x86::has_avx2() && x86::has_gfni() {
                    unsafe {
                        x86::transpose_1024_avx2_gfni(black_box(input), black_box(output));
                    }
                }
            }
            #[cfg(target_arch = "x86_64")]
            "avx512_gfni" => {
                use vortex_fastlanes::transpose::x86;
                if x86::has_avx512() && x86::has_gfni() {
                    unsafe {
                        x86::transpose_1024_avx512_gfni(black_box(input), black_box(output));
                    }
                }
            }
            #[cfg(target_arch = "x86_64")]
            "avx512_vbmi" => {
                use vortex_fastlanes::transpose::x86;
                if x86::has_vbmi() {
                    unsafe {
                        x86::transpose_1024_vbmi(black_box(input), black_box(output));
                    }
                }
            }
            #[cfg(target_arch = "x86_64")]
            "vbmi_dual" => {
                use vortex_fastlanes::transpose::x86;
                if x86::has_vbmi() {
                    let input2 = *input;
                    let mut output2 = [0u8; 128];
                    unsafe {
                        x86::transpose_1024x2_vbmi(
                            black_box(input),
                            black_box(&input2),
                            black_box(output),
                            black_box(&mut output2),
                        );
                    }
                }
            }
            #[cfg(target_arch = "x86_64")]
            "vbmi_quad" => {
                use vortex_fastlanes::transpose::x86;
                if x86::has_vbmi() {
                    let input2 = *input;
                    let input3 = *input;
                    let input4 = *input;
                    let mut output2 = [0u8; 128];
                    let mut output3 = [0u8; 128];
                    let mut output4 = [0u8; 128];
                    unsafe {
                        x86::transpose_1024x4_vbmi(
                            black_box(input),
                            black_box(&input2),
                            black_box(&input3),
                            black_box(&input4),
                            black_box(output),
                            black_box(&mut output2),
                            black_box(&mut output3),
                            black_box(&mut output4),
                        );
                    }
                }
            }
            _ => {}
        }
    }

    // Measure
    let start = rdtsc();

    match mode {
        "baseline" => {
            for _ in 0..MEASURE_ITERATIONS {
                transpose::transpose_1024_baseline(black_box(input), black_box(output));
            }
        }
        "scalar" => {
            for _ in 0..MEASURE_ITERATIONS {
                transpose::transpose_1024_scalar(black_box(input), black_box(output));
            }
        }
        "scalar_fast" => {
            for _ in 0..MEASURE_ITERATIONS {
                transpose::transpose_1024_scalar_fast(black_box(input), black_box(output));
            }
        }
        #[cfg(target_arch = "x86_64")]
        "bmi2" => {
            use vortex_fastlanes::transpose::x86;
            if x86::has_bmi2() {
                for _ in 0..MEASURE_ITERATIONS {
                    unsafe {
                        x86::transpose_1024_bmi2(black_box(input), black_box(output));
                    }
                }
            } else {
                println!("{:15} BMI2 not available", mode);
                return;
            }
        }
        #[cfg(target_arch = "x86_64")]
        "avx2" => {
            use vortex_fastlanes::transpose::x86;
            if x86::has_avx2() {
                for _ in 0..MEASURE_ITERATIONS {
                    unsafe {
                        x86::transpose_1024_avx2(black_box(input), black_box(output));
                    }
                }
            } else {
                println!("{:15} AVX2 not available", mode);
                return;
            }
        }
        #[cfg(target_arch = "x86_64")]
        "avx2_gfni" => {
            use vortex_fastlanes::transpose::x86;
            if x86::has_avx2() && x86::has_gfni() {
                for _ in 0..MEASURE_ITERATIONS {
                    unsafe {
                        x86::transpose_1024_avx2_gfni(black_box(input), black_box(output));
                    }
                }
            } else {
                println!("{:15} AVX2+GFNI not available", mode);
                return;
            }
        }
        #[cfg(target_arch = "x86_64")]
        "avx512_gfni" => {
            use vortex_fastlanes::transpose::x86;
            if x86::has_avx512() && x86::has_gfni() {
                for _ in 0..MEASURE_ITERATIONS {
                    unsafe {
                        x86::transpose_1024_avx512_gfni(black_box(input), black_box(output));
                    }
                }
            } else {
                println!("{:15} AVX-512+GFNI not available", mode);
                return;
            }
        }
        #[cfg(target_arch = "x86_64")]
        "avx512_vbmi" => {
            use vortex_fastlanes::transpose::x86;
            if x86::has_vbmi() {
                for _ in 0..MEASURE_ITERATIONS {
                    unsafe {
                        x86::transpose_1024_vbmi(black_box(input), black_box(output));
                    }
                }
            } else {
                println!("{:15} AVX-512 VBMI not available", mode);
                return;
            }
        }
        #[cfg(target_arch = "x86_64")]
        "vbmi_dual" => {
            use vortex_fastlanes::transpose::x86;
            if x86::has_vbmi() {
                let input2 = *input;
                let mut output2 = [0u8; 128];
                // Note: we do MEASURE_ITERATIONS/2 since each call processes 2 blocks
                for _ in 0..MEASURE_ITERATIONS / 2 {
                    unsafe {
                        x86::transpose_1024x2_vbmi(
                            black_box(input),
                            black_box(&input2),
                            black_box(output),
                            black_box(&mut output2),
                        );
                    }
                }
            } else {
                println!("{:15} AVX-512 VBMI not available", mode);
                return;
            }
        }
        #[cfg(target_arch = "x86_64")]
        "vbmi_quad" => {
            use vortex_fastlanes::transpose::x86;
            if x86::has_vbmi() {
                let input2 = *input;
                let input3 = *input;
                let input4 = *input;
                let mut output2 = [0u8; 128];
                let mut output3 = [0u8; 128];
                let mut output4 = [0u8; 128];
                // Note: we do MEASURE_ITERATIONS/4 since each call processes 4 blocks
                for _ in 0..MEASURE_ITERATIONS / 4 {
                    unsafe {
                        x86::transpose_1024x4_vbmi(
                            black_box(input),
                            black_box(&input2),
                            black_box(&input3),
                            black_box(&input4),
                            black_box(output),
                            black_box(&mut output2),
                            black_box(&mut output3),
                            black_box(&mut output4),
                        );
                    }
                }
            } else {
                println!("{:15} AVX-512 VBMI not available", mode);
                return;
            }
        }
        _ => {
            println!("Unknown mode: {}", mode);
            return;
        }
    }

    let end = rdtsc();
    let total_cycles = end - start;
    let cycles_per_call = total_cycles as f64 / MEASURE_ITERATIONS as f64;

    println!(
        "{:15} {:>12} total cycles, {:>8.1} cycles/call",
        mode, total_cycles, cycles_per_call
    );
}
