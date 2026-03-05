// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for FastLanes 1024-bit transpose implementations.
//!
//! Run with: cargo bench -p vortex-fastlanes --bench transpose_bench

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_fastlanes::transpose;

fn main() {
    divan::main();
}

/// Generate deterministic test data.
fn generate_test_data(seed: u8) -> [u8; 128] {
    let mut data = [0u8; 128];
    for (i, byte) in data.iter_mut().enumerate() {
        *byte = seed.wrapping_mul(17).wrapping_add(i as u8).wrapping_mul(31);
    }
    data
}

// ============================================================================
// Transpose benchmarks
// ============================================================================

#[divan::bench]
fn transpose_baseline(bencher: Bencher) {
    let input = generate_test_data(42);
    let mut output = [0u8; 128];

    bencher.bench_local(|| {
        transpose::transpose_1024_baseline(&input, &mut output);
        divan::black_box(&output);
    });
}

#[divan::bench]
fn transpose_scalar(bencher: Bencher) {
    let input = generate_test_data(42);
    let mut output = [0u8; 128];

    bencher.bench_local(|| {
        transpose::transpose_1024_scalar(&input, &mut output);
        divan::black_box(&output);
    });
}

#[divan::bench]
fn transpose_scalar_fast(bencher: Bencher) {
    let input = generate_test_data(42);
    let mut output = [0u8; 128];

    bencher.bench_local(|| {
        transpose::transpose_1024_scalar_fast(&input, &mut output);
        divan::black_box(&output);
    });
}

#[divan::bench]
fn transpose_best(bencher: Bencher) {
    let input = generate_test_data(42);
    let mut output = [0u8; 128];

    bencher.bench_local(|| {
        transpose::transpose_1024_best(&input, &mut output);
        divan::black_box(&output);
    });
}

#[cfg(target_arch = "x86_64")]
mod x86_benches {
    use vortex_fastlanes::transpose::x86;

    use super::*;

    #[divan::bench]
    fn transpose_bmi2(bencher: Bencher) {
        if !x86::has_bmi2() {
            eprintln!("BMI2 not available, skipping benchmark");
            return;
        }

        let input = generate_test_data(42);
        let mut output = [0u8; 128];

        bencher.bench_local(|| {
            unsafe { x86::transpose_1024_bmi2(&input, &mut output) };
            divan::black_box(&output);
        });
    }

    #[divan::bench]
    fn transpose_avx2(bencher: Bencher) {
        if !x86::has_avx2() {
            eprintln!("AVX2 not available, skipping benchmark");
            return;
        }

        let input = generate_test_data(42);
        let mut output = [0u8; 128];

        bencher.bench_local(|| {
            unsafe { x86::transpose_1024_avx2(&input, &mut output) };
            divan::black_box(&output);
        });
    }

    #[divan::bench]
    fn transpose_avx2_gfni(bencher: Bencher) {
        if !x86::has_avx2() || !x86::has_gfni() {
            eprintln!("AVX2+GFNI not available, skipping benchmark");
            return;
        }

        let input = generate_test_data(42);
        let mut output = [0u8; 128];

        bencher.bench_local(|| {
            unsafe { x86::transpose_1024_avx2_gfni(&input, &mut output) };
            divan::black_box(&output);
        });
    }

    #[divan::bench]
    fn transpose_avx512_gfni(bencher: Bencher) {
        if !x86::has_avx512() || !x86::has_gfni() {
            eprintln!("AVX-512+GFNI not available, skipping benchmark");
            return;
        }

        let input = generate_test_data(42);
        let mut output = [0u8; 128];

        bencher.bench_local(|| {
            unsafe { x86::transpose_1024_avx512_gfni(&input, &mut output) };
            divan::black_box(&output);
        });
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64_benches {
    use vortex_fastlanes::transpose::aarch64;

    use super::*;

    #[divan::bench]
    fn transpose_neon(bencher: Bencher) {
        let input = generate_test_data(42);
        let mut output = [0u8; 128];

        bencher.bench_local(|| {
            unsafe { aarch64::transpose_1024_neon(&input, &mut output) };
            divan::black_box(&output);
        });
    }

    #[divan::bench]
    fn transpose_neon_tbl(bencher: Bencher) {
        let input = generate_test_data(42);
        let mut output = [0u8; 128];

        bencher.bench_local(|| {
            unsafe { aarch64::transpose_1024_neon_tbl(&input, &mut output) };
            divan::black_box(&output);
        });
    }

    #[divan::bench]
    fn transpose_neon_tbl_throughput(bencher: Bencher) {
        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                unsafe { aarch64::transpose_1024_neon_tbl(&input, output) };
            }
            divan::black_box(&outputs);
        });
    }

    #[divan::bench]
    fn untranspose_neon(bencher: Bencher) {
        let input = generate_test_data(42);
        let mut output = [0u8; 128];

        bencher.bench_local(|| {
            unsafe { aarch64::untranspose_1024_neon(&input, &mut output) };
            divan::black_box(&output);
        });
    }

    #[divan::bench]
    fn untranspose_neon_tbl(bencher: Bencher) {
        let input = generate_test_data(42);
        let mut output = [0u8; 128];

        bencher.bench_local(|| {
            unsafe { aarch64::untranspose_1024_neon_tbl(&input, &mut output) };
            divan::black_box(&output);
        });
    }

    #[divan::bench]
    fn transpose_neon_dual_block(bencher: Bencher) {
        let input0 = generate_test_data(42);
        let input1 = generate_test_data(99);
        let mut output0 = [0u8; 128];
        let mut output1 = [0u8; 128];

        bencher.bench_local(|| {
            unsafe { aarch64::transpose_1024x2_neon(&input0, &input1, &mut output0, &mut output1) };
            divan::black_box((&output0, &output1));
        });
    }

    #[divan::bench]
    fn transpose_neon_dual_block_throughput(bencher: Bencher) {
        let inputs0: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let inputs1: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let mut outputs0 = vec![[0u8; 128]; BATCH_SIZE];
        let mut outputs1 = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for ((input0, input1), (output0, output1)) in inputs0
                .iter()
                .zip(inputs1.iter())
                .zip(outputs0.iter_mut().zip(outputs1.iter_mut()))
            {
                unsafe { aarch64::transpose_1024x2_neon(&input0, &input1, output0, output1) };
                divan::black_box((&output0, &output1));
            }
        });
    }

    #[divan::bench]
    fn transpose_sve(bencher: Bencher) {
        if !aarch64::has_sme() {
            eprintln!("SME not available, skipping benchmark");
            return;
        }

        let input = generate_test_data(42);
        let mut output = [0u8; 128];

        bencher.bench_local(|| {
            unsafe { aarch64::transpose_1024_sve(&input, &mut output) };
            divan::black_box(&output);
        });
    }

    #[divan::bench]
    fn transpose_sve_throughput(bencher: Bencher) {
        if !aarch64::has_sme() {
            eprintln!("SME not available, skipping benchmark");
            return;
        }
        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                unsafe { aarch64::transpose_1024_sve(input, output) };
            }
            divan::black_box(&outputs);
        });
    }

    #[divan::bench]
    fn untranspose_sve(bencher: Bencher) {
        if !aarch64::has_sme() {
            eprintln!("SME not available, skipping benchmark");
            return;
        }

        let input = generate_test_data(42);
        let mut output = [0u8; 128];

        bencher.bench_local(|| {
            unsafe { aarch64::untranspose_1024_sve(&input, &mut output) };
            divan::black_box(&output);
        });
    }

    #[divan::bench]
    fn transpose_sve_batch(bencher: Bencher) {
        if !aarch64::has_sme() {
            eprintln!("SME not available, skipping benchmark");
            return;
        }
        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            unsafe { aarch64::transpose_1024_batch_sve(&inputs, &mut outputs) };
            divan::black_box(&outputs);
        });
    }

    #[divan::bench]
    fn untranspose_sve_batch(bencher: Bencher) {
        if !aarch64::has_sme() {
            eprintln!("SME not available, skipping benchmark");
            return;
        }
        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            unsafe { aarch64::untranspose_1024_batch_sve(&inputs, &mut outputs) };
            divan::black_box(&outputs);
        });
    }

    #[divan::bench]
    fn untranspose_neon_dual_block(bencher: Bencher) {
        let input0 = generate_test_data(42);
        let input1 = generate_test_data(99);
        let mut output0 = [0u8; 128];
        let mut output1 = [0u8; 128];

        bencher.bench_local(|| {
            unsafe {
                aarch64::untranspose_1024x2_neon(&input0, &input1, &mut output0, &mut output1)
            };
            divan::black_box((&output0, &output1));
        });
    }
}

// ============================================================================
// Untranspose benchmarks
// ============================================================================

#[divan::bench]
fn untranspose_baseline(bencher: Bencher) {
    let input = generate_test_data(42);
    let mut output = [0u8; 128];

    bencher.bench_local(|| {
        transpose::untranspose_1024_baseline(&input, &mut output);
        divan::black_box(&output);
    });
}

#[divan::bench]
fn untranspose_scalar(bencher: Bencher) {
    let input = generate_test_data(42);
    let mut output = [0u8; 128];

    bencher.bench_local(|| {
        transpose::untranspose_1024_scalar(&input, &mut output);
        divan::black_box(&output);
    });
}

#[divan::bench]
fn untranspose_best(bencher: Bencher) {
    let input = generate_test_data(42);
    let mut output = [0u8; 128];

    bencher.bench_local(|| {
        transpose::untranspose_1024_best(&input, &mut output);
        divan::black_box(&output);
    });
}

#[cfg(target_arch = "x86_64")]
mod x86_untranspose_benches {
    use vortex_fastlanes::transpose::x86;

    use super::*;

    #[divan::bench]
    fn untranspose_bmi2(bencher: Bencher) {
        if !x86::has_bmi2() {
            eprintln!("BMI2 not available, skipping benchmark");
            return;
        }

        let input = generate_test_data(42);
        let mut output = [0u8; 128];

        bencher.bench_local(|| {
            unsafe { x86::untranspose_1024_bmi2(&input, &mut output) };
            divan::black_box(&output);
        });
    }

    #[divan::bench]
    fn untranspose_avx2_gfni(bencher: Bencher) {
        if !x86::has_avx2() || !x86::has_gfni() {
            eprintln!("AVX2+GFNI not available, skipping benchmark");
            return;
        }

        let input = generate_test_data(42);
        let mut output = [0u8; 128];

        bencher.bench_local(|| {
            unsafe { x86::untranspose_1024_avx2_gfni(&input, &mut output) };
            divan::black_box(&output);
        });
    }

    #[divan::bench]
    fn untranspose_avx512_gfni(bencher: Bencher) {
        if !x86::has_avx512() || !x86::has_gfni() {
            eprintln!("AVX-512+GFNI not available, skipping benchmark");
            return;
        }

        let input = generate_test_data(42);
        let mut output = [0u8; 128];

        bencher.bench_local(|| {
            unsafe { x86::untranspose_1024_avx512_gfni(&input, &mut output) };
            divan::black_box(&output);
        });
    }
}

// ============================================================================
// Throughput benchmarks (multiple iterations to measure GB/s)
// ============================================================================

const BATCH_SIZE: usize = 1000;

#[divan::bench]
fn transpose_baseline_throughput(bencher: Bencher) {
    let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
    let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

    bencher.bench_local(|| {
        for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
            transpose::transpose_1024_baseline(input, output);
        }
        divan::black_box(&outputs);
    });
}

#[divan::bench]
fn transpose_scalar_throughput(bencher: Bencher) {
    let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
    let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

    bencher.bench_local(|| {
        for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
            transpose::transpose_1024_scalar(input, output);
        }
        divan::black_box(&outputs);
    });
}

#[divan::bench]
fn transpose_scalar_fast_throughput(bencher: Bencher) {
    let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
    let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

    bencher.bench_local(|| {
        for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
            transpose::transpose_1024_scalar_fast(input, output);
        }
        divan::black_box(&outputs);
    });
}

#[divan::bench]
fn transpose_best_throughput(bencher: Bencher) {
    let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
    let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

    bencher.bench_local(|| {
        for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
            transpose::transpose_1024_best(input, output);
        }
        divan::black_box(&outputs);
    });
}

#[cfg(target_arch = "x86_64")]
mod x86_throughput_benches {
    use vortex_fastlanes::transpose::x86;

    use super::*;

    #[divan::bench]
    fn transpose_bmi2_throughput(bencher: Bencher) {
        if !x86::has_bmi2() {
            eprintln!("BMI2 not available, skipping benchmark");
            return;
        }

        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                unsafe { x86::transpose_1024_bmi2(input, output) };
            }
            divan::black_box(&outputs);
        });
    }

    #[divan::bench]
    fn transpose_avx2_throughput(bencher: Bencher) {
        if !x86::has_avx2() {
            eprintln!("AVX2 not available, skipping benchmark");
            return;
        }

        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                unsafe { x86::transpose_1024_avx2(input, output) };
            }
            divan::black_box(&outputs);
        });
    }

    #[divan::bench]
    fn transpose_avx2_gfni_throughput(bencher: Bencher) {
        if !x86::has_avx2() || !x86::has_gfni() {
            eprintln!("AVX2+GFNI not available, skipping benchmark");
            return;
        }

        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                unsafe { x86::transpose_1024_avx2_gfni(input, output) };
            }
            divan::black_box(&outputs);
        });
    }

    #[divan::bench]
    fn transpose_avx512_gfni_throughput(bencher: Bencher) {
        if !x86::has_avx512() || !x86::has_gfni() {
            eprintln!("AVX-512+GFNI not available, skipping benchmark");
            return;
        }

        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                unsafe { x86::transpose_1024_avx512_gfni(input, output) };
            }
            divan::black_box(&outputs);
        });
    }
}
