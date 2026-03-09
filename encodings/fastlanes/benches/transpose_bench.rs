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

const BATCH_SIZE: usize = 1000;

// ============================================================================
// Transpose: single array
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

// ============================================================================
// Transpose: throughput (1000 arrays)
// ============================================================================

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

// ============================================================================
// Untranspose: single array
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
fn untranspose_scalar_fast(bencher: Bencher) {
    let input = generate_test_data(42);
    let mut output = [0u8; 128];

    bencher.bench_local(|| {
        transpose::untranspose_1024_scalar_fast(&input, &mut output);
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

// ============================================================================
// Untranspose: throughput (1000 arrays)
// ============================================================================

#[divan::bench]
fn untranspose_baseline_throughput(bencher: Bencher) {
    let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
    let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

    bencher.bench_local(|| {
        for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
            transpose::untranspose_1024_baseline(input, output);
        }
        divan::black_box(&outputs);
    });
}

#[divan::bench]
fn untranspose_scalar_throughput(bencher: Bencher) {
    let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
    let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

    bencher.bench_local(|| {
        for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
            transpose::untranspose_1024_scalar(input, output);
        }
        divan::black_box(&outputs);
    });
}

#[divan::bench]
fn untranspose_scalar_fast_throughput(bencher: Bencher) {
    let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
    let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

    bencher.bench_local(|| {
        for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
            transpose::untranspose_1024_scalar_fast(input, output);
        }
        divan::black_box(&outputs);
    });
}

#[divan::bench]
fn untranspose_best_throughput(bencher: Bencher) {
    let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
    let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

    bencher.bench_local(|| {
        for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
            transpose::untranspose_1024_best(input, output);
        }
        divan::black_box(&outputs);
    });
}

// ============================================================================
// x86_64 benchmarks
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod x86_benches {
    use vortex_fastlanes::transpose::x86;

    use super::*;

    // --- Transpose: single array ---

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

    #[divan::bench]
    fn transpose_vbmi(bencher: Bencher) {
        if !x86::has_vbmi() {
            eprintln!("AVX512VBMI not available, skipping benchmark");
            return;
        }

        let input = generate_test_data(42);
        let mut output = [0u8; 128];

        bencher.bench_local(|| {
            unsafe { x86::transpose_1024_vbmi(&input, &mut output) };
            divan::black_box(&output);
        });
    }

    #[divan::bench]
    fn transpose_x2_avx512(bencher: Bencher) {
        if !x86::has_avx512() || !x86::has_gfni() {
            eprintln!("AVX-512+GFNI not available, skipping benchmark");
            return;
        }

        let input0 = generate_test_data(42);
        let input1 = generate_test_data(98);
        let mut output0 = [0u8; 128];
        let mut output1 = [0u8; 128];

        bencher.bench_local(|| {
            unsafe { x86::transpose_1024x2_avx512(&input0, &input1, &mut output0, &mut output1) };
            divan::black_box((&output0, &output1));
        });
    }

    #[divan::bench]
    fn transpose_x2_vbmi(bencher: Bencher) {
        if !x86::has_vbmi() {
            eprintln!("AVX512VBMI not available, skipping benchmark");
            return;
        }

        let input0 = generate_test_data(42);
        let input1 = generate_test_data(98);
        let mut output0 = [0u8; 128];
        let mut output1 = [0u8; 128];

        bencher.bench_local(|| {
            unsafe { x86::transpose_1024x2_vbmi(&input0, &input1, &mut output0, &mut output1) };
            divan::black_box((&output0, &output1));
        });
    }

    #[divan::bench]
    fn transpose_x4_vbmi(bencher: Bencher) {
        if !x86::has_vbmi() {
            eprintln!("AVX512VBMI not available, skipping benchmark");
            return;
        }

        let input0 = generate_test_data(42);
        let input1 = generate_test_data(98);
        let input2 = generate_test_data(13);
        let input3 = generate_test_data(77);
        let mut output0 = [0u8; 128];
        let mut output1 = [0u8; 128];
        let mut output2 = [0u8; 128];
        let mut output3 = [0u8; 128];

        bencher.bench_local(|| {
            unsafe {
                x86::transpose_1024x4_vbmi(
                    &input0,
                    &input1,
                    &input2,
                    &input3,
                    &mut output0,
                    &mut output1,
                    &mut output2,
                    &mut output3,
                )
            };
            divan::black_box((&output0, &output1, &output2, &output3));
        });
    }

    // --- Untranspose: single array ---

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

    #[divan::bench]
    fn untranspose_vbmi(bencher: Bencher) {
        if !x86::has_vbmi() {
            eprintln!("AVX512VBMI not available, skipping benchmark");
            return;
        }

        let input = generate_test_data(42);
        let mut output = [0u8; 128];

        bencher.bench_local(|| {
            unsafe { x86::untranspose_1024_vbmi(&input, &mut output) };
            divan::black_box(&output);
        });
    }

    #[divan::bench]
    fn untranspose_x2_avx512(bencher: Bencher) {
        if !x86::has_avx512() || !x86::has_gfni() {
            eprintln!("AVX-512+GFNI not available, skipping benchmark");
            return;
        }

        let input0 = generate_test_data(42);
        let input1 = generate_test_data(98);
        let mut output0 = [0u8; 128];
        let mut output1 = [0u8; 128];

        bencher.bench_local(|| {
            unsafe { x86::untranspose_1024x2_avx512(&input0, &input1, &mut output0, &mut output1) };
            divan::black_box((&output0, &output1));
        });
    }

    #[divan::bench]
    fn untranspose_x2_vbmi(bencher: Bencher) {
        if !x86::has_vbmi() {
            eprintln!("AVX512VBMI not available, skipping benchmark");
            return;
        }

        let input0 = generate_test_data(42);
        let input1 = generate_test_data(98);
        let mut output0 = [0u8; 128];
        let mut output1 = [0u8; 128];

        bencher.bench_local(|| {
            unsafe { x86::untranspose_1024x2_vbmi(&input0, &input1, &mut output0, &mut output1) };
            divan::black_box((&output0, &output1));
        });
    }

    // --- Transpose: throughput (1000 arrays) ---

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

    #[divan::bench]
    fn transpose_vbmi_throughput(bencher: Bencher) {
        if !x86::has_vbmi() {
            eprintln!("AVX512VBMI not available, skipping benchmark");
            return;
        }

        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                unsafe { x86::transpose_1024_vbmi(input, output) };
            }
            divan::black_box(&outputs);
        });
    }

    #[divan::bench]
    fn transpose_x2_avx512_throughput(bencher: Bencher) {
        if !x86::has_avx512() || !x86::has_gfni() {
            eprintln!("AVX-512+GFNI not available, skipping benchmark");
            return;
        }

        let inputs0: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let inputs1: Vec<[u8; 128]> = (0..BATCH_SIZE as u8)
            .map(|i| generate_test_data(i.wrapping_add(128)))
            .collect();
        let mut outputs0 = vec![[0u8; 128]; BATCH_SIZE];
        let mut outputs1 = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for ((in0, in1), (out0, out1)) in inputs0
                .iter()
                .zip(inputs1.iter())
                .zip(outputs0.iter_mut().zip(outputs1.iter_mut()))
            {
                unsafe { x86::transpose_1024x2_avx512(in0, in1, out0, out1) };
            }
            divan::black_box((&outputs0, &outputs1));
        });
    }

    #[divan::bench]
    fn transpose_x2_vbmi_throughput(bencher: Bencher) {
        if !x86::has_vbmi() {
            eprintln!("AVX512VBMI not available, skipping benchmark");
            return;
        }

        let inputs0: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let inputs1: Vec<[u8; 128]> = (0..BATCH_SIZE as u8)
            .map(|i| generate_test_data(i.wrapping_add(128)))
            .collect();
        let mut outputs0 = vec![[0u8; 128]; BATCH_SIZE];
        let mut outputs1 = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for ((in0, in1), (out0, out1)) in inputs0
                .iter()
                .zip(inputs1.iter())
                .zip(outputs0.iter_mut().zip(outputs1.iter_mut()))
            {
                unsafe { x86::transpose_1024x2_vbmi(in0, in1, out0, out1) };
            }
            divan::black_box((&outputs0, &outputs1));
        });
    }

    #[divan::bench]
    fn transpose_x4_vbmi_throughput(bencher: Bencher) {
        if !x86::has_vbmi() {
            eprintln!("AVX512VBMI not available, skipping benchmark");
            return;
        }

        let inputs0: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let inputs1: Vec<[u8; 128]> = (0..BATCH_SIZE as u8)
            .map(|i| generate_test_data(i.wrapping_add(64)))
            .collect();
        let inputs2: Vec<[u8; 128]> = (0..BATCH_SIZE as u8)
            .map(|i| generate_test_data(i.wrapping_add(128)))
            .collect();
        let inputs3: Vec<[u8; 128]> = (0..BATCH_SIZE as u8)
            .map(|i| generate_test_data(i.wrapping_add(192)))
            .collect();
        let mut outputs0 = vec![[0u8; 128]; BATCH_SIZE];
        let mut outputs1 = vec![[0u8; 128]; BATCH_SIZE];
        let mut outputs2 = vec![[0u8; 128]; BATCH_SIZE];
        let mut outputs3 = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for ((((in0, in1), (in2, in3)), (out0, out1)), (out2, out3)) in inputs0
                .iter()
                .zip(inputs1.iter())
                .zip(inputs2.iter().zip(inputs3.iter()))
                .zip(outputs0.iter_mut().zip(outputs1.iter_mut()))
                .zip(outputs2.iter_mut().zip(outputs3.iter_mut()))
            {
                unsafe { x86::transpose_1024x4_vbmi(in0, in1, in2, in3, out0, out1, out2, out3) };
            }
            divan::black_box((&outputs0, &outputs1, &outputs2, &outputs3));
        });
    }

    // --- Untranspose: throughput (1000 arrays) ---

    #[divan::bench]
    fn untranspose_bmi2_throughput(bencher: Bencher) {
        if !x86::has_bmi2() {
            eprintln!("BMI2 not available, skipping benchmark");
            return;
        }

        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                unsafe { x86::untranspose_1024_bmi2(input, output) };
            }
            divan::black_box(&outputs);
        });
    }

    #[divan::bench]
    fn untranspose_avx512_gfni_throughput(bencher: Bencher) {
        if !x86::has_avx512() || !x86::has_gfni() {
            eprintln!("AVX-512+GFNI not available, skipping benchmark");
            return;
        }

        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                unsafe { x86::untranspose_1024_avx512_gfni(input, output) };
            }
            divan::black_box(&outputs);
        });
    }

    #[divan::bench]
    fn untranspose_vbmi_throughput(bencher: Bencher) {
        if !x86::has_vbmi() {
            eprintln!("AVX512VBMI not available, skipping benchmark");
            return;
        }

        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                unsafe { x86::untranspose_1024_vbmi(input, output) };
            }
            divan::black_box(&outputs);
        });
    }

    #[divan::bench]
    fn untranspose_x2_avx512_throughput(bencher: Bencher) {
        if !x86::has_avx512() || !x86::has_gfni() {
            eprintln!("AVX-512+GFNI not available, skipping benchmark");
            return;
        }

        let inputs0: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let inputs1: Vec<[u8; 128]> = (0..BATCH_SIZE as u8)
            .map(|i| generate_test_data(i.wrapping_add(128)))
            .collect();
        let mut outputs0 = vec![[0u8; 128]; BATCH_SIZE];
        let mut outputs1 = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for ((in0, in1), (out0, out1)) in inputs0
                .iter()
                .zip(inputs1.iter())
                .zip(outputs0.iter_mut().zip(outputs1.iter_mut()))
            {
                unsafe { x86::untranspose_1024x2_avx512(in0, in1, out0, out1) };
            }
            divan::black_box((&outputs0, &outputs1));
        });
    }

    #[divan::bench]
    fn untranspose_x2_vbmi_throughput(bencher: Bencher) {
        if !x86::has_vbmi() {
            eprintln!("AVX512VBMI not available, skipping benchmark");
            return;
        }

        let inputs0: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let inputs1: Vec<[u8; 128]> = (0..BATCH_SIZE as u8)
            .map(|i| generate_test_data(i.wrapping_add(128)))
            .collect();
        let mut outputs0 = vec![[0u8; 128]; BATCH_SIZE];
        let mut outputs1 = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for ((in0, in1), (out0, out1)) in inputs0
                .iter()
                .zip(inputs1.iter())
                .zip(outputs0.iter_mut().zip(outputs1.iter_mut()))
            {
                unsafe { x86::untranspose_1024x2_vbmi(in0, in1, out0, out1) };
            }
            divan::black_box((&outputs0, &outputs1));
        });
    }
}

// ============================================================================
// aarch64 benchmarks
// ============================================================================

#[cfg(target_arch = "aarch64")]
mod aarch64_benches {
    use vortex_fastlanes::transpose::aarch64;

    use super::*;

    // --- Transpose: single array ---

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
    fn transpose_x2_neon(bencher: Bencher) {
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

    // --- Untranspose: single array ---

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
    fn untranspose_x2_neon(bencher: Bencher) {
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

    // --- Transpose: throughput (1000 arrays) ---

    #[divan::bench]
    fn transpose_neon_throughput(bencher: Bencher) {
        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                unsafe { aarch64::transpose_1024_neon(input, output) };
            }
            divan::black_box(&outputs);
        });
    }

    #[divan::bench]
    fn transpose_neon_tbl_throughput(bencher: Bencher) {
        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                unsafe { aarch64::transpose_1024_neon_tbl(input, output) };
            }
            divan::black_box(&outputs);
        });
    }

    #[divan::bench]
    fn transpose_x2_neon_throughput(bencher: Bencher) {
        let inputs0: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let inputs1: Vec<[u8; 128]> = (0..BATCH_SIZE as u8)
            .map(|i| generate_test_data(i.wrapping_add(128)))
            .collect();
        let mut outputs0 = vec![[0u8; 128]; BATCH_SIZE];
        let mut outputs1 = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for ((in0, in1), (out0, out1)) in inputs0
                .iter()
                .zip(inputs1.iter())
                .zip(outputs0.iter_mut().zip(outputs1.iter_mut()))
            {
                unsafe { aarch64::transpose_1024x2_neon(in0, in1, out0, out1) };
            }
            divan::black_box((&outputs0, &outputs1));
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

    // --- Untranspose: throughput (1000 arrays) ---

    #[divan::bench]
    fn untranspose_neon_throughput(bencher: Bencher) {
        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                unsafe { aarch64::untranspose_1024_neon(input, output) };
            }
            divan::black_box(&outputs);
        });
    }

    #[divan::bench]
    fn untranspose_neon_tbl_throughput(bencher: Bencher) {
        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                unsafe { aarch64::untranspose_1024_neon_tbl(input, output) };
            }
            divan::black_box(&outputs);
        });
    }

    #[divan::bench]
    fn untranspose_x2_neon_throughput(bencher: Bencher) {
        let inputs0: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let inputs1: Vec<[u8; 128]> = (0..BATCH_SIZE as u8)
            .map(|i| generate_test_data(i.wrapping_add(128)))
            .collect();
        let mut outputs0 = vec![[0u8; 128]; BATCH_SIZE];
        let mut outputs1 = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for ((in0, in1), (out0, out1)) in inputs0
                .iter()
                .zip(inputs1.iter())
                .zip(outputs0.iter_mut().zip(outputs1.iter_mut()))
            {
                unsafe { aarch64::untranspose_1024x2_neon(in0, in1, out0, out1) };
            }
            divan::black_box((&outputs0, &outputs1));
        });
    }

    #[divan::bench]
    fn untranspose_sve_throughput(bencher: Bencher) {
        if !aarch64::has_sme() {
            eprintln!("SME not available, skipping benchmark");
            return;
        }
        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE as u8).map(generate_test_data).collect();
        let mut outputs = vec![[0u8; 128]; BATCH_SIZE];

        bencher.bench_local(|| {
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                unsafe { aarch64::untranspose_1024_sve(input, output) };
            }
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
}
