// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use divan::Bencher;
use vortex_fastlanes::bit_transpose::scalar::transpose_bits_scalar;
use vortex_fastlanes::bit_transpose::scalar::untranspose_bits_scalar;

mod shared;

fn main() {
    divan::main();
}

/// Generate deterministic test data.
#[expect(clippy::cast_possible_truncation)]
fn generate_test_data(seed: usize) -> [u8; 128] {
    let mut data = [0u8; 128];
    for (i, byte) in data.iter_mut().enumerate() {
        *byte = seed.wrapping_mul(17).wrapping_add(i).wrapping_mul(31) as u8;
    }
    data
}

const BATCH_SIZE: usize = 1000;

// ============================================================================
// Transpose: single array
//
// Scalar benchmarks are architecture-neutral, so they run on every leg as a
// per-architecture baseline.
// ============================================================================

#[divan::bench(
    name = variant!("transpose_scalar"),
    ignore = ignore_unless_variant!(simulation, x86_64, aarch64),
)]
fn transpose_scalar(bencher: Bencher) {
    let input = generate_test_data(42);

    bencher
        .with_inputs(|| (&input, [0u8; 128]))
        .bench_refs(|(input, output)| {
            transpose_bits_scalar(input, output);
        });
}

// ============================================================================
// Transpose: throughput (1000 arrays)
// ============================================================================

#[divan::bench(
    name = variant!("transpose_scalar_throughput"),
    ignore = ignore_unless_variant!(simulation, x86_64, aarch64),
)]
fn transpose_scalar_throughput(bencher: Bencher) {
    let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE).map(generate_test_data).collect();

    bencher
        .with_inputs(|| (&inputs, vec![[0u8; 128]; BATCH_SIZE]))
        .bench_refs(|(inputs, outputs)| {
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                transpose_bits_scalar(input, output);
            }
        });
}

// ============================================================================
// Untranspose: single array
// ============================================================================

#[divan::bench(
    name = variant!("untranspose_scalar"),
    ignore = ignore_unless_variant!(simulation, x86_64, aarch64),
)]
fn untranspose_scalar(bencher: Bencher) {
    let input = generate_test_data(42);

    bencher
        .with_inputs(|| (&input, [0u8; 128]))
        .bench_refs(|(input, output)| {
            untranspose_bits_scalar(input, output);
        });
}

// ============================================================================
// Untranspose: throughput (1000 arrays)
// ============================================================================

#[divan::bench(
    name = variant!("untranspose_scalar_throughput"),
    ignore = ignore_unless_variant!(simulation, x86_64, aarch64),
)]
fn untranspose_scalar_throughput(bencher: Bencher) {
    let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE).map(generate_test_data).collect();

    bencher
        .with_inputs(|| (&inputs, vec![[0u8; 128]; BATCH_SIZE]))
        .bench_refs(|(inputs, outputs)| {
            for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                untranspose_bits_scalar(input, output);
            }
        });
}

// ============================================================================
// x86_64 benchmarks
//
// BMI2 and VBMI share the `x86_64` walltime leg (and the `simulation` leg); the
// `#[target_feature]` intrinsics are selected at runtime via `has_bmi2` /
// `has_vbmi`, so a single x86 build covers both.
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod x86 {
    use divan::Bencher;
    use vortex_fastlanes::bit_transpose::x86::has_bmi2;
    use vortex_fastlanes::bit_transpose::x86::has_vbmi;
    use vortex_fastlanes::bit_transpose::x86::transpose_bits_bmi2;
    use vortex_fastlanes::bit_transpose::x86::transpose_bits_vbmi;
    use vortex_fastlanes::bit_transpose::x86::untranspose_bits_bmi2;
    use vortex_fastlanes::bit_transpose::x86::untranspose_bits_vbmi;

    use super::BATCH_SIZE;
    use super::generate_test_data;

    // --- Transpose: single array ---

    #[divan::bench(
        name = crate::variant!("transpose_bmi2"),
        ignore = crate::ignore_unless_variant!(simulation, x86_64),
    )]
    fn transpose_bmi2(bencher: Bencher) {
        if !has_bmi2() {
            return;
        }

        let input = generate_test_data(42);

        bencher
            .with_inputs(|| (&input, [0u8; 128]))
            .bench_refs(|(input, output)| {
                unsafe { transpose_bits_bmi2(input, output) };
            });
    }

    #[divan::bench(
        name = crate::variant!("transpose_vbmi"),
        ignore = crate::ignore_unless_variant!(simulation, x86_64),
    )]
    fn transpose_vbmi(bencher: Bencher) {
        if !has_vbmi() {
            return;
        }

        let input = generate_test_data(42);

        bencher
            .with_inputs(|| (&input, [0u8; 128]))
            .bench_refs(|(input, output)| {
                unsafe { transpose_bits_vbmi(input, output) };
            });
    }

    // --- Untranspose: single array ---

    #[divan::bench(
        name = crate::variant!("untranspose_bmi2"),
        ignore = crate::ignore_unless_variant!(simulation, x86_64),
    )]
    fn untranspose_bmi2(bencher: Bencher) {
        if !has_bmi2() {
            return;
        }

        let input = generate_test_data(42);

        bencher
            .with_inputs(|| (&input, [0u8; 128]))
            .bench_refs(|(input, output)| {
                unsafe { untranspose_bits_bmi2(input, output) };
            });
    }

    #[divan::bench(
        name = crate::variant!("untranspose_vbmi"),
        ignore = crate::ignore_unless_variant!(simulation, x86_64),
    )]
    fn untranspose_vbmi(bencher: Bencher) {
        if !has_vbmi() {
            return;
        }

        let input = generate_test_data(42);

        bencher
            .with_inputs(|| (&input, [0u8; 128]))
            .bench_refs(|(input, output)| {
                unsafe { untranspose_bits_vbmi(input, output) };
            });
    }

    // --- Transpose: throughput (1000 arrays) ---

    #[divan::bench(
        name = crate::variant!("transpose_bmi2_throughput"),
        ignore = crate::ignore_unless_variant!(simulation, x86_64),
    )]
    fn transpose_bmi2_throughput(bencher: Bencher) {
        if !has_bmi2() {
            return;
        }

        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE).map(generate_test_data).collect();

        bencher
            .with_inputs(|| (&inputs, vec![[0u8; 128]; BATCH_SIZE]))
            .bench_refs(|(inputs, outputs)| {
                for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                    unsafe { transpose_bits_bmi2(input, output) };
                }
            });
    }

    #[divan::bench(
        name = crate::variant!("transpose_vbmi_throughput"),
        ignore = crate::ignore_unless_variant!(simulation, x86_64),
    )]
    fn transpose_vbmi_throughput(bencher: Bencher) {
        if !has_vbmi() {
            return;
        }

        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE).map(generate_test_data).collect();

        bencher
            .with_inputs(|| (&inputs, vec![[0u8; 128]; BATCH_SIZE]))
            .bench_refs(|(inputs, outputs)| {
                for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                    unsafe { transpose_bits_vbmi(input, output) };
                }
            });
    }

    // --- Untranspose: throughput (1000 arrays) ---

    #[divan::bench(
        name = crate::variant!("untranspose_bmi2_throughput"),
        ignore = crate::ignore_unless_variant!(simulation, x86_64),
    )]
    fn untranspose_bmi2_throughput(bencher: Bencher) {
        if !has_bmi2() {
            return;
        }

        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE).map(generate_test_data).collect();

        bencher
            .with_inputs(|| (&inputs, vec![[0u8; 128]; BATCH_SIZE]))
            .bench_refs(|(inputs, outputs)| {
                for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                    unsafe { untranspose_bits_bmi2(input, output) };
                }
            });
    }

    #[divan::bench(
        name = crate::variant!("untranspose_vbmi_throughput"),
        ignore = crate::ignore_unless_variant!(simulation, x86_64),
    )]
    fn untranspose_vbmi_throughput(bencher: Bencher) {
        if !has_vbmi() {
            return;
        }

        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE).map(generate_test_data).collect();

        bencher
            .with_inputs(|| (&inputs, vec![[0u8; 128]; BATCH_SIZE]))
            .bench_refs(|(inputs, outputs)| {
                for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                    unsafe { untranspose_bits_vbmi(input, output) };
                }
            });
    }
}

// ============================================================================
// aarch64 benchmarks
//
// NEON has its own walltime leg; the scalar baselines above also run there.
// ============================================================================

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use vortex_fastlanes::bit_transpose::aarch64::transpose_bits_neon;
    use vortex_fastlanes::bit_transpose::aarch64::untranspose_bits_neon;

    use super::BATCH_SIZE;
    use super::Bencher;
    use super::generate_test_data;

    // --- Transpose: single array ---

    #[divan::bench(
        name = crate::variant!("transpose_neon"),
        ignore = crate::ignore_unless_variant!(aarch64),
    )]
    fn transpose_neon(bencher: Bencher) {
        let input = generate_test_data(42);

        bencher
            .with_inputs(|| (&input, [0u8; 128]))
            .bench_refs(|(input, output)| {
                unsafe { transpose_bits_neon(input, output) };
            });
    }

    // --- Untranspose: single array ---

    #[divan::bench(
        name = crate::variant!("untranspose_neon"),
        ignore = crate::ignore_unless_variant!(aarch64),
    )]
    fn untranspose_neon(bencher: Bencher) {
        let input = generate_test_data(42);

        bencher
            .with_inputs(|| (&input, [0u8; 128]))
            .bench_refs(|(input, output)| {
                unsafe { untranspose_bits_neon(input, output) };
            });
    }

    // --- Transpose: throughput (1000 arrays) ---

    #[divan::bench(
        name = crate::variant!("transpose_neon_throughput"),
        ignore = crate::ignore_unless_variant!(aarch64),
    )]
    fn transpose_neon_throughput(bencher: Bencher) {
        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE).map(generate_test_data).collect();

        bencher
            .with_inputs(|| (&inputs, vec![[0u8; 128]; BATCH_SIZE]))
            .bench_refs(|(inputs, outputs)| {
                for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                    unsafe { transpose_bits_neon(input, output) };
                }
            });
    }

    // --- Untranspose: throughput (1000 arrays) ---

    #[divan::bench(
        name = crate::variant!("untranspose_neon_throughput"),
        ignore = crate::ignore_unless_variant!(aarch64),
    )]
    fn untranspose_neon_throughput(bencher: Bencher) {
        let inputs: Vec<[u8; 128]> = (0..BATCH_SIZE).map(generate_test_data).collect();

        bencher
            .with_inputs(|| (&inputs, vec![[0u8; 128]; BATCH_SIZE]))
            .bench_refs(|(inputs, outputs)| {
                for (input, output) in inputs.iter().zip(outputs.iter_mut()) {
                    unsafe { untranspose_bits_neon(input, output) };
                }
            });
    }
}
