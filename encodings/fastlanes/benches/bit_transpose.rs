// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use divan::Bencher;
use fastlanes::transpose_bits;
use fastlanes::untranspose_bits;

fn main() {
    divan::main();
}

fn generate_test_data(seed: usize) -> [u64; 16] {
    std::array::from_fn(|idx| seed.wrapping_mul(17).wrapping_add(idx).wrapping_mul(31) as u64)
}

const BATCH_SIZE: usize = 1000;

#[divan::bench]
fn transpose(bencher: Bencher) {
    let input = generate_test_data(42);
    bencher
        .with_inputs(|| (&input, [0u64; 16]))
        .bench_refs(|(input, output)| transpose_bits(input, output));
}

#[divan::bench]
fn transpose_throughput(bencher: Bencher) {
    let inputs: Vec<[u64; 16]> = (0..BATCH_SIZE).map(generate_test_data).collect();
    bencher
        .with_inputs(|| (&inputs, vec![[0u64; 16]; BATCH_SIZE]))
        .bench_refs(|(inputs, outputs)| {
            for (input, output) in inputs.iter().zip(outputs) {
                transpose_bits(input, output);
            }
        });
}

#[divan::bench]
fn untranspose(bencher: Bencher) {
    let input = generate_test_data(42);
    bencher
        .with_inputs(|| (&input, [0u64; 16]))
        .bench_refs(|(input, output)| untranspose_bits::<u64>(input, output));
}

#[divan::bench]
fn untranspose_throughput(bencher: Bencher) {
    let inputs: Vec<[u64; 16]> = (0..BATCH_SIZE).map(generate_test_data).collect();
    bencher
        .with_inputs(|| (&inputs, vec![[0u64; 16]; BATCH_SIZE]))
        .bench_refs(|(inputs, outputs)| {
            for (input, output) in inputs.iter().zip(outputs) {
                untranspose_bits::<u64>(input, output);
            }
        });
}
