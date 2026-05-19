// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cross-check the JIT, composed, and hand-fused paths agree for many
//! `(bit_width, threshold)` combinations.

use vortex_jit_experiment::CHUNK_SIZE;
use vortex_jit_experiment::MASK_WORDS;
use vortex_jit_experiment::composed;
use vortex_jit_experiment::fused;
use vortex_jit_experiment::jit;
use vortex_jit_experiment::pack::pack_chunk;

fn run_one(bit_width: u32, threshold_fraction: f64) {
    let max = (1u32 << bit_width) - 1;
    let threshold = (max as f64 * threshold_fraction) as u32;

    let mut values = [0u32; CHUNK_SIZE];
    for (i, v) in values.iter_mut().enumerate() {
        // Three different patterns mixed to exercise both halves of the threshold.
        *v = ((i as u32).wrapping_mul(0x9E37_79B1) ^ (i as u32 >> 1)) & max;
    }
    let mut packed = pack_chunk(&values, bit_width);
    packed.extend(std::iter::repeat_n(0u32, jit::REQUIRED_PAD_WORDS));

    let mut mask_composed = [0u64; MASK_WORDS];
    composed::unpack_then_compare(&packed, bit_width, threshold, &mut mask_composed);

    let mut mask_fused = [0u64; MASK_WORDS];
    fused::unpack_compare_fused(&packed, bit_width, threshold, &mut mask_fused);

    let kernel = jit::compile(bit_width).unwrap();
    let mut mask_jit = [0u64; MASK_WORDS];
    unsafe { kernel.run(&packed, threshold, &mut mask_jit) };

    // Spot-check against the values directly so we're not just comparing two
    // implementations that happen to share a bug.
    let mut expected = [0u64; MASK_WORDS];
    for (i, &v) in values.iter().enumerate() {
        if v > threshold {
            expected[i / 64] |= 1u64 << (i % 64);
        }
    }

    assert_eq!(
        mask_composed, expected,
        "composed wrong @ bit_width={bit_width} k={threshold}",
    );
    assert_eq!(
        mask_fused, expected,
        "fused wrong @ bit_width={bit_width} k={threshold}",
    );
    assert_eq!(
        mask_jit, expected,
        "JIT wrong @ bit_width={bit_width} k={threshold}",
    );
}

#[test]
fn all_widths_all_thresholds() {
    for bit_width in [1u32, 2, 3, 5, 7, 8, 11, 13, 16, 17, 19, 24, 31] {
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0] {
            run_one(bit_width, frac);
        }
    }
}

#[test]
fn jit_handles_word_boundary_straddling() {
    // bit_width = 7 means values cross 32-bit word boundaries frequently
    // (every ~4-5 values), exercising the unaligned 8-byte load path heavily.
    run_one(7, 0.5);
}

#[test]
fn jit_threshold_above_max_yields_empty_mask() {
    let bit_width = 8u32;
    let values = [42u32; CHUNK_SIZE]; // every value is 42
    let mut packed = pack_chunk(&values, bit_width);
    packed.extend(std::iter::repeat_n(0u32, jit::REQUIRED_PAD_WORDS));

    let kernel = jit::compile(bit_width).unwrap();
    let mut mask = [0u64; MASK_WORDS];
    unsafe { kernel.run(&packed, 200, &mut mask) };
    assert!(
        mask.iter().all(|&w| w == 0),
        "no value should exceed 200, got mask = {:?}",
        mask,
    );
}

#[test]
fn jit_threshold_zero_selects_all_nonzero() {
    let bit_width = 8u32;
    let mut values = [0u32; CHUNK_SIZE];
    for (i, v) in values.iter_mut().enumerate() {
        *v = (i as u32 + 1) & 0xFF; // 1..=255, 0, 1, 2, ... (one zero per 256)
    }
    let mut packed = pack_chunk(&values, bit_width);
    packed.extend(std::iter::repeat_n(0u32, jit::REQUIRED_PAD_WORDS));

    let kernel = jit::compile(bit_width).unwrap();
    let mut mask = [0u64; MASK_WORDS];
    unsafe { kernel.run(&packed, 0, &mut mask) };

    let popcount: u32 = mask.iter().map(|w| w.count_ones()).sum();
    let expected_zeros = values.iter().filter(|&&v| v == 0).count() as u32;
    assert_eq!(popcount, CHUNK_SIZE as u32 - expected_zeros);
}
