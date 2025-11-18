// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Hand-rolled BP -> FoR -> ALP decode pipeline
//! Divan benchmark comparing decompress and decompress_inlined implementations

#![allow(
    clippy::unwrap_used,
    clippy::uninit_vec,
    clippy::cast_possible_truncation
)]

use std::time::Duration;

use divan::Bencher;
use fastlanes::BitPacking;
use rand::Rng;
use vortex_alp::{ALPFloat, Exponents};

/// Size of each chunk.
const N: usize = 1024;

/// The width of each bitpacked value.
const W: usize = 10;

/// The width of the unpacked i32 values.
const T: usize = 32;

/// The bitpacked stride that makes up 1024 bits.
const S: usize = N * W / T;

fn main() {
    divan::main();
}

struct SetupData {
    bitpacked: Vec<u32>,
    reference: i32,
    exponents: Exponents,
    for_decoded: Vec<i32>,
    alp_decoded: Vec<f32>,
}

fn setup(size: usize) -> SetupData {
    let original = create_random_values(size);
    let (alp_encoded, exponents, _patches) = alp_compress(&original);
    let (for_encoded, reference) = for_compress(&alp_encoded);
    let bitpacked = bitpack_10(cast_i32_as_u32(&for_encoded));

    let for_decoded: Vec<i32> = vec![0i32; size];
    let alp_decoded: Vec<f32> = vec![0.0f32; size];

    SetupData {
        bitpacked,
        reference,
        exponents,
        for_decoded,
        alp_decoded,
    }
}

fn decompress_batch(
    bitpacked: &[u32],
    reference: i32,
    exponents: Exponents,
    bitpacked_output: &mut [u32],
    for_decoded: &mut [i32],
    alp_decoded: &mut [f32],
) {
    unpack_10(bitpacked, bitpacked_output);
    for_decompress(cast_u32_as_i32(bitpacked_output), reference, for_decoded);
    alp_decompress(for_decoded, exponents, alp_decoded);
}

/// Inlined version of decompress that processes data chunk by chunk without intermediate
/// allocations.
fn decompress_inlined(bitpacked: &[u32], reference: i32, exponents: Exponents, output: &mut [f32]) {
    debug_assert!(bitpacked.len().is_multiple_of(S));
    debug_assert_eq!(output.len(), bitpacked.len() * T / W);

    // let num_chunks = bitpacked.len() / S;

    // Stack-allocated buffer for one chunk of unpacked values.
    let mut chunk_buffer: [u32; N] = [0; N];

    let mut input_offset = 0;
    let mut output_offset = 0;

    // Process each 1024-element chunk.
    while input_offset < bitpacked.len() {
        // SAFETY: We've verified:
        // - bitpacked.len() is a multiple of S
        // - input_offset + S <= bitpacked.len() (by loop bounds)
        // - chunk_buffer has N elements
        unsafe {
            let input = bitpacked.get_unchecked(input_offset..input_offset + S);
            BitPacking::unchecked_unpack(W, input, &mut chunk_buffer);
        }

        // Stages 2 & 3: Apply FoR and ALP decompression in a single pass.
        // SAFETY: We've verified output.len() matches bitpacked data
        unsafe {
            let output_chunk = output.get_unchecked_mut(output_offset..output_offset + N);

            let mut i = 0;
            while i < N {
                // SAFETY: i < N and chunk_buffer has N elements
                let unpacked = *chunk_buffer.get_unchecked(i) as i32;

                // Apply FoR decompression (add reference).
                let for_decoded = unpacked.wrapping_add(reference);

                // Apply ALP decompression (convert to float with exponent scaling).
                // SAFETY: i < N and output_chunk.len() == N
                *output_chunk.get_unchecked_mut(i) = f32::decode_single(for_decoded, exponents);

                i += 1;
            }
        }

        input_offset += S;
        output_offset += N;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Benchmarks
////////////////////////////////////////////////////////////////////////////////////////////////////

#[divan::bench_group(min_time = Duration::from_secs(1))]
mod decompress_benchmarks {
    use super::*;

    /// Benchmark sizes to test: 1K, 16K, 64K, 256K, 1M, 4M.
    const BENCHMARK_SIZES: [usize; 6] = [
        1024,    // 1K
        16384,   // 16K
        65536,   // 64K
        262144,  // 256K
        1048576, // 1M
        4194304, // 4M
    ];

    #[divan::bench(consts = BENCHMARK_SIZES)]
    fn decompress_original<const SIZE: usize>(bencher: Bencher) {
        let mut data = setup(SIZE);
        let mut bitpacked_output = vec![0u32; SIZE];

        bencher.bench_local(|| {
            decompress_batch(
                &data.bitpacked,
                data.reference,
                data.exponents,
                &mut bitpacked_output,
                &mut data.for_decoded,
                &mut data.alp_decoded,
            );
        });
    }

    #[divan::bench(consts = BENCHMARK_SIZES)]
    fn decompress_pipeline<const SIZE: usize>(bencher: Bencher) {
        let data = setup(SIZE);
        let mut output = vec![0f32; SIZE];

        bencher.bench_local(|| {
            decompress_inlined(&data.bitpacked, data.reference, data.exponents, &mut output);
        });
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Bitpacking
////////////////////////////////////////////////////////////////////////////////////////////////////

fn bitpack_10(values: &[u32]) -> Vec<u32> {
    let len = values.len();
    debug_assert!(len.is_multiple_of(N));

    let mut bitpacked = Vec::with_capacity(len * W / T);
    // SAFETY: We're setting the length to the exact capacity we just allocated
    unsafe { bitpacked.set_len(len * W / T) };

    let mut input_offset = 0;
    let mut output_offset = 0;

    while input_offset < len {
        // SAFETY: We've verified that:
        // - len is a multiple of N
        // - bitpacked.len() == len * W / T
        // - input_offset < len ensures we have at least N elements left
        unsafe {
            let input = values.get_unchecked(input_offset..input_offset + N);
            let output = bitpacked.get_unchecked_mut(output_offset..output_offset + S);
            BitPacking::unchecked_pack(W, input, output);
        }

        input_offset += N;
        output_offset += S;
    }

    bitpacked
}

fn unpack_10(bitpacked: &[u32], unpacked: &mut [u32]) {
    debug_assert!(bitpacked.len().is_multiple_of(S));
    let len = bitpacked.len() * T / W;
    debug_assert_eq!(unpacked.len(), len);

    let mut input_offset = 0;
    let mut output_offset = 0;

    while output_offset < len {
        // SAFETY: We've verified that:
        // - bitpacked.len() is a multiple of S
        // - unpacked.len() == bitpacked.len() * T / W
        // - output_offset < len ensures we have at least N elements left
        unsafe {
            let input = bitpacked.get_unchecked(input_offset..input_offset + S);
            let output = unpacked.get_unchecked_mut(output_offset..output_offset + N);
            BitPacking::unchecked_unpack(W, input, output);
        }

        input_offset += S;
        output_offset += N;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// ALP
////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct Patches {
    indices: Vec<u64>,
    values: Vec<f32>,
}

fn alp_compress(values: &[f32]) -> (Vec<i32>, Exponents, Patches) {
    let (exponents, encoded, patch_indices, patch_values, _) = f32::encode(values, None);

    let indices = patch_indices.into_iter().collect();
    let values = patch_values.into_iter().collect();

    let alp_vec: Vec<i32> = encoded.into_iter().collect();
    (alp_vec, exponents, Patches { indices, values })
}

fn alp_decompress(encoded: &[i32], exponents: Exponents, output: &mut [f32]) {
    f32::decode_into(encoded, exponents, output)
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// FoR
////////////////////////////////////////////////////////////////////////////////////////////////////

fn for_compress(values: &[i32]) -> (Vec<i32>, i32) {
    let min = values.iter().min().copied().unwrap();
    (values.iter().map(|x| x.wrapping_sub(min)).collect(), min)
}

#[inline(never)]
fn for_decompress(for_values: &[i32], reference: i32, output: &mut [i32]) {
    debug_assert_eq!(for_values.len(), output.len());
    let len = for_values.len();

    // SAFETY: We've verified that for_values.len() == output.len()
    unsafe {
        for i in 0..len {
            *output.get_unchecked_mut(i) = for_values.get_unchecked(i).wrapping_add(reference);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

fn create_random_values(len: usize) -> Vec<f32> {
    assert!(len.is_multiple_of(N));

    let mut rng = rand::rng();
    (0..len)
        .map(|_| rng.random_range(0..1024))
        .map(|x| x as f32 / 100.0)
        .collect()
}

fn cast_i32_as_u32(slice: &[i32]) -> &[u32] {
    // SAFETY: i32 and u32 have the same size and alignment, so this transmute is safe.
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u32, slice.len()) }
}

fn cast_u32_as_i32(slice: &[u32]) -> &[i32] {
    // SAFETY: i32 and u32 have the same size and alignment, so this transmute is safe.
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const i32, slice.len()) }
}

#[allow(dead_code)]
fn verify(
    for_decoded: &[i32],
    alp_decoded: &[f32],
    alp_encoded: &[i32],
    original: &[f32],
    patches: &Patches,
) {
    // Verification

    for i in 0..for_decoded.len() {
        assert_eq!(
            for_decoded[i], alp_encoded[i],
            "FoR decode mismatch at index {}: decoded={}, expected={}",
            i, for_decoded[i], alp_encoded[i]
        );
    }

    for i in 0..alp_decoded.len() {
        if let Some(patch_idx) = patches.indices.iter().position(|&idx| idx == i as u64) {
            assert_eq!(
                patches.values[patch_idx], original[i],
                "Patch value mismatch at index {}: patch={}, expected={}",
                i, patches.values[patch_idx], original[i]
            );
        } else {
            assert_eq!(
                alp_decoded[i], original[i],
                "ALP decode mismatch at index {}: decoded={}, expected={}",
                i, alp_decoded[i], original[i]
            );
        }
    }
}

/// Compare outputs from original and inlined decompress functions.
#[allow(dead_code)]
fn compare_outputs(original: &[f32], inlined: &[f32]) {
    assert_eq!(
        original.len(),
        inlined.len(),
        "Output length mismatch: original={}, inlined={}",
        original.len(),
        inlined.len()
    );

    for i in 0..original.len() {
        assert_eq!(
            original[i], inlined[i],
            "Output mismatch at index {}: original={}, inlined={}",
            i, original[i], inlined[i]
        );
    }
}
