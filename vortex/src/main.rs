// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Hand-rolled BP -> FoR -> ALP decode pipeline

#![allow(
    clippy::unwrap_used,
    clippy::uninit_vec,
    clippy::cast_possible_truncation
)]

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
    let size = 1024 * 1024;
    let mut setup_data = setup(size);

    decompress(
        &setup_data.bitpacked,
        setup_data.reference,
        setup_data.exponents,
        &mut setup_data.for_decoded,
        &mut setup_data.alp_decoded,
    );

    println!("Decompression completed");

    verify(
        &setup_data.for_decoded,
        &setup_data.alp_decoded,
        &setup_data.alp_encoded,
        &setup_data.original,
        &setup_data.patches,
    );
}

struct SetupData {
    original: Vec<f32>,
    bitpacked: Vec<u32>,
    reference: i32,
    exponents: Exponents,
    patches: Patches,
    alp_encoded: Vec<i32>,
    for_decoded: Vec<i32>,
    alp_decoded: Vec<f32>,
}

fn setup(size: usize) -> SetupData {
    let original = create_random_values(size);
    println!("Created random values");
    let (alp_encoded, exponents, patches) = alp_compress(&original);
    println!("ALP compression completed");
    let (for_encoded, reference) = for_compress(&alp_encoded);
    println!("FoR compression completed");
    let bitpacked = bitpack_10(cast_i32_as_u32(&for_encoded));
    println!("Bitpacking completed");

    let for_decoded: Vec<i32> = vec![0i32; size];
    let alp_decoded: Vec<f32> = vec![0.0f32; size];

    SetupData {
        original,
        bitpacked,
        reference,
        exponents,
        patches,
        alp_encoded,
        for_decoded,
        alp_decoded,
    }
}

fn decompress(
    bitpacked: &[u32],
    reference: i32,
    exponents: Exponents,
    for_decoded: &mut [i32],
    alp_decoded: &mut [f32],
) {
    let unpacked = unpack_10(bitpacked);
    for_decompress(cast_u32_as_i32(&unpacked), reference, for_decoded);
    alp_decompress(for_decoded, exponents, alp_decoded);
}

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

////////////////////////////////////////////////////////////////////////////////////////////////////
// Bitpacking
////////////////////////////////////////////////////////////////////////////////////////////////////

fn bitpack_10(values: &[u32]) -> Vec<u32> {
    let len = values.len();

    let mut bitpacked = Vec::with_capacity(len * W / T);
    // SAFETY: TODO
    unsafe { bitpacked.set_len(len * W / T) };

    for chunk in 0..len / N {
        let input_offset = chunk * N;
        let input = &values[input_offset..][..N];

        let output_offset = chunk * S;
        let output = &mut bitpacked[output_offset..][..S];

        // SAFETY: TODO
        unsafe { BitPacking::unchecked_pack(W, input, output) };
    }

    bitpacked
}

fn unpack_10(bitpacked: &[u32]) -> Vec<u32> {
    assert!(bitpacked.len().is_multiple_of(S));
    let len = bitpacked.len() * T / W;

    let mut unpacked = Vec::with_capacity(len);
    // SAFETY: TODO
    unsafe { unpacked.set_len(len) };

    for chunk in 0..len / N {
        let input_offset = chunk * S;
        let input = &bitpacked[input_offset..][..S];

        let output_offset = chunk * N;
        let output = &mut unpacked[output_offset..][..N];

        // SAFETY: TODO
        unsafe { BitPacking::unchecked_unpack(W, input, output) };
    }

    unpacked
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

fn for_decompress(for_values: &[i32], reference: i32, output: &mut [i32]) {
    assert_eq!(for_values.len(), output.len());

    for i in 0..for_values.len() {
        output[i] = for_values[i].wrapping_add(reference);
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
