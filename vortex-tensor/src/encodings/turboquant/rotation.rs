// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Deterministic random rotation for TurboQuant.
//!
//! Uses a Structured Random Hadamard Transform (SRHT) for O(d log d) rotation
//! instead of a full d×d matrix multiply. The SRHT applies the sequence
//! D₃ · H · D₂ · H · D₁ where H is the Walsh-Hadamard Transform (WHT) and Dₖ are
//! random diagonal ±1 sign matrices. Three rounds of HD provide sufficient
//! randomness for near-uniform distribution on the sphere.
//!
//! For dimensions that are not powers of 2, the input is zero-padded to the
//! next power of 2 before the transform and truncated afterward.
//!
//! # Sign representation
//!
//! Signs are stored internally as `u32` XOR masks: `0x00000000` for +1 (no-op)
//! and `0x80000000` for -1 (flip IEEE 754 sign bit). The sign application
//! function uses integer XOR instead of floating-point multiply, which avoids
//! FP dependency chains and auto-vectorizes into `vpxor`/`veor`.

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

/// IEEE 754 sign bit mask for f32.
const F32_SIGN_BIT: u32 = 0x8000_0000;

/// A structured random Hadamard transform for O(d log d) pseudo-random rotation.
pub struct RotationMatrix {
    /// XOR masks for each of the 3 diagonal matrices, each of length `padded_dim`.
    /// `0x00000000` = multiply by +1 (no-op), `0x80000000` = multiply by -1 (flip sign bit).
    sign_masks: [Vec<u32>; 3],
    /// The padded dimension (next power of 2 >= dimension).
    padded_dim: usize,
    /// Normalization factor: 1/(padded_dim * sqrt(padded_dim)), applied once at the end.
    norm_factor: f32,
}

impl RotationMatrix {
    /// Create a new SRHT rotation from a deterministic seed.
    pub fn try_new(seed: u64, dimension: usize) -> VortexResult<Self> {
        let padded_dim = dimension.next_power_of_two();
        let mut rng = StdRng::seed_from_u64(seed);

        let sign_masks = std::array::from_fn(|_| gen_random_sign_masks(&mut rng, padded_dim));
        let norm_factor = 1.0 / (padded_dim as f32 * (padded_dim as f32).sqrt());

        Ok(Self {
            sign_masks,
            padded_dim,
            norm_factor,
        })
    }

    /// Apply forward rotation: `output = SRHT(input)`.
    ///
    /// Both `input` and `output` must have length `padded_dim()`. The caller
    /// is responsible for zero-padding input beyond `dim` positions.
    pub fn rotate(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.padded_dim);
        debug_assert_eq!(output.len(), self.padded_dim);

        output.copy_from_slice(input);
        self.apply_srht(output);
    }

    /// Apply inverse rotation: `output = SRHT⁻¹(input)`.
    ///
    /// Both `input` and `output` must have length `padded_dim()`.
    pub fn inverse_rotate(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.padded_dim);
        debug_assert_eq!(output.len(), self.padded_dim);

        output.copy_from_slice(input);
        self.apply_inverse_srht(output);
    }

    /// Returns the padded dimension (next power of 2 >= dim).
    ///
    /// All rotate/inverse_rotate buffers must be this length.
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Apply the SRHT: D₃ · H · D₂ · H · D₁ · x, with normalization.
    fn apply_srht(&self, buf: &mut [f32]) {
        apply_signs_xor(buf, &self.sign_masks[0]);
        walsh_hadamard_transform(buf);

        apply_signs_xor(buf, &self.sign_masks[1]);
        walsh_hadamard_transform(buf);

        apply_signs_xor(buf, &self.sign_masks[2]);
        walsh_hadamard_transform(buf);

        let norm = self.norm_factor;
        buf.iter_mut().for_each(|val| *val *= norm);
    }

    /// Apply the inverse SRHT.
    ///
    /// Forward is: norm · H · D₃ · H · D₂ · H · D₁
    /// Inverse is: norm · D₁ · H · D₂ · H · D₃ · H
    fn apply_inverse_srht(&self, buf: &mut [f32]) {
        walsh_hadamard_transform(buf);
        apply_signs_xor(buf, &self.sign_masks[2]);

        walsh_hadamard_transform(buf);
        apply_signs_xor(buf, &self.sign_masks[1]);

        walsh_hadamard_transform(buf);
        apply_signs_xor(buf, &self.sign_masks[0]);

        let norm = self.norm_factor;
        buf.iter_mut().for_each(|val| *val *= norm);
    }

    /// Export the 3 sign vectors as a flat `Vec<u8>` of 0/1 values in inverse
    /// application order `[D₃ | D₂ | D₁]`.
    ///
    /// Convention: `1` = positive (+1), `0` = negative (-1).
    /// The output has length `3 * padded_dim` and is suitable for bitpacking
    /// via FastLanes `bitpack_encode(..., 1, None)`.
    pub fn export_inverse_signs_u8(&self) -> Vec<u8> {
        let total = 3 * self.padded_dim;
        let mut out = Vec::with_capacity(total);

        // Store in inverse order: sign_masks[2] (D₃), sign_masks[1] (D₂), sign_masks[0] (D₁)
        for sign_idx in [2, 1, 0] {
            for &mask in &self.sign_masks[sign_idx] {
                out.push(if mask == 0 { 1u8 } else { 0u8 });
            }
        }
        out
    }

    /// Reconstruct a `RotationMatrix` from unpacked `u8` 0/1 values.
    ///
    /// The input must have length `3 * padded_dim` with signs in inverse
    /// application order `[D₃ | D₂ | D₁]` (as produced by [`export_inverse_signs_u8`]).
    /// Convention: `1` = positive, `0` = negative.
    ///
    /// This is the decode-time reconstruction path: FastLanes SIMD-unpacks the
    /// stored `BitPackedArray` into `&[u8]`, which is passed here.
    pub fn from_u8_slice(signs_u8: &[u8], dimension: usize) -> VortexResult<Self> {
        let padded_dim = dimension.next_power_of_two();
        vortex_ensure!(
            signs_u8.len() == 3 * padded_dim,
            "Expected {} sign bytes, got {}",
            3 * padded_dim,
            signs_u8.len()
        );

        // Reconstruct in storage order (inverse): [D₃, D₂, D₁] → sign_masks[2], [1], [0]
        let mut sign_masks: [Vec<u32>; 3] = std::array::from_fn(|_| Vec::with_capacity(padded_dim));

        for (round, sign_idx) in [2, 1, 0].into_iter().enumerate() {
            let offset = round * padded_dim;
            sign_masks[sign_idx] = signs_u8[offset..offset + padded_dim]
                .iter()
                .map(|&v| if v != 0 { 0u32 } else { F32_SIGN_BIT })
                .collect();
        }

        let norm_factor = 1.0 / (padded_dim as f32 * (padded_dim as f32).sqrt());

        Ok(Self {
            sign_masks,
            padded_dim,
            norm_factor,
        })
    }
}

/// Generate a vector of random XOR sign masks.
fn gen_random_sign_masks(rng: &mut StdRng, len: usize) -> Vec<u32> {
    (0..len)
        .map(|_| {
            if rng.random_bool(0.5) {
                0u32 // +1: no-op
            } else {
                F32_SIGN_BIT // -1: flip sign bit
            }
        })
        .collect()
}

/// Apply sign masks via XOR on the IEEE 754 sign bit.
///
/// This is branchless and auto-vectorizes into `vpxor` (x86) / `veor` (ARM).
/// Equivalent to multiplying each element by ±1.0, but avoids FP dependency chains.
#[inline]
fn apply_signs_xor(buf: &mut [f32], masks: &[u32]) {
    for (val, &mask) in buf.iter_mut().zip(masks.iter()) {
        *val = f32::from_bits(val.to_bits() ^ mask);
    }
}

/// In-place Walsh-Hadamard Transform (unnormalized, iterative).
///
/// Input length must be a power of 2. Runs in O(n log n).
///
/// Uses a fixed-size chunk strategy: for each stage, the buffer is processed
/// in `CHUNK`-element blocks with a compile-time-known butterfly function.
/// This lets LLVM unroll and auto-vectorize the butterfly into NEON/AVX SIMD.
fn walsh_hadamard_transform(buf: &mut [f32]) {
    let len = buf.len();
    debug_assert!(len.is_power_of_two());

    let mut half = 1;
    while half < len {
        let stride = half * 2;
        // Process in chunks of `stride` elements. Within each chunk,
        // split into non-overlapping (lo, hi) halves for the butterfly.
        for chunk in buf.chunks_exact_mut(stride) {
            let (lo, hi) = chunk.split_at_mut(half);
            butterfly(lo, hi);
        }
        half *= 2;
    }
}

/// Butterfly: `lo[i], hi[i] = lo[i] + hi[i], lo[i] - hi[i]`.
///
/// Separate function so LLVM can see the slice lengths match and auto-vectorize.
#[inline(always)]
fn butterfly(lo: &mut [f32], hi: &mut [f32]) {
    debug_assert_eq!(lo.len(), hi.len());
    for (a, b) in lo.iter_mut().zip(hi.iter_mut()) {
        let sum = *a + *b;
        let diff = *a - *b;
        *a = sum;
        *b = diff;
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;

    use super::*;

    #[test]
    fn deterministic_from_seed() -> VortexResult<()> {
        let r1 = RotationMatrix::try_new(42, 64)?;
        let r2 = RotationMatrix::try_new(42, 64)?;
        let pd = r1.padded_dim();

        let mut input = vec![0.0f32; pd];
        for i in 0..64 {
            input[i] = i as f32;
        }
        let mut out1 = vec![0.0f32; pd];
        let mut out2 = vec![0.0f32; pd];

        r1.rotate(&input, &mut out1);
        r2.rotate(&input, &mut out2);

        assert_eq!(out1, out2);
        Ok(())
    }

    /// Verify roundtrip is exact to f32 precision across many dimensions,
    /// including non-power-of-two dimensions that require padding.
    #[rstest]
    #[case(32)]
    #[case(64)]
    #[case(100)]
    #[case(128)]
    #[case(256)]
    #[case(512)]
    #[case(768)]
    #[case(1024)]
    fn roundtrip_exact(#[case] dim: usize) -> VortexResult<()> {
        let rot = RotationMatrix::try_new(42, dim)?;
        let padded_dim = rot.padded_dim();

        let mut input = vec![0.0f32; padded_dim];
        for i in 0..dim {
            input[i] = (i as f32 + 1.0) * 0.01;
        }
        let mut rotated = vec![0.0f32; padded_dim];
        let mut recovered = vec![0.0f32; padded_dim];

        rot.rotate(&input, &mut rotated);
        rot.inverse_rotate(&rotated, &mut recovered);

        let max_err: f32 = input
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_val: f32 = input.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let rel_err = max_err / max_val;

        // SRHT roundtrip should be exact up to f32 precision (~1e-6).
        assert!(
            rel_err < 1e-5,
            "roundtrip relative error too large for dim={dim}: {rel_err:.2e}"
        );
        Ok(())
    }

    /// Verify norm preservation across dimensions.
    #[rstest]
    #[case(128)]
    #[case(768)]
    fn preserves_norm(#[case] dim: usize) -> VortexResult<()> {
        let rot = RotationMatrix::try_new(7, dim)?;
        let padded_dim = rot.padded_dim();

        let mut input = vec![0.0f32; padded_dim];
        for i in 0..dim {
            input[i] = (i as f32) * 0.01;
        }
        let input_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();

        let mut rotated = vec![0.0f32; padded_dim];
        rot.rotate(&input, &mut rotated);
        let rotated_norm: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            (input_norm - rotated_norm).abs() / input_norm < 1e-5,
            "norm not preserved for dim={dim}: {} vs {} (rel err: {:.2e})",
            input_norm,
            rotated_norm,
            (input_norm - rotated_norm).abs() / input_norm
        );
        Ok(())
    }

    /// Verify that export → from_u8_slice produces identical rotation output.
    #[rstest]
    #[case(64)]
    #[case(128)]
    #[case(768)]
    fn sign_export_import_roundtrip(#[case] dim: usize) -> VortexResult<()> {
        let rot = RotationMatrix::try_new(42, dim)?;
        let padded_dim = rot.padded_dim();

        let signs_u8 = rot.export_inverse_signs_u8();
        let rot2 = RotationMatrix::from_u8_slice(&signs_u8, dim)?;

        let mut input = vec![0.0f32; padded_dim];
        for i in 0..dim {
            input[i] = (i as f32 + 1.0) * 0.01;
        }

        let mut out1 = vec![0.0f32; padded_dim];
        let mut out2 = vec![0.0f32; padded_dim];
        rot.rotate(&input, &mut out1);
        rot2.rotate(&input, &mut out2);
        assert_eq!(out1, out2, "Forward rotation mismatch after export/import");

        rot.inverse_rotate(&out1, &mut out2);
        let mut out3 = vec![0.0f32; padded_dim];
        rot2.inverse_rotate(&out1, &mut out3);
        assert_eq!(out2, out3, "Inverse rotation mismatch after export/import");

        Ok(())
    }

    #[test]
    fn wht_basic() {
        // WHT of [1, 0, 0, 0] should be [1, 1, 1, 1]
        let mut buf = vec![1.0f32, 0.0, 0.0, 0.0];
        walsh_hadamard_transform(&mut buf);
        assert_eq!(buf, vec![1.0, 1.0, 1.0, 1.0]);

        // WHT is self-inverse (up to scaling by n)
        walsh_hadamard_transform(&mut buf);
        // After two WHTs: each element multiplied by n=4
        assert_eq!(buf, vec![4.0, 0.0, 0.0, 0.0]);
    }
}
