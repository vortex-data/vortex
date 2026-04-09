// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Deterministic random rotation for TurboQuant.
//!
//! The TurboQuant paper analyzes a full random orthogonal rotation. The current implementation
//! uses a cheaper structured Walsh-Hadamard-based surrogate instead of a dense d x d matrix.
//!
//! Concretely, this applies three rounds of random sign diagonals interleaved with the Fast
//! Walsh-Hadamard Transform (FWHT): `D3 * H * D2 * H * D1 * H`, followed by normalization. This is
//! a SORF-style structured approximation to a random orthogonal matrix, chosen for O(d log d)
//! encode/decode cost and compact serialized parameters.
//!
//! The FWHT exploits the Kronecker product structure of the Hadamard matrix (`H_n = H_2 (x) H_2
//! (x) ... (x) H_2`, with `log2(n)` factors) to compute the matrix-vector product in O(n log n)
//! time using only in-place 2-element butterfly operations. No row of the full n x n Hadamard
//! matrix is ever materialized.
//!
//! For dimensions that are not powers of 2, the input is zero-padded to the next power of 2 before
//! the transform and truncated afterward.
//!
//! # Sign representation
//!
//! Signs are stored internally as `u32` XOR masks: `0x00000000` for +1 (no-op) and `0x80000000` for
//! -1 (flip IEEE 754 sign bit). The sign application function uses integer XOR instead of
//! floating-point multiply, which avoids FP dependency chains and auto-vectorizes into
//! `vpxor`/`veor`.

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

/// IEEE 754 sign bit mask for f32.
const F32_SIGN_BIT: u32 = 0x8000_0000;

/// A Walsh-Hadamard-based structured surrogate for a random orthogonal rotation.
pub struct RotationMatrix {
    /// Flat XOR masks for all `num_rounds` diagonal matrices, total length `num_rounds * padded_dim`.
    /// Indexed as `round * padded_dim + i`. `0x00000000` = multiply by +1 (no-op), `0x80000000` =
    /// multiply by -1 (flip sign bit).
    sign_masks: Vec<u32>,
    /// The number of sign-diagonal + WHT rounds.
    num_rounds: usize,
    /// The padded dimension (next power of 2 >= dimension).
    padded_dim: usize,
    /// Normalization factor: `padded_dim^(-num_rounds/2)`, applied once at the end.
    norm_factor: f32,
}

impl RotationMatrix {
    /// Create a new structured Walsh-Hadamard-based rotation from a deterministic seed.
    pub fn try_new(seed: u64, dimension: usize, num_rounds: usize) -> VortexResult<Self> {
        vortex_ensure!(num_rounds >= 1, "num_rounds must be >= 1, got {num_rounds}");

        let padded_dim = dimension.next_power_of_two();
        let mut rng = StdRng::seed_from_u64(seed);

        let mut sign_masks = Vec::with_capacity(num_rounds * padded_dim);
        for _ in 0..num_rounds {
            sign_masks.extend(gen_random_sign_masks(&mut rng, padded_dim));
        }

        #[expect(
            clippy::cast_possible_truncation,
            reason = "Intentional f64 -> f32 truncation for normalization factor."
        )]
        let norm_factor = (padded_dim as f64).powf(-(num_rounds as f64) / 2.0) as f32;

        Ok(Self {
            sign_masks,
            num_rounds,
            padded_dim,
            norm_factor,
        })
    }

    /// Apply forward rotation: `output = R(input)`.
    ///
    /// Both `input` and `output` must have length [`padded_dim()`](Self::padded_dim). The caller is
    /// responsible for zero-padding input beyond `dim` positions.
    pub fn rotate(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.padded_dim);
        debug_assert_eq!(output.len(), self.padded_dim);

        output.copy_from_slice(input);
        self.apply_srht(output);
    }

    /// Apply inverse rotation: `output = R⁻¹(input)`.
    ///
    /// Both `input` and `output` must have length `padded_dim()`.
    pub fn inverse_rotate(&self, input: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), self.padded_dim);
        debug_assert_eq!(output.len(), self.padded_dim);

        output.copy_from_slice(input);
        self.apply_inverse_srht(output);
    }

    /// Returns the number of sign-diagonal + WHT rounds.
    pub fn num_rounds(&self) -> usize {
        self.num_rounds
    }

    /// Returns the padded dimension (next power of 2 >= dim).
    ///
    /// All rotate/inverse_rotate buffers must be this length.
    pub fn padded_dim(&self) -> usize {
        self.padded_dim
    }

    /// Apply the structured rotation: `D_k · H · ... · D₁ · H · x`, with normalization.
    fn apply_srht(&self, buf: &mut [f32]) {
        for round in 0..self.num_rounds {
            let offset = round * self.padded_dim;
            apply_signs_xor(buf, &self.sign_masks[offset..offset + self.padded_dim]);
            walsh_hadamard_transform(buf);
        }

        let norm = self.norm_factor;
        buf.iter_mut().for_each(|val| *val *= norm);
    }

    /// Apply the inverse structured rotation.
    ///
    /// Forward is: `norm · H · D_k · H · ... · D₁ · H`.
    /// Inverse is: `norm · D₁ · H · ... · D_k · H`.
    fn apply_inverse_srht(&self, buf: &mut [f32]) {
        for round in (0..self.num_rounds).rev() {
            walsh_hadamard_transform(buf);
            let offset = round * self.padded_dim;
            apply_signs_xor(buf, &self.sign_masks[offset..offset + self.padded_dim]);
        }

        let norm = self.norm_factor;
        buf.iter_mut().for_each(|val| *val *= norm);
    }

    /// Export the sign vectors as a flat `Vec<u8>` of 0/1 values in inverse application order
    /// `[D_k | ... | D₁]`.
    ///
    /// Convention: `1` = positive (+1), `0` = negative (-1). The output has length
    /// `num_rounds * padded_dim` and is suitable for bitpacking via FastLanes
    /// `bitpack_encode(..., 1, None)`.
    pub fn export_inverse_signs_u8(&self) -> Vec<u8> {
        let total = self.num_rounds * self.padded_dim;
        let mut out = Vec::with_capacity(total);

        // Store in inverse order: round k-1 first, then k-2, ..., then 0.
        for round in (0..self.num_rounds).rev() {
            let offset = round * self.padded_dim;
            for &mask in &self.sign_masks[offset..offset + self.padded_dim] {
                out.push(if mask == 0 { 1u8 } else { 0u8 });
            }
        }
        out
    }

    /// Reconstruct a [`RotationMatrix`] from unpacked `u8` 0/1 values.
    ///
    /// The input must have length `num_rounds * padded_dim` with signs in inverse application
    /// order `[D_k | ... | D₁]` (as produced by [`export_inverse_signs_u8`]). Convention:
    /// `1` = positive, `0` = negative.
    ///
    /// This is the decode-time reconstruction path: FastLanes SIMD-unpacks the stored
    /// [`BitPackedArray`] into `&[u8]`, which is passed here.
    pub fn from_u8_slice(
        signs_u8: &[u8],
        dimension: usize,
        num_rounds: usize,
    ) -> VortexResult<Self> {
        vortex_ensure!(num_rounds >= 1, "num_rounds must be >= 1, got {num_rounds}");
        let padded_dim = dimension.next_power_of_two();
        vortex_ensure!(
            signs_u8.len() == num_rounds * padded_dim,
            "Expected {} sign bytes, got {}",
            num_rounds * padded_dim,
            signs_u8.len()
        );

        // The storage is in inverse application order: round k-1 first, then k-2, ..., 0.
        // We reconstruct into forward order (round 0 at the start of the flat vec).
        let mut sign_masks = vec![0u32; num_rounds * padded_dim];
        for storage_idx in 0..num_rounds {
            let round = num_rounds - 1 - storage_idx;
            let src_offset = storage_idx * padded_dim;
            let dst_offset = round * padded_dim;
            for i in 0..padded_dim {
                sign_masks[dst_offset + i] = if signs_u8[src_offset + i] != 0 {
                    0u32
                } else {
                    F32_SIGN_BIT
                };
            }
        }

        #[expect(
            clippy::cast_possible_truncation,
            reason = "Intentional f64 -> f32 truncation for normalization factor."
        )]
        let norm_factor = (padded_dim as f64).powf(-(num_rounds as f64) / 2.0) as f32;

        Ok(Self {
            sign_masks,
            num_rounds,
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
/// This is branchless and auto-vectorizes into `vpxor` (x86) / `veor` (ARM). Equivalent to
/// multiplying each element by +/-1.0, but avoids FP dependency chains.
#[inline]
fn apply_signs_xor(buf: &mut [f32], masks: &[u32]) {
    for (val, &mask) in buf.iter_mut().zip(masks.iter()) {
        *val = f32::from_bits(val.to_bits() ^ mask);
    }
}

/// In-place Fast Walsh-Hadamard Transform (FWHT), unnormalized and iterative.
///
/// Input length must be a power of 2. Runs in O(n log n) via `log2(n)` stages of `n / 2`
/// [`butterfly`] operations each. See the [module-level docs](self) for why this avoids
/// materializing the full Hadamard matrix.
///
/// The chunk-based iteration gives LLVM enough structure to auto-vectorize each butterfly call
/// into NEON/AVX SIMD instructions.
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

/// Butterfly: `(lo[i], hi[i]) -> (lo[i] + hi[i], lo[i] - hi[i])`.
///
/// This is multiplication by the 2x2 Hadamard kernel `H_2 = [[1, 1], [1, -1]]` on each element
/// pair. Factored into a separate function so LLVM can see the slice lengths match and
/// auto-vectorize.
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
        let r1 = RotationMatrix::try_new(42, 64, 3)?;
        let r2 = RotationMatrix::try_new(42, 64, 3)?;
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

    /// Verify roundtrip is exact to f32 precision across many dimensions and round counts,
    /// including non-power-of-two dimensions that require padding.
    #[rstest]
    #[case(32, 3)]
    #[case(64, 3)]
    #[case(100, 3)]
    #[case(128, 1)]
    #[case(128, 2)]
    #[case(128, 3)]
    #[case(128, 5)]
    #[case(256, 3)]
    #[case(512, 3)]
    #[case(768, 3)]
    #[case(1024, 3)]
    fn roundtrip_exact(#[case] dim: usize, #[case] num_rounds: usize) -> VortexResult<()> {
        let rot = RotationMatrix::try_new(42, dim, num_rounds)?;
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
            "roundtrip relative error too large for dim={dim}, rounds={num_rounds}: {rel_err:.2e}"
        );
        Ok(())
    }

    /// Verify norm preservation across dimensions and round counts.
    #[rstest]
    #[case(128, 1)]
    #[case(128, 3)]
    #[case(128, 5)]
    #[case(768, 3)]
    fn preserves_norm(#[case] dim: usize, #[case] num_rounds: usize) -> VortexResult<()> {
        let rot = RotationMatrix::try_new(7, dim, num_rounds)?;
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

    /// Verify that export -> [`from_u8_slice`] produces identical rotation output.
    #[rstest]
    #[case(64, 3)]
    #[case(128, 1)]
    #[case(128, 3)]
    #[case(128, 5)]
    #[case(768, 3)]
    fn sign_export_import_roundtrip(
        #[case] dim: usize,
        #[case] num_rounds: usize,
    ) -> VortexResult<()> {
        let rot = RotationMatrix::try_new(42, dim, num_rounds)?;
        let padded_dim = rot.padded_dim();

        let signs_u8 = rot.export_inverse_signs_u8();
        let rot2 = RotationMatrix::from_u8_slice(&signs_u8, dim, num_rounds)?;

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
