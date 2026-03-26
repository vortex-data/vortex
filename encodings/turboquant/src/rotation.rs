// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Deterministic random rotation for TurboQuant.
//!
//! Uses a Structured Random Hadamard Transform (SRHT) for O(d log d) rotation
//! instead of a full d×d matrix multiply. The SRHT applies the sequence
//! D₃ · H · D₂ · H · D₁ where H is the Walsh-Hadamard transform and Dₖ are
//! random diagonal ±1 sign matrices. Three rounds of HD provide sufficient
//! randomness for near-uniform distribution on the sphere.
//!
//! For dimensions that are not powers of 2, the input is zero-padded to the
//! next power of 2 before the transform and truncated afterward.

use rand::SeedableRng;
use rand::rngs::StdRng;
use vortex_array::arrays::BoolArray;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

/// A structured random Hadamard transform for O(d log d) pseudo-random rotation.
pub struct RotationMatrix {
    /// Random ±1 signs for each of the 3 diagonal matrices, each of length `padded_dim`.
    signs: [Vec<f32>; 3],
    /// The original (unpadded) dimension.
    dim: usize,
    /// The padded dimension (next power of 2 >= dim).
    padded_dim: usize,
    /// Normalization factor: 1/padded_dim per Hadamard, applied once at the end.
    norm_factor: f32,
}

impl RotationMatrix {
    /// Create a new SRHT rotation from a deterministic seed.
    pub fn try_new(seed: u64, dimension: usize) -> VortexResult<Self> {
        let padded_dim = dimension.next_power_of_two();
        let mut rng = StdRng::seed_from_u64(seed);

        // Generate 3 random sign vectors (±1).
        let signs = std::array::from_fn(|_| gen_random_signs(&mut rng, padded_dim));

        // Each Hadamard transform has a normalization factor of 1/sqrt(padded_dim).
        // With 3 Hadamard transforms: (1/sqrt(n))^3 = 1/(n * sqrt(n)).
        // But we want an orthogonal-like transform that preserves norms. The
        // standard WHT without normalization scales by sqrt(n) each time. With 3
        // applications: output ~ n^(3/2) * input. To normalize: divide by n^(3/2).
        // Equivalently, divide by n after each WHT (making each one orthonormal).
        // We fold all normalization into a single factor applied at the end.
        let norm_factor = 1.0 / (padded_dim as f32 * (padded_dim as f32).sqrt());

        Ok(Self {
            signs,
            dim: dimension,
            padded_dim,
            norm_factor,
        })
    }

    /// Apply forward rotation: `output = SRHT(input)`.
    ///
    /// Both `input` and `output` must have length `padded_dim()`. The caller
    /// is responsible for zero-padding input beyond `dim` positions.
    pub fn rotate(&self, input: &[f32], output: &mut [f32]) {
        let pd = self.padded_dim;
        debug_assert_eq!(input.len(), pd);
        debug_assert_eq!(output.len(), pd);

        output.copy_from_slice(input);
        self.apply_srht(output);
    }

    /// Apply inverse rotation: `output = SRHT⁻¹(input)`.
    ///
    /// Both `input` and `output` must have length `padded_dim()`.
    pub fn inverse_rotate(&self, input: &[f32], output: &mut [f32]) {
        let pd = self.padded_dim;
        debug_assert_eq!(input.len(), pd);
        debug_assert_eq!(output.len(), pd);

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
        // Round 1: D₁ then H
        apply_signs(buf, &self.signs[0]);
        walsh_hadamard_transform(buf);

        // Round 2: D₂ then H
        apply_signs(buf, &self.signs[1]);
        walsh_hadamard_transform(buf);

        // Round 3: D₃ then normalize
        apply_signs(buf, &self.signs[2]);
        walsh_hadamard_transform(buf);

        // Apply combined normalization factor.
        let norm = self.norm_factor;
        for val in buf.iter_mut() {
            *val *= norm;
        }
    }

    /// Apply the inverse SRHT.
    ///
    /// Forward is: norm · H · D₃ · H · D₂ · H · D₁
    /// Inverse is: norm · D₁ · H · D₂ · H · D₃ · H
    fn apply_inverse_srht(&self, buf: &mut [f32]) {
        walsh_hadamard_transform(buf);
        apply_signs(buf, &self.signs[2]);

        walsh_hadamard_transform(buf);
        apply_signs(buf, &self.signs[1]);

        walsh_hadamard_transform(buf);
        apply_signs(buf, &self.signs[0]);

        let norm = self.norm_factor;
        for val in buf.iter_mut() {
            *val *= norm;
        }
    }

    /// Returns the dimension of this rotation.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Returns the normalization factor for this transform.
    pub fn norm_factor(&self) -> f32 {
        self.norm_factor
    }

    /// Export the 3 sign vectors as a single `BoolArray` in inverse-application order.
    ///
    /// The output `BoolArray` has length `3 * padded_dim` and stores `[D₃ | D₂ | D₁]`
    /// so that decompression (which applies the inverse transform) iterates sign arrays
    /// 0→1→2 sequentially. Convention: `true` = +1, `false` = -1.
    pub fn export_inverse_signs_bool_array(&self) -> BoolArray {
        let total_bits = 3 * self.padded_dim;
        let mut bits = BitBufferMut::new_unset(total_bits);

        // Store in inverse order: signs[2] (D₃), signs[1] (D₂), signs[0] (D₁)
        for (round, sign_idx) in [2, 1, 0].iter().enumerate() {
            let offset = round * self.padded_dim;
            for j in 0..self.padded_dim {
                if self.signs[*sign_idx][j] > 0.0 {
                    bits.set(offset + j);
                }
            }
        }

        BoolArray::new(bits.freeze(), Validity::NonNullable)
    }

    /// Reconstruct a `RotationMatrix` from a stored `BoolArray` of signs.
    ///
    /// The `BoolArray` must have length `3 * padded_dim` with signs in inverse
    /// application order `[D₃ | D₂ | D₁]` (as produced by
    /// [`export_inverse_signs_bool_array`]).
    pub fn from_bool_array(signs_array: &BoolArray, dim: usize) -> VortexResult<Self> {
        let padded_dim = dim.next_power_of_two();
        vortex_ensure!(
            signs_array.len() == 3 * padded_dim,
            "Expected BoolArray of length {}, got {}",
            3 * padded_dim,
            signs_array.len()
        );

        let bit_buf = signs_array.to_bit_buffer();

        // Reconstruct in storage order (inverse): [D₃, D₂, D₁] → signs[2], signs[1], signs[0]
        let mut signs: [Vec<f32>; 3] = std::array::from_fn(|_| Vec::with_capacity(padded_dim));

        for (round, sign_idx) in [2, 1, 0].iter().enumerate() {
            let offset = round * padded_dim;
            signs[*sign_idx] = (0..padded_dim)
                .map(|j| {
                    if bit_buf.value(offset + j) {
                        1.0f32
                    } else {
                        -1.0f32
                    }
                })
                .collect();
        }

        let norm_factor = 1.0 / (padded_dim as f32 * (padded_dim as f32).sqrt());

        Ok(Self {
            signs,
            dim,
            padded_dim,
            norm_factor,
        })
    }
}

/// Apply the inverse SRHT using sign bits stored in a raw byte slice.
///
/// This is the hot-path function for decompression. The `signs_bytes` buffer
/// contains `3 * padded_dim` bits in inverse-application order `[D₃ | D₂ | D₁]`.
/// Convention: bit set (1) = +1, bit unset (0) = -1 (negate).
///
/// Applies: H → D₃ → H → D₂ → H → D₁ → scale
#[inline]
pub fn apply_inverse_srht_from_bits(
    buf: &mut [f32],
    signs_bytes: &[u8],
    padded_dim: usize,
    norm_factor: f32,
) {
    debug_assert!(padded_dim.is_power_of_two());
    debug_assert_eq!(buf.len(), padded_dim);

    for round in 0..3 {
        walsh_hadamard_transform(buf);
        apply_signs_from_bits(buf, signs_bytes, round * padded_dim);
    }

    for val in buf.iter_mut() {
        *val *= norm_factor;
    }
}

/// Element-wise negate coordinates where the sign bit is unset (0 = -1).
#[inline]
fn apply_signs_from_bits(buf: &mut [f32], signs_bytes: &[u8], bit_offset: usize) {
    for (j, val) in buf.iter_mut().enumerate() {
        let idx = bit_offset + j;
        let is_positive = (signs_bytes[idx / 8] >> (idx % 8)) & 1 == 1;
        if !is_positive {
            *val = -*val;
        }
    }
}

/// Generate a vector of random ±1 signs.
fn gen_random_signs(rng: &mut StdRng, len: usize) -> Vec<f32> {
    use rand::RngExt;
    (0..len)
        .map(|_| {
            if rng.random_bool(0.5) {
                1.0f32
            } else {
                -1.0f32
            }
        })
        .collect()
}

/// Element-wise multiply by ±1 signs.
#[inline]
fn apply_signs(buf: &mut [f32], signs: &[f32]) {
    for (val, &sign) in buf.iter_mut().zip(signs.iter()) {
        *val *= sign;
    }
}

/// In-place Walsh-Hadamard Transform (unnormalized, iterative).
///
/// Input length must be a power of 2. Runs in O(n log n).
fn walsh_hadamard_transform(buf: &mut [f32]) {
    let len = buf.len();
    debug_assert!(len.is_power_of_two());

    let mut half = 1;
    while half < len {
        for block_start in (0..len).step_by(half * 2) {
            for idx in block_start..block_start + half {
                let sum = buf[idx] + buf[idx + half];
                let diff = buf[idx] - buf[idx + half];
                buf[idx] = sum;
                buf[idx + half] = diff;
            }
        }
        half *= 2;
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

    /// Verify that export → from_bool_array produces identical rotation output.
    #[rstest]
    #[case(64)]
    #[case(128)]
    #[case(768)]
    fn sign_export_import_roundtrip(#[case] dim: usize) -> VortexResult<()> {
        let rot = RotationMatrix::try_new(42, dim)?;
        let padded_dim = rot.padded_dim();

        let signs_array = rot.export_inverse_signs_bool_array();
        let rot2 = RotationMatrix::from_bool_array(&signs_array, dim)?;

        // Verify both produce identical rotation and inverse rotation.
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

    /// Verify that the hot-path `apply_inverse_srht_from_bits` matches `inverse_rotate`.
    #[rstest]
    #[case(64)]
    #[case(128)]
    #[case(768)]
    fn hot_path_matches_inverse_rotate(#[case] dim: usize) -> VortexResult<()> {
        let rot = RotationMatrix::try_new(99, dim)?;
        let padded_dim = rot.padded_dim();
        let norm_factor = rot.norm_factor();

        let signs_array = rot.export_inverse_signs_bool_array();
        let bit_buf = signs_array.to_bit_buffer();
        let (_, _, raw_buf) = bit_buf.into_inner();

        // Create some rotated input.
        let mut input = vec![0.0f32; padded_dim];
        for i in 0..dim {
            input[i] = (i as f32 + 1.0) * 0.01;
        }
        let mut rotated = vec![0.0f32; padded_dim];
        rot.rotate(&input, &mut rotated);

        // Inverse via the struct method.
        let mut recovered1 = vec![0.0f32; padded_dim];
        rot.inverse_rotate(&rotated, &mut recovered1);

        // Inverse via the hot-path function.
        let mut recovered2 = rotated.clone();
        apply_inverse_srht_from_bits(&mut recovered2, raw_buf.as_ref(), padded_dim, norm_factor);

        for i in 0..padded_dim {
            assert!(
                (recovered1[i] - recovered2[i]).abs() < 1e-10,
                "Hot-path mismatch at {i}: {} vs {}",
                recovered1[i],
                recovered2[i]
            );
        }

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
