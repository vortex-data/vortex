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
use vortex_error::VortexResult;

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

    #[test]
    fn roundtrip_rotation() -> VortexResult<()> {
        let dim = 64;
        let rot = RotationMatrix::try_new(99, dim)?;
        let pd = rot.padded_dim();

        let mut input = vec![0.0f32; pd];
        for i in 0..dim {
            input[i] = (i as f32) * 0.1;
        }
        let mut rotated = vec![0.0f32; pd];
        let mut recovered = vec![0.0f32; pd];

        rot.rotate(&input, &mut rotated);
        rot.inverse_rotate(&rotated, &mut recovered);

        for i in 0..dim {
            assert!(
                (input[i] - recovered[i]).abs() < 1e-3,
                "roundtrip mismatch at {i}: {} vs {}",
                input[i],
                recovered[i]
            );
        }
        Ok(())
    }

    #[test]
    fn roundtrip_non_power_of_two() -> VortexResult<()> {
        let dim = 100;
        let rot = RotationMatrix::try_new(77, dim)?;
        let pd = rot.padded_dim();
        assert_eq!(pd, 128); // 100 rounds up to 128

        let mut input = vec![0.0f32; pd];
        for i in 0..dim {
            input[i] = (i as f32) * 0.01;
        }
        let mut rotated = vec![0.0f32; pd];
        let mut recovered = vec![0.0f32; pd];

        rot.rotate(&input, &mut rotated);
        rot.inverse_rotate(&rotated, &mut recovered);

        for i in 0..dim {
            assert!(
                (input[i] - recovered[i]).abs() < 1e-2,
                "roundtrip mismatch at {i}: {} vs {}",
                input[i],
                recovered[i]
            );
        }
        Ok(())
    }

    #[test]
    fn preserves_norm() -> VortexResult<()> {
        let dim = 128;
        let rot = RotationMatrix::try_new(7, dim)?;
        let pd = rot.padded_dim();

        let mut input = vec![0.0f32; pd];
        for i in 0..dim {
            input[i] = (i as f32) * 0.01;
        }
        let input_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();

        let mut rotated = vec![0.0f32; pd];
        rot.rotate(&input, &mut rotated);
        let rotated_norm: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            (input_norm - rotated_norm).abs() / input_norm < 0.01,
            "norm not preserved: {} vs {} (ratio: {})",
            input_norm,
            rotated_norm,
            rotated_norm / input_norm
        );
        Ok(())
    }

    #[test]
    fn preserves_norm_dim768() -> VortexResult<()> {
        let dim = 768;
        let rot = RotationMatrix::try_new(42, dim)?;
        let pd = rot.padded_dim();

        let mut input = vec![0.0f32; pd];
        for i in 0..dim {
            input[i] = (i as f32) * 0.001;
        }
        let input_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();

        let mut rotated = vec![0.0f32; pd];
        rot.rotate(&input, &mut rotated);
        let rotated_norm: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            (input_norm - rotated_norm).abs() / input_norm < 0.01,
            "norm not preserved for dim768: {} vs {} (ratio: {})",
            input_norm,
            rotated_norm,
            rotated_norm / input_norm
        );
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
