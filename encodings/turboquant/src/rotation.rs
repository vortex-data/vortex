// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Deterministic random rotation matrix for TurboQuant.
//!
//! Generates a d×d orthogonal rotation matrix Π from a seed, using QR decomposition
//! of a random Normal(0,1) matrix. The same seed always produces the same matrix,
//! enabling reproducible encode/decode across sessions.

use nalgebra::DMatrix;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::Distribution;
use rand_distr::Normal;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

/// A deterministic d×d orthogonal rotation matrix generated from a seed.
pub struct RotationMatrix {
    /// The orthogonal matrix Q from QR decomposition.
    matrix: DMatrix<f32>,
}

impl RotationMatrix {
    /// Generate a rotation matrix from a seed via QR decomposition of a random Normal(0,1) matrix.
    pub fn try_new(seed: u64, dimension: usize) -> VortexResult<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0f32, 1.0)
            .map_err(|err| vortex_err!("Failed to create Normal distribution: {err}"))?;

        // Generate random d×d matrix with i.i.d. N(0,1) entries.
        let random_matrix = DMatrix::from_fn(dimension, dimension, |_, _| normal.sample(&mut rng));

        // QR decomposition to get an orthogonal matrix.
        let qr = random_matrix.qr();
        let q = qr.q();

        // Ensure the matrix is a proper rotation (det = +1) by adjusting signs
        // based on the diagonal of R. This makes the decomposition unique.
        let r = qr.r();
        let signs: Vec<f32> = (0..dimension)
            .map(|i| if r[(i, i)] >= 0.0 { 1.0 } else { -1.0 })
            .collect();

        let sign_matrix = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(signs));
        let matrix = q * sign_matrix;

        Ok(Self { matrix })
    }

    /// Apply forward rotation: `output = Π · input`.
    pub fn rotate(&self, input: &[f32], output: &mut [f32]) {
        let d = self.matrix.nrows();
        debug_assert_eq!(input.len(), d);
        debug_assert_eq!(output.len(), d);

        let input_vec = nalgebra::DVector::from_column_slice(input);
        let result = &self.matrix * &input_vec;
        output.copy_from_slice(result.as_slice());
    }

    /// Apply inverse rotation: `output = Πᵀ · input`.
    pub fn inverse_rotate(&self, input: &[f32], output: &mut [f32]) {
        let d = self.matrix.nrows();
        debug_assert_eq!(input.len(), d);
        debug_assert_eq!(output.len(), d);

        let input_vec = nalgebra::DVector::from_column_slice(input);
        let result = self.matrix.transpose() * &input_vec;
        output.copy_from_slice(result.as_slice());
    }

    /// Returns the dimension of this rotation matrix.
    pub fn dimension(&self) -> usize {
        self.matrix.nrows()
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

        let input: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let mut out1 = vec![0.0f32; 64];
        let mut out2 = vec![0.0f32; 64];

        r1.rotate(&input, &mut out1);
        r2.rotate(&input, &mut out2);

        assert_eq!(out1, out2);
        Ok(())
    }

    #[test]
    fn orthogonality() -> VortexResult<()> {
        let d = 32;
        let rot = RotationMatrix::try_new(123, d)?;

        // Π^T · Π should be approximately identity.
        let product = rot.matrix.transpose() * &rot.matrix;
        let identity = DMatrix::<f32>::identity(d, d);

        for i in 0..d {
            for j in 0..d {
                let diff: f32 = product[(i, j)] - identity[(i, j)];
                assert!(
                    diff.abs() < 1e-5,
                    "Πᵀ·Π[{i},{j}] = {}, expected {}",
                    product[(i, j)],
                    identity[(i, j)]
                );
            }
        }
        Ok(())
    }

    #[test]
    fn roundtrip_rotation() -> VortexResult<()> {
        let d = 64;
        let rot = RotationMatrix::try_new(99, d)?;

        let input: Vec<f32> = (0..d).map(|i| (i as f32) * 0.1).collect();
        let mut rotated = vec![0.0f32; d];
        let mut recovered = vec![0.0f32; d];

        rot.rotate(&input, &mut rotated);
        rot.inverse_rotate(&rotated, &mut recovered);

        for i in 0..d {
            assert!(
                (input[i] - recovered[i]).abs() < 1e-4,
                "roundtrip mismatch at {i}: {} vs {}",
                input[i],
                recovered[i]
            );
        }
        Ok(())
    }

    #[test]
    fn preserves_norm() -> VortexResult<()> {
        let d = 128;
        let rot = RotationMatrix::try_new(7, d)?;

        let input: Vec<f32> = (0..d).map(|i| (i as f32) * 0.01).collect();
        let input_norm: f32 = input.iter().map(|x| x * x).sum::<f32>().sqrt();

        let mut rotated = vec![0.0f32; d];
        rot.rotate(&input, &mut rotated);
        let rotated_norm: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();

        assert!(
            (input_norm - rotated_norm).abs() < 1e-3,
            "norm not preserved: {} vs {}",
            input_norm,
            rotated_norm
        );
        Ok(())
    }
}
