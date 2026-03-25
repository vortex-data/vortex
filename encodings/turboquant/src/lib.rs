// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant vector quantization encoding for Vortex.
//!
//! Implements the TurboQuant algorithm for lossy compression of high-dimensional vector data.
//! Supports two variants:
//! - **MSE**: Optimal for mean-squared error reconstruction
//! - **Prod**: Optimal for inner product preservation (unbiased)
//!
//! The encoding operates on `FixedSizeList` arrays of floats (the storage format of
//! `Vector` and `FixedShapeTensor` extension types).

pub use array::TurboQuant;
pub use array::TurboQuantArray;
pub use array::TurboQuantVariant;
pub use compress::TurboQuantConfig;
pub use compress::turboquant_encode;
mod array;
pub mod centroids;
mod compress;
mod decompress;
pub mod rotation;
mod rules;

use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Initialize the TurboQuant encoding in the given session.
pub fn initialize(session: &mut VortexSession) {
    session.arrays().register(TurboQuant);
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::FixedSizeListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::BufferMut;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::TurboQuantConfig;
    use crate::TurboQuantVariant;
    use crate::turboquant_encode;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    /// Create a FixedSizeListArray of random f32 vectors.
    fn make_fsl(num_rows: usize, dim: usize, seed: u64) -> FixedSizeListArray {
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use rand_distr::Distribution;
        use rand_distr::Normal;

        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0f32, 1.0).unwrap();

        let mut buf = BufferMut::<f32>::with_capacity(num_rows * dim);
        for _ in 0..(num_rows * dim) {
            buf.push(normal.sample(&mut rng));
        }

        let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
        FixedSizeListArray::try_new(
            elements.into_array(),
            dim as u32,
            Validity::NonNullable,
            num_rows,
        )
        .unwrap()
    }

    /// Theoretical MSE distortion bound from the TurboQuant paper (Theorem 1):
    ///   D_mse <= (sqrt(3) * pi / 2) * (1 / 4^b)
    ///
    /// This is the per-coordinate normalized MSE for a unit-norm vector after
    /// quantization with b bits using optimal scalar quantizers on a random rotation.
    ///
    /// The paper's bound is an upper bound; with fixed seeds our results are
    /// deterministic and empirically 0.5x-0.9x of the theoretical limit.
    fn theoretical_mse_bound(bit_width: u8) -> f32 {
        let sqrt3_pi_over_2 = (3.0f32).sqrt() * std::f32::consts::PI / 2.0;
        sqrt3_pi_over_2 / (4.0f32).powi(bit_width as i32)
    }

    /// Compute per-vector normalized MSE: average over vectors of ||x - x_hat||^2 / ||x||^2.
    fn per_vector_normalized_mse(
        original: &[f32],
        reconstructed: &[f32],
        dim: usize,
        num_rows: usize,
    ) -> f32 {
        let mut total = 0.0f32;
        for row in 0..num_rows {
            let orig = &original[row * dim..(row + 1) * dim];
            let recon = &reconstructed[row * dim..(row + 1) * dim];
            let norm_sq: f32 = orig.iter().map(|&v| v * v).sum();
            if norm_sq < 1e-10 {
                continue;
            }
            let err_sq: f32 = orig
                .iter()
                .zip(recon.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum();
            total += err_sq / norm_sq;
        }
        total / num_rows as f32
    }

    /// Helper to encode and decode, returning (original_elements, decoded_elements).
    fn encode_decode(
        fsl: &FixedSizeListArray,
        config: &TurboQuantConfig,
    ) -> VortexResult<(Vec<f32>, Vec<f32>)> {
        let original: Vec<f32> = {
            let prim = fsl.elements().to_canonical().unwrap().into_primitive();
            prim.as_slice::<f32>().to_vec()
        };
        let encoded = turboquant_encode(fsl, config)?;
        let mut ctx = SESSION.create_execution_ctx();
        let decoded = encoded
            .into_array()
            .execute::<FixedSizeListArray>(&mut ctx)?;
        let decoded_elements: Vec<f32> = {
            let prim = decoded.elements().to_canonical().unwrap().into_primitive();
            prim.as_slice::<f32>().to_vec()
        };
        Ok((original, decoded_elements))
    }

    #[rstest]
    #[case(32, 1)]
    #[case(32, 2)]
    #[case(32, 3)]
    #[case(32, 4)]
    #[case(128, 2)]
    #[case(128, 4)]
    #[case(256, 2)]
    fn roundtrip_mse(#[case] dim: usize, #[case] bit_width: u8) -> VortexResult<()> {
        let num_rows = 10;
        let fsl = make_fsl(num_rows, dim, 42);
        let config = TurboQuantConfig {
            bit_width,
            variant: TurboQuantVariant::Mse,
            seed: Some(123),
        };
        let (original, decoded) = encode_decode(&fsl, &config)?;
        assert_eq!(decoded.len(), original.len());
        Ok(())
    }

    /// Verify that MSE distortion is within theoretical bounds.
    ///
    /// Paper Theorem 1: D_mse <= (sqrt(3)*pi/2) / 4^b for the normalized
    /// per-coordinate MSE of unit-norm vectors. We use a relaxed bound since
    /// the SRHT is an approximation.
    #[rstest]
    #[case(128, 1)]
    #[case(128, 2)]
    #[case(128, 3)]
    #[case(128, 4)]
    #[case(256, 2)]
    #[case(256, 4)]
    fn mse_within_theoretical_bound(#[case] dim: usize, #[case] bit_width: u8) -> VortexResult<()> {
        let num_rows = 200;
        let fsl = make_fsl(num_rows, dim, 42);
        let config = TurboQuantConfig {
            bit_width,
            variant: TurboQuantVariant::Mse,
            seed: Some(123),
        };
        let (original, decoded) = encode_decode(&fsl, &config)?;

        let normalized_mse = per_vector_normalized_mse(&original, &decoded, dim, num_rows);
        let bound = theoretical_mse_bound(bit_width);

        assert!(
            normalized_mse < bound,
            "Normalized MSE {normalized_mse:.6} exceeds theoretical bound {bound:.6} \
             (theoretical {:.6}) for dim={dim}, bits={bit_width}",
            theoretical_mse_bound(bit_width)
        );

        Ok(())
    }

    #[rstest]
    #[case(32, 2)]
    #[case(32, 3)]
    #[case(128, 2)]
    #[case(128, 4)]
    fn roundtrip_prod(#[case] dim: usize, #[case] bit_width: u8) -> VortexResult<()> {
        let num_rows = 10;
        let fsl = make_fsl(num_rows, dim, 42);
        let config = TurboQuantConfig {
            bit_width,
            variant: TurboQuantVariant::Prod,
            seed: Some(456),
        };
        let (original, decoded) = encode_decode(&fsl, &config)?;
        assert_eq!(decoded.len(), original.len());
        Ok(())
    }

    /// Verify that the Prod variant produces approximately unbiased inner products.
    ///
    /// For random query y and quantized x_hat, the paper guarantees:
    ///   E[<y, x_hat>] = <y, x>
    ///
    /// We test by computing inner products between all pairs of original and
    /// reconstructed vectors and checking that the mean relative error is small.
    #[rstest]
    #[case(128, 2)]
    #[case(128, 3)]
    #[case(128, 4)]
    fn prod_inner_product_bias(#[case] dim: usize, #[case] bit_width: u8) -> VortexResult<()> {
        let num_rows = 100;
        let fsl = make_fsl(num_rows, dim, 42);
        let config = TurboQuantConfig {
            bit_width,
            variant: TurboQuantVariant::Prod,
            seed: Some(789),
        };
        let (original, decoded) = encode_decode(&fsl, &config)?;

        // Compute inner products between pairs of vectors: <x_i, x_hat_j> vs <x_i, x_j>
        // for i != j. Check that the mean signed error is close to zero (unbiased).
        let num_pairs = 500;
        let mut rng = {
            use rand::SeedableRng;
            rand::rngs::StdRng::seed_from_u64(0)
        };
        let mut signed_errors = Vec::with_capacity(num_pairs);

        for _ in 0..num_pairs {
            use rand::RngExt;
            let qi = rng.random_range(0..num_rows);
            let xi = rng.random_range(0..num_rows);
            if qi == xi {
                continue;
            }

            let query = &original[qi * dim..(qi + 1) * dim];
            let orig_vec = &original[xi * dim..(xi + 1) * dim];
            let quant_vec = &decoded[xi * dim..(xi + 1) * dim];

            let true_ip: f32 = query.iter().zip(orig_vec).map(|(&a, &b)| a * b).sum();
            let quant_ip: f32 = query.iter().zip(quant_vec).map(|(&a, &b)| a * b).sum();

            if true_ip.abs() > 1e-6 {
                signed_errors.push((quant_ip - true_ip) / true_ip.abs());
            }
        }

        if signed_errors.is_empty() {
            return Ok(());
        }

        let mean_rel_error: f32 = signed_errors.iter().sum::<f32>() / signed_errors.len() as f32;

        // The mean relative error should be close to zero for an unbiased estimator.
        // We allow up to 0.3 absolute mean relative error (generous for finite samples).
        assert!(
            mean_rel_error.abs() < 0.3,
            "Prod inner product bias too high: mean relative error = {mean_rel_error:.4} \
             for dim={dim}, bits={bit_width} ({} pairs)",
            signed_errors.len()
        );

        Ok(())
    }

    /// Verify that MSE distortion decreases with more bits (Prod variant too).
    #[rstest]
    #[case(TurboQuantVariant::Mse)]
    #[case(TurboQuantVariant::Prod)]
    fn mse_decreases_with_bits(#[case] variant: TurboQuantVariant) -> VortexResult<()> {
        let dim = 128;
        let num_rows = 50;
        let fsl = make_fsl(num_rows, dim, 99);

        let min_bits = match variant {
            TurboQuantVariant::Mse => 1,
            TurboQuantVariant::Prod => 2,
        };

        let mut prev_mse = f32::MAX;
        for bit_width in min_bits..=4u8 {
            let config = TurboQuantConfig {
                bit_width,
                variant,
                seed: Some(123),
            };
            let (original, decoded) = encode_decode(&fsl, &config)?;
            let mse = per_vector_normalized_mse(&original, &decoded, dim, num_rows);

            assert!(
                mse <= prev_mse * 1.01, // allow tiny floating point noise
                "MSE should decrease with more bits ({variant:?}): \
                 {bit_width}-bit MSE={mse:.6} > previous={prev_mse:.6}"
            );
            prev_mse = mse;
        }

        Ok(())
    }

    #[test]
    fn roundtrip_empty() -> VortexResult<()> {
        let fsl = make_fsl(0, 128, 0);
        let config = TurboQuantConfig {
            bit_width: 2,
            variant: TurboQuantVariant::Mse,
            seed: Some(0),
        };

        let encoded = turboquant_encode(&fsl, &config)?;
        let mut ctx = SESSION.create_execution_ctx();
        let decoded = encoded
            .into_array()
            .execute::<FixedSizeListArray>(&mut ctx)?;
        assert_eq!(decoded.len(), 0);

        Ok(())
    }
}
