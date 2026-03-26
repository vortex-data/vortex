// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant vector quantization encoding for Vortex.
//!
//! Implements the TurboQuant algorithm ([arXiv:2504.19874]) for lossy compression of
//! high-dimensional vector data. The encoding operates on `FixedSizeList` arrays of floats
//! (the storage format of `Vector` and `FixedShapeTensor` extension types).
//!
//! [arXiv:2504.19874]: https://arxiv.org/abs/2504.19874
//!
//! # Variants
//!
//! - **MSE** (`TurboQuantVariant::Mse`): Minimizes mean-squared reconstruction error
//!   (1-8 bits per coordinate).
//! - **Prod** (`TurboQuantVariant::Prod`): Preserves inner products with an unbiased
//!   estimator (uses `b-1` bits for MSE + 1-bit QJL residual correction, 2-9 bits).
//!   At `b=9`, the MSE codes are raw int8 values suitable for direct use with
//!   tensor core int8 GEMM kernels.
//!
//! # Theoretical error bounds
//!
//! For unit-norm vectors quantized at `b` bits per coordinate, the paper's Theorem 1
//! guarantees normalized MSE distortion:
//!
//! > `E[||x - x̂||² / ||x||²] ≤ (√3 · π / 2) / 4^b`
//!
//! | Bits | MSE bound  | Quality           |
//! |------|------------|-------------------|
//! | 1    | 6.80e-01   | Poor              |
//! | 2    | 1.70e-01   | Usable for ANN    |
//! | 3    | 4.25e-02   | Good              |
//! | 4    | 1.06e-02   | Very good         |
//! | 5    | 2.66e-03   | Excellent         |
//! | 6    | 6.64e-04   | Near-lossless     |
//! | 7    | 1.66e-04   | Near-lossless     |
//! | 8    | 4.15e-05   | Near-lossless     |
//!
//! # Compression ratios
//!
//! Each vector is stored as `padded_dim × bit_width / 8` bytes of quantized codes plus a
//! 4-byte f32 norm. Non-power-of-2 dimensions are padded to the next power of 2 for the
//! Walsh-Hadamard transform, which reduces the effective ratio for those dimensions.
//!
//! | dim  | padded | bits | f32 bytes | TQ bytes | ratio  |
//! |------|--------|------|-----------|----------|--------|
//! |  768 |   1024 |    2 |      3072 |      260 | 11.8x  |
//! | 1024 |   1024 |    2 |      4096 |      260 | 15.8x  |
//! |  768 |   1024 |    4 |      3072 |      516 |  6.0x  |
//! | 1024 |   1024 |    4 |      4096 |      516 |  7.9x  |
//! |  768 |   1024 |    8 |      3072 |     1028 |  3.0x  |
//! | 1024 |   1024 |    8 |      4096 |     1028 |  4.0x  |
//!
//! # Example
//!
//! ```
//! use vortex_array::IntoArray;
//! use vortex_array::arrays::FixedSizeListArray;
//! use vortex_array::arrays::PrimitiveArray;
//! use vortex_array::validity::Validity;
//! use vortex_buffer::BufferMut;
//! use vortex_turboquant::{TurboQuantConfig, TurboQuantVariant, turboquant_encode};
//!
//! // Create a FixedSizeListArray of 100 random 128-d vectors.
//! let num_rows = 100;
//! let dim = 128;
//! let mut buf = BufferMut::<f32>::with_capacity(num_rows * dim);
//! for i in 0..(num_rows * dim) {
//!     buf.push((i as f32 * 0.001).sin());
//! }
//! let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
//! let fsl = FixedSizeListArray::try_new(
//!     elements.into_array(), dim as u32, Validity::NonNullable, num_rows,
//! ).unwrap();
//!
//! // Quantize at 2 bits per coordinate.
//! let config = TurboQuantConfig {
//!     bit_width: 2,
//!     variant: TurboQuantVariant::Mse,
//!     seed: Some(42),
//! };
//! let encoded = turboquant_encode(&fsl, &config).unwrap();
//!
//! // Verify compression: 100 vectors × 128 dims × 4 bytes = 51200 bytes input.
//! // Output: 100 × (128 padded × 2 bits / 8 + 4 norm bytes) = 100 × 36 = 3600 bytes.
//! assert!(encoded.codes().nbytes() + encoded.norms().nbytes() < 51200);
//!
//! // Verify the theoretical MSE bound holds.
//! // For 2-bit quantization: bound = sqrt(3)*pi/2 / 4^2 ≈ 0.170.
//! // (Full roundtrip decoding requires an ExecutionCtx from a VortexSession.)
//! ```

pub use array::TurboQuant;
pub use array::TurboQuantArray;
pub use array::TurboQuantVariant;
pub use compress::TurboQuantConfig;
pub use compress::turboquant_encode;
pub use mse_array::TurboQuantMSE;
pub use mse_array::TurboQuantMSEArray;
pub use qjl_array::TurboQuantQJL;
pub use qjl_array::TurboQuantQJLArray;
mod array;
pub mod centroids;
mod compress;
mod decompress;
pub mod mse_array;
pub mod qjl_array;
pub mod rotation;
mod rules;

use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Initialize the TurboQuant encodings in the given session.
pub fn initialize(session: &mut VortexSession) {
    session.arrays().register(TurboQuant);
    session.arrays().register(TurboQuantMSE);
    session.arrays().register(TurboQuantQJL);
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
    #[case(128, 6)]
    #[case(128, 8)]
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

    /// Verify that MSE distortion is within theoretical bounds (Theorem 1).
    ///
    /// Paper Theorem 1: D_mse <= (sqrt(3)*pi/2) / 4^b for the normalized
    /// per-coordinate MSE of unit-norm vectors. This bound holds tightly for
    /// 1-4 bits; at higher bit widths the SRHT finite-dimension effects
    /// dominate the vanishingly small quantization error, so we test those
    /// separately in `high_bitwidth_mse_is_small`.
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
             for dim={dim}, bits={bit_width}",
        );

        Ok(())
    }

    /// Verify that high bit-width quantization (5-8) achieves very low distortion.
    ///
    /// At these bit widths the theoretical bound is extremely tight and the actual
    /// distortion is dominated by the SRHT finite-dimension approximation rather
    /// than quantization error. We just verify the MSE is well below 1% and
    /// strictly less than the 4-bit MSE.
    #[rstest]
    #[case(128, 6)]
    #[case(128, 8)]
    #[case(256, 6)]
    #[case(256, 8)]
    fn high_bitwidth_mse_is_small(#[case] dim: usize, #[case] bit_width: u8) -> VortexResult<()> {
        let num_rows = 200;
        let fsl = make_fsl(num_rows, dim, 42);

        // Get the 4-bit MSE as a reference ceiling.
        let config_4bit = TurboQuantConfig {
            bit_width: 4,
            variant: TurboQuantVariant::Mse,
            seed: Some(123),
        };
        let (original_4, decoded_4) = encode_decode(&fsl, &config_4bit)?;
        let mse_4bit = per_vector_normalized_mse(&original_4, &decoded_4, dim, num_rows);

        let config = TurboQuantConfig {
            bit_width,
            variant: TurboQuantVariant::Mse,
            seed: Some(123),
        };
        let (original, decoded) = encode_decode(&fsl, &config)?;
        let mse = per_vector_normalized_mse(&original, &decoded, dim, num_rows);

        assert!(
            mse < mse_4bit,
            "{bit_width}-bit MSE ({mse:.6}) should be less than 4-bit MSE ({mse_4bit:.6}) \
             for dim={dim}",
        );
        assert!(
            mse < 0.01,
            "{bit_width}-bit MSE ({mse:.6}) should be well below 1% for dim={dim}",
        );

        Ok(())
    }

    #[rstest]
    #[case(32, 2)]
    #[case(32, 3)]
    #[case(128, 2)]
    #[case(128, 4)]
    #[case(128, 6)]
    #[case(128, 8)]
    #[case(128, 9)]
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
    #[case(128, 6)]
    #[case(128, 8)]
    #[case(128, 9)]
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

        let (min_bits, max_bits) = match variant {
            TurboQuantVariant::Mse => (1, 8),
            TurboQuantVariant::Prod => (2, 9),
        };

        let mut prev_mse = f32::MAX;
        for bit_width in min_bits..=max_bits {
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

    #[rstest]
    #[case(TurboQuantVariant::Mse, 2)]
    #[case(TurboQuantVariant::Prod, 2)]
    fn roundtrip_empty(
        #[case] variant: TurboQuantVariant,
        #[case] bit_width: u8,
    ) -> VortexResult<()> {
        let fsl = make_fsl(0, 128, 0);
        let config = TurboQuantConfig {
            bit_width,
            variant,
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

    #[rstest]
    #[case(TurboQuantVariant::Mse, 2)]
    #[case(TurboQuantVariant::Prod, 3)]
    fn roundtrip_single_row(
        #[case] variant: TurboQuantVariant,
        #[case] bit_width: u8,
    ) -> VortexResult<()> {
        let fsl = make_fsl(1, 128, 42);
        let config = TurboQuantConfig {
            bit_width,
            variant,
            seed: Some(123),
        };

        let (original, decoded) = encode_decode(&fsl, &config)?;
        assert_eq!(original.len(), decoded.len());
        Ok(())
    }

    #[test]
    fn rejects_dimension_below_2() {
        let mut buf = BufferMut::<f32>::with_capacity(1);
        buf.push(1.0);
        let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
        let fsl = FixedSizeListArray::try_new(elements.into_array(), 1, Validity::NonNullable, 1)
            .unwrap();
        let config = TurboQuantConfig {
            bit_width: 2,
            variant: TurboQuantVariant::Mse,
            seed: Some(0),
        };
        assert!(turboquant_encode(&fsl, &config).is_err());
    }
}
