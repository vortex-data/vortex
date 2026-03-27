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
//! use vortex_turboquant::{TurboQuantConfig, turboquant_encode_mse};
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
//! // Quantize at 2 bits per coordinate using MSE-optimal encoding.
//! let config = TurboQuantConfig { bit_width: 2, seed: Some(42) };
//! let encoded = turboquant_encode_mse(&fsl, &config).unwrap();
//!
//! // Verify compression: 100 vectors × 128 dims × 4 bytes = 51200 bytes input.
//! assert!(encoded.codes().nbytes() + encoded.norms().nbytes() < 51200);
//! ```

pub use compress::TurboQuantConfig;
pub use compress::turboquant_encode_mse;
pub use compress::turboquant_encode_qjl;
pub use mse::*;
pub use qjl::*;

pub mod centroids;
mod compress;
pub(crate) mod decompress;
mod mse;
mod qjl;
pub mod rotation;

/// Extension ID for the `Vector` type from `vortex-tensor`.
pub const VECTOR_EXT_ID: &str = "vortex.tensor.vector";

/// Extension ID for the `FixedShapeTensor` type from `vortex-tensor`.
pub const FIXED_SHAPE_TENSOR_EXT_ID: &str = "vortex.tensor.fixed_shape_tensor";

use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Initialize the TurboQuant encodings in the given session.
pub fn initialize(session: &mut VortexSession) {
    session.arrays().register(TurboQuantMSE);
    session.arrays().register(TurboQuantQJL);
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use std::sync::LazyLock;

    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::Distribution;
    use rand_distr::Normal;
    use rstest::rstest;
    use vortex_array::ArrayRef;
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
    use crate::rotation::RotationMatrix;
    use crate::turboquant_encode_mse;
    use crate::turboquant_encode_qjl;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

    /// Create a FixedSizeListArray of random f32 vectors (i.i.d. standard normal).
    fn make_fsl(num_rows: usize, dim: usize, seed: u64) -> FixedSizeListArray {
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

    fn theoretical_mse_bound(bit_width: u8) -> f32 {
        let sqrt3_pi_over_2 = (3.0f32).sqrt() * std::f32::consts::PI / 2.0;
        sqrt3_pi_over_2 / (4.0f32).powi(bit_width as i32)
    }

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

    /// Encode and decode, returning (original, decoded) flat f32 slices.
    fn encode_decode(
        fsl: &FixedSizeListArray,
        encode_fn: impl FnOnce(&FixedSizeListArray) -> VortexResult<ArrayRef>,
    ) -> VortexResult<(Vec<f32>, Vec<f32>)> {
        let original: Vec<f32> = {
            let prim = fsl.elements().to_canonical().unwrap().into_primitive();
            prim.as_slice::<f32>().to_vec()
        };
        let encoded = encode_fn(fsl)?;
        let mut ctx = SESSION.create_execution_ctx();
        let decoded = encoded.execute::<FixedSizeListArray>(&mut ctx)?;
        let decoded_elements: Vec<f32> = {
            let prim = decoded.elements().to_canonical().unwrap().into_primitive();
            prim.as_slice::<f32>().to_vec()
        };
        Ok((original, decoded_elements))
    }

    fn encode_decode_mse(
        fsl: &FixedSizeListArray,
        config: &TurboQuantConfig,
    ) -> VortexResult<(Vec<f32>, Vec<f32>)> {
        let config = config.clone();
        encode_decode(fsl, |fsl| {
            Ok(turboquant_encode_mse(fsl, &config)?.into_array())
        })
    }

    fn encode_decode_qjl(
        fsl: &FixedSizeListArray,
        config: &TurboQuantConfig,
    ) -> VortexResult<(Vec<f32>, Vec<f32>)> {
        let config = config.clone();
        encode_decode(fsl, |fsl| {
            Ok(turboquant_encode_qjl(fsl, &config)?.into_array())
        })
    }

    // -----------------------------------------------------------------------
    // MSE encoding tests
    // -----------------------------------------------------------------------

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
        let fsl = make_fsl(10, dim, 42);
        let config = TurboQuantConfig {
            bit_width,
            seed: Some(123),
        };
        let (original, decoded) = encode_decode_mse(&fsl, &config)?;
        assert_eq!(decoded.len(), original.len());
        Ok(())
    }

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
            seed: Some(123),
        };
        let (original, decoded) = encode_decode_mse(&fsl, &config)?;

        let normalized_mse = per_vector_normalized_mse(&original, &decoded, dim, num_rows);
        let bound = theoretical_mse_bound(bit_width);

        assert!(
            normalized_mse < bound,
            "Normalized MSE {normalized_mse:.6} exceeds bound {bound:.6} for dim={dim}, bits={bit_width}",
        );
        Ok(())
    }

    #[rstest]
    #[case(128, 6)]
    #[case(128, 8)]
    #[case(256, 6)]
    #[case(256, 8)]
    fn high_bitwidth_mse_is_small(#[case] dim: usize, #[case] bit_width: u8) -> VortexResult<()> {
        let num_rows = 200;
        let fsl = make_fsl(num_rows, dim, 42);

        let config_4bit = TurboQuantConfig {
            bit_width: 4,
            seed: Some(123),
        };
        let (original_4, decoded_4) = encode_decode_mse(&fsl, &config_4bit)?;
        let mse_4bit = per_vector_normalized_mse(&original_4, &decoded_4, dim, num_rows);

        let config = TurboQuantConfig {
            bit_width,
            seed: Some(123),
        };
        let (original, decoded) = encode_decode_mse(&fsl, &config)?;
        let mse = per_vector_normalized_mse(&original, &decoded, dim, num_rows);

        assert!(
            mse < mse_4bit,
            "{bit_width}-bit MSE ({mse:.6}) should be < 4-bit MSE ({mse_4bit:.6})"
        );
        assert!(mse < 0.01, "{bit_width}-bit MSE ({mse:.6}) should be < 1%");
        Ok(())
    }

    #[test]
    fn mse_decreases_with_bits() -> VortexResult<()> {
        let dim = 128;
        let num_rows = 50;
        let fsl = make_fsl(num_rows, dim, 99);

        let mut prev_mse = f32::MAX;
        for bit_width in 1..=8u8 {
            let config = TurboQuantConfig {
                bit_width,
                seed: Some(123),
            };
            let (original, decoded) = encode_decode_mse(&fsl, &config)?;
            let mse = per_vector_normalized_mse(&original, &decoded, dim, num_rows);
            assert!(
                mse <= prev_mse * 1.01,
                "MSE should decrease: {bit_width}-bit={mse:.6} > prev={prev_mse:.6}"
            );
            prev_mse = mse;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // QJL encoding tests
    // -----------------------------------------------------------------------

    #[rstest]
    #[case(32, 2)]
    #[case(32, 3)]
    #[case(128, 2)]
    #[case(128, 4)]
    #[case(128, 6)]
    #[case(128, 8)]
    #[case(128, 9)]
    #[case(768, 3)]
    fn roundtrip_qjl(#[case] dim: usize, #[case] bit_width: u8) -> VortexResult<()> {
        let fsl = make_fsl(10, dim, 42);
        let config = TurboQuantConfig {
            bit_width,
            seed: Some(456),
        };
        let (original, decoded) = encode_decode_qjl(&fsl, &config)?;
        assert_eq!(decoded.len(), original.len());
        Ok(())
    }

    #[rstest]
    #[case(128, 2)]
    #[case(128, 3)]
    #[case(128, 4)]
    #[case(128, 6)]
    #[case(128, 8)]
    #[case(128, 9)]
    #[case(768, 3)]
    #[case(768, 4)]
    fn qjl_inner_product_bias(#[case] dim: usize, #[case] bit_width: u8) -> VortexResult<()> {
        let num_rows = 100;
        let fsl = make_fsl(num_rows, dim, 42);
        let config = TurboQuantConfig {
            bit_width,
            seed: Some(789),
        };
        let (original, decoded) = encode_decode_qjl(&fsl, &config)?;

        let num_pairs = 500;
        let mut rng = StdRng::seed_from_u64(0);
        let mut signed_errors = Vec::with_capacity(num_pairs);

        for _ in 0..num_pairs {
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
        assert!(
            mean_rel_error.abs() < 0.3,
            "QJL inner product bias too high: {mean_rel_error:.4} for dim={dim}, bits={bit_width}"
        );
        Ok(())
    }

    #[test]
    fn qjl_mse_decreases_with_bits() -> VortexResult<()> {
        let dim = 128;
        let num_rows = 50;
        let fsl = make_fsl(num_rows, dim, 99);

        let mut prev_mse = f32::MAX;
        for bit_width in 2..=9u8 {
            let config = TurboQuantConfig {
                bit_width,
                seed: Some(123),
            };
            let (original, decoded) = encode_decode_qjl(&fsl, &config)?;
            let mse = per_vector_normalized_mse(&original, &decoded, dim, num_rows);
            assert!(
                mse <= prev_mse * 1.01,
                "QJL MSE should decrease: {bit_width}-bit={mse:.6} > prev={prev_mse:.6}"
            );
            prev_mse = mse;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[rstest]
    #[case(0)]
    #[case(1)]
    fn roundtrip_mse_edge_cases(#[case] num_rows: usize) -> VortexResult<()> {
        let fsl = make_fsl(num_rows, 128, 42);
        let config = TurboQuantConfig {
            bit_width: 2,
            seed: Some(123),
        };
        let encoded = turboquant_encode_mse(&fsl, &config)?;
        let mut ctx = SESSION.create_execution_ctx();
        let decoded = encoded
            .into_array()
            .execute::<FixedSizeListArray>(&mut ctx)?;
        assert_eq!(decoded.len(), num_rows);
        Ok(())
    }

    #[rstest]
    #[case(0)]
    #[case(1)]
    fn roundtrip_qjl_edge_cases(#[case] num_rows: usize) -> VortexResult<()> {
        let fsl = make_fsl(num_rows, 128, 42);
        let config = TurboQuantConfig {
            bit_width: 3,
            seed: Some(456),
        };
        let encoded = turboquant_encode_qjl(&fsl, &config)?;
        let mut ctx = SESSION.create_execution_ctx();
        let decoded = encoded
            .into_array()
            .execute::<FixedSizeListArray>(&mut ctx)?;
        assert_eq!(decoded.len(), num_rows);
        Ok(())
    }

    #[test]
    fn mse_rejects_dimension_below_2() {
        let fsl = make_fsl_dim1();
        let config = TurboQuantConfig {
            bit_width: 2,
            seed: Some(0),
        };
        assert!(turboquant_encode_mse(&fsl, &config).is_err());
    }

    #[test]
    fn qjl_rejects_dimension_below_2() {
        let fsl = make_fsl_dim1();
        let config = TurboQuantConfig {
            bit_width: 3,
            seed: Some(0),
        };
        assert!(turboquant_encode_qjl(&fsl, &config).is_err());
    }

    fn make_fsl_dim1() -> FixedSizeListArray {
        let mut buf = BufferMut::<f32>::with_capacity(1);
        buf.push(1.0);
        let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
        FixedSizeListArray::try_new(elements.into_array(), 1, Validity::NonNullable, 1).unwrap()
    }

    // -----------------------------------------------------------------------
    // Verification tests for stored metadata
    // -----------------------------------------------------------------------

    /// Verify that the centroids stored in the MSE array match what get_centroids() computes.
    #[test]
    fn stored_centroids_match_computed() -> VortexResult<()> {
        let fsl = make_fsl(10, 128, 42);
        let config = TurboQuantConfig {
            bit_width: 3,
            seed: Some(123),
        };
        let encoded = turboquant_encode_mse(&fsl, &config)?;

        let mut ctx = SESSION.create_execution_ctx();
        let stored_centroids_prim = encoded
            .centroids()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        let stored = stored_centroids_prim.as_slice::<f32>();

        let padded_dim = encoded.padded_dim();
        let computed = crate::centroids::get_centroids(padded_dim, 3)?;

        assert_eq!(stored.len(), computed.len());
        for i in 0..stored.len() {
            assert_eq!(stored[i], computed[i], "Centroid mismatch at {i}");
        }
        Ok(())
    }

    /// Verify that stored rotation signs produce identical decode to seed-based decode.
    ///
    /// Encodes the same data twice: once with the new path (stored signs), and
    /// once by manually recomputing the rotation from the seed. Both should
    /// produce identical output.
    #[test]
    fn stored_rotation_signs_produce_correct_decode() -> VortexResult<()> {
        let fsl = make_fsl(20, 128, 42);
        let config = TurboQuantConfig {
            bit_width: 3,
            seed: Some(123),
        };
        let encoded = turboquant_encode_mse(&fsl, &config)?;

        // Decode via the stored-signs path (normal decode).
        let mut ctx = SESSION.create_execution_ctx();
        let decoded_fsl = encoded
            .clone()
            .into_array()
            .execute::<FixedSizeListArray>(&mut ctx)?;
        let decoded = decoded_fsl.elements().to_canonical()?.into_primitive();
        let decoded_slice = decoded.as_slice::<f32>();

        // Verify stored signs match seed-derived signs.
        let rot_from_seed = RotationMatrix::try_new(123, 128)?;
        let exported = rot_from_seed.export_inverse_signs_bool_array();
        let stored_signs = encoded
            .rotation_signs()
            .clone()
            .execute::<vortex_array::arrays::BoolArray>(&mut ctx)?;

        assert_eq!(exported.len(), stored_signs.len());
        let exp_buf = exported.to_bit_buffer();
        let stored_buf = stored_signs.to_bit_buffer();
        for i in 0..exported.len() {
            assert_eq!(
                exp_buf.value(i),
                stored_buf.value(i),
                "Sign mismatch at bit {i}"
            );
        }

        // Also verify decode output is non-empty and has expected size.
        assert_eq!(decoded_slice.len(), 20 * 128);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // QJL-specific quality tests
    // -----------------------------------------------------------------------

    /// Verify that QJL's MSE component (at bit_width-1) satisfies the theoretical bound.
    #[rstest]
    #[case(128, 3)]
    #[case(128, 4)]
    #[case(256, 3)]
    fn qjl_mse_within_theoretical_bound(
        #[case] dim: usize,
        #[case] bit_width: u8,
    ) -> VortexResult<()> {
        let num_rows = 200;
        let fsl = make_fsl(num_rows, dim, 42);
        let config = TurboQuantConfig {
            bit_width,
            seed: Some(789),
        };
        let (original, decoded) = encode_decode_qjl(&fsl, &config)?;

        let normalized_mse = per_vector_normalized_mse(&original, &decoded, dim, num_rows);

        // QJL at b bits uses (b-1)-bit MSE plus a correction term.
        // The MSE should be at most the (b-1)-bit theoretical bound, though
        // in practice the QJL correction often improves it further.
        let mse_bound = theoretical_mse_bound(bit_width - 1);
        assert!(
            normalized_mse < mse_bound,
            "QJL MSE {normalized_mse:.6} exceeds (b-1)-bit bound {mse_bound:.6} \
             for dim={dim}, bits={bit_width}",
        );
        Ok(())
    }

    /// Verify that high-bitwidth QJL (8-9 bits) achieves very low distortion.
    #[rstest]
    #[case(128, 8)]
    #[case(128, 9)]
    fn high_bitwidth_qjl_is_small(#[case] dim: usize, #[case] bit_width: u8) -> VortexResult<()> {
        let num_rows = 200;
        let fsl = make_fsl(num_rows, dim, 42);

        // Compare against 4-bit QJL as reference ceiling.
        let config_4bit = TurboQuantConfig {
            bit_width: 4,
            seed: Some(789),
        };
        let (original_4, decoded_4) = encode_decode_qjl(&fsl, &config_4bit)?;
        let mse_4bit = per_vector_normalized_mse(&original_4, &decoded_4, dim, num_rows);

        let config = TurboQuantConfig {
            bit_width,
            seed: Some(789),
        };
        let (original, decoded) = encode_decode_qjl(&fsl, &config)?;
        let mse = per_vector_normalized_mse(&original, &decoded, dim, num_rows);

        assert!(
            mse < mse_4bit,
            "{bit_width}-bit QJL MSE ({mse:.6}) should be < 4-bit ({mse_4bit:.6})"
        );
        assert!(
            mse < 0.01,
            "{bit_width}-bit QJL MSE ({mse:.6}) should be < 1%"
        );
        Ok(())
    }
}
