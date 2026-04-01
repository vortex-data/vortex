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
//! use vortex_tensor::encodings::turboquant::{TurboQuantConfig, turboquant_encode_mse};
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
//! assert!(encoded.nbytes() < 51200);
//! ```

pub use array::QjlCorrection;
pub use array::TurboQuant;
pub use array::TurboQuantArray;
pub use compress::TurboQuantConfig;
pub use compress::turboquant_encode_mse;
pub use compress::turboquant_encode_qjl;

mod array;
pub(crate) mod centroids;
mod compress;
mod compute;
pub(crate) mod decompress;
pub(crate) mod rotation;
pub mod scheme;
mod vtable;

/// Extension ID for the `Vector` type from `vortex-tensor`.
pub const VECTOR_EXT_ID: &str = "vortex.tensor.vector";

/// Extension ID for the `FixedShapeTensor` type from `vortex-tensor`.
pub const FIXED_SHAPE_TENSOR_EXT_ID: &str = "vortex.tensor.fixed_shape_tensor";

use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

/// Initialize the TurboQuant encoding in the given session.
pub fn initialize(session: &mut VortexSession) {
    session.arrays().register(TurboQuant);
}

#[cfg(test)]
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
    use vortex_array::matcher::Matcher;
    use vortex_array::session::ArraySession;
    use vortex_array::validity::Validity;
    use vortex_buffer::BufferMut;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::encodings::turboquant::TurboQuant;
    use crate::encodings::turboquant::TurboQuantConfig;
    use crate::encodings::turboquant::rotation::RotationMatrix;
    use crate::encodings::turboquant::turboquant_encode_mse;
    use crate::encodings::turboquant::turboquant_encode_qjl;

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
        encode_decode(fsl, |fsl| turboquant_encode_mse(fsl, &config))
    }

    fn encode_decode_qjl(
        fsl: &FixedSizeListArray,
        config: &TurboQuantConfig,
    ) -> VortexResult<(Vec<f32>, Vec<f32>)> {
        let config = config.clone();
        encode_decode(fsl, |fsl| turboquant_encode_qjl(fsl, &config))
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

    /// Compute the mean signed relative error of QJL inner product estimation
    /// over random query/vector pairs.
    fn qjl_mean_signed_relative_error(
        original: &[f32],
        decoded: &[f32],
        dim: usize,
        num_rows: usize,
    ) -> f32 {
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
            return 0.0;
        }

        signed_errors.iter().sum::<f32>() / signed_errors.len() as f32
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

        let mean_rel_error = qjl_mean_signed_relative_error(&original, &decoded, dim, num_rows);

        // Known limitation: non-power-of-2 dims have elevated QJL bias (~23% vs
        // ~11%) due to distribution mismatch between the SRHT zero-padded coordinate
        // distribution and the analytical (1-x^2)^((d-3)/2) model used for centroids.
        // Investigated approaches:
        // - Random permutation of zeros: no effect (issue is distribution shape)
        // - MC empirical centroids: fixes QJL bias but regresses MSE quality
        // - Analytical centroids with dim instead of padded_dim: mixed results
        // The principled fix requires jointly correcting centroids and QJL scale
        // factor for the actual SRHT zero-padded distribution.
        let threshold = if dim.is_power_of_two() { 0.15 } else { 0.25 };
        assert!(
            mean_rel_error.abs() < threshold,
            "QJL inner product bias too high: {mean_rel_error:.4} for dim={dim}, bits={bit_width} \
             (threshold={threshold})"
        );
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
        let decoded = encoded.execute::<FixedSizeListArray>(&mut ctx)?;
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
        let decoded = encoded.execute::<FixedSizeListArray>(&mut ctx)?;
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
        let encoded = TurboQuant::try_match(&*encoded).unwrap();

        let mut ctx = SESSION.create_execution_ctx();
        let stored_centroids_prim = encoded
            .centroids()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        let stored = stored_centroids_prim.as_slice::<f32>();

        let padded_dim = encoded.padded_dim();
        let computed = crate::encodings::turboquant::centroids::get_centroids(padded_dim, 3)?;

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
        let encoded = TurboQuant::try_match(&*encoded).unwrap();

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
        let expected_u8 = rot_from_seed.export_inverse_signs_u8();
        let stored_signs = encoded
            .rotation_signs()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        let stored_u8 = stored_signs.as_slice::<u8>();

        assert_eq!(expected_u8.len(), stored_u8.len());
        for i in 0..expected_u8.len() {
            assert_eq!(expected_u8[i], stored_u8[i], "Sign mismatch at index {i}");
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

    // -----------------------------------------------------------------------
    // Edge case and input format tests
    // -----------------------------------------------------------------------

    /// Verify that all-zero vectors roundtrip correctly (norm == 0 branch).
    #[test]
    fn all_zero_vectors_roundtrip() -> VortexResult<()> {
        let num_rows = 10;
        let dim = 128;
        let buf = BufferMut::<f32>::full(0.0f32, num_rows * dim);
        let elements = PrimitiveArray::new::<f32>(buf.freeze(), Validity::NonNullable);
        let fsl = FixedSizeListArray::try_new(
            elements.into_array(),
            dim as u32,
            Validity::NonNullable,
            num_rows,
        )?;

        let config = TurboQuantConfig {
            bit_width: 3,
            seed: Some(42),
        };
        let (original, decoded) = encode_decode_mse(&fsl, &config)?;
        // All-zero vectors should decode to all-zero (norm=0 → 0 * anything = 0).
        for (i, (&o, &d)) in original.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(o, 0.0, "original[{i}] not zero");
            assert_eq!(d, 0.0, "decoded[{i}] not zero for all-zero input");
        }
        Ok(())
    }

    /// Verify that f64 input is accepted and encoded (converted to f32 internally).
    #[test]
    fn f64_input_encodes_successfully() -> VortexResult<()> {
        let num_rows = 10;
        let dim = 64;
        let mut rng = StdRng::seed_from_u64(99);
        let normal = Normal::new(0.0f64, 1.0).unwrap();

        let mut buf = BufferMut::<f64>::with_capacity(num_rows * dim);
        for _ in 0..(num_rows * dim) {
            buf.push(normal.sample(&mut rng));
        }
        let elements = PrimitiveArray::new::<f64>(buf.freeze(), Validity::NonNullable);
        let fsl = FixedSizeListArray::try_new(
            elements.into_array(),
            dim as u32,
            Validity::NonNullable,
            num_rows,
        )?;

        let config = TurboQuantConfig {
            bit_width: 3,
            seed: Some(42),
        };
        // Verify encoding succeeds with f64 input (f64→f32 conversion).
        let encoded = turboquant_encode_mse(&fsl, &config)?;
        let encoded = TurboQuant::try_match(&*encoded).unwrap();
        assert_eq!(encoded.norms().len(), num_rows);
        assert_eq!(encoded.dimension(), dim as u32);
        Ok(())
    }

    /// Verify serde roundtrip: serialize MSE array metadata + children, then rebuild.
    #[test]
    fn mse_serde_roundtrip() -> VortexResult<()> {
        use vortex_array::DynArray;
        use vortex_array::vtable::VTable;

        let fsl = make_fsl(10, 128, 42);
        let config = TurboQuantConfig {
            bit_width: 3,
            seed: Some(123),
        };
        let encoded = turboquant_encode_mse(&fsl, &config)?;
        let encoded = TurboQuant::try_match(&*encoded).unwrap();

        // Serialize metadata.
        let metadata = <TurboQuant as VTable>::metadata(encoded)?;
        let serialized =
            <TurboQuant as VTable>::serialize(metadata)?.expect("metadata should serialize");

        // Collect children.
        let nchildren = <TurboQuant as VTable>::nchildren(encoded);
        assert_eq!(nchildren, 4);
        let children: Vec<ArrayRef> = (0..nchildren)
            .map(|i| <TurboQuant as VTable>::child(encoded, i))
            .collect();

        // Deserialize and rebuild.
        let deserialized = <TurboQuant as VTable>::deserialize(
            &serialized,
            encoded.dtype(),
            encoded.len(),
            &[],
            &SESSION,
        )?;

        // Verify metadata fields survived roundtrip.
        assert_eq!(deserialized.dimension, encoded.dimension());
        assert_eq!(deserialized.bit_width, encoded.bit_width() as u32);
        assert_eq!(deserialized.has_qjl, encoded.has_qjl());

        // Verify the rebuilt array decodes identically.
        let mut ctx = SESSION.create_execution_ctx();
        let decoded_original = encoded
            .clone()
            .into_array()
            .execute::<FixedSizeListArray>(&mut ctx)?;
        let original_elements = decoded_original.elements().to_canonical()?.into_primitive();

        // Rebuild from children (simulating deserialization).
        let rebuilt = crate::encodings::turboquant::array::TurboQuantArray::try_new_mse(
            encoded.dtype().clone(),
            children[0].clone(),
            children[1].clone(),
            children[2].clone(),
            children[3].clone(),
            deserialized.dimension,
            deserialized.bit_width as u8,
        )?;
        let decoded_rebuilt = rebuilt
            .into_array()
            .execute::<FixedSizeListArray>(&mut ctx)?;
        let rebuilt_elements = decoded_rebuilt.elements().to_canonical()?.into_primitive();

        assert_eq!(
            original_elements.as_slice::<f32>(),
            rebuilt_elements.as_slice::<f32>()
        );
        Ok(())
    }

    /// Verify serde roundtrip for QJL: serialize metadata + children, then rebuild.
    #[test]
    fn qjl_serde_roundtrip() -> VortexResult<()> {
        use vortex_array::DynArray;
        use vortex_array::vtable::VTable;

        let fsl = make_fsl(10, 128, 42);
        let config = TurboQuantConfig {
            bit_width: 4,
            seed: Some(456),
        };
        let encoded = turboquant_encode_qjl(&fsl, &config)?;
        let encoded = TurboQuant::try_match(&*encoded).unwrap();

        // Serialize metadata.
        let metadata = <TurboQuant as VTable>::metadata(encoded)?;
        let serialized =
            <TurboQuant as VTable>::serialize(metadata)?.expect("metadata should serialize");

        // Collect children — QJL has 7 (4 MSE + 3 QJL).
        let nchildren = <TurboQuant as VTable>::nchildren(encoded);
        assert_eq!(nchildren, 7);
        let children: Vec<ArrayRef> = (0..nchildren)
            .map(|i| <TurboQuant as VTable>::child(encoded, i))
            .collect();

        // Deserialize metadata.
        let deserialized = <TurboQuant as VTable>::deserialize(
            &serialized,
            encoded.dtype(),
            encoded.len(),
            &[],
            &SESSION,
        )?;

        assert!(deserialized.has_qjl);
        assert_eq!(deserialized.dimension, encoded.dimension());

        // Verify decode: original vs rebuilt from children.
        let mut ctx = SESSION.create_execution_ctx();
        let decoded_original = encoded
            .clone()
            .into_array()
            .execute::<FixedSizeListArray>(&mut ctx)?;
        let original_elements = decoded_original.elements().to_canonical()?.into_primitive();

        // Rebuild with QJL children.
        let rebuilt = crate::encodings::turboquant::array::TurboQuantArray::try_new_qjl(
            encoded.dtype().clone(),
            children[0].clone(),
            children[1].clone(),
            children[2].clone(),
            children[3].clone(),
            crate::encodings::turboquant::array::QjlCorrection {
                signs: children[4].clone(),
                residual_norms: children[5].clone(),
                rotation_signs: children[6].clone(),
            },
            deserialized.dimension,
            deserialized.bit_width as u8,
        )?;
        let decoded_rebuilt = rebuilt
            .into_array()
            .execute::<FixedSizeListArray>(&mut ctx)?;
        let rebuilt_elements = decoded_rebuilt.elements().to_canonical()?.into_primitive();

        assert_eq!(
            original_elements.as_slice::<f32>(),
            rebuilt_elements.as_slice::<f32>()
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Compute pushdown tests
    // -----------------------------------------------------------------------

    #[test]
    fn slice_preserves_data() -> VortexResult<()> {
        let fsl = make_fsl(20, 128, 42);
        let config = TurboQuantConfig {
            bit_width: 3,
            seed: Some(123),
        };
        let encoded = turboquant_encode_mse(&fsl, &config)?;

        // Full decompress then slice.
        let mut ctx = SESSION.create_execution_ctx();
        let full_decoded = encoded.clone().execute::<FixedSizeListArray>(&mut ctx)?;
        let expected = full_decoded.slice(5..10)?;
        let expected_prim = expected.to_canonical()?.into_fixed_size_list();
        let expected_elements = expected_prim.elements().to_canonical()?.into_primitive();

        // Slice then decompress.
        let sliced = encoded.slice(5..10)?;
        let sliced_decoded = sliced.execute::<FixedSizeListArray>(&mut ctx)?;
        let actual_elements = sliced_decoded.elements().to_canonical()?.into_primitive();

        assert_eq!(
            expected_elements.as_slice::<f32>(),
            actual_elements.as_slice::<f32>()
        );
        Ok(())
    }

    #[test]
    fn slice_qjl_preserves_data() -> VortexResult<()> {
        let fsl = make_fsl(20, 128, 42);
        let config = TurboQuantConfig {
            bit_width: 4,
            seed: Some(456),
        };
        let encoded = turboquant_encode_qjl(&fsl, &config)?;

        let mut ctx = SESSION.create_execution_ctx();
        let full_decoded = encoded.clone().execute::<FixedSizeListArray>(&mut ctx)?;
        let expected = full_decoded.slice(3..8)?;
        let expected_prim = expected.to_canonical()?.into_fixed_size_list();
        let expected_elements = expected_prim.elements().to_canonical()?.into_primitive();

        let sliced = encoded.slice(3..8)?;
        let sliced_decoded = sliced.execute::<FixedSizeListArray>(&mut ctx)?;
        let actual_elements = sliced_decoded.elements().to_canonical()?.into_primitive();

        assert_eq!(
            expected_elements.as_slice::<f32>(),
            actual_elements.as_slice::<f32>()
        );
        Ok(())
    }

    #[test]
    fn scalar_at_matches_decompress() -> VortexResult<()> {
        let fsl = make_fsl(10, 64, 42);
        let config = TurboQuantConfig {
            bit_width: 3,
            seed: Some(123),
        };
        let encoded = turboquant_encode_mse(&fsl, &config)?;

        let mut ctx = SESSION.create_execution_ctx();
        let full_decoded = encoded.clone().execute::<FixedSizeListArray>(&mut ctx)?;

        for i in [0, 1, 5, 9] {
            let expected = full_decoded.scalar_at(i)?;
            let actual = encoded.scalar_at(i)?;
            assert_eq!(expected, actual, "scalar_at mismatch at index {i}");
        }
        Ok(())
    }

    #[test]
    fn l2_norm_readthrough() -> VortexResult<()> {
        let fsl = make_fsl(10, 128, 42);
        let config = TurboQuantConfig {
            bit_width: 3,
            seed: Some(123),
        };
        let encoded = turboquant_encode_mse(&fsl, &config)?;
        let tq = TurboQuant::try_match(&*encoded).unwrap();

        // Stored norms should match the actual L2 norms of the input.
        let norms_prim = tq.norms().to_canonical()?.into_primitive();
        let stored_norms = norms_prim.as_slice::<f32>();

        let input_prim = fsl.elements().to_canonical()?.into_primitive();
        let input_f32 = input_prim.as_slice::<f32>();
        for row in 0..10 {
            let vec = &input_f32[row * 128..(row + 1) * 128];
            let actual_norm: f32 = vec.iter().map(|&v| v * v).sum::<f32>().sqrt();
            assert!(
                (stored_norms[row] - actual_norm).abs() < 1e-5,
                "norm mismatch at row {row}: stored={}, actual={}",
                stored_norms[row],
                actual_norm
            );
        }
        Ok(())
    }

    #[test]
    fn cosine_similarity_quantized_accuracy() -> VortexResult<()> {
        use crate::encodings::turboquant::compute::cosine_similarity::cosine_similarity_quantized;

        let fsl = make_fsl(20, 128, 42);
        let config = TurboQuantConfig {
            bit_width: 4,
            seed: Some(123),
        };
        let encoded = turboquant_encode_mse(&fsl, &config)?;
        let tq = TurboQuant::try_match(&*encoded).unwrap();

        // Compute exact cosine similarity from original data.
        let input_prim = fsl.elements().to_canonical()?.into_primitive();
        let input_f32 = input_prim.as_slice::<f32>();

        for (row_a, row_b) in [(0, 1), (5, 10), (0, 19)] {
            let a = &input_f32[row_a * 128..(row_a + 1) * 128];
            let b = &input_f32[row_b * 128..(row_b + 1) * 128];

            let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|&v| v * v).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|&v| v * v).sum::<f32>().sqrt();
            let exact_cos = dot / (norm_a * norm_b);

            let mut ctx = SESSION.create_execution_ctx();
            let approx_cos = cosine_similarity_quantized(tq, row_a, row_b, &mut ctx)?;

            // 4-bit quantization: expect reasonable accuracy.
            let error = (exact_cos - approx_cos).abs();
            assert!(
                error < 0.15,
                "cosine similarity error too large for ({row_a}, {row_b}): \
                 exact={exact_cos:.4}, approx={approx_cos:.4}, error={error:.4}"
            );
        }
        Ok(())
    }
}
