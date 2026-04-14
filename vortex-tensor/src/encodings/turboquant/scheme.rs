// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant compression scheme.
//!
//! The scheme first normalizes the input via [`normalize_as_l2_denorm`], then encodes the
//! normalized child via [`turboquant_encode_unchecked`]. The result is:
//!
//! ```text
//! ScalarFnArray(L2Denorm, [
//!     ScalarFnArray(
//!         SorfTransform,
//!         FSL(Dict(codes, centroids))
//!     ),
//!     norms
//! ])
//! ```
//!
//! Decompression is automatic: executing the outer array walks the ScalarFn tree.
//!
//! [`normalize_as_l2_denorm`]: crate::scalar_fns::l2_denorm::normalize_as_l2_denorm
//! [`turboquant_encode_unchecked`]: crate::encodings::turboquant::turboquant_encode_unchecked

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::arrays::Extension;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_compressor::CascadingCompressor;
use vortex_compressor::ctx::CompressorContext;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_compressor::scheme::Scheme;
use vortex_compressor::stats::ArrayAndStats;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::encodings::turboquant::MAX_CENTROIDS;
use crate::encodings::turboquant::TurboQuantConfig;
use crate::encodings::turboquant::tq_validate_vector_dtype;
use crate::encodings::turboquant::turboquant_encode_unchecked;
use crate::scalar_fns::l2_denorm::L2Denorm;
use crate::scalar_fns::l2_denorm::normalize_as_l2_denorm;

/// TurboQuant compression scheme for [`Vector`] extension types.
///
/// Applies lossy vector quantization to [`Vector`] extension arrays using the TurboQuant algorithm
/// with MSE-optimal encoding.
///
/// Register this scheme with the compressor builder via `with_scheme`:
///
/// ```ignore
/// use vortex_btrblocks::BtrBlocksCompressorBuilder;
/// use vortex_tensor::encodings::turboquant::TurboQuantScheme;
///
/// let compressor = BtrBlocksCompressorBuilder::default()
///     .with_new_scheme(&TurboQuantScheme)
///     .build();
/// ```
///
/// [`Vector`]: crate::vector::Vector
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TurboQuantScheme;

impl Scheme for TurboQuantScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.tensor.turboquant"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        let Canonical::Extension(ext) = canonical else {
            return false;
        };

        tq_validate_vector_dtype(ext.dtype()).is_ok()
    }

    fn expected_compression_ratio(
        &self,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> CompressionEstimate {
        let len = data.array().len();
        let dtype = data.array().dtype();

        let vector_metadata =
            tq_validate_vector_dtype(dtype).vortex_expect("invalid dtype for TurboQuant");
        let element_ptype = vector_metadata.element_ptype();
        let element_bit_width: u8 = element_ptype
            .bit_width()
            .try_into()
            .vortex_expect("invalid bit width for TurboQuant");
        let dimension = vector_metadata.dimensions();

        CompressionEstimate::Verdict(EstimateVerdict::Ratio(estimate_compression_ratio(
            element_bit_width,
            dimension,
            len,
        )))
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let ext_array = data
            .array()
            .as_opt::<Extension>()
            .vortex_expect("expected an extension array");

        let mut ctx = compressor.execution_ctx();

        // 1. Normalize: produces L2Denorm(normalized_vectors, norms).
        let l2_denorm = normalize_as_l2_denorm(ext_array.as_ref().clone(), &mut ctx)?;
        let normalized = l2_denorm.child_at(0).clone();
        let norms = l2_denorm.child_at(1).clone();
        let num_rows = l2_denorm.len();

        // 2. Quantize the normalized child: SorfTransform(FSL(Dict)).
        let normalized_ext = normalized
            .as_opt::<Extension>()
            .vortex_expect("normalized child should be an Extension array");

        let config = TurboQuantConfig::default();
        // SAFETY: We just normalized the input via `normalize_as_l2_denorm`, so all rows are
        // guaranteed to be unit-norm (or zero for originally-null rows).
        let sorf_dict = unsafe { turboquant_encode_unchecked(normalized_ext, &config, &mut ctx)? };

        // 3. Wrap back in L2Denorm: the SorfTransform is the "normalized" child.
        // SAFETY: TurboQuant is a lossy approximation of the normalized child, so we intentionally
        // bypass the strict normalized-row validation when reattaching the stored norms.
        Ok(unsafe { L2Denorm::new_array_unchecked(sorf_dict, norms, num_rows) }?.into_array())
    }
}

// TODO(connor): If we ever add scheme vtables with metadata, we would need to pass in the config as
// a parameter here.
/// Estimate the compression ratio for TurboQuant MSE encoding with the default config.
fn estimate_compression_ratio(element_bit_width: u8, dimensions: u32, num_vectors: usize) -> f64 {
    let config = TurboQuantConfig::default();
    let padded_dim = dimensions.next_power_of_two() as usize;

    // Per-vector: MSE codes per padded coordinate, plus one stored norm in the input element
    // float width.
    let compressed_bits_per_vector =
        usize::from(element_bit_width) + usize::from(config.bit_width) * padded_dim;

    // Shared overhead: codebook centroids (2^bit_width f32 values).
    // Note: rotation signs are no longer stored — rotation is deterministic from seed.
    let num_centroids = 1usize << config.bit_width;
    debug_assert!(num_centroids <= MAX_CENTROIDS);
    let overhead_bits = num_centroids * 32; // centroids are always f32

    let compressed_size_bits = compressed_bits_per_vector * num_vectors + overhead_bits;

    let uncompressed_size_bits = usize::from(element_bit_width) * dimensions as usize * num_vectors;
    uncompressed_size_bits as f64 / compressed_size_bits as f64
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// Verify compression ratio for typical embedding dimensions.
    ///
    /// f32 input at 768-d (padded to 1024) with 1000 vectors should give ~3x.
    /// f32 input at 1024-d (no padding) should give ~4x since no padding waste.
    #[rstest]
    #[case::f32_768d(32, 768, 1000, 2.5, 4.5)]
    #[case::f32_1024d(32, 1024, 1000, 3.5, 5.0)]
    #[case::f32_1536d(32, 1536, 1000, 2.5, 4.5)]
    #[case::f32_128d(32, 128, 1000, 3.0, 5.0)]
    #[case::f64_768d(64, 768, 1000, 5.0, 9.0)]
    #[case::f16_768d(16, 768, 1000, 1.2, 2.5)]
    fn compression_ratio_in_expected_range(
        #[case] element_bit_width: u8,
        #[case] dim: u32,
        #[case] num_vectors: usize,
        #[case] min_ratio: f64,
        #[case] max_ratio: f64,
    ) {
        let ratio = estimate_compression_ratio(element_bit_width, dim, num_vectors);
        assert!(
            ratio > min_ratio && ratio < max_ratio,
            "ratio {ratio:.2} not in [{min_ratio}, {max_ratio}] for \
             {element_bit_width}-bit elements, dim={dim}, n={num_vectors}"
        );
    }

    /// Compression ratio must always be > 1 for reasonable inputs,
    /// otherwise TurboQuant makes things bigger and should not be selected.
    #[rstest]
    #[case(32, 128, 100)]
    #[case(32, 768, 10)]
    #[case(64, 256, 50)]
    fn ratio_always_greater_than_one(
        #[case] element_bit_width: u8,
        #[case] dim: u32,
        #[case] num_vectors: usize,
    ) {
        let ratio = estimate_compression_ratio(element_bit_width, dim, num_vectors);
        assert!(
            ratio > 1.0,
            "ratio {ratio:.4} <= 1.0 for {element_bit_width}-bit, dim={dim}, n={num_vectors}"
        );
    }

    #[rstest]
    #[case(16)]
    #[case(32)]
    #[case(64)]
    fn ratio_accounts_for_norm_storage_width(#[case] element_bit_width: u8) {
        let dim = 128u32;
        let num_vectors = 1usize;
        let padded_dim = dim.next_power_of_two() as usize;
        let config = TurboQuantConfig::default();
        let num_centroids = 1usize << config.bit_width;

        let expected_compressed_bits = usize::from(element_bit_width)
            + usize::from(config.bit_width) * padded_dim
            + num_centroids * 32;
        let expected_uncompressed_bits =
            usize::from(element_bit_width) * dim as usize * num_vectors;
        let expected = expected_uncompressed_bits as f64 / expected_compressed_bits as f64;

        assert_eq!(
            estimate_compression_ratio(element_bit_width, dim, num_vectors),
            expected
        );
    }

    /// Power-of-2 dimensions should have better ratios than their non-power-of-2
    /// predecessors due to no padding waste.
    #[test]
    fn power_of_two_has_better_ratio() {
        let ratio_768 = estimate_compression_ratio(32, 768, 1000);
        let ratio_1024 = estimate_compression_ratio(32, 1024, 1000);
        assert!(
            ratio_1024 > ratio_768,
            "1024-d ratio ({ratio_1024:.2}) should exceed 768-d ({ratio_768:.2})"
        );
    }
}
