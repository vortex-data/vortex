// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant compression scheme for the pluggable compressor.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_compressor::CascadingCompressor;
use vortex_compressor::ctx::CompressorContext;
use vortex_compressor::scheme::Scheme;
use vortex_compressor::stats::ArrayAndStats;
use vortex_error::VortexResult;

use crate::encodings::turboquant::TurboQuant;
use crate::encodings::turboquant::TurboQuantConfig;
use crate::encodings::turboquant::turboquant_encode;
use crate::utils::tensor_element_ptype;
use crate::utils::tensor_list_size;

/// TurboQuant compression scheme for [`Vector`] extension types.
///
/// Applies lossy vector quantization to [`Vector`] extension arrays using the TurboQuant
/// algorithm with MSE-optimal encoding.
///
/// Register this scheme with the compressor builder via `with_scheme`:
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

        TurboQuant::validate_dtype(ext.dtype()).is_ok()
    }

    fn expected_compression_ratio(
        &self,
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<f64> {
        let dtype = data.array().dtype();
        let len = data.array().len();

        let ext = TurboQuant::validate_dtype(dtype)?;
        let element_ptype = tensor_element_ptype(ext)?;
        let dimension = tensor_list_size(ext)?;

        Ok(estimate_compression_ratio(
            element_ptype.bit_width(),
            dimension,
            len,
        ))
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        // TODO(connor): Fix this once we ensure that the data array is always canonical.
        let ext_array = data.array().to_canonical()?.into_extension();

        let config = TurboQuantConfig::default();
        turboquant_encode(&ext_array, &config, &mut compressor.execution_ctx())
    }
}

/// Estimate the compression ratio for TurboQuant MSE encoding with the default config.
fn estimate_compression_ratio(bits_per_element: usize, dimensions: u32, num_vectors: usize) -> f64 {
    let config = TurboQuantConfig::default();
    let padded_dim = dimensions.next_power_of_two() as usize;

    // Per-vector: MSE codes per padded coordinate, plus one f32 norm.
    let compressed_bits_per_vector = 32 // norm is always f32
        + (config.bit_width as usize) * padded_dim; // MSE codes

    // Shared overhead: codebook centroids (2^bit_width f32 values) and
    // rotation signs (3 * padded_dim bits).
    let num_centroids = 1usize << config.bit_width;
    let overhead_bits = num_centroids * 32 // centroids are always f32
        + 3 * padded_dim; // rotation signs, 1 bit each

    let compressed_size_bits = compressed_bits_per_vector * num_vectors + overhead_bits;
    let uncompressed_size_bits = bits_per_element * num_vectors * dimensions as usize;
    uncompressed_size_bits as f64 / compressed_size_bits as f64
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// Verify compression ratio for typical embedding dimensions.
    ///
    /// f32 input at 768-d (padded to 1024) with 1000 vectors should give ~4-6x.
    /// f32 input at 1024-d (no padding) should give higher ratio since no waste.
    #[rstest]
    #[case::f32_768d(32, 768, 1000, 2.5, 4.0)]
    #[case::f32_1024d(32, 1024, 1000, 3.5, 5.0)]
    #[case::f32_1536d(32, 1536, 1000, 2.5, 4.0)]
    #[case::f32_128d(32, 128, 1000, 3.0, 5.0)]
    #[case::f64_768d(64, 768, 1000, 5.0, 7.0)]
    #[case::f16_768d(16, 768, 1000, 1.2, 2.0)]
    fn compression_ratio_in_expected_range(
        #[case] bits_per_element: usize,
        #[case] dim: u32,
        #[case] num_vectors: usize,
        #[case] min_ratio: f64,
        #[case] max_ratio: f64,
    ) {
        let ratio = estimate_compression_ratio(bits_per_element, dim, num_vectors);
        assert!(
            ratio > min_ratio && ratio < max_ratio,
            "ratio {ratio:.2} not in [{min_ratio}, {max_ratio}] for \
             {bits_per_element}-bit elements, dim={dim}, n={num_vectors}"
        );
    }

    /// Compression ratio must always be > 1 for reasonable inputs,
    /// otherwise TurboQuant makes things bigger and should not be selected.
    #[rstest]
    #[case(32, 128, 100)]
    #[case(32, 768, 10)]
    #[case(64, 256, 50)]
    fn ratio_always_greater_than_one(
        #[case] bits_per_element: usize,
        #[case] dim: u32,
        #[case] num_vectors: usize,
    ) {
        let ratio = estimate_compression_ratio(bits_per_element, dim, num_vectors);
        assert!(
            ratio > 1.0,
            "ratio {ratio:.4} <= 1.0 for {bits_per_element}-bit, dim={dim}, n={num_vectors}"
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
