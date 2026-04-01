// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant compression scheme for the pluggable compressor.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_compressor::CascadingCompressor;
use vortex_compressor::ctx::CompressorContext;
use vortex_compressor::scheme::Scheme;
use vortex_compressor::stats::ArrayAndStats;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use super::FIXED_SHAPE_TENSOR_EXT_ID;
use super::TurboQuantConfig;
use super::VECTOR_EXT_ID;
use super::turboquant_encode_qjl;

/// TurboQuant compression scheme for tensor extension types.
///
/// Applies lossy vector quantization to `Vector` and `FixedShapeTensor` extension
/// arrays using the TurboQuant algorithm with QJL correction for unbiased inner
/// product estimation.
///
/// Register this scheme with the compressor builder via `with_scheme`:
/// ```ignore
/// use vortex_btrblocks::BtrBlocksCompressorBuilder;
/// use vortex_tensor::encodings::turboquant::scheme::TURBOQUANT_SCHEME;
///
/// let compressor = BtrBlocksCompressorBuilder::default()
///     .with_scheme(&TURBOQUANT_SCHEME)
///     .build();
/// ```
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TurboQuantScheme;

/// Static instance for registration with `BtrBlocksCompressorBuilder::with_scheme`.
pub static TURBOQUANT_SCHEME: TurboQuantScheme = TurboQuantScheme;

impl Scheme for TurboQuantScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.tensor.turboquant"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        let Canonical::Extension(ext) = canonical else {
            return false;
        };

        get_tensor_element_ptype_and_length(ext.dtype()).is_ok()
    }

    fn expected_compression_ratio(
        &self,
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<f64> {
        let dtype = data.array().dtype();
        let len = data.array().len();
        let (element_ptype, dimensions) = get_tensor_element_ptype_and_length(dtype)?;
        let padded_dim = dimensions.next_power_of_two() as usize;
        let bits_per_element = element_ptype.bit_width();

        // Conservative estimate for 5-bit QJL (the default config): ~4x compression
        // for typical embedding dimensions (768-1536). The actual ratio varies with
        // dimension and padding overhead, but 4x is a reasonable lower bound that
        // ensures TurboQuant is preferred over generic float compression for tensor data.
        let compressed_bits_per_vector = 2 * bits_per_element // 2 of the original ptype for norm and qjl residual norms
            + 5 * padded_dim; // 5 bits per coordinate for TurboQuant with QJL
        let overhead_bits: usize = 2_usize.pow(bits_per_element as u32) * bits_per_element // 2^bits_per_element centroids (codebook)
            + 2 * 3 * padded_dim; // 2 * 3 * padded_dim bits for rotation signs and QJL rotation signs

        let compressed_size_bits = compressed_bits_per_vector * len + overhead_bits;
        let uncompressed_size_bits = bits_per_element * len * dimensions as usize;
        Ok(uncompressed_size_bits as f64 / compressed_size_bits as f64)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let array = data.array().clone();
        let ext_array = array.to_canonical()?.into_extension();
        let storage = ext_array.storage_array();
        let fsl = storage.to_canonical()?.into_fixed_size_list();

        let config = TurboQuantConfig::default();
        let encoded = turboquant_encode_qjl(&fsl, &config)?;

        Ok(ExtensionArray::new(ext_array.ext_dtype().clone(), encoded).into_array())
    }
}

fn get_tensor_element_ptype_and_length(dtype: &DType) -> VortexResult<(PType, u32)> {
    let ext_id = dtype.as_extension().id();
    let is_tensor = dtype.is_extension()
        && (ext_id.as_ref() == VECTOR_EXT_ID || ext_id.as_ref() == FIXED_SHAPE_TENSOR_EXT_ID);
    vortex_ensure!(is_tensor, "expected tensor extension dtype, got {}", dtype);

    let storage_dtype = dtype.as_extension().storage_dtype();
    let (element_dtype, fsl_len) = match storage_dtype {
        DType::FixedSizeList(element_dtype, list_size, _) => (element_dtype, list_size),
        _ => vortex_bail!(
            "expected FixedSizeList storage dtype, got {}",
            storage_dtype
        ),
    };

    if let &DType::Primitive(ptype, Nullability::NonNullable) = element_dtype.as_ref() {
        Ok((ptype, *fsl_len))
    } else {
        vortex_bail!(
            "expected non-nullable primitive element type, got {}",
            element_dtype
        );
    }
}
