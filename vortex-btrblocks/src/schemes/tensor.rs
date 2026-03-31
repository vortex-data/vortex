// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant compression scheme for tensor extension types (Vector, FixedShapeTensor).

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::CanonicalValidity;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::ExtensionArray;
use vortex_error::VortexResult;
use vortex_tensor::encodings::turboquant::FIXED_SHAPE_TENSOR_EXT_ID;
use vortex_tensor::encodings::turboquant::TurboQuantConfig;
use vortex_tensor::encodings::turboquant::VECTOR_EXT_ID;
use vortex_tensor::encodings::turboquant::turboquant_encode_qjl;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;

/// TurboQuant compression scheme for tensor extension types.
///
/// Applies lossy vector quantization to `Vector` and `FixedShapeTensor` extension
/// arrays using the TurboQuant algorithm with QJL correction for unbiased inner
/// product estimation.
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

        let ext_id = ext.ext_dtype().id();
        let is_tensor =
            ext_id.as_ref() == VECTOR_EXT_ID || ext_id.as_ref() == FIXED_SHAPE_TENSOR_EXT_ID;

        // TurboQuant requires non-nullable storage.
        is_tensor && !ext.storage_array().dtype().is_nullable()
    }

    fn expected_compression_ratio(
        &self,
        _compressor: &CascadingCompressor,
        _data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<f64> {
        // TurboQuant at 5-bit MSE + QJL ≈ 5x compression from f32.
        // Return a high ratio to prefer this for tensor data.
        Ok(f64::MAX)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &mut ArrayAndStats,
        _ctx: CompressorContext,
    ) -> VortexResult<ArrayRef> {
        let array = data.array().clone();
        let ext_array = array.to_extension();
        let storage = ext_array.storage_array();
        let fsl = storage.to_canonical()?.into_fixed_size_list();

        let config = TurboQuantConfig::default();
        let encoded = turboquant_encode_qjl(&fsl, &config)?;

        Ok(ExtensionArray::new(ext_array.ext_dtype().clone(), encoded).into_array())
    }
}
