// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Specialized compressor for TurboQuant vector quantization of tensor extension types.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::DynArray;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_turboquant::TurboQuantConfig;
use vortex_turboquant::turboquant_encode;

use crate::BtrBlocksCompressor;
use crate::CanonicalCompressor;
use crate::CompressorContext;
use crate::Excludes;

/// Extension IDs for tensor types (from vortex-tensor).
const VECTOR_EXT_ID: &str = "vortex.tensor.vector";
const FIXED_SHAPE_TENSOR_EXT_ID: &str = "vortex.tensor.fixed_shape_tensor";

/// Check if an extension array has a tensor extension type.
pub(crate) fn is_tensor_extension(ext_array: &ExtensionArray) -> bool {
    let ext_id = ext_array.ext_dtype().id();
    ext_id.as_ref() == VECTOR_EXT_ID || ext_id.as_ref() == FIXED_SHAPE_TENSOR_EXT_ID
}

/// Compress a tensor extension array using TurboQuant.
///
/// Applies TurboQuant encoding to the FixedSizeList storage, then recursively
/// compresses each child (codes, norms, etc.) via the BtrBlocks compressor.
pub(crate) fn compress_turboquant(
    compressor: &BtrBlocksCompressor,
    ext_array: &ExtensionArray,
    config: &TurboQuantConfig,
) -> VortexResult<ArrayRef> {
    let storage = ext_array.storage_array();
    let fsl = storage.to_canonical()?.into_fixed_size_list();
    let tq_array = turboquant_encode(&fsl, config)?;

    let ctx = CompressorContext::default().descend();

    // Recursively compress each child via the standard BtrBlocks pipeline.
    let compressed_codes =
        compressor.compress_canonical(tq_array.codes().to_canonical()?, ctx, Excludes::none())?;
    let compressed_norms = compressor.compress_canonical(
        Canonical::Primitive(tq_array.norms().to_canonical()?.into_primitive()),
        ctx,
        Excludes::none(),
    )?;

    let compressed_tq = match tq_array.variant() {
        vortex_turboquant::TurboQuantVariant::Mse => {
            vortex_turboquant::TurboQuantArray::try_new_mse(
                fsl.dtype().clone(),
                compressed_codes,
                compressed_norms,
                tq_array.dimension(),
                tq_array.bit_width(),
                tq_array.rotation_seed(),
            )?
        }
        vortex_turboquant::TurboQuantVariant::Prod => {
            let compressed_qjl = compressor.compress_canonical(
                tq_array
                    .qjl_signs()
                    .vortex_expect("Prod variant must have qjl_signs")
                    .to_canonical()?,
                ctx,
                Excludes::none(),
            )?;
            let compressed_res_norms = compressor.compress_canonical(
                Canonical::Primitive(
                    tq_array
                        .residual_norms()
                        .vortex_expect("Prod variant must have residual_norms")
                        .to_canonical()?
                        .into_primitive(),
                ),
                ctx,
                Excludes::none(),
            )?;

            vortex_turboquant::TurboQuantArray::try_new_prod(
                fsl.dtype().clone(),
                compressed_codes,
                compressed_norms,
                compressed_qjl,
                compressed_res_norms,
                tq_array.dimension(),
                tq_array.bit_width(),
                tq_array.rotation_seed(),
            )?
        }
    };

    Ok(ExtensionArray::new(ext_array.ext_dtype().clone(), compressed_tq.into_array()).into_array())
}
