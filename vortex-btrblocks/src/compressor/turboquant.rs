// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Specialized compressor for TurboQuant vector quantization of tensor extension types.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::matcher::Matcher;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_turboquant::FIXED_SHAPE_TENSOR_EXT_ID;
use vortex_turboquant::TurboQuantConfig;
use vortex_turboquant::TurboQuantMSE;
use vortex_turboquant::TurboQuantMSEArray;
use vortex_turboquant::TurboQuantQJL;
use vortex_turboquant::TurboQuantQJLArray;
use vortex_turboquant::VECTOR_EXT_ID;
use vortex_turboquant::turboquant_encode_qjl;

/// Check if an extension array has a tensor extension type.
pub(crate) fn is_tensor_extension(ext_array: &ExtensionArray) -> bool {
    let ext_id = ext_array.ext_dtype().id();
    ext_id.as_ref() == VECTOR_EXT_ID || ext_id.as_ref() == FIXED_SHAPE_TENSOR_EXT_ID
}

/// Try to compress a tensor extension array using TurboQuant.
///
/// Returns `Ok(Some(...))` on success, or `Ok(None)` if the storage is nullable
/// (TurboQuant requires non-nullable input). The caller should fall through to
/// default compression when `None` is returned.
///
/// Produces a `TurboQuantQJLArray` wrapping a `TurboQuantMSEArray`, stored inside
/// the Extension wrapper. The MSE codes child is bitpacked for storage efficiency.
pub(crate) fn compress_turboquant(
    ext_array: &ExtensionArray,
    config: &TurboQuantConfig,
) -> VortexResult<Option<ArrayRef>> {
    let storage = ext_array.storage_array();
    let fsl = storage.to_canonical()?.into_fixed_size_list();

    if fsl.dtype().is_nullable() {
        return Ok(None);
    }
    if fsl.is_empty() {
        return Ok(None);
    }

    // Produce the cascaded QJL(MSE) structure.
    let encoded = turboquant_encode_qjl(&fsl, config)?;
    let encoded = encoded.as_opt::<TurboQuantQJL>().expect("encoded should be a QJL array");

    // Bitpack the MSE codes child for storage efficiency.
    let encoded = bitpack_mse_codes(encoded)?;

    Ok(Some(
        ExtensionArray::new(ext_array.ext_dtype().clone(), encoded).into_array(),
    ))
}

/// Bitpack the codes child of the MSE array within a QJL array.
///
/// The encode functions produce raw `PrimitiveArray<u8>` codes. This function
/// applies bitpacking to compress them based on the MSE bit_width.
fn bitpack_mse_codes(qjl: &TurboQuantQJLArray) -> VortexResult<ArrayRef> {
    // If this is a QJL array, descend into its MSE inner child.
        let mse_inner = qjl.mse_inner().as_opt::<TurboQuantMSE>().vortex_expect("mse_inner should be a TurboQuantMSE array");
        let bit_width = mse.bit_width();
        if bit_width < 8 {
            let codes_prim: PrimitiveArray = mse.codes().to_canonical()?.into_primitive();
            let packed = bitpack_encode(&codes_prim, bit_width, None)?.into_array();
            let new_mse = TurboQuantMSEArray::try_new(
                mse.dtype().clone(),
                packed,
                mse.norms().clone(),
                mse.centroids().clone(),
                mse.rotation_signs().clone(),
                mse.dimension(),
                bit_width,
                mse.padded_dim(),
                mse.rotation_seed(),
            );
        return Ok(TurboQuantQJLArray::try_new(
            qjl.dtype().clone(),
            new_mse,
            qjl.qjl_signs().clone(),
            qjl.residual_norms().clone(),
            qjl.rotation_signs().clone(),
            qjl.bit_width(),
            qjl.padded_dim(),
        )?
        .into_array());
        }

    // No bitpacking needed (8-bit codes or unrecognized array).
    Ok(array.clone())
}
