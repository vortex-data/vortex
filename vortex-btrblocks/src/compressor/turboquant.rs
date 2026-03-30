// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Specialized compressor for TurboQuant vector quantization of tensor extension types.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_fastlanes::bitpack_compress::bitpack_encode;
use vortex_turboquant::FIXED_SHAPE_TENSOR_EXT_ID;
use vortex_turboquant::TurboQuant;
use vortex_turboquant::TurboQuantArray;
use vortex_turboquant::TurboQuantConfig;
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
/// Produces a `TurboQuantArray` with QJL correction, stored inside the Extension
/// wrapper. The MSE codes child is bitpacked for storage efficiency.
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

    // Produce the TurboQuant array with QJL correction.
    let encoded_ref = turboquant_encode_qjl(&fsl, config)?;
    let encoded = encoded_ref
        .as_opt::<TurboQuant>()
        .vortex_expect("encoded should be a TurboQuantArray");

    // Bitpack the codes child for storage efficiency.
    let result = bitpack_codes(encoded)?;

    Ok(Some(
        ExtensionArray::new(ext_array.ext_dtype().clone(), result).into_array(),
    ))
}

/// Bitpack the codes child of a TurboQuant array.
///
/// The encode functions produce raw `PrimitiveArray<u8>` codes. This function
/// applies bitpacking to compress them based on the bit_width.
fn bitpack_codes(array: &TurboQuantArray) -> VortexResult<ArrayRef> {
    let bit_width = array.bit_width();

    if bit_width >= 8 {
        // 8-bit codes are stored as raw u8, no bitpacking needed.
        return Ok(array.clone().into_array());
    }

    let codes_prim: PrimitiveArray = array.codes().to_canonical()?.into_primitive();
    let packed = bitpack_encode(&codes_prim, bit_width, None)?.into_array();

    // Rebuild the array with the bitpacked codes.
    let rebuilt = if let Some(qjl) = array.qjl() {
        TurboQuantArray::try_new_qjl(
            array.dtype().clone(),
            packed,
            array.norms().clone(),
            array.centroids().clone(),
            array.rotation_signs().clone(),
            qjl.clone(),
            array.dimension(),
            bit_width,
        )?
    } else {
        TurboQuantArray::try_new_mse(
            array.dtype().clone(),
            packed,
            array.norms().clone(),
            array.centroids().clone(),
            array.rotation_signs().clone(),
            array.dimension(),
            bit_width,
        )?
    };

    Ok(rebuilt.into_array())
}
