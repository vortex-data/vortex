// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Specialized compressor for TurboQuant vector quantization of tensor extension types.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_error::VortexResult;
use vortex_turboquant::FIXED_SHAPE_TENSOR_EXT_ID;
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
/// wrapper. The per-row children (codes, QJL signs) are `FixedSizeListArray`s
/// whose inner elements will be cascading-compressed by the layout writer.
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

    let encoded = turboquant_encode_qjl(&fsl, config)?;

    Ok(Some(
        ExtensionArray::new(ext_array.ext_dtype().clone(), encoded).into_array(),
    ))
}
