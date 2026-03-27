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

/// Compress a tensor extension array using TurboQuant.
///
/// Produces a `TurboQuantQJLArray` wrapping a `TurboQuantMSEArray`, stored inside
/// the Extension wrapper. All children (codes, norms, centroids, rotation signs,
/// QJL signs, residual norms) are left for the standard BtrBlocks recursive
/// compression pipeline to handle during layout serialization.
pub(crate) fn compress_turboquant(
    ext_array: &ExtensionArray,
    config: &TurboQuantConfig,
) -> VortexResult<ArrayRef> {
    let storage = ext_array.storage_array();
    let fsl = storage.to_canonical()?.into_fixed_size_list();

    // Produce the cascaded QJL(MSE) structure. The layout writer will
    // recursively descend into children and compress each one.
    let qjl_array = turboquant_encode_qjl(&fsl, config)?;

    Ok(ExtensionArray::new(ext_array.ext_dtype().clone(), qjl_array.into_array()).into_array())
}
