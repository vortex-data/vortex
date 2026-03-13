// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::Constant;
use vortex::array::arrays::ConstantArray;
use vortex::array::arrays::Extension;
use vortex::array::arrays::PrimitiveArray;
use vortex::dtype::DType;
use vortex::dtype::PType;
use vortex::dtype::extension::ExtDTypeRef;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;

/// Extracts the list size from a tensor-like extension dtype.
///
/// The storage dtype must be a `FixedSizeList`.
pub(crate) fn extension_list_size(ext: &ExtDTypeRef) -> VortexResult<usize> {
    let DType::FixedSizeList(_, list_size, _) = ext.storage_dtype() else {
        vortex_bail!(
            "expected FixedSizeList storage dtype, got {}",
            ext.storage_dtype()
        );
    };

    Ok(*list_size as usize)
}

/// Extracts the float element [`PType`] from a tensor-like extension dtype.
///
/// The storage dtype must be a `FixedSizeList` of non-nullable primitives.
pub(crate) fn extension_element_ptype(ext: &ExtDTypeRef) -> VortexResult<PType> {
    let element_dtype = ext
        .storage_dtype()
        .as_fixed_size_list_element_opt()
        .ok_or_else(|| {
            vortex_err!(
                "expected FixedSizeList storage dtype, got {}",
                ext.storage_dtype()
            )
        })?;

    vortex_ensure!(
        !element_dtype.is_nullable(),
        "element dtype must be non-nullable"
    );

    Ok(element_dtype.as_ptype())
}

/// Extracts the storage array from an extension array without canonicalizing.
pub(crate) fn extension_storage(array: &ArrayRef) -> VortexResult<ArrayRef> {
    let ext = array
        .as_opt::<Extension>()
        .ok_or_else(|| vortex_err!("scalar_fn input must be an extension array"))?;

    Ok(ext.storage_array().clone())
}

// TODO(connor): it would be nicer if this took a generic parameter and a FnMut arg that we run
// directly on the values without having to return this ugly stride.
/// Extracts the flat primitive elements from a tensor storage array (FixedSizeList).
///
/// When the input is a [`ConstantArray`] (e.g., a literal query vector), only a single row is
/// materialized to avoid expanding it to the full column length. Returns `(elements, stride)`
/// where `stride` is `list_size` for a full array and `0` for a constant.
pub(crate) fn extract_flat_elements(
    storage: &ArrayRef,
    list_size: usize,
) -> VortexResult<(PrimitiveArray, usize)> {
    if let Some(constant) = storage.as_opt::<Constant>() {
        // Rewrite the array as a length 1 array so when we canonicalize, we do not duplicate a huge
        // amount of data.
        let single = ConstantArray::new(constant.scalar().clone(), 1).into_array();
        let fsl = single.to_canonical()?.into_fixed_size_list();
        let elems = fsl.elements().to_canonical()?.into_primitive();
        return Ok((elems, 0));
    }

    // Otherwise we have to fully expand all of the data.
    let fsl = storage.to_canonical()?.into_fixed_size_list();
    let elems = fsl.elements().to_canonical()?.into_primitive();
    Ok((elems, list_size))
}
