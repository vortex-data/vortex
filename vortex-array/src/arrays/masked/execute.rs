// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for MaskedArray - applies a validity mask to canonical arrays.

use std::ops::BitAnd;

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Canonical;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::BoolArrayParts;
use crate::arrays::DecimalArray;
use crate::arrays::DecimalArrayParts;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListViewArray;
use crate::arrays::ListViewArrayParts;
use crate::arrays::NullArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveArrayParts;
use crate::arrays::StructArray;
use crate::arrays::StructArrayParts;
use crate::arrays::VarBinViewArray;
use crate::arrays::VarBinViewArrayParts;
use crate::dtype::Nullability;
use crate::executor::ExecutionCtx;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

/// TODO: replace usage of compute fn.
/// Apply a validity mask to a canonical array, ANDing with existing validity.
///
/// This is the core operation for MaskedArray execution - it intersects the child's
/// validity with the provided mask, marking additional positions as invalid.
pub fn mask_validity_canonical(
    canonical: Canonical,
    validity_mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Canonical> {
    Ok(match canonical {
        Canonical::Null(a) => Canonical::Null(mask_validity_null(a, validity_mask)),
        Canonical::Bool(a) => Canonical::Bool(mask_validity_bool(a, validity_mask)),
        Canonical::Primitive(a) => Canonical::Primitive(mask_validity_primitive(a, validity_mask)),
        Canonical::Decimal(a) => Canonical::Decimal(mask_validity_decimal(a, validity_mask)),
        Canonical::VarBinView(a) => {
            Canonical::VarBinView(mask_validity_varbinview(a, validity_mask))
        }
        Canonical::List(a) => Canonical::List(mask_validity_listview(a, validity_mask)),
        Canonical::FixedSizeList(a) => {
            Canonical::FixedSizeList(mask_validity_fixed_size_list(a, validity_mask))
        }
        Canonical::Struct(a) => Canonical::Struct(mask_validity_struct(a, validity_mask)),
        Canonical::Extension(a) => {
            Canonical::Extension(mask_validity_extension(&a, validity_mask, ctx)?)
        }
    })
}

fn combine_validity(validity: &Validity, mask: &Mask, len: usize) -> Validity {
    let current_mask = validity.to_mask(len);
    let combined = current_mask.bitand(mask);
    Validity::from_mask(combined, Nullability::Nullable)
}

fn mask_validity_null(array: NullArray, _mask: &Mask) -> NullArray {
    array
}

fn mask_validity_bool(array: BoolArray, mask: &Mask) -> BoolArray {
    let new_validity = combine_validity(array.validity(), mask, array.len());
    let BoolArrayParts {
        bits,
        offset,
        len,
        validity: _,
    } = array.into_parts();
    BoolArray::new_handle(bits, offset, len, new_validity)
}

fn mask_validity_primitive(array: PrimitiveArray, mask: &Mask) -> PrimitiveArray {
    let new_validity = combine_validity(array.validity(), mask, array.len());
    let PrimitiveArrayParts {
        ptype,
        buffer,
        validity: _,
    } = array.into_parts();
    // SAFETY: validity has same length as values
    unsafe { PrimitiveArray::new_unchecked_from_handle(buffer, ptype, new_validity) }
}

fn mask_validity_decimal(array: DecimalArray, mask: &Mask) -> DecimalArray {
    let new_validity = combine_validity(array.validity(), mask, array.len());
    let DecimalArrayParts {
        decimal_dtype,
        values,
        values_type,
        validity: _,
    } = array.into_parts();
    // SAFETY: We're only changing validity, not the data structure
    unsafe { DecimalArray::new_unchecked_handle(values, values_type, decimal_dtype, new_validity) }
}

/// Mask validity for VarBinViewArray.
fn mask_validity_varbinview(array: VarBinViewArray, mask: &Mask) -> VarBinViewArray {
    let new_validity = combine_validity(array.validity(), mask, array.len());
    let VarBinViewArrayParts {
        dtype,
        buffers,
        views,
        validity: _,
    } = array.into_parts();
    // SAFETY: We're only changing validity, not the data structure
    unsafe {
        VarBinViewArray::new_handle_unchecked(views, buffers, dtype.as_nullable(), new_validity)
    }
}

fn mask_validity_listview(array: ListViewArray, mask: &Mask) -> ListViewArray {
    let new_validity = combine_validity(array.validity(), mask, array.len());
    let ListViewArrayParts {
        elements_dtype: _,
        elements,
        offsets,
        sizes,
        validity: _,
    } = array.into_parts();
    // SAFETY: We're only changing validity, not the data structure
    unsafe { ListViewArray::new_unchecked(elements, offsets, sizes, new_validity) }
}

fn mask_validity_fixed_size_list(array: FixedSizeListArray, mask: &Mask) -> FixedSizeListArray {
    let len = array.len();
    let list_size = array.list_size();
    let new_validity = combine_validity(array.validity(), mask, len);
    let (elements, ..) = array.into_parts();
    // SAFETY: We're only changing validity, not the data structure
    unsafe { FixedSizeListArray::new_unchecked(elements, list_size, new_validity, len) }
}

fn mask_validity_struct(array: StructArray, mask: &Mask) -> StructArray {
    let len = array.len();
    let new_validity = combine_validity(array.validity(), mask, len);
    let StructArrayParts {
        struct_fields,
        fields,
        validity: _,
    } = array.into_parts();
    // SAFETY: We're only changing validity, not the data structure
    unsafe { StructArray::new_unchecked(fields, struct_fields, len, new_validity) }
}

fn mask_validity_extension(
    array: &ExtensionArray,
    mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExtensionArray> {
    // For extension arrays, we need to mask the underlying storage
    let storage = array.storage().clone().execute::<Canonical>(ctx)?;
    let masked_storage = mask_validity_canonical(storage, mask, ctx)?;
    let masked_storage = masked_storage.into_array();
    Ok(ExtensionArray::new(
        array
            .ext_dtype()
            .with_nullability(masked_storage.dtype().nullability()),
        masked_storage,
    ))
}
