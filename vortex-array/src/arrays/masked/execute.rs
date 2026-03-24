// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for MaskedArray - applies a validity mask to canonical arrays.

use std::ops::BitAnd;

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Canonical;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::DecimalArray;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListViewArray;
use crate::arrays::NullArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::dtype::Nullability;
use crate::executor::ExecutionCtx;
use crate::match_each_decimal_value_type;
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
        Canonical::Bool(a) => Canonical::Bool(mask_validity_bool(a, validity_mask, ctx)?),
        Canonical::Primitive(a) => {
            Canonical::Primitive(mask_validity_primitive(a, validity_mask, ctx)?)
        }
        Canonical::Decimal(a) => Canonical::Decimal(mask_validity_decimal(a, validity_mask, ctx)?),
        Canonical::VarBinView(a) => {
            Canonical::VarBinView(mask_validity_varbinview(a, validity_mask, ctx)?)
        }
        Canonical::List(a) => Canonical::List(mask_validity_listview(a, validity_mask, ctx)?),
        Canonical::FixedSizeList(a) => {
            Canonical::FixedSizeList(mask_validity_fixed_size_list(a, validity_mask, ctx)?)
        }
        Canonical::Struct(a) => Canonical::Struct(mask_validity_struct(a, validity_mask, ctx)?),
        Canonical::Extension(a) => {
            Canonical::Extension(mask_validity_extension(a, validity_mask, ctx)?)
        }
    })
}

fn combine_validity(
    validity: &Validity,
    mask: &Mask,
    len: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Validity> {
    let current_mask = validity.execute_mask(len, ctx)?;
    let combined = current_mask.bitand(mask);
    Ok(Validity::from_mask(combined, Nullability::Nullable))
}

fn mask_validity_null(array: NullArray, _mask: &Mask) -> NullArray {
    array
}

fn mask_validity_bool(
    array: BoolArray,
    mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<BoolArray> {
    let len = array.len();
    let new_validity = combine_validity(array.validity(), mask, len, ctx)?;
    Ok(BoolArray::new(array.to_bit_buffer(), new_validity))
}

fn mask_validity_primitive(
    array: PrimitiveArray,
    mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let len = array.len();
    let ptype = array.ptype();
    let new_validity = combine_validity(array.validity(), mask, len, ctx)?;
    // SAFETY: validity has same length as values
    Ok(unsafe {
        PrimitiveArray::new_unchecked_from_handle(
            array.buffer_handle().clone(),
            ptype,
            new_validity,
        )
    })
}

fn mask_validity_decimal(
    array: DecimalArray,
    mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<DecimalArray> {
    let len = array.len();
    let dec_dtype = array.decimal_dtype();
    let values_type = array.values_type();
    let new_validity = combine_validity(array.validity(), mask, len, ctx)?;
    // SAFETY: We're only changing validity, not the data structure
    Ok(match_each_decimal_value_type!(values_type, |T| {
        let buffer = array.buffer::<T>();
        unsafe { DecimalArray::new_unchecked(buffer, dec_dtype, new_validity) }
    }))
}

/// Mask validity for VarBinViewArray.
fn mask_validity_varbinview(
    array: VarBinViewArray,
    mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<VarBinViewArray> {
    let len = array.len();
    let dtype = array.dtype().as_nullable();
    let new_validity = combine_validity(array.validity(), mask, len, ctx)?;
    // SAFETY: We're only changing validity, not the data structure
    Ok(unsafe {
        VarBinViewArray::new_handle_unchecked(
            array.views_handle().clone(),
            array.buffers().clone(),
            dtype,
            new_validity,
        )
    })
}

fn mask_validity_listview(
    array: ListViewArray,
    mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ListViewArray> {
    let len = array.len();
    let new_validity = combine_validity(array.validity(), mask, len, ctx)?;
    // SAFETY: We're only changing validity, not the data structure
    Ok(unsafe {
        ListViewArray::new_unchecked(
            array.elements().clone(),
            array.offsets().clone(),
            array.sizes().clone(),
            new_validity,
        )
    })
}

fn mask_validity_fixed_size_list(
    array: FixedSizeListArray,
    mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<FixedSizeListArray> {
    let len = array.len();
    let list_size = array.list_size();
    let new_validity = combine_validity(array.validity(), mask, len, ctx)?;
    // SAFETY: We're only changing validity, not the data structure
    Ok(unsafe {
        FixedSizeListArray::new_unchecked(array.elements().clone(), list_size, new_validity, len)
    })
}

fn mask_validity_struct(
    array: StructArray,
    mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<StructArray> {
    let len = array.len();
    let new_validity = combine_validity(array.validity(), mask, len, ctx)?;
    let fields = array.unmasked_fields().clone();
    let struct_fields = array.struct_fields().clone();
    // SAFETY: We're only changing validity, not the data structure
    Ok(unsafe { StructArray::new_unchecked(fields, struct_fields, len, new_validity) })
}

fn mask_validity_extension(
    array: ExtensionArray,
    mask: &Mask,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ExtensionArray> {
    // For extension arrays, we need to mask the underlying storage
    let storage = array.storage_array().clone().execute::<Canonical>(ctx)?;
    let masked_storage = mask_validity_canonical(storage, mask, ctx)?;
    let masked_storage = masked_storage.into_array();
    Ok(ExtensionArray::new(
        array
            .ext_dtype()
            .with_nullability(masked_storage.dtype().nullability()),
        masked_storage,
    ))
}
