// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for MaskedArray - applies a validity mask to canonical arrays.

use std::ops::BitAnd;

use vortex_dtype::Nullability;
use vortex_dtype::match_each_decimal_value_type;
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
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

/// TODO: replace usage of compute fn.
/// Apply a validity mask to a canonical array, ANDing with existing validity.
///
/// This is the core operation for MaskedArray execution - it intersects the child's
/// validity with the provided mask, marking additional positions as invalid.
pub fn mask_validity_canonical(canonical: Canonical, validity_mask: &Mask) -> Canonical {
    match canonical {
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
        Canonical::Extension(a) => Canonical::Extension(mask_validity_extension(a, validity_mask)),
    }
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
    let len = array.len();
    let new_validity = combine_validity(array.validity(), mask, len);
    BoolArray::new(array.bit_buffer().clone(), new_validity)
}

fn mask_validity_primitive(array: PrimitiveArray, mask: &Mask) -> PrimitiveArray {
    let len = array.len();
    let ptype = array.ptype();
    let new_validity = combine_validity(array.validity(), mask, len);
    PrimitiveArray::from_byte_buffer(array.into_byte_buffer(), ptype, new_validity)
}

fn mask_validity_decimal(array: DecimalArray, mask: &Mask) -> DecimalArray {
    let len = array.len();
    let dec_dtype = array.decimal_dtype();
    let values_type = array.values_type();
    let new_validity = combine_validity(array.validity(), mask, len);
    // SAFETY: We're only changing validity, not the data structure
    match_each_decimal_value_type!(values_type, |T| {
        let buffer = array.buffer::<T>();
        unsafe { DecimalArray::new_unchecked(buffer, dec_dtype, new_validity) }
    })
}

/// Mask validity for VarBinViewArray.
fn mask_validity_varbinview(array: VarBinViewArray, mask: &Mask) -> VarBinViewArray {
    let len = array.len();
    let dtype = array.dtype().clone();
    let new_validity = combine_validity(array.validity(), mask, len);
    // SAFETY: We're only changing validity, not the data structure
    unsafe {
        VarBinViewArray::new_unchecked(
            array.views().clone(),
            array.buffers().clone(),
            dtype,
            new_validity,
        )
    }
}

fn mask_validity_listview(array: ListViewArray, mask: &Mask) -> ListViewArray {
    let len = array.len();
    let new_validity = combine_validity(array.validity(), mask, len);
    // SAFETY: We're only changing validity, not the data structure
    unsafe {
        ListViewArray::new_unchecked(
            array.elements().clone(),
            array.offsets().clone(),
            array.sizes().clone(),
            new_validity,
        )
    }
}

fn mask_validity_fixed_size_list(array: FixedSizeListArray, mask: &Mask) -> FixedSizeListArray {
    let len = array.len();
    let list_size = array.list_size();
    let new_validity = combine_validity(array.validity(), mask, len);
    // SAFETY: We're only changing validity, not the data structure
    unsafe {
        FixedSizeListArray::new_unchecked(array.elements().clone(), list_size, new_validity, len)
    }
}

fn mask_validity_struct(array: StructArray, mask: &Mask) -> StructArray {
    let len = array.len();
    let new_validity = combine_validity(array.validity(), mask, len);
    let fields = array.fields().clone();
    let struct_fields = array.struct_fields().clone();
    // SAFETY: We're only changing validity, not the data structure
    unsafe { StructArray::new_unchecked(fields, struct_fields, len, new_validity) }
}

fn mask_validity_extension(array: ExtensionArray, mask: &Mask) -> ExtensionArray {
    // For extension arrays, we need to mask the underlying storage
    let storage = array.storage().to_canonical();
    let masked_storage = mask_validity_canonical(storage, mask);
    ExtensionArray::new(array.ext_dtype().clone(), masked_storage.into_array())
}
