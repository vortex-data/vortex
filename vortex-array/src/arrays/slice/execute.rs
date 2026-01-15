// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for SliceArray - slices canonical arrays by a range.

use std::ops::Range;

use itertools::Itertools;
use vortex_dtype::match_each_decimal_value_type;
use vortex_dtype::match_each_native_ptype;

use crate::Array;
use crate::Canonical;
use crate::arrays::BoolArray;
use crate::arrays::DecimalArray;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListViewArray;
use crate::arrays::NullArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::VarBinViewArray;
use crate::vtable::ValidityHelper;

/// Slice a canonical array by a range, returning a new canonical array.
pub fn slice_canonical(canonical: Canonical, range: Range<usize>) -> Canonical {
    match canonical {
        Canonical::Null(a) => Canonical::Null(slice_null(&a, range)),
        Canonical::Bool(a) => Canonical::Bool(slice_bool(&a, range)),
        Canonical::Primitive(a) => Canonical::Primitive(slice_primitive(&a, range)),
        Canonical::Decimal(a) => Canonical::Decimal(slice_decimal(&a, range)),
        Canonical::VarBinView(a) => Canonical::VarBinView(slice_varbinview(&a, range)),
        Canonical::List(a) => Canonical::List(slice_listview(&a, range)),
        Canonical::FixedSizeList(a) => Canonical::FixedSizeList(slice_fixed_size_list(&a, range)),
        Canonical::Struct(a) => Canonical::Struct(slice_struct(&a, range)),
        Canonical::Extension(a) => Canonical::Extension(slice_extension(&a, range)),
    }
}

fn slice_null(_array: &NullArray, range: Range<usize>) -> NullArray {
    NullArray::new(range.len())
}

fn slice_bool(array: &BoolArray, range: Range<usize>) -> BoolArray {
    BoolArray::from_bit_buffer(
        array.bit_buffer().slice(range.clone()),
        array.validity().slice(range),
    )
}

fn slice_primitive(array: &PrimitiveArray, range: Range<usize>) -> PrimitiveArray {
    match_each_native_ptype!(array.ptype(), |T| {
        PrimitiveArray::from_buffer_handle(
            array.buffer_handle().slice_typed::<T>(range.clone()),
            array.ptype(),
            array.validity().slice(range),
        )
    })
}

fn slice_decimal(array: &DecimalArray, range: Range<usize>) -> DecimalArray {
    match_each_decimal_value_type!(array.values_type(), |D| {
        let sliced = array.buffer::<D>().slice(range.clone());
        let validity = array.validity().clone().slice(range);
        // SAFETY: Slicing preserves all DecimalArray invariants
        unsafe { DecimalArray::new_unchecked(sliced, array.decimal_dtype(), validity) }
    })
}

fn slice_varbinview(array: &VarBinViewArray, range: Range<usize>) -> VarBinViewArray {
    VarBinViewArray::new(
        array.views().slice(range.clone()),
        array.buffers().clone(),
        array.dtype().clone(),
        array.validity().slice(range),
    )
}

fn slice_listview(array: &ListViewArray, range: Range<usize>) -> ListViewArray {
    // SAFETY: Slicing the components of an existing valid array is still valid.
    unsafe {
        ListViewArray::new_unchecked(
            array.elements().clone(),
            array.offsets().slice(range.clone()),
            array.sizes().slice(range.clone()),
            array.validity().slice(range),
        )
        .with_zero_copy_to_list(array.is_zero_copy_to_list())
    }
}

fn slice_fixed_size_list(array: &FixedSizeListArray, range: Range<usize>) -> FixedSizeListArray {
    let new_len = range.len();
    let list_size = array.list_size() as usize;

    // SAFETY: Slicing preserves FixedSizeListArray invariants
    unsafe {
        FixedSizeListArray::new_unchecked(
            array
                .elements()
                .slice(range.start * list_size..range.end * list_size),
            array.list_size(),
            array.validity().slice(range),
            new_len,
        )
    }
}

fn slice_struct(array: &StructArray, range: Range<usize>) -> StructArray {
    let fields = array
        .fields()
        .iter()
        .map(|field| field.slice(range.clone()))
        .collect_vec();

    // SAFETY: Slicing preserves all StructArray invariants
    unsafe {
        StructArray::new_unchecked(
            fields,
            array.struct_fields().clone(),
            range.len(),
            array.validity().slice(range),
        )
    }
}

fn slice_extension(array: &ExtensionArray, range: Range<usize>) -> ExtensionArray {
    ExtensionArray::new(array.ext_dtype().clone(), array.storage().slice(range))
}
