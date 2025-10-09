// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use itertools::Itertools;
use vortex_scalar::Scalar;

use crate::arrays::struct_::{
    StructArray,
    StructVTable,
};
use crate::vtable::{
    OperationsVTable,
    ValidityHelper,
};
use crate::{
    ArrayRef,
    IntoArray,
};

impl OperationsVTable<StructVTable> for StructVTable {
    fn slice(array: &StructArray, range: Range<usize>) -> ArrayRef {
        let fields = array
            .fields()
            .iter()
            .map(|field| field.slice(range.clone()))
            .collect_vec();
        // SAFETY: All invariants are preserved:
        // - fields.len() == dtype.names().len() (same struct fields)
        // - Every field has length == range.len() (all sliced to same range)
        // - Each field's dtype matches the struct dtype (unchanged from original)
        // - Validity length matches array length (both sliced to same range)
        unsafe {
            StructArray::new_unchecked(
                fields,
                array.struct_fields().clone(),
                range.len(),
                array.validity().slice(range),
            )
        }
        .into_array()
    }

    fn scalar_at(array: &StructArray, index: usize) -> Scalar {
        Scalar::struct_(
            array.dtype().clone(),
            array
                .fields()
                .iter()
                .map(|field| field.scalar_at(index))
                .collect_vec(),
        )
    }
}
