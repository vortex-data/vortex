// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::Nullability;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::StructArray;
use crate::arrays::StructVTable;
use crate::compute::TakeKernel;
use crate::compute::TakeKernelAdapter;
use crate::compute::{self};
use crate::register_kernel;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl TakeKernel for StructVTable {
    fn take(&self, array: &StructArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        // If the struct array is empty then the indices must be all null, otherwise it will access
        // an out of bounds element
        if array.is_empty() {
            return StructArray::try_new_with_dtype(
                array.unmasked_fields().clone(),
                array.struct_fields().clone(),
                indices.len(),
                Validity::AllInvalid,
            )
            .map(StructArray::into_array);
        }
        // The validity is applied to the struct validity,
        let inner_indices = &compute::fill_null(
            indices,
            &Scalar::default_value(indices.dtype().with_nullability(Nullability::NonNullable)),
        )?;
        StructArray::try_new_with_dtype(
            array
                .unmasked_fields()
                .iter()
                .map(|field| compute::take(field, inner_indices))
                .collect::<Result<Vec<_>, _>>()?,
            array.struct_fields().clone(),
            indices.len(),
            array.validity().take(indices)?,
        )
        .map(|a| a.into_array())
    }
}

register_kernel!(TakeKernelAdapter(StructVTable).lift());
