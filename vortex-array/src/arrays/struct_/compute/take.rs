// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::Nullability;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::Array;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::StructArray;
use crate::arrays::StructVTable;
use crate::arrays::TakeExecute;
use crate::compute;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

impl TakeExecute for StructVTable {
    fn take(
        array: &StructArray,
        indices: &dyn Array,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // If the struct array is empty then the indices must be all null, otherwise it will access
        // an out of bounds element
        if array.is_empty() {
            let null_fields: Vec<ArrayRef> = array
                .unmasked_fields()
                .iter()
                .map(|field| {
                    ConstantArray::new(
                        Scalar::default_value(field.dtype().clone()),
                        indices.len(),
                    )
                        .into_array()
                })
                .collect();
            return StructArray::try_new_with_dtype(
                null_fields,
                array.struct_fields().clone(),
                indices.len(),
                Validity::AllInvalid,
            )
            .map(StructArray::into_array)
            .map(Some);
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
                .map(|field| field.take(inner_indices.to_array()))
                .collect::<Result<Vec<_>, _>>()?,
            array.struct_fields().clone(),
            indices.len(),
            array.validity().take(indices)?,
        )
        .map(|a| a.into_array())
        .map(Some)
    }
}
