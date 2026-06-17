// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::dict::TakeReduce;
use crate::arrays::struct_::StructArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::scalar::Scalar;

impl TakeReduce for Struct {
    fn take(array: ArrayView<'_, Struct>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // If the struct array is empty then the indices must be all null, otherwise it will access
        // an out of bounds element. The result is therefore an all-null struct.
        if array.is_empty() {
            return Ok(Some(
                ConstantArray::null(array.dtype().as_nullable(), indices.len()).into_array(),
            ));
        }

        // TODO(connor): This could be bad for cache locality...

        // Fill null indices with zero so they point at a valid row.
        // Note that we strip nullability so that `Take::return_dtype` doesn't union nullable into
        // each field's dtype (the struct-level validity already captures which rows are null).
        let fill_scalar = Scalar::zero_value(&indices.dtype().as_nonnullable());
        let inner_indices = indices.clone().fill_null(fill_scalar)?;

        StructArray::try_new_with_dtype(
            array
                .iter_unmasked_fields()
                .map(|field| field.take(inner_indices.clone()))
                .collect::<Result<Vec<_>, _>>()?,
            array.struct_fields().clone(),
            indices.len(),
            array.validity()?.take(indices)?,
        )
        .map(|a| a.into_array())
        .map(Some)
    }
}
