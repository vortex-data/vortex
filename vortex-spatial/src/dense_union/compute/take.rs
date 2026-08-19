// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::arrays::dict::TakeReduce;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use super::with_selectors;
use crate::dense_union::DenseUnion;
use crate::dense_union::DenseUnionArraySlotsExt;

impl TakeReduce for DenseUnion {
    fn take(array: ArrayView<'_, Self>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // A null index makes the row null through the type IDs, so the offset it gathers is never
        // read; gather offset zero rather than propagating the null into a non-nullable array.
        let fill_scalar = Scalar::zero_value(&indices.dtype().as_nonnullable());
        let offset_indices = indices.clone().fill_null(fill_scalar)?;

        with_selectors(
            array,
            array.type_ids().take(indices.clone())?,
            array.offsets().take(offset_indices)?,
        )
    }
}
