// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Selector-only take for dense unions.
//!
//! For performance, take gathers only the row selectors and retains every compact child in full.
//! This is O(selected rows) but may reorder per-child offsets; Arrow dense unions instead require
//! those offsets to increase.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::dict::TakeReduce;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::dense_union::DenseUnion;
use crate::dense_union::DenseUnionArrayExt;
use crate::dense_union::DenseUnionArraySlotsExt;

impl TakeReduce for DenseUnion {
    fn take(array: ArrayView<'_, Self>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let type_ids = array.type_ids().take(indices.clone())?;
        let fill_scalar = Scalar::zero_value(&indices.dtype().as_nonnullable());
        let offset_indices = indices.clone().fill_null(fill_scalar)?;
        let offsets = array.offsets().take(offset_indices)?;

        DenseUnion::try_new(
            type_ids,
            offsets,
            array.variants().clone(),
            array.iter_children().cloned(),
        )
        .map(|array| Some(array.into_array()))
    }
}
