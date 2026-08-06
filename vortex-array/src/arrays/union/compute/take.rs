// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Union;
use crate::arrays::UnionArray;
use crate::arrays::dict::TakeReduce;
use crate::arrays::union::UnionArrayExt;
use crate::arrays::union::UnionArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::scalar::Scalar;

/// Structural take for [`UnionArray`]: gathers the type IDs and every sparse child at `indices`.
///
/// A sparse union keeps every child row-aligned with the union, so a gather must visit all of them
/// even though at most one is active per row. Take therefore costs `O(variants * indices)`.
/// Reducing that cost requires the dense union encoding, not a different sparse gather.
///
/// The type IDs carry the union's validity, so gathering them with the original `indices` is what
/// turns a null index into an outer union null. The children are gathered with the null indices
/// filled in, which keeps each child's dtype exactly as the variant schema declares it.
impl TakeReduce for Union {
    fn take(array: ArrayView<'_, Union>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // An empty union has no row for a child to point at, so the only legal indices are all
        // null and every output row is an outer union null.
        if array.is_empty() {
            return UnionArray::constant(&Scalar::null(array.dtype().as_nullable()), indices.len())
                .map(UnionArray::into_array)
                .map(Some);
        }

        let type_ids = array.type_ids().take(indices.clone())?;

        // Nullability is stripped so that the children keep their declared variant dtypes. The
        // type IDs already record which rows are null.
        //
        // This stays a lazy node that every child then executes for itself, so the fill runs once
        // per variant. `TakeReduce` has no `ExecutionCtx` to materialize it with, and the cost is
        // proportional to the indices rather than to the data, so it is left alone. Non-nullable
        // indices skip it entirely because `FillNull::simplify` returns its input unchanged.
        let fill_scalar = Scalar::zero_value(&indices.dtype().as_nonnullable());
        let child_indices = indices.clone().fill_null(fill_scalar)?;

        let children: Vec<ArrayRef> = array
            .iter_children()
            .map(|child| child.take(child_indices.clone()))
            .try_collect()?;

        UnionArray::try_new(type_ids, array.variants().clone(), children)
            .map(UnionArray::into_array)
            .map(Some)
    }
}
