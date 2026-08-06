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

/// Gathers the type IDs and every sparse child at `indices`.
///
/// Sparse children are row-aligned with the union, so a gather must visit all of them. Take costs
/// `O(variants * indices)`, which only the dense encoding fixes.
///
/// The type IDs carry the union's validity, so gathering them with the original `indices` turns a
/// null index into an outer union null. The children are gathered with the nulls filled in, which
/// keeps their declared variant dtypes.
impl TakeReduce for Union {
    fn take(array: ArrayView<'_, Union>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        // An empty union has no row to point at, so the indices must be all null.
        if array.is_empty() {
            return UnionArray::constant(&Scalar::null(array.dtype().as_nullable()), indices.len())
                .map(UnionArray::into_array)
                .map(Some);
        }

        let type_ids = array.type_ids().take(indices.clone())?;

        // This stays a lazy node, so the fill runs once per child. `TakeReduce` has no
        // `ExecutionCtx` to materialize it with, and the cost is per index rather than per element.
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
