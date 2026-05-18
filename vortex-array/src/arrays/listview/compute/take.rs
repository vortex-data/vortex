// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use num_traits::Zero;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::dict::TakeReduce;
use crate::arrays::listview::ListViewArrayExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::Nullability;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;

/// Metadata-only take for [`ListViewArray`].
///
/// Always succeeds: we filter `offsets`/`sizes`/`validity` and keep the original `elements`
/// buffer. The result carries a `reachable_elements_bound` (sum of kept sizes when cheap to
/// compute) which the export-time prune helper consults to decide whether to compact.
impl TakeReduce for ListView {
    fn take(array: ArrayView<'_, ListView>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(apply_take(array, indices)?.into_array()))
    }
}

/// Execution-path take for [`ListViewArray`].
///
/// Identical to [`TakeReduce`]: metadata-only. Compaction is deferred to the operator tree's
/// root (the export / canonicalisation site), which consults the propagated
/// `reachable_elements_bound` to decide whether to commit to a rebuild. Eagerly rebuilding
/// here duplicates that work whenever a downstream `take` / `filter` would narrow further.
impl TakeExecute for ListView {
    fn take(
        array: ArrayView<'_, ListView>,
        indices: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(apply_take(array, indices)?.into_array()))
    }
}

/// Shared metadata-only take: take `offsets`, `sizes` and `validity` at `indices` while reusing
/// the original `elements` buffer as-is.
fn apply_take(array: ArrayView<'_, ListView>, indices: &ArrayRef) -> VortexResult<ListViewArray> {
    let elements = array.elements();
    let offsets = array.offsets();
    let sizes = array.sizes();

    // Combine the array's validity with the indices' validity.
    let new_validity = array.validity()?.take(indices)?;

    // Take can reorder offsets, create gaps, and may introduce overlaps if `indices` contain
    // duplicates.
    let nullable_new_offsets = offsets.take(indices.clone())?;
    let nullable_new_sizes = sizes.take(indices.clone())?;

    // `take` returns nullable arrays; cast back to non-nullable (filling with zeros to represent
    // the null lists — the validity mask tracks nullness separately).
    let new_offsets = match_each_integer_ptype!(nullable_new_offsets.dtype().as_ptype(), |O| {
        nullable_new_offsets.fill_null(Scalar::primitive(O::zero(), Nullability::NonNullable))?
    });
    let new_sizes = match_each_integer_ptype!(nullable_new_sizes.dtype().as_ptype(), |S| {
        nullable_new_sizes.fill_null(Scalar::primitive(S::zero(), Nullability::NonNullable))?
    });

    // SAFETY: Take operation maintains all `ListViewArray` invariants:
    // - `new_offsets` and `new_sizes` are derived from existing valid child arrays.
    // - `new_offsets` and `new_sizes` are non-nullable.
    // - `new_offsets` and `new_sizes` have the same length (both taken with the same `indices`).
    // - Validity correctly reflects the combination of array and indices validity.
    let result = unsafe {
        ListViewArray::new_unchecked(elements.clone(), new_offsets, new_sizes, new_validity)
    };
    // Stamp an upper bound on reachable elements: it can't exceed the sum of the new sizes
    // (overlaps among `indices` can shrink the true count further, never grow it). We compute
    // this opportunistically — if the new sizes already canonicalise as a host primitive we
    // sum them in O(kept_rows); otherwise we leave the bound unset.
    let bound = sum_sizes_if_cheap(result.sizes());
    Ok(result.with_reachable_elements_bound(bound))
}

/// Best-effort sum-of-sizes computation. Returns `Some(sum)` only when the sizes array is
/// already a host primitive — we never force an execute purely to refine the hint.
pub(crate) fn sum_sizes_if_cheap(sizes: &ArrayRef) -> Option<u64> {
    if !sizes.is_host() {
        return None;
    }
    let prim = sizes.as_opt::<crate::arrays::Primitive>()?;
    let s: u64 = match_each_integer_ptype!(prim.ptype(), |S| {
        prim.as_slice::<S>()
            .iter()
            .map(|s| AsPrimitive::<u64>::as_(*s))
            .sum()
    });
    Some(s)
}
