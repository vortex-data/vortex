// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::Filter;
use vortex_array::arrays::Slice;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_error::VortexResult;

use crate::encodings::normalized::Normalized;
use crate::encodings::normalized::array::NormalizedArraySlotsExt;

pub(super) const RULES: ParentRuleSet<Normalized> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&NormalizedSliceRule),
    ParentRuleSet::lift(&NormalizedFilterRule),
]);

/// Pushes a slice into the encoded slots.
///
/// The norm split is row-wise, so any row subset of a [`Normalized`] array is itself a valid
/// [`Normalized`] array. Rewriting the slice as two child slices keeps the column encoded instead of
/// canonicalizing it just to throw most of the rows away.
#[derive(Debug)]
struct NormalizedSliceRule;

impl ArrayParentReduceRule<Normalized> for NormalizedSliceRule {
    type Parent = Slice;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Normalized>,
        parent: ArrayView<'_, Slice>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let range = parent.slice_range();

        // SAFETY: Slicing both children and the validity preserves their structure.
        Ok(Some(
            unsafe {
                Normalized::new_unchecked(
                    array.normalized().slice(range.clone())?,
                    array.norms().slice(range.clone())?,
                    array.validity()?.slice(range.clone())?,
                )
            }
            .into_array(),
        ))
    }
}

/// Pushes a filter into the encoded slots.
///
/// Same row-wise argument as [`NormalizedSliceRule`]. Unlike the generic scalar-function push-down,
/// this always fires: both children are physically per-row, so filtering them is strictly less
/// work than reconstructing the tensor column and filtering that.
#[derive(Debug)]
struct NormalizedFilterRule;

impl ArrayParentReduceRule<Normalized> for NormalizedFilterRule {
    type Parent = Filter;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Normalized>,
        parent: ArrayView<'_, Filter>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let mask = parent.filter_mask();

        // SAFETY: Filtering both children and the validity with the same mask preserves their
        // structure.
        Ok(Some(
            unsafe {
                Normalized::new_unchecked(
                    array.normalized().filter(mask.clone())?,
                    array.norms().filter(mask.clone())?,
                    array.validity()?.filter(mask)?,
                )
            }
            .into_array(),
        ))
    }
}
