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

use crate::encodings::l2_denorm::L2Denorm;
use crate::encodings::l2_denorm::array::L2DenormArraySlotsExt;

pub(super) const RULES: ParentRuleSet<L2Denorm> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&L2DenormSliceRule),
    ParentRuleSet::lift(&L2DenormFilterRule),
]);

/// Pushes a slice through the encoding into both children.
///
/// The norm split is row-wise, so any row subset of an [`L2Denorm`] array is itself a valid
/// [`L2Denorm`] array. Rewriting the slice as two child slices keeps the column encoded instead of
/// canonicalizing it just to throw most of the rows away.
#[derive(Debug)]
struct L2DenormSliceRule;

impl ArrayParentReduceRule<L2Denorm> for L2DenormSliceRule {
    type Parent = Slice;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, L2Denorm>,
        parent: ArrayView<'_, Slice>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let range = parent.slice_range();

        // SAFETY: Slicing both children preserves their structure.
        Ok(Some(
            unsafe {
                L2Denorm::new_unchecked(
                    array.normalized().slice(range.clone())?,
                    array.norms().slice(range.clone())?,
                )
            }
            .into_array(),
        ))
    }
}

/// Pushes a filter through the encoding into both children.
///
/// Same row-wise argument as [`L2DenormSliceRule`]. Unlike the generic scalar-function push-down,
/// this always fires: both children are physically per-row, so filtering them is strictly less
/// work than reconstructing the tensor column and filtering that.
#[derive(Debug)]
struct L2DenormFilterRule;

impl ArrayParentReduceRule<L2Denorm> for L2DenormFilterRule {
    type Parent = Filter;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, L2Denorm>,
        parent: ArrayView<'_, Filter>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let mask = parent.filter_mask();

        // SAFETY: Filtering both children with the same mask preserves their structure.
        Ok(Some(
            unsafe {
                L2Denorm::new_unchecked(
                    array.normalized().filter(mask.clone())?,
                    array.norms().filter(mask.clone())?,
                )
            }
            .into_array(),
        ))
    }
}
