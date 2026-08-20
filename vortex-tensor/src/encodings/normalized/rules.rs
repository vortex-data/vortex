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

        // SAFETY: Slicing every slot with the same range preserves their dtypes and lengths.
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

        // SAFETY: Filtering every slot with the same mask preserves their dtypes and lengths.
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
