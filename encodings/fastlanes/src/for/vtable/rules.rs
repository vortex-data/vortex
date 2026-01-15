// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::FilterVTable;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::SliceVTable;
use vortex_array::matchers::Exact;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_error::VortexResult;

use crate::FoRArray;
use crate::FoRVTable;

pub(super) const PARENT_RULES: ParentRuleSet<FoRVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&FoRSlicePushDownRule),
    ParentRuleSet::lift(&FoRFilterPushDownRule),
]);

/// Push slice operations through FoR encoding.
#[derive(Debug)]
struct FoRSlicePushDownRule;

impl ArrayParentReduceRule<FoRVTable> for FoRSlicePushDownRule {
    type Parent = Exact<SliceVTable>;

    fn parent(&self) -> Exact<SliceVTable> {
        // SAFETY: SliceVTable is a valid VTable with a stable ID
        unsafe { Exact::new_unchecked(SliceVTable.id()) }
    }

    fn reduce_parent(
        &self,
        for_arr: &FoRArray,
        parent: &SliceArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(FoRVTable::slice(
            for_arr,
            parent.slice_range().clone(),
        )))
    }
}

#[derive(Debug)]
struct FoRFilterPushDownRule;

impl ArrayParentReduceRule<FoRVTable> for FoRFilterPushDownRule {
    type Parent = Exact<FilterVTable>;

    fn parent(&self) -> Self::Parent {
        Exact::from(&FilterVTable)
    }

    fn reduce_parent(
        &self,
        child: &FoRArray,
        parent: &FilterArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let new_array = unsafe {
            FoRArray::new_unchecked(
                child.encoded.filter(parent.filter_mask().clone())?,
                child.reference.clone(),
            )
        };
        Ok(Some(new_array.into_array()))
    }
}
