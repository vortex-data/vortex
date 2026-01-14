// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::SliceVTable;
use vortex_array::matchers::Exact;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_error::VortexResult;

use crate::BitPackedArray;
use crate::BitPackedVTable;

pub(super) const RULES: ParentRuleSet<BitPackedVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&BitPackedSliceRule)]);

/// A rule to push slice operations through BitPacked encoding.
/// This delegates to the BitPackedVTable's slice operation which handles
/// the complex offset and packed bits calculations.
#[derive(Debug)]
struct BitPackedSliceRule;

impl ArrayParentReduceRule<BitPackedVTable> for BitPackedSliceRule {
    type Parent = Exact<SliceVTable>;

    fn parent(&self) -> Exact<SliceVTable> {
        // SAFETY: SliceVTable is a valid VTable with a stable ID
        unsafe { Exact::new_unchecked(SliceVTable.id()) }
    }

    fn reduce_parent(
        &self,
        bitpacked: &BitPackedArray,
        parent: &SliceArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(BitPackedVTable::slice(
            bitpacked,
            parent.slice_range().clone(),
        )))
    }
}
