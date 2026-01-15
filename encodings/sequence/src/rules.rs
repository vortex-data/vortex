// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::SliceVTable;
use vortex_array::matchers::Exact;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;

use crate::SequenceArray;
use crate::SequenceVTable;

pub(super) const RULES: ParentRuleSet<SequenceVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SequenceSliceRule)]);

/// Push slice operations through Sequence encoding.
#[derive(Debug)]
struct SequenceSliceRule;

impl ArrayParentReduceRule<SequenceVTable> for SequenceSliceRule {
    type Parent = Exact<SliceVTable>;

    fn parent(&self) -> Exact<SliceVTable> {
        Exact::from(&SliceVTable)
    }

    fn reduce_parent(
        &self,
        seq: &SequenceArray,
        parent: &SliceArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(SequenceVTable::slice(
            seq,
            parent.slice_range().clone(),
        )))
    }
}
