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

use crate::DeltaArray;
use crate::DeltaVTable;

pub(super) const RULES: ParentRuleSet<DeltaVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&DeltaSliceRule)]);

/// Push slice operations through Delta encoding.
#[derive(Debug)]
struct DeltaSliceRule;

impl ArrayParentReduceRule<DeltaVTable> for DeltaSliceRule {
    type Parent = Exact<SliceVTable>;

    fn parent(&self) -> Exact<SliceVTable> {
        Exact::from(&SliceVTable)
    }

    fn reduce_parent(
        &self,
        delta: &DeltaArray,
        parent: &SliceArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(DeltaVTable::slice(
            delta,
            parent.slice_range().clone(),
        )))
    }
}
