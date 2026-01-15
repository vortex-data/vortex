// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::SliceVTable;
use vortex_array::matchers::Exact;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::vtable::VTable;
use vortex_error::VortexResult;

use crate::ALPArray;
use crate::ALPVTable;

pub(super) const RULES: ParentRuleSet<ALPVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&ALPSliceRule)]);

/// Push slice operations through ALP encoding.
#[derive(Debug)]
struct ALPSliceRule;

impl ArrayParentReduceRule<ALPVTable> for ALPSliceRule {
    type Parent = Exact<SliceVTable>;

    fn parent(&self) -> Exact<SliceVTable> {
        Exact::from(&SliceVTable)
    }

    fn reduce_parent(
        &self,
        alp: &ALPArray,
        parent: &SliceArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        ALPVTable::slice(alp, parent.slice_range().clone())
    }
}
