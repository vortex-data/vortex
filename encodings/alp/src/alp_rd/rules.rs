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

use crate::ALPRDArray;
use crate::ALPRDVTable;

pub(super) const RULES: ParentRuleSet<ALPRDVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&ALPRDSliceRule)]);

/// Push slice operations through ALP-RD encoding.
#[derive(Debug)]
struct ALPRDSliceRule;

impl ArrayParentReduceRule<ALPRDVTable> for ALPRDSliceRule {
    type Parent = Exact<SliceVTable>;

    fn parent(&self) -> Exact<SliceVTable> {
        Exact::from(&SliceVTable)
    }

    fn reduce_parent(
        &self,
        alprd: &ALPRDArray,
        parent: &SliceArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(ALPRDVTable::slice(
            alprd,
            parent.slice_range().clone(),
        )))
    }
}
