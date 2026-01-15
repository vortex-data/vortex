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

use crate::ByteBoolArray;
use crate::ByteBoolVTable;

pub(super) const RULES: ParentRuleSet<ByteBoolVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&ByteBoolSliceRule)]);

/// Push slice operations through ByteBool encoding.
#[derive(Debug)]
struct ByteBoolSliceRule;

impl ArrayParentReduceRule<ByteBoolVTable> for ByteBoolSliceRule {
    type Parent = Exact<SliceVTable>;

    fn parent(&self) -> Exact<SliceVTable> {
        Exact::from(&SliceVTable)
    }

    fn reduce_parent(
        &self,
        bytebool: &ByteBoolArray,
        parent: &SliceArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        ByteBoolVTable::slice(bytebool, parent.slice_range().clone())
    }
}
