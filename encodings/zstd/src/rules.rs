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

use crate::ZstdArray;
use crate::ZstdVTable;

pub(super) const RULES: ParentRuleSet<ZstdVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&ZstdSliceRule)]);

/// Push slice operations through Zstd encoding.
#[derive(Debug)]
struct ZstdSliceRule;

impl ArrayParentReduceRule<ZstdVTable> for ZstdSliceRule {
    type Parent = Exact<SliceVTable>;

    fn parent(&self) -> Exact<SliceVTable> {
        Exact::from(&SliceVTable)
    }

    fn reduce_parent(
        &self,
        zstd: &ZstdArray,
        parent: &SliceArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        ZstdVTable::slice(zstd, parent.slice_range().clone())
    }
}
