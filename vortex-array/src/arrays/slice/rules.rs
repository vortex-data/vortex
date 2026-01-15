// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::arrays::slice::SliceArray;
use crate::arrays::slice::SliceVTable;
use crate::optimizer::rules::ArrayReduceRule;
use crate::optimizer::rules::ReduceRuleSet;

pub(super) const RULES: ReduceRuleSet<SliceVTable> = ReduceRuleSet::new(&[&SliceVTableRule]);

/// Generic reduce rule that calls VTable::slice on the child.
/// This allows all encodings to implement their own slice logic.
#[derive(Debug)]
struct SliceVTableRule;

impl ArrayReduceRule<SliceVTable> for SliceVTableRule {
    fn reduce(&self, array: &SliceArray) -> VortexResult<Option<ArrayRef>> {
        // Try the child's VTable::slice implementation
        array
            .child()
            .encoding()
            .slice(array.child(), array.slice_range().clone())
    }
}
