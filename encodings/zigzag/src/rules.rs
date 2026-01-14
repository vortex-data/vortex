// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::SliceVTable;
use vortex_array::matchers::Exact;
use vortex_array::optimizer::rules::ArrayParentReduceRule;
use vortex_array::optimizer::rules::ParentRuleSet;
use vortex_array::vtable::VTable;
use vortex_error::VortexResult;

use crate::ZigZagArray;
use crate::ZigZagVTable;

pub(super) const RULES: ParentRuleSet<ZigZagVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&ZigZagSliceRule)]);

/// A rule to push slice operations through ZigZag encoding.
///
/// Transforms: Slice(ZigZag(encoded)) -> ZigZag(Slice(encoded))
///
/// This works because ZigZag encoding is element-wise, so slicing can be
/// pushed directly to the encoded child array.
#[derive(Debug)]
struct ZigZagSliceRule;

impl ArrayParentReduceRule<ZigZagVTable> for ZigZagSliceRule {
    type Parent = Exact<SliceVTable>;

    fn parent(&self) -> Exact<SliceVTable> {
        // SAFETY: SliceVTable is a valid VTable with a stable ID
        unsafe { Exact::new_unchecked(SliceVTable.id()) }
    }

    fn reduce_parent(
        &self,
        zigzag: &ZigZagArray,
        parent: &SliceArray,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        assert_eq!(child_idx, 0);
        let sliced_encoded = parent.slice_range().clone();
        let new_encoded = zigzag.encoded().slice(sliced_encoded);

        Ok(Some(ZigZagArray::new(new_encoded).into_array()))
    }
}
