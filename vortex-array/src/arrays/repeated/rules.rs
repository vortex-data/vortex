// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ChunkedArray;
use crate::arrays::Repeated;
use crate::arrays::RepeatedArray;
use crate::arrays::repeated::RepeatedArraySlotsExt;
use crate::arrays::slice::SliceReduce;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ParentRuleSet;

pub(super) const PARENT_RULES: ParentRuleSet<Repeated> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(Repeated))]);

impl SliceReduce for Repeated {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // The range is non-empty, so the child is too.
        let child = array.child();
        let child_len = child.len();
        let first_copy = range.start / child_len;
        let offset = first_copy * child_len;

        // A range inside one copy is a slice of the child.
        if range.end - offset <= child_len {
            return Ok(Some(child.slice(range.start - offset..range.end - offset)?));
        }

        // A range whose length is a whole number of copies repeats the child rotated to start where
        // the range does.
        if range.len().is_multiple_of(child_len) {
            let rotation = range.start - offset;
            let rotated = if rotation == 0 {
                child.clone()
            } else {
                ChunkedArray::try_new(
                    [child.slice(rotation..child_len)?, child.slice(0..rotation)?],
                    child.dtype().clone(),
                )?
                .into_array()
            };
            let count = range.len() / child_len;
            return Ok(Some(RepeatedArray::try_new(rotated, count)?.into_array()));
        }

        Ok(None)
    }
}
