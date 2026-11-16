// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Tiled;
use crate::arrays::TiledArray;
use crate::arrays::slice::SliceReduce;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::arrays::tiled::TiledArraySlotsExt;
use crate::optimizer::rules::ParentRuleSet;

pub(super) const PARENT_RULES: ParentRuleSet<Tiled> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&SliceReduceAdaptor(Tiled))]);

impl SliceReduce for Tiled {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // The range is non-empty, so the child is too.
        let child = array.child();
        let child_len = child.len();
        let first_tile = range.start / child_len;
        let offset = first_tile * child_len;

        // A range inside one tile is a slice of the child.
        if range.end - offset <= child_len {
            return Ok(Some(child.slice(range.start - offset..range.end - offset)?));
        }

        // A range of whole tiles is a shorter tiling of the child.
        if range.start % child_len == 0 && range.end % child_len == 0 {
            let count = range.len() / child_len;
            return Ok(Some(TiledArray::try_new(child.clone(), count)?.into_array()));
        }

        Ok(None)
    }
}
