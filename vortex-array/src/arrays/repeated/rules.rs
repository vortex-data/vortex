// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ChunkedArray;
use crate::arrays::Repeated;
use crate::arrays::RepeatedArray;
use crate::arrays::Slice;
use crate::arrays::repeated::RepeatedArraySlotsExt;
use crate::arrays::slice::SliceReduce;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;

pub(super) fn register_parent_rules(session: &VortexSession) {
    session.kernels().register_reduce_parent_rule(
        Slice,
        Repeated,
        const { &SliceReduceAdaptor(Repeated) },
    );
}

impl SliceReduce for Repeated {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        // The range is non-empty, so the child is too.
        let child = array.child();
        let child_len = child.len();
        let rotation = range.start % child_len;

        // A range inside one copy is a slice of the child.
        if rotation + range.len() <= child_len {
            return Ok(Some(child.slice(rotation..rotation + range.len())?));
        }

        // Any other range repeats the child rotated to start where the range does.
        let rotated = if rotation == 0 {
            child.clone()
        } else {
            ChunkedArray::try_new(
                [child.slice(rotation..child_len)?, child.slice(0..rotation)?],
                child.dtype().clone(),
            )?
            .into_array()
        };
        Ok(Some(
            RepeatedArray::try_new(rotated, range.len())?.into_array(),
        ))
    }
}
