// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::DictArray;
use crate::arrays::DictVTable;
use crate::arrays::SliceReduce;

impl SliceReduce for DictVTable {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let sliced_code = array.codes().slice(range)?;
        /// TODO(joe): if the range is size 1 replace with a constant array
        // SAFETY: slicing the codes preserves invariants.
        Ok(Some(
            unsafe { DictArray::new_unchecked(sliced_code, array.values().clone()) }.into_array(),
        ))
    }
}
