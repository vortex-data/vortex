// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use super::with_selectors;
use crate::dense_union::DenseUnion;
use crate::dense_union::DenseUnionArraySlotsExt;

impl SliceReduce for DenseUnion {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        with_selectors(
            array,
            array.type_ids().slice(range.clone())?,
            array.offsets().slice(range)?,
        )
    }
}
