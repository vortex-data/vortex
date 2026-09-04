// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::DecimalByteParts;
use crate::decimal_byte_parts::map_parts;

impl SliceReduce for DecimalByteParts {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        map_parts(array, |part| part.slice(range.clone())).map(|d| Some(d.into_array()))
    }
}
