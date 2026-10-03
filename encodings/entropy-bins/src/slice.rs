// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::EntropyBins;
use crate::EntropyBinsArrayExt;

impl SliceReduce for EntropyBins {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(
            EntropyBins::try_new(
                array.dtype().clone(),
                array.data().sliced(range.start, range.end),
                array.unsliced_validity(),
            )?
            .into_array(),
        ))
    }
}
