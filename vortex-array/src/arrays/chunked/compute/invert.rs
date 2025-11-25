// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ChunkedVTable;
use crate::compute::InvertKernel;
use crate::compute::InvertKernelAdapter;
use crate::compute::invert;
use crate::register_kernel;

impl InvertKernel for ChunkedVTable {
    fn invert(&self, array: &ChunkedArray) -> VortexResult<ArrayRef> {
        let chunks = array.chunks().iter().map(|c| invert(c)).try_collect()?;
        // SAFETY: Invert operation preserves the dtype of each chunk.
        // All inverted chunks maintain the same dtype as the original array.
        unsafe { Ok(ChunkedArray::new_unchecked(chunks, array.dtype().clone()).into_array()) }
    }
}

register_kernel!(InvertKernelAdapter(ChunkedVTable).lift());
