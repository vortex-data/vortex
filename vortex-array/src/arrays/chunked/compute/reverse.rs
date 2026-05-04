// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray as _;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::chunked::ChunkedArrayExt as _;
use crate::arrays::reversed::ReverseReduce;

/// Reverses a `ChunkedArray` by reversing the chunk order and lazily reversing each chunk.
///
/// Transforms `Reversed(Chunked([c0, c1, …, cn]))` into
/// `Chunked([reverse(cn), …, reverse(c1), reverse(c0)])`.
///
/// This avoids eagerly merging all chunks into a single canonical array before reversing.
/// Each per-chunk `reverse()` call goes through the optimizer, so further reduce rules
/// (e.g. `Dict` codes-only reversal) still fire on individual chunks.
impl ReverseReduce for Chunked {
    fn reverse(array: ArrayView<'_, Self>) -> VortexResult<Option<ArrayRef>> {
        let dtype = array.as_ref().dtype().clone();
        let reversed_chunks = array
            .chunks()
            .into_iter()
            .rev()
            .map(|chunk| chunk.reverse())
            .collect::<VortexResult<Vec<ArrayRef>>>()?;
        // SAFETY: all chunks come from the original ChunkedArray and share its DType;
        // reversing order and wrapping in Reversed preserves the invariant.
        Ok(Some(
            unsafe { ChunkedArray::new_unchecked(reversed_chunks, dtype) }.into_array(),
        ))
    }
}
