// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::ffi_stream;
use vortex_error::VortexError;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::arrow::FromArrowArray;
use crate::dtype::DType;
use crate::dtype::arrow::FromArrowType;
use crate::iter::ArrayIterator;

/// An adapter for converting an `ArrowArrayStreamReader` into a Vortex `ArrayStream`.
pub struct ArrowArrayStreamAdapter {
    stream: ffi_stream::ArrowArrayStreamReader,
    dtype: DType,
}

impl ArrowArrayStreamAdapter {
    pub fn new(stream: ffi_stream::ArrowArrayStreamReader, dtype: DType) -> Self {
        Self { stream, dtype }
    }
}

impl ArrayIterator for ArrowArrayStreamAdapter {
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl Iterator for ArrowArrayStreamAdapter {
    type Item = VortexResult<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = self.stream.next()?;

        Some(batch.map_err(VortexError::from).and_then(|b| {
            debug_assert_eq!(&self.dtype, &DType::from_arrow(b.schema()));
            ArrayRef::from_arrow(b, false)
        }))
    }
}
