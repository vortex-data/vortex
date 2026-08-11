// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::ffi_stream;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::iter::ArrayIterator;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::FromArrowArray;
use crate::dtype::from_arrow_schema_naive;

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
            debug_assert_eq!(
                &self.dtype,
                &from_arrow_schema_naive(b.schema().as_ref())
                    .vortex_expect("arrow schema to dtype")
            );
            ArrayRef::from_arrow(b, false)
        }))
    }
}
