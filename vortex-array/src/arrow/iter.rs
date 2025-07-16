// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{RecordBatch, RecordBatchReader, ffi_stream};
use arrow_schema::{ArrowError, DataType, SchemaRef};
use vortex_dtype::DType;
use vortex_dtype::arrow::FromArrowType;
use vortex_error::{VortexError, VortexResult};

use crate::ArrayRef;
use crate::arrow::FromArrowArray;
use crate::arrow::compute::to_arrow;
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

        Some(batch.map_err(VortexError::from).map(|b| {
            debug_assert_eq!(&self.dtype, &DType::from_arrow(b.schema()));
            ArrayRef::from_arrow(b, false)
        }))
    }
}

/// Adapter for converting a [`ArrayIterator`] into an Arrow [`RecordBatchReader`].
pub struct VortexRecordBatchReader<I> {
    iter: I,
    arrow_schema: SchemaRef,
    arrow_dtype: DataType,
}

impl<I: ArrayIterator> VortexRecordBatchReader<I> {
    pub fn try_new(iter: I) -> VortexResult<Self> {
        let arrow_schema = Arc::new(iter.dtype().to_arrow_schema()?);
        let arrow_dtype = DataType::Struct(arrow_schema.fields().clone());
        Ok(VortexRecordBatchReader {
            iter,
            arrow_schema,
            arrow_dtype,
        })
    }
}

impl<I: ArrayIterator> Iterator for VortexRecordBatchReader<I> {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|result| {
            result
                .and_then(|array| to_arrow(&array, &self.arrow_dtype))
                .map_err(|e| ArrowError::ExternalError(Box::new(e)))
                .map(|array| RecordBatch::from(array.as_struct()))
        })
    }
}

impl<I: ArrayIterator> RecordBatchReader for VortexRecordBatchReader<I> {
    fn schema(&self) -> SchemaRef {
        self.arrow_schema.clone()
    }
}
