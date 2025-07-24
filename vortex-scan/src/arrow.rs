// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::cast::AsArray;
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, DataType, SchemaRef};
use vortex_array::ArrayRef;
use vortex_array::arrow::IntoArrowArray;
use vortex_error::{VortexError, VortexResult};

use crate::ScanBuilder;

impl ScanBuilder<ArrayRef> {
    /// Creates a new thread-safe `RecordBatchReader` from the scan builder.
    ///
    /// This reader can be cloned and passed to multiple threads for concurrent processing.
    ///
    /// The `schema` parameter is used to define the schema of the resulting record batches. In
    /// general, it is not possible to exactly infer an Arrow schema from a Vortex
    /// [`vortex_dtype::DType`], therefore it is required to be provided explicitly.
    pub fn into_record_batch_reader(
        self,
        schema: SchemaRef,
    ) -> VortexResult<impl RecordBatchReader + Send + Clone + 'static> {
        let data_type = DataType::Struct(schema.fields().clone());

        let iter = self
            .into_array_iter()?
            .map(move |chunk| to_record_batch(chunk, &data_type));

        Ok(RecordBatchIteratorAdapter { iter, schema })
    }

    /// Creates a new `RecordBatchReader` from the scan builder that internally drives the scan
    /// on multithreaded pool of workers.
    #[cfg(feature = "tokio")]
    pub fn into_record_batch_reader_multithread(
        self,
        schema: SchemaRef,
    ) -> VortexResult<impl RecordBatchReader + Send + 'static> {
        use arrow_array::RecordBatchIterator;

        let data_type = DataType::Struct(schema.fields().clone());

        let iter = self.into_iter_multithread(move |chunk| to_record_batch(chunk, &data_type))?;

        Ok(RecordBatchIterator::new(iter, schema))
    }
}

fn to_record_batch(
    chunk: VortexResult<ArrayRef>,
    data_type: &DataType,
) -> Result<RecordBatch, ArrowError> {
    chunk
        .and_then(|array| {
            let arrow = array.into_arrow(data_type)?;
            Ok::<_, VortexError>(RecordBatch::from(arrow.as_struct().clone()))
        })
        .map_err(|e| ArrowError::ExternalError(Box::new(e)))
}

/// We create an adapter for record batch iterators that supports clone.
/// This allows us to create thread-safe [`RecordBatchIterator`].
#[derive(Clone)]
struct RecordBatchIteratorAdapter<I> {
    iter: I,
    schema: SchemaRef,
}

impl<I> Iterator for RecordBatchIteratorAdapter<I>
where
    I: Iterator<Item = Result<RecordBatch, ArrowError>>,
{
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<I> RecordBatchReader for RecordBatchIteratorAdapter<I>
where
    I: Iterator<Item = Result<RecordBatch, ArrowError>>,
{
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
