// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_array::RecordBatchReader;
use arrow_array::cast::AsArray;
use arrow_schema::ArrowError;
use arrow_schema::DataType;
use arrow_schema::SchemaRef;
use futures::Stream;
use futures::TryStreamExt;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VortexSessionExecute;
use vortex_array::arrow::ArrowArrayExecutor;
use vortex_error::VortexResult;
use vortex_io::runtime::BlockingRuntime;

use crate::scan::scan_builder::ScanBuilder;

impl ScanBuilder<ArrayRef> {
    /// Creates a new `RecordBatchReader` from the scan builder.
    ///
    /// The `schema` parameter is used to define the schema of the resulting record batches. In
    /// general, it is not possible to exactly infer an Arrow schema from a Vortex
    /// [`vortex_array::dtype::DType`], therefore it is required to be provided explicitly.
    pub fn into_record_batch_reader<B: BlockingRuntime>(
        self,
        schema: SchemaRef,
        runtime: &B,
    ) -> VortexResult<impl RecordBatchReader + 'static> {
        let data_type = DataType::Struct(schema.fields().clone());
        let session = self.session().clone();

        let iter = self
            .map(move |chunk| {
                let mut ctx = session.create_execution_ctx();
                to_record_batch(chunk, &data_type, &mut ctx)
            })
            .into_iter(runtime)?
            .map(|result| result.map_err(|e| ArrowError::ExternalError(Box::new(e))));

        Ok(RecordBatchIteratorAdapter { iter, schema })
    }

    pub fn into_record_batch_stream(
        self,
        schema: SchemaRef,
    ) -> VortexResult<impl Stream<Item = Result<RecordBatch, ArrowError>> + Send + 'static> {
        let data_type = DataType::Struct(schema.fields().clone());
        let session = self.session().clone();

        let stream = self
            .map(move |chunk| {
                let mut ctx = session.create_execution_ctx();
                to_record_batch(chunk, &data_type, &mut ctx)
            })
            .into_stream()?
            .map_err(|e| ArrowError::ExternalError(Box::new(e)));

        Ok(stream)
    }
}

fn to_record_batch(
    chunk: ArrayRef,
    data_type: &DataType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<RecordBatch> {
    let arrow = chunk.execute_arrow(Some(data_type), ctx)?;
    Ok(RecordBatch::from(arrow.as_struct().clone()))
}

/// We create an adapter for record batch iterators that supports clone.
/// This allows us to create thread-safe [`arrow_array::RecordBatchIterator`].
#[derive(Clone)]
pub struct RecordBatchIteratorAdapter<I> {
    iter: I,
    schema: SchemaRef,
}

impl<I> RecordBatchIteratorAdapter<I> {
    /// Creates a new `RecordBatchIteratorAdapter`.
    pub fn new(iter: I, schema: SchemaRef) -> Self {
        Self { iter, schema }
    }
}

impl<I> Iterator for RecordBatchIteratorAdapter<I>
where
    I: Iterator<Item = Result<RecordBatch, ArrowError>>,
{
    type Item = Result<RecordBatch, ArrowError>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<I> RecordBatchReader for RecordBatchIteratorAdapter<I>
where
    I: Iterator<Item = Result<RecordBatch, ArrowError>>,
{
    #[inline]
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::Array;
    use arrow_array::ArrayRef as ArrowArrayRef;
    use arrow_array::Int32Array;
    use arrow_array::RecordBatch;
    use arrow_array::StringArray;
    use arrow_array::StructArray;
    use arrow_array::cast::AsArray;
    use arrow_schema::ArrowError;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use vortex_array::ArrayRef;
    use vortex_array::arrow::FromArrowArray;
    use vortex_error::VortexResult;

    use super::*;
    use crate::scan::test::SCAN_SESSION;

    fn create_test_struct_array() -> VortexResult<ArrayRef> {
        // Create Arrow arrays
        let id_array = Int32Array::from(vec![Some(1), Some(2), None, Some(4)]);
        let name_array = StringArray::from(vec![Some("Alice"), Some("Bob"), Some("Charlie"), None]);

        // Create Arrow struct array
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("name", DataType::Utf8, true),
        ]));

        let struct_array = StructArray::new(
            schema.fields().clone(),
            vec![
                Arc::new(id_array) as ArrowArrayRef,
                Arc::new(name_array) as ArrowArrayRef,
            ],
            None,
        );

        // Convert to Vortex
        ArrayRef::from_arrow(&struct_array, true)
    }

    fn create_arrow_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("name", DataType::Utf8, true),
        ]))
    }

    #[test]
    fn test_record_batch_conversion() -> VortexResult<()> {
        let vortex_array = create_test_struct_array()?;
        let schema = create_arrow_schema();
        let data_type = DataType::Struct(schema.fields().clone());
        let mut ctx = SCAN_SESSION.create_execution_ctx();

        let batch = to_record_batch(vortex_array, &data_type, &mut ctx)?;
        assert_eq!(batch.num_columns(), 2);
        assert_eq!(batch.num_rows(), 4);

        // Check id column
        let id_col = batch
            .column(0)
            .as_primitive::<arrow_array::types::Int32Type>();
        assert_eq!(id_col.value(0), 1);
        assert_eq!(id_col.value(1), 2);
        assert!(id_col.is_null(2));
        assert_eq!(id_col.value(3), 4);

        // Check name column
        let name_col = batch.column(1).as_string::<i32>();
        assert_eq!(name_col.value(0), "Alice");
        assert_eq!(name_col.value(1), "Bob");
        assert_eq!(name_col.value(2), "Charlie");
        assert!(name_col.is_null(3));

        Ok(())
    }

    #[test]
    fn test_record_batch_iterator_adapter() -> VortexResult<()> {
        let schema = create_arrow_schema();
        let batch1 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![Some(1), Some(2)])) as ArrowArrayRef,
                Arc::new(StringArray::from(vec![Some("Alice"), Some("Bob")])) as ArrowArrayRef,
            ],
        )?;
        let batch2 = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![None, Some(4)])) as ArrowArrayRef,
                Arc::new(StringArray::from(vec![Some("Charlie"), None])) as ArrowArrayRef,
            ],
        )?;

        let iter = vec![Ok(batch1), Ok(batch2)].into_iter();
        let mut adapter = RecordBatchIteratorAdapter {
            iter,
            schema: Arc::clone(&schema),
        };

        // Test RecordBatchReader trait
        assert_eq!(adapter.schema(), schema);

        // Test Iterator trait
        let first = adapter.next().unwrap()?;
        assert_eq!(first.num_rows(), 2);

        let second = adapter.next().unwrap()?;
        assert_eq!(second.num_rows(), 2);

        assert!(adapter.next().is_none());

        Ok(())
    }

    #[test]
    fn test_error_in_iterator() {
        let schema = create_arrow_schema();
        let error = ArrowError::ComputeError("test error".to_string());

        let iter = vec![Err(error)].into_iter();
        let mut adapter = RecordBatchIteratorAdapter {
            iter,
            schema: Arc::clone(&schema),
        };

        // Test that error is propagated
        assert_eq!(adapter.schema(), schema);
        let result = adapter.next().unwrap();
        assert!(result.is_err());
    }

    #[test]
    fn test_mixed_success_and_error() {
        let schema = create_arrow_schema();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![Some(1)])) as ArrowArrayRef,
                Arc::new(StringArray::from(vec![Some("Test")])) as ArrowArrayRef,
            ],
        )
        .unwrap();

        let error = ArrowError::ComputeError("test error".to_string());

        let iter = vec![Ok(batch.clone()), Err(error), Ok(batch)].into_iter();
        let mut adapter = RecordBatchIteratorAdapter { iter, schema };

        // First batch succeeds
        let first = adapter.next().unwrap();
        assert!(first.is_ok());

        // Second batch errors
        let second = adapter.next().unwrap();
        assert!(second.is_err());

        // Third batch succeeds
        let third = adapter.next().unwrap();
        assert!(third.is_ok());
    }
}
