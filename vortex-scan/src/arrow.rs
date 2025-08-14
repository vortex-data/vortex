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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::cast::AsArray;
    use arrow_array::{
        Array, ArrayRef as ArrowArrayRef, Int32Array, RecordBatch, StringArray, StructArray,
    };
    use arrow_schema::{ArrowError, DataType, Field, Schema};
    use vortex_array::ArrayRef;
    use vortex_array::arrow::FromArrowArray;
    use vortex_error::{VortexResult, vortex_err};

    use super::*;

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
        Ok(ArrayRef::from_arrow(&struct_array, true))
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

        let result = to_record_batch(Ok(vortex_array), &data_type);
        assert!(result.is_ok());

        let batch = result.unwrap();
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
    fn test_record_batch_conversion_error() {
        let error = vortex_err!("test error");
        let data_type = DataType::Struct(create_arrow_schema().fields().clone());

        let result = to_record_batch(Err(error), &data_type);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ArrowError::ExternalError(_)));
    }

    #[test]
    fn test_record_batch_iterator_adapter() {
        let schema = create_arrow_schema();
        let batch1 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![Some(1), Some(2)])) as ArrowArrayRef,
                Arc::new(StringArray::from(vec![Some("Alice"), Some("Bob")])) as ArrowArrayRef,
            ],
        )
        .unwrap();
        let batch2 = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![None, Some(4)])) as ArrowArrayRef,
                Arc::new(StringArray::from(vec![Some("Charlie"), None])) as ArrowArrayRef,
            ],
        )
        .unwrap();

        let iter = vec![Ok(batch1), Ok(batch2)].into_iter();
        let mut adapter = RecordBatchIteratorAdapter {
            iter,
            schema: schema.clone(),
        };

        // Test RecordBatchReader trait
        assert_eq!(adapter.schema(), schema);

        // Test Iterator trait
        let first = adapter.next().unwrap().unwrap();
        assert_eq!(first.num_rows(), 2);

        let second = adapter.next().unwrap().unwrap();
        assert_eq!(second.num_rows(), 2);

        assert!(adapter.next().is_none());
    }

    #[test]
    fn test_error_in_iterator() {
        let schema = create_arrow_schema();
        let error = ArrowError::ComputeError("test error".to_string());

        let iter = vec![Err(error)].into_iter();
        let mut adapter = RecordBatchIteratorAdapter {
            iter,
            schema: schema.clone(),
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
            schema.clone(),
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
