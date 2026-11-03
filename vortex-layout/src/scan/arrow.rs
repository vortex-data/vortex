// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future;
use std::iter;
use std::sync::Arc;

use arrow_array::Array;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::RecordBatch;
use arrow_array::RecordBatchOptions;
use arrow_array::RecordBatchReader;
use arrow_array::cast::AsArray;
use arrow_schema::ArrowError;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::FieldRef;
use arrow_schema::Fields;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use futures::FutureExt;
use futures::Stream;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use futures::future::join_all;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::Struct;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructDataParts;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_io::runtime::BlockingRuntime;
use vortex_io::runtime::Handle;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

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
        let struct_field = Field::new_struct("", schema.fields().clone(), false);
        let session = self.session().clone();

        let iter = self
            .map(move |chunk| {
                let mut ctx = session.create_execution_ctx();
                to_record_batch(chunk, &struct_field, &mut ctx)
            })
            .into_iter(runtime)?
            .map(|result| result.map_err(|e| ArrowError::ExternalError(Box::new(e))));

        Ok(RecordBatchIteratorAdapter { iter, schema })
    }

    pub fn into_record_batch_stream(
        self,
        schema: SchemaRef,
    ) -> VortexResult<impl Stream<Item = Result<RecordBatch, ArrowError>> + Send + 'static> {
        let struct_field = Field::new_struct("", schema.fields().clone(), false);
        let session = self.session().clone();

        let stream = self
            .map(move |chunk| record_batch_future(chunk, &struct_field, &session))
            .into_stream()?
            .and_then(|batch| batch)
            .map_err(|e| ArrowError::ExternalError(Box::new(e)));

        Ok(stream)
    }
}

pub(super) fn to_record_batch(
    chunk: ArrayRef,
    data_type: &Field,
    ctx: &mut ExecutionCtx,
) -> VortexResult<RecordBatch> {
    let session = ctx.session().clone();
    let arrow = session.arrow().execute_arrow(chunk, Some(data_type), ctx)?;
    Ok(RecordBatch::from(arrow.as_struct().clone()))
}

/// The number of top-level columns a batch needs before its columns are exported in parallel.
const PARALLEL_EXPORT_COLUMNS: usize = 128;

/// The minimum number of columns exported by each runtime task of a parallel export.
const MIN_COLUMNS_PER_TASK: usize = 32;

/// Start converting `chunk` into a [`RecordBatch`].
///
/// A wide struct without nulls has its columns exported in groups spawned on the session runtime,
/// so that exporting, and dropping, thousands of columns does not run on a single thread. Every
/// other chunk is converted in place.
fn record_batch_future(
    chunk: ArrayRef,
    struct_field: &Field,
    session: &VortexSession,
) -> VortexResult<BoxFuture<'static, VortexResult<RecordBatch>>> {
    let DataType::Struct(target_fields) = struct_field.data_type() else {
        vortex_bail!("Record batch target must be a struct, got {struct_field}");
    };
    let handle = (target_fields.len() >= PARALLEL_EXPORT_COLUMNS)
        .then(|| session.try_handle())
        .flatten();
    let chunk = match (handle, chunk.try_downcast::<Struct>()) {
        (Some(handle), Ok(array))
            if matches!(
                array.validity()?,
                Validity::NonNullable | Validity::AllValid
            ) =>
        {
            return Ok(parallel_record_batch(
                array,
                target_fields,
                &handle,
                session,
            ));
        }
        (_, Ok(array)) => array.into_array(),
        (_, Err(chunk)) => chunk,
    };

    let mut ctx = session.create_execution_ctx();
    Ok(future::ready(to_record_batch(chunk, struct_field, &mut ctx)).boxed())
}

/// Export the columns of a struct without nulls in groups spawned on the runtime.
fn parallel_record_batch(
    array: StructArray,
    target_fields: &Fields,
    handle: &Handle,
    session: &VortexSession,
) -> BoxFuture<'static, VortexResult<RecordBatch>> {
    let len = array.len();
    let StructDataParts { fields, .. } = array.into_data_parts();
    if fields.len() != target_fields.len() {
        return future::ready(Err(vortex_err!(
            "StructArray field count {} must match the target Arrow type {}",
            fields.len(),
            target_fields.len()
        )))
        .boxed();
    }

    let parallelism = get_available_parallelism().unwrap_or(1);
    let ntasks = (fields.len() / MIN_COLUMNS_PER_TASK).clamp(1, parallelism * 4);
    let group_size = fields.len().div_ceil(ntasks);

    let mut columns = fields.into_iter().zip(target_fields.iter().cloned());
    let tasks: Vec<_> = iter::from_fn(|| {
        let group: Vec<_> = columns.by_ref().take(group_size).collect();
        (!group.is_empty()).then(|| {
            let session = session.clone();
            handle.spawn_cpu(move || export_columns(group, &session))
        })
    })
    .collect();

    let schema = Arc::new(Schema::new(target_fields.clone()));
    async move {
        let mut columns = Vec::with_capacity(schema.fields().len());
        for group in join_all(tasks).await {
            columns.extend(group?);
        }
        Ok(RecordBatch::try_new_with_options(
            schema,
            columns,
            &RecordBatchOptions::new().with_row_count(Some(len)),
        )?)
    }
    .boxed()
}

/// Export each column to its target Arrow field.
fn export_columns(
    columns: Vec<(ArrayRef, FieldRef)>,
    session: &VortexSession,
) -> VortexResult<Vec<ArrowArrayRef>> {
    let mut ctx = session.create_execution_ctx();
    columns
        .into_iter()
        .map(|(column, field)| {
            let arrow = session
                .arrow()
                .execute_arrow(column, Some(field.as_ref()), &mut ctx)?;
            vortex_ensure!(
                field.is_nullable() || arrow.null_count() == 0,
                "Cannot convert field '{}' to non-nullable Arrow field because it contains nulls",
                field.name()
            );
            Ok(arrow)
        })
        .collect()
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
    use vortex_array::VortexSessionExecute;
    use vortex_arrow::ArrowSessionExt;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;

    use super::*;
    use crate::scan::test::SCAN_SESSION;
    use crate::scan::test::session_with_handle;

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
        SCAN_SESSION
            .arrow()
            .from_arrow_array(Arc::new(struct_array), true)
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
        let struct_field = Field::new_struct("", schema.fields().clone(), false);
        let mut ctx = SCAN_SESSION.create_execution_ctx();

        let batch = to_record_batch(vortex_array, &struct_field, &mut ctx)?;
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

        let iter = vec![Ok(batch1.clone()), Ok(batch2.clone())].into_iter();
        let mut adapter = RecordBatchIteratorAdapter {
            iter,
            schema: Arc::clone(&schema),
        };

        // Test RecordBatchReader trait
        assert_eq!(adapter.schema(), schema);

        // Test Iterator trait
        let first = adapter.next().unwrap()?;
        assert_eq!(first, batch1);

        let second = adapter.next().unwrap()?;
        assert_eq!(second, batch2);

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

    /// A wide struct exports its columns in parallel groups, in column order.
    #[test]
    fn test_wide_record_batch_parallel_export() -> VortexResult<()> {
        let ncols = i32::try_from(PARALLEL_EXPORT_COLUMNS)? + 72;
        let chunk = vortex_array::arrays::StructArray::from_fields(
            &(0..ncols)
                .map(|i| {
                    (
                        format!("c{i}"),
                        vortex_array::arrays::PrimitiveArray::from_iter([i, i + 1]).into_array(),
                    )
                })
                .collect::<Vec<_>>(),
        )?
        .into_array();
        let fields: Fields = (0..ncols)
            .map(|i| Field::new(format!("c{i}"), DataType::Int32, false))
            .collect();
        let struct_field = Field::new_struct("", fields, false);

        let batch = block_on(|handle| {
            let session = session_with_handle(handle);
            async move { record_batch_future(chunk, &struct_field, &session)?.await }
        })?;

        assert_eq!(batch.num_columns(), PARALLEL_EXPORT_COLUMNS + 72);
        assert_eq!(batch.num_rows(), 2);
        for (i, column) in (0..ncols).zip(batch.columns()) {
            assert_eq!(
                column
                    .as_primitive::<arrow_array::types::Int32Type>()
                    .values(),
                &[i, i + 1]
            );
        }
        Ok(())
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

        let iter = vec![Ok(batch.clone()), Err(error), Ok(batch.clone())].into_iter();
        let mut adapter = RecordBatchIteratorAdapter { iter, schema };

        // First batch succeeds
        let first = adapter.next().unwrap();
        assert_eq!(first.unwrap(), batch);

        // Second batch errors
        let second = adapter.next().unwrap();
        assert!(
            matches!(second, Err(ArrowError::ComputeError(message)) if message == "test error")
        );

        // Third batch succeeds
        let third = adapter.next().unwrap();
        assert_eq!(third.unwrap(), batch);
        assert!(adapter.next().is_none());
    }
}
