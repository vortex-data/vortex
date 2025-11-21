// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::{Arc, Weak};

use arrow_schema::{ArrowError, DataType, Field, SchemaRef};
use datafusion_common::arrow::array::RecordBatch;
use datafusion_common::{DataFusionError, Result as DFResult};
use datafusion_datasource::file_meta::FileMeta;
use datafusion_datasource::file_stream::{FileOpenFuture, FileOpener};
use datafusion_datasource::schema_adapter::SchemaAdapterFactory;
use datafusion_datasource::{FileRange, PartitionedFile};
use datafusion_physical_expr::simplifier::PhysicalExprSimplifier;
use datafusion_physical_expr::{PhysicalExprRef, split_conjunction};
use datafusion_physical_expr_adapter::PhysicalExprAdapterFactory;
use datafusion_physical_expr_common::physical_expr::is_dynamic_physical_expr;
use datafusion_physical_plan::metrics::Count;
use datafusion_pruning::FilePruner;
use futures::{FutureExt, StreamExt, TryStreamExt, stream};
use object_store::ObjectStore;
use object_store::path::Path;
use tracing::Instrument;
use vortex::ArrayRef;
use vortex::dtype::FieldName;
use vortex::error::VortexError;
use vortex::expr::{root, select};
use vortex::layout::LayoutReader;
use vortex::metrics::VortexMetrics;
use vortex::scan::ScanBuilder;
use vortex::session::VortexSession;
use vortex_utils::aliases::dash_map::{DashMap, Entry};

use super::cache::VortexFileCache;
use crate::convert::exprs::{can_be_pushed_down, make_vortex_predicate};

#[derive(Clone)]
pub(crate) struct VortexOpener {
    pub session: VortexSession,
    pub object_store: Arc<dyn ObjectStore>,
    /// Projection by index of the file's columns
    pub projection: Option<Arc<[usize]>>,
    /// Filter expression optimized for pushdown into Vortex scan operations.
    /// This may be a subset of file_pruning_predicate containing only expressions
    /// that Vortex can efficiently evaluate.
    pub filter: Option<PhysicalExprRef>,
    /// Filter expression used by DataFusion's FilePruner to eliminate files based on
    /// statistics and partition values without opening them.
    pub file_pruning_predicate: Option<PhysicalExprRef>,
    pub expr_adapter_factory: Option<Arc<dyn PhysicalExprAdapterFactory>>,
    pub schema_adapter_factory: Arc<dyn SchemaAdapterFactory>,
    /// Hive-style partitioning columns
    pub partition_fields: Vec<Arc<Field>>,
    pub file_cache: VortexFileCache,
    /// This is the table's schema without partition columns. It might be different than
    /// the physical schema, and the stream's type will be a projection of it.
    pub logical_schema: SchemaRef,
    pub batch_size: usize,
    pub limit: Option<usize>,
    pub metrics: VortexMetrics,
    pub layout_readers: Arc<DashMap<Path, Weak<dyn LayoutReader>>>,
    /// Whether the query has output ordering specified
    pub has_output_ordering: bool,
}

/// Merges the data types of two fields, preferring the logical type from the
/// table field.
fn merge_field_types(physical_field: &Field, table_field: &Field) -> DataType {
    match (physical_field.data_type(), table_field.data_type()) {
        (DataType::Struct(phys_fields), DataType::Struct(table_fields)) => {
            let merged_fields = merge_fields(phys_fields, table_fields);
            DataType::Struct(merged_fields.into())
        }
        (DataType::List(phys_field), DataType::List(table_field)) => {
            DataType::List(Arc::new(Field::new(
                phys_field.name(),
                merge_field_types(phys_field, table_field),
                phys_field.is_nullable(),
            )))
        }
        (DataType::LargeList(phys_field), DataType::LargeList(table_field)) => {
            DataType::LargeList(Arc::new(Field::new(
                phys_field.name(),
                merge_field_types(phys_field, table_field),
                phys_field.is_nullable(),
            )))
        }
        _ => table_field.data_type().clone(),
    }
}

/// Merges two field collections, using logical types from table_fields where available.
/// Falls back to physical field types when no matching table field is found.
fn merge_fields(
    physical_fields: &arrow_schema::Fields,
    table_fields: &arrow_schema::Fields,
) -> Vec<Field> {
    physical_fields
        .iter()
        .map(|phys_field| {
            table_fields
                .iter()
                .find(|f| f.name() == phys_field.name())
                .map(|table_field| {
                    Field::new(
                        phys_field.name(),
                        merge_field_types(phys_field, table_field),
                        phys_field.is_nullable(),
                    )
                })
                .unwrap_or_else(|| (**phys_field).clone())
        })
        .collect()
}

/// Computes a logical file schema from the physical file schema and the table
/// schema.
///
/// For each field in the physical file schema, looks up the corresponding field
/// in the table schema and uses its logical type.
fn compute_logical_file_schema(
    physical_file_schema: &SchemaRef,
    table_schema: &SchemaRef,
) -> SchemaRef {
    let logical_fields: Vec<Field> = physical_file_schema
        .fields()
        .iter()
        .map(|physical_field| {
            table_schema
                .fields()
                .find(physical_field.name())
                .map(|(_, table_field)| {
                    Field::new(
                        physical_field.name(),
                        merge_field_types(physical_field, table_field),
                        physical_field.is_nullable(),
                    )
                    .with_metadata(physical_field.metadata().clone())
                })
                .unwrap_or_else(|| (**physical_field).clone())
        })
        .collect();

    Arc::new(arrow_schema::Schema::new(logical_fields))
}

impl FileOpener for VortexOpener {
    fn open(&self, file_meta: FileMeta, file: PartitionedFile) -> DFResult<FileOpenFuture> {
        let session = self.session.clone();
        let object_store = self.object_store.clone();
        let projection = self.projection.clone();
        let mut filter = self.filter.clone();
        let file_pruning_predicate = self.file_pruning_predicate.clone();
        let expr_adapter_factory = self.expr_adapter_factory.clone();
        let partition_fields = self.partition_fields.clone();
        let file_cache = self.file_cache.clone();
        let logical_schema = self.logical_schema.clone();
        let batch_size = self.batch_size;
        let limit = self.limit;
        let metrics = self.metrics.clone();
        let layout_reader = self.layout_readers.clone();
        let has_output_ordering = self.has_output_ordering;

        let projected_schema = match projection.as_ref() {
            None => logical_schema.clone(),
            Some(indices) => Arc::new(logical_schema.project(indices)?),
        };

        let mut predicate_file_schema = logical_schema.clone();

        let schema_adapter = self
            .schema_adapter_factory
            .create(projected_schema, logical_schema.clone());

        Ok(async move {
            // Create FilePruner when we have a predicate and either dynamic expressions
            // or file statistics available. The pruner can eliminate files without
            // opening them based on:
            // - Partition column values (e.g., date=2024-01-01)
            // - File-level statistics (min/max values per column)
            let mut file_pruner = file_pruning_predicate
                .map(|predicate| {
                    // Only create pruner if we have dynamic expressions or file statistics
                    // to work with. Static predicates without stats won't benefit from pruning.
                    Ok::<_, DataFusionError>(
                        (is_dynamic_physical_expr(&predicate) | file.has_statistics()).then_some(
                            FilePruner::new(
                                predicate.clone(),
                                &logical_schema,
                                partition_fields.clone(),
                                file.clone(),
                                Count::default(),
                            )?,
                        ),
                    )
                })
                .transpose()?
                .flatten();

            // Check if this file should be pruned based on statistics/partition values.
            // Returns empty stream if file can be skipped entirely.
            if let Some(file_pruner) = &mut file_pruner
                && file_pruner.should_prune()?
            {
                return Ok(stream::empty().boxed());
            }

            let vxf = file_cache
                .try_get(&file_meta.object_meta, object_store)
                .await
                .map_err(|e| {
                    DataFusionError::Execution(format!("Failed to open Vortex file {e}"))
                })?;

            let physical_file_schema = Arc::new(vxf.dtype().to_arrow_schema().map_err(|e| {
                DataFusionError::Execution(format!("Failed to convert file schema to arrow: {e}"))
            })?);

            if let Some(expr_adapter_factory) = expr_adapter_factory {
                let partition_values = partition_fields
                    .iter()
                    .cloned()
                    .zip(file.partition_values)
                    .collect::<Vec<_>>();

                // The adapter rewrites the expression to the local file schema, allowing
                // for schema evolution and divergence between the table's schema and individual files.
                filter = filter
                    .map(|filter| {
                        let logical_file_schema = compute_logical_file_schema(
                            &physical_file_schema.clone(),
                            &logical_schema,
                        );

                        let expr = expr_adapter_factory
                            .create(logical_file_schema, physical_file_schema.clone())
                            .with_partition_values(partition_values)
                            .rewrite(filter)?;

                        // Expression might now reference columns that don't exist in the file, so we can give it
                        // another simplification pass.
                        PhysicalExprSimplifier::new(&physical_file_schema).simplify(expr)
                    })
                    .transpose()?;

                predicate_file_schema = physical_file_schema.clone();
            }

            // Create the initial mapping from physical file schema to projected schema.
            // This gives us the field reordering and tells us which logical schema fields
            // to select.
            let (_schema_mapping, adapted_projections) =
                schema_adapter.map_schema(&physical_file_schema)?;

            // Build the Vortex projection expression using the adapted projections.
            // This will reorder the fields to match the target order.
            let fields = adapted_projections
                .iter()
                .map(|idx| {
                    let field = logical_schema.field(*idx);
                    FieldName::from(field.name().as_str())
                })
                .collect::<Vec<_>>();
            let projection_expr = select(fields, root());

            // After Vortex applies the projection, the batch will have fields in the target
            // order (matching adapted_projections), but with the physical file types.
            // We need a second schema mapping for type casting only, not reordering.
            // Build a schema that represents what Vortex will return: fields in target order
            // with physical types.
            let projected_physical_fields: Vec<Field> = adapted_projections
                .iter()
                .map(|&idx| {
                    let logical_field = logical_schema.field(idx);
                    let field_name = logical_field.name();

                    // Find this field in the physical schema to get its physical type
                    physical_file_schema
                        .field_with_name(field_name)
                        .map(|phys_field| {
                            Field::new(
                                field_name,
                                merge_field_types(phys_field, logical_field),
                                phys_field.is_nullable(),
                            )
                        })
                        .unwrap_or_else(|_| (*logical_field).clone())
                })
                .collect();

            let projected_physical_schema =
                Arc::new(arrow_schema::Schema::new(projected_physical_fields));

            // Create a second mapping from the projected physical schema (what Vortex returns)
            // to the final projected schema. This mapping will handle type casting without reordering.
            let (batch_schema_mapping, _) =
                schema_adapter.map_schema(&projected_physical_schema)?;

            // We share our layout readers with others partitions in the scan, so we can only need to read each layout in each file once.
            let layout_reader = match layout_reader.entry(file_meta.object_meta.location.clone()) {
                Entry::Occupied(mut occupied_entry) => {
                    if let Some(reader) = occupied_entry.get().upgrade() {
                        log::trace!("reusing layout reader for {}", occupied_entry.key());
                        reader
                    } else {
                        log::trace!("creating layout reader for {}", occupied_entry.key());
                        let reader = vxf.layout_reader().map_err(|e| {
                            DataFusionError::Execution(format!(
                                "Failed to create layout reader: {e}"
                            ))
                        })?;
                        occupied_entry.insert(Arc::downgrade(&reader));
                        reader
                    }
                }
                Entry::Vacant(vacant_entry) => {
                    log::trace!("creating layout reader for {}", vacant_entry.key());
                    let reader = vxf.layout_reader().map_err(|e| {
                        DataFusionError::Execution(format!("Failed to create layout reader: {e}"))
                    })?;
                    vacant_entry.insert(Arc::downgrade(&reader));

                    reader
                }
            };

            let mut scan_builder = ScanBuilder::new(session, layout_reader);
            if let Some(file_range) = file_meta.range {
                scan_builder = apply_byte_range(
                    file_range,
                    file_meta.object_meta.size,
                    vxf.row_count(),
                    scan_builder,
                );
            }

            let filter = filter
                .and_then(|f| {
                    let exprs = split_conjunction(&f)
                        .into_iter()
                        .filter(|expr| can_be_pushed_down(expr, &predicate_file_schema))
                        .collect::<Vec<_>>();

                    make_vortex_predicate(&exprs).transpose()
                })
                .transpose()
                .map_err(|e| DataFusionError::External(e.into()))?;

            if let Some(limit) = limit
                && filter.is_none()
            {
                scan_builder = scan_builder.with_limit(limit);
            }

            let stream = scan_builder
                .with_metrics(metrics)
                .with_projection(projection_expr)
                .with_some_filter(filter)
                .with_ordered(has_output_ordering)
                .map(|chunk| RecordBatch::try_from(chunk.as_ref()))
                .into_stream()
                .map_err(|e| {
                    DataFusionError::Execution(format!("Failed to create Vortex stream: {e}"))
                })?
                .map_ok(move |rb| {
                    // We try and slice the stream into respecting datafusion's configured batch size.
                    stream::iter(
                        (0..rb.num_rows().div_ceil(batch_size * 2))
                            .flat_map(move |block_idx| {
                                let offset = block_idx * batch_size * 2;

                                // If we have less than two batches worth of rows left, we keep them together as a single batch.
                                if rb.num_rows() - offset < 2 * batch_size {
                                    let length = rb.num_rows() - offset;
                                    [Some(rb.slice(offset, length)), None].into_iter()
                                } else {
                                    let first = rb.slice(offset, batch_size);
                                    let second = rb.slice(offset + batch_size, batch_size);
                                    [Some(first), Some(second)].into_iter()
                                }
                            })
                            .flatten()
                            .map(Ok),
                    )
                })
                .map_err(move |e: VortexError| {
                    ArrowError::ExternalError(Box::new(e.with_context(format!(
                        "Failed to read Vortex file: {}",
                        file_meta.object_meta.location
                    ))))
                })
                .try_flatten()
                .map(move |batch| batch.and_then(|b| batch_schema_mapping.map_batch(b)))
                .boxed();

            Ok(stream)
        }
        .in_current_span()
        .boxed())
    }
}

/// If the file has a [`FileRange`](datafusion::datasource::listing::FileRange), we translate it into a row range in the file for the scan.
fn apply_byte_range(
    file_range: FileRange,
    total_size: u64,
    row_count: u64,
    scan_builder: ScanBuilder<ArrayRef>,
) -> ScanBuilder<ArrayRef> {
    let row_range = byte_range_to_row_range(
        file_range.start as u64..file_range.end as u64,
        row_count,
        total_size,
    );

    scan_builder.with_row_range(row_range)
}

fn byte_range_to_row_range(byte_range: Range<u64>, row_count: u64, total_size: u64) -> Range<u64> {
    let average_row = total_size / row_count;
    assert!(average_row > 0, "A row must always have at least one byte");

    let start_row = byte_range.start / average_row;
    let end_row = byte_range.end / average_row;

    // We take the min here as `end_row` might overshoot
    start_row..u64::min(row_count, end_row)
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use arrow_schema::Fields;
    use chrono::Utc;
    use datafusion::arrow::array::{RecordBatch, StringArray, StructArray};
    use datafusion::arrow::datatypes::{DataType, Schema};
    use datafusion::arrow::util::display::FormatOptions;
    use datafusion::common::record_batch;
    use datafusion::datasource::schema_adapter::DefaultSchemaAdapterFactory;
    use datafusion::logical_expr::{col, lit};
    use datafusion::physical_expr::planner::logical2physical;
    use datafusion::physical_expr_adapter::DefaultPhysicalExprAdapterFactory;
    use datafusion::scalar::ScalarValue;
    use insta::assert_snapshot;
    use itertools::Itertools;
    use object_store::ObjectMeta;
    use object_store::memory::InMemory;
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::arrow::FromArrowArray;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::io::{ObjectStoreWriter, VortexWrite};
    use vortex::session::VortexSession;

    use super::*;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

    #[rstest]
    #[case(0..100, 100, 100, 0..100)]
    #[case(0..105, 100, 105, 0..100)]
    #[case(0..50, 100, 105, 0..50)]
    #[case(50..105, 100, 105, 50..100)]
    #[case(0..1, 4, 8, 0..0)]
    #[case(1..8, 4, 8, 0..4)]
    fn test_range_translation(
        #[case] byte_range: Range<u64>,
        #[case] row_count: u64,
        #[case] total_size: u64,
        #[case] expected: Range<u64>,
    ) {
        assert_eq!(
            byte_range_to_row_range(byte_range, row_count, total_size),
            expected
        );
    }

    #[test]
    fn test_consecutive_ranges() {
        let row_count = 100;
        let total_size = 429;
        let bytes_a = 0..143;
        let bytes_b = 143..286;
        let bytes_c = 286..429;

        let rows_a = byte_range_to_row_range(bytes_a, row_count, total_size);
        let rows_b = byte_range_to_row_range(bytes_b, row_count, total_size);
        let rows_c = byte_range_to_row_range(bytes_c, row_count, total_size);

        assert_eq!(rows_a.end - rows_a.start, 35);
        assert_eq!(rows_b.end - rows_b.start, 36);
        assert_eq!(rows_c.end - rows_c.start, 29);

        assert_eq!(rows_a.start, 0);
        assert_eq!(rows_c.end, 100);
        for (left, right) in [rows_a, rows_b, rows_c].iter().tuple_windows() {
            assert_eq!(left.end, right.start);
        }
    }

    async fn write_arrow_to_vortex(
        object_store: Arc<dyn ObjectStore>,
        path: &str,
        rb: RecordBatch,
    ) -> anyhow::Result<u64> {
        let array = ArrayRef::from_arrow(rb, false);
        let path = Path::parse(path)?;

        let mut write = ObjectStoreWriter::new(object_store, &path).await?;
        let summary = SESSION
            .write_options()
            .write(&mut write, array.to_array_stream())
            .await?;
        write.shutdown().await?;

        Ok(summary.size())
    }

    fn make_meta(path: &str, data_size: u64) -> FileMeta {
        FileMeta {
            object_meta: ObjectMeta {
                location: Path::from(path),
                last_modified: Utc::now(),
                size: data_size,
                e_tag: None,
                version: None,
            },
            range: None,
            extensions: None,
            metadata_size_hint: None,
        }
    }

    #[rstest]
    #[case(Some(Arc::new(DefaultPhysicalExprAdapterFactory) as _), (1, 3), (0, 0))]
    // If we don't have a physical expr adapter, we just drop filters on partition values
    #[case(None, (1, 3), (1, 3))]
    #[tokio::test]
    async fn test_adapter_optimization_partition_column(
        #[case] expr_adapter_factory: Option<Arc<dyn PhysicalExprAdapterFactory>>,
        #[case] expected_result1: (usize, usize),
        #[case] expected_result2: (usize, usize),
    ) -> anyhow::Result<()> {
        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let file_path = "part=1/file.vortex";
        let batch = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)])).unwrap();
        let data_size =
            write_arrow_to_vortex(object_store.clone(), file_path, batch.clone()).await?;

        let file_schema = batch.schema();
        let mut file = PartitionedFile::new(file_path.to_string(), data_size);
        file.partition_values = vec![ScalarValue::Int32(Some(1))];

        let table_schema = Arc::new(Schema::new(vec![
            Field::new("part", DataType::Int32, false),
            Field::new("a", DataType::Int32, false),
        ]));

        let make_opener = |filter| VortexOpener {
            session: SESSION.clone(),
            object_store: object_store.clone(),
            projection: Some([0].into()),
            filter: Some(filter),
            file_pruning_predicate: None,
            expr_adapter_factory: expr_adapter_factory.clone(),
            schema_adapter_factory: Arc::new(DefaultSchemaAdapterFactory),
            partition_fields: vec![Arc::new(Field::new("part", DataType::Int32, false))],
            file_cache: VortexFileCache::new(1, 1, SESSION.clone()),
            logical_schema: file_schema.clone(),
            batch_size: 100,
            limit: None,
            metrics: Default::default(),
            layout_readers: Default::default(),
            has_output_ordering: false,
        };

        // filter matches partition value
        let filter = col("part").eq(lit(1));
        let filter = logical2physical(&filter, table_schema.as_ref());

        let opener = make_opener(filter);
        let stream = opener
            .open(make_meta(file_path, data_size), file.clone())
            .unwrap()
            .await
            .unwrap();

        let data = stream.try_collect::<Vec<_>>().await?;
        let num_batches = data.len();
        let num_rows = data.iter().map(|rb| rb.num_rows()).sum::<usize>();

        assert_eq!((num_batches, num_rows), expected_result1);

        // filter doesn't matches partition value
        let filter = col("part").eq(lit(2));
        let filter = logical2physical(&filter, table_schema.as_ref());

        let opener = make_opener(filter);
        let stream = opener
            .open(make_meta(file_path, data_size), file.clone())
            .unwrap()
            .await
            .unwrap();

        let data = stream.try_collect::<Vec<_>>().await?;
        let num_batches = data.len();
        let num_rows = data.iter().map(|rb| rb.num_rows()).sum::<usize>();
        assert_eq!((num_batches, num_rows), expected_result2);

        Ok(())
    }

    #[rstest]
    #[case(Some(Arc::new(DefaultPhysicalExprAdapterFactory) as _))]
    // If we don't have a physical expr adapter, we just drop filters on partition values.
    // This is currently not supported, the work to support it requires to rewrite the predicate with appropriate casts.
    // Seems like datafusion is moving towards having DefaultPhysicalExprAdapterFactory be always provided, which would make it work OOTB.
    // See: https://github.com/apache/datafusion/issues/16800
    // #[case(None)]
    #[tokio::test]
    async fn test_open_files_different_table_schema(
        #[case] expr_adapter_factory: Option<Arc<dyn PhysicalExprAdapterFactory>>,
    ) -> anyhow::Result<()> {
        use datafusion::arrow::util::pretty::pretty_format_batches_with_options;

        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let file1_path = "/path/file1.vortex";
        let batch1 = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)])).unwrap();
        let data_size1 = write_arrow_to_vortex(object_store.clone(), file1_path, batch1).await?;
        let file1 = PartitionedFile::new(file1_path.to_string(), data_size1);

        let file2_path = "/path/file2.vortex";
        let batch2 = record_batch!(("a", Int16, vec![Some(-1), Some(-2), Some(-3)])).unwrap();
        let data_size2 = write_arrow_to_vortex(object_store.clone(), file2_path, batch2).await?;
        let file2 = PartitionedFile::new(file1_path.to_string(), data_size1);

        // Table schema has can accommodate both files
        let table_schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));

        let make_opener = |filter| VortexOpener {
            session: SESSION.clone(),
            object_store: object_store.clone(),
            projection: Some([0].into()),
            filter: Some(filter),
            file_pruning_predicate: None,
            expr_adapter_factory: expr_adapter_factory.clone(),
            schema_adapter_factory: Arc::new(DefaultSchemaAdapterFactory),
            partition_fields: vec![],
            file_cache: VortexFileCache::new(1, 1, SESSION.clone()),
            logical_schema: table_schema.clone(),
            batch_size: 100,
            limit: None,
            metrics: Default::default(),
            layout_readers: Default::default(),
            has_output_ordering: false,
        };

        let filter = col("a").lt(lit(100_i32));
        let filter = logical2physical(&filter, table_schema.as_ref());

        let opener1 = make_opener(filter.clone());
        let stream = opener1
            .open(make_meta(file1_path, data_size1), file1)?
            .await?;

        let format_opts = FormatOptions::new().with_types_info(true);

        let data = stream.try_collect::<Vec<_>>().await?;
        assert_snapshot!(pretty_format_batches_with_options(&data, &format_opts)?.to_string(), @r"
        +-------+
        | a     |
        | Int32 |
        +-------+
        | 1     |
        | 2     |
        | 3     |
        +-------+
        ");

        let opener2 = make_opener(filter.clone());
        let stream = opener2
            .open(make_meta(file2_path, data_size2), file2)?
            .await?;

        let data = stream.try_collect::<Vec<_>>().await?;
        assert_snapshot!(pretty_format_batches_with_options(&data, &format_opts)?.to_string(), @r"
        +-------+
        | a     |
        | Int32 |
        +-------+
        | -1    |
        | -2    |
        | -3    |
        +-------+
        ");

        Ok(())
    }

    #[tokio::test]
    // This test verifies that files with different column order than the
    // table schema can be opened without errors. The fix ensures that the
    // schema mapper is only used for type casting, not for reordering,
    // since the vortex projection already handles reordering.
    async fn test_schema_different_column_order() -> anyhow::Result<()> {
        use datafusion::arrow::util::pretty::pretty_format_batches_with_options;

        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let file_path = "/path/file.vortex";

        // File has columns in order: c, b, a
        let batch = record_batch!(
            ("c", Int32, vec![Some(300), Some(301), Some(302)]),
            ("b", Int32, vec![Some(200), Some(201), Some(202)]),
            ("a", Int32, vec![Some(100), Some(101), Some(102)])
        )
        .unwrap();
        let data_size = write_arrow_to_vortex(object_store.clone(), file_path, batch).await?;
        let file = PartitionedFile::new(file_path.to_string(), data_size);

        // Table schema has columns in different order: a, b, c
        let table_schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
            Field::new("c", DataType::Int32, true),
        ]));

        let opener = VortexOpener {
            session: SESSION.clone(),
            object_store: object_store.clone(),
            projection: Some([0, 1, 2].into()),
            filter: None,
            file_pruning_predicate: None,
            expr_adapter_factory: Some(Arc::new(DefaultPhysicalExprAdapterFactory) as _),
            schema_adapter_factory: Arc::new(DefaultSchemaAdapterFactory),
            partition_fields: vec![],
            file_cache: VortexFileCache::new(1, 1, SESSION.clone()),
            logical_schema: table_schema.clone(),
            batch_size: 100,
            limit: None,
            metrics: Default::default(),
            layout_readers: Default::default(),
            has_output_ordering: false,
        };

        // The opener should successfully open the file and reorder columns
        let stream = opener.open(make_meta(file_path, data_size), file)?.await?;

        let format_opts = FormatOptions::new().with_types_info(true);
        let data = stream.try_collect::<Vec<_>>().await?;

        // Verify the output has columns in table schema order (a, b, c)
        // not file order (c, b, a)
        assert_snapshot!(pretty_format_batches_with_options(&data, &format_opts)?.to_string(), @r"
        +-------+-------+-------+
        | a     | b     | c     |
        | Int32 | Int32 | Int32 |
        +-------+-------+-------+
        | 100   | 200   | 300   |
        | 101   | 201   | 301   |
        | 102   | 202   | 302   |
        +-------+-------+-------+
        ");

        Ok(())
    }

    #[tokio::test]
    // This test verifies that expression rewriting doesn't fail when there is
    // a nested schema mismatch between the physical file schema and logical
    // table schema.
    async fn test_adapter_logical_physical_struct_mismatch() -> anyhow::Result<()> {
        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let file_path = "/path/file.vortex";
        let file_struct_fields = Fields::from(vec![
            Field::new("field1", DataType::Utf8, true),
            Field::new("field2", DataType::Utf8, true),
        ]);
        let struct_array = StructArray::new(
            file_struct_fields.clone(),
            vec![
                Arc::new(StringArray::from(vec!["value1", "value2", "value3"])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ],
            None,
        );
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "my_struct",
                DataType::Struct(file_struct_fields),
                true,
            )])),
            vec![Arc::new(struct_array)],
        )?;
        let data_size = write_arrow_to_vortex(object_store.clone(), file_path, batch).await?;

        // Table schema has an extra utf8 field.
        let table_schema = Arc::new(Schema::new(vec![Field::new(
            "my_struct",
            DataType::Struct(Fields::from(vec![
                Field::new(
                    "field1",
                    DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                    true,
                ),
                Field::new(
                    "field2",
                    DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                    true,
                ),
                Field::new("field3", DataType::Utf8, true),
            ])),
            true,
        )]));

        let opener = VortexOpener {
            session: SESSION.clone(),
            object_store: object_store.clone(),
            projection: None,
            filter: Some(logical2physical(
                &col("my_struct").is_not_null(),
                &table_schema,
            )),
            file_pruning_predicate: None,
            expr_adapter_factory: Some(Arc::new(DefaultPhysicalExprAdapterFactory) as _),
            schema_adapter_factory: Arc::new(DefaultSchemaAdapterFactory),
            partition_fields: vec![],
            file_cache: VortexFileCache::new(1, 1, SESSION.clone()),
            logical_schema: table_schema,
            batch_size: 100,
            limit: None,
            metrics: Default::default(),
            layout_readers: Default::default(),
            has_output_ordering: false,
        };

        // The opener should be able to open the file with a filter on the
        // struct column.
        let data = opener
            .open(
                make_meta(file_path, data_size),
                PartitionedFile::new(file_path.to_string(), data_size),
            )?
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        assert_eq!(data.len(), 1);
        assert_eq!(data[0].num_rows(), 3);

        Ok(())
    }
}
