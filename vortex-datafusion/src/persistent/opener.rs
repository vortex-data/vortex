// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::Weak;

use arrow_schema::ArrowError;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::SchemaRef;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::arrow::array::RecordBatch;
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::file_meta::FileMeta;
use datafusion_datasource::file_stream::FileOpenFuture;
use datafusion_datasource::file_stream::FileOpener;
use datafusion_datasource::schema_adapter::SchemaAdapterFactory;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr::simplifier::PhysicalExprSimplifier;
use datafusion_physical_expr::split_conjunction;
use datafusion_physical_expr_adapter::PhysicalExprAdapterFactory;
use datafusion_physical_expr_common::physical_expr::is_dynamic_physical_expr;
use datafusion_physical_plan::metrics::Count;
use datafusion_pruning::FilePruner;
use futures::FutureExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use object_store::ObjectStore;
use object_store::path::Path;
use tracing::Instrument;
use vortex::dtype::FieldName;
use vortex::error::VortexError;
use vortex::expr::root;
use vortex::expr::select;
use vortex::layout::LayoutReader;
use vortex::metrics::VortexMetrics;
use vortex::scan::ScanBuilder;
use vortex::session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::dash_map::Entry;

use super::cache::VortexFileCache;
use crate::convert::exprs::can_be_pushed_down;
use crate::convert::exprs::make_vortex_predicate;
use crate::convert::ranges::apply_byte_range;

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

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use arrow_schema::Fields;
    use chrono::DateTime;
    use chrono::Utc;
    use datafusion::arrow::array::RecordBatch;
    use datafusion::arrow::array::StructArray;
    use datafusion::arrow::datatypes::DataType;
    use datafusion::arrow::datatypes::Schema;
    use datafusion::arrow::util::display::FormatOptions;
    use datafusion::arrow::util::pretty::pretty_format_batches_with_options;
    use datafusion::common::record_batch;
    use datafusion::logical_expr::col;
    use datafusion::logical_expr::lit;
    use datafusion::physical_expr::planner::logical2physical;
    use datafusion::prelude::SessionContext;
    use datafusion_common::config::ConfigOptions;
    use datafusion_common::create_array;
    use datafusion_datasource::file::FileSource;
    use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
    use datafusion_execution::object_store::ObjectStoreUrl;
    use datafusion_physical_plan::filter_pushdown::PushedDown;
    use futures::pin_mut;
    use insta::assert_snapshot;
    use object_store::ObjectMeta;
    use object_store::memory::InMemory;
    use rstest::rstest;
    use url::Url;
    use vortex::ArrayRef;
    use vortex::VortexSessionDefault;
    use vortex::arrow::FromArrowArray;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::io::ObjectStoreWriter;
    use vortex::io::VortexWrite;
    use vortex::session::VortexSession;
    use vortex_utils::aliases::hash_map::HashMap;

    use super::*;
    use crate::VortexSource;

    /// Fixtures used for integration testing the FileSource and FileOpener
    struct TestFixtures {
        object_store: Arc<dyn ObjectStore>,
        // We need to return this to the caller to prevent the session context from
        // being dropped and the object_store from being removed
        #[allow(dead_code)]
        session_context: SessionContext,
        source: Arc<dyn FileSource>,
        files: Files,
    }

    struct Files {
        file_meta: HashMap<String, FileMeta>,
    }

    impl Files {
        fn get(&self, path: &str) -> (FileMeta, PartitionedFile) {
            let file = self
                .file_meta
                .get(path)
                .unwrap_or_else(|| panic!("Missing file {}", path));
            (
                file.clone(),
                PartitionedFile::new(path, file.object_meta.size),
            )
        }
    }

    // Make a set of files and record batches
    async fn make_source(
        files: HashMap<String, RecordBatch>,
        file_schema: &SchemaRef,
    ) -> anyhow::Result<TestFixtures> {
        let session = VortexSession::default();

        let ctx = SessionContext::new();

        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());

        ctx.register_object_store(&Url::from_str("s3://in-memory")?, store.clone());

        // "write" all the record batches to the named file paths
        let mut file_meta = HashMap::with_capacity(files.len());

        // TODO: make file schema by superset of fields?
        for (path_str, rb) in files.iter() {
            let array = ArrayRef::from_arrow(rb, false);
            let path = Path::from_url_path(path_str.as_str())?;
            let mut write = ObjectStoreWriter::new(store.clone(), &path).await?;
            let summary = session
                .write_options()
                .write(&mut write, array.to_array_stream())
                .await?;
            write.shutdown().await?;

            file_meta.insert(
                path_str.clone(),
                FileMeta::from(ObjectMeta {
                    location: path.clone(),
                    size: summary.size(),
                    e_tag: None,
                    version: None,
                    last_modified: DateTime::<Utc>::from_timestamp_secs(0).unwrap(),
                }),
            );
        }

        let source = VortexSource::new(
            session.clone(),
            VortexFileCache::new(1024, 1024, session.clone()),
        );
        let source = source
            .with_schema(Arc::clone(file_schema))
            .with_batch_size(100);

        Ok(TestFixtures {
            session_context: ctx,
            object_store: store,
            files: Files { file_meta },
            source,
        })
    }

    #[rstest]
    #[tokio::test]
    async fn test_do_not_pushdown_filter_on_partition_columns() -> anyhow::Result<()> {
        let batch = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)]))?;
        let file_schema = batch.schema().clone();
        let files = HashMap::from_iter([("part=1/file.vortex".to_string(), batch)]);

        let table_schema = Arc::new(Schema::new(vec![
            Field::new("part", DataType::Int32, false),
            Field::new("a", DataType::Int32, false),
        ]));

        let TestFixtures {
            source,
            object_store,
            files,
            ..
        } = make_source(files, &file_schema).await?;

        // Attempting to push filters over partitions should fail.
        let filter_partition_col = col("part").eq(lit(1i32));
        let filter_partition_col = logical2physical(&filter_partition_col, table_schema.as_ref());

        let push_filters =
            source.try_pushdown_filters(vec![filter_partition_col], &ConfigOptions::default())?;

        assert!(matches!(push_filters.filters[0], PushedDown::No));

        let base_config = FileScanConfigBuilder::new(
            ObjectStoreUrl::parse("s3://in-memory")?,
            file_schema.clone(),
            source.clone(),
        )
        .build();

        // Create an opener with this
        let opener = source.create_file_opener(object_store.clone(), &base_config, 0);

        let (file1, part_file1) = files.get("part=1/file.vortex");

        let open_result = opener.open(file1, part_file1)?.await?;

        pin_mut!(open_result);
        let mut rbs = open_result.try_collect::<Vec<_>>().await?;
        assert_eq!(rbs.len(), 1);

        let rb = rbs.pop().unwrap();
        assert_eq!(rb.num_rows(), 3);
        let expected = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)]))?;

        assert_eq!(rb, expected);

        Ok(())
    }

    // Seems like datafusion is moving towards having DefaultPhysicalExprAdapterFactory be always provided, which would make it work OOTB.
    // See: https://github.com/apache/datafusion/issues/16800
    #[tokio::test]
    async fn test_open_files_different_table_schema() -> anyhow::Result<()> {
        let batch1 = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)]))?;
        let batch2 = record_batch!(("a", Int16, vec![Some(-1), Some(-2), Some(-3)]))?;
        let files = HashMap::from_iter([
            ("path/file1.vortex".to_string(), batch1),
            ("path/file2.vortex".to_string(), batch2),
        ]);

        // Table schema has can accommodate both files
        let table_schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));

        let TestFixtures {
            object_store,
            source,
            files,
            ..
        } = make_source(files, &table_schema).await?;

        let (file1, part_file1) = files.get("path/file1.vortex");
        let (file2, part_file2) = files.get("path/file2.vortex");

        // Try and push filters into the source.
        let filter = col("a").lt(lit(100_i32));
        let filter = logical2physical(&filter, table_schema.as_ref());
        let pushdown_result =
            source.try_pushdown_filters(vec![filter], &ConfigOptions::default())?;
        // filter should've succeeded pushing
        assert!(matches!(pushdown_result.filters[0], PushedDown::Yes));

        let base_config = FileScanConfigBuilder::new(
            ObjectStoreUrl::parse("s3://in-memory")?,
            table_schema.clone(),
            source.clone(),
        )
        .build();

        let opener = source.create_file_opener(object_store.clone(), &base_config, 0);
        let stream = opener.open(file1, part_file1)?.await?;

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

        let stream = opener.open(file2, part_file2)?.await?;
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

        // File has field order c,b,a
        let batch = record_batch!(
            ("c", Int32, vec![Some(300), Some(301), Some(302)]),
            ("b", Int32, vec![Some(200), Some(201), Some(202)]),
            ("a", Int32, vec![Some(100), Some(101), Some(102)])
        )?;

        // table schema has field order a,b,c
        let table_schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
            Field::new("c", DataType::Int32, true),
        ]));

        let files = HashMap::from_iter([("path/file1.vortex".to_string(), batch)]);

        let TestFixtures {
            source,
            files,
            object_store,
            ..
        } = make_source(files, &table_schema).await?;

        let (file1, part_file1) = files.get("path/file1.vortex");

        // Table schema has columns in different order: a, b, c
        let table_schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
            Field::new("c", DataType::Int32, true),
        ]));

        let base_config = FileScanConfigBuilder::new(
            ObjectStoreUrl::parse("s3://in-memory")?,
            table_schema.clone(),
            source.clone(),
        )
        .build();
        let opener = source.create_file_opener(object_store, &base_config, 0);

        // The opener should successfully open the file and reorder columns
        let stream = opener.open(file1, part_file1)?.await?;

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
        let field1 = create_array!(Utf8, vec![Some("value1"), Some("value2"), Some("value3")]);
        let field2 = create_array!(Utf8, vec![Some("a"), Some("b"), Some("c")]);

        let struct_array = StructArray::new(
            Fields::from(vec![
                Field::new("field1", DataType::Utf8, true),
                Field::new("field2", DataType::Utf8, true),
            ]),
            vec![field1.clone(), field2.clone()],
            None,
        );

        // file schema reflects the data
        let file_schema = Arc::new(Schema::new(vec![Field::new(
            "my_struct",
            DataType::Struct(Fields::from(vec![
                Field::new("field1", DataType::Utf8, true),
                Field::new("field2", DataType::Utf8, true),
            ])),
            true,
        )]));

        // Table schema has an extra inner utf8 field.
        let table_schema = Arc::new(Schema::new(vec![Field::new(
            "my_struct",
            DataType::Struct(Fields::from(vec![
                Field::new("field1", DataType::Utf8, true),
                Field::new("field2", DataType::Utf8, true),
                Field::new("field3", DataType::Utf8, true),
            ])),
            true,
        )]));

        let batch = RecordBatch::try_new(file_schema.clone(), vec![Arc::new(struct_array)])?;

        let files = HashMap::from_iter([("path/file.vortex".to_string(), batch.clone())]);

        let TestFixtures {
            source,
            files,
            object_store,
            ..
        } = make_source(files, &table_schema).await?;

        let filter = logical2physical(&col("my_struct").is_not_null(), &table_schema);
        let pushdown_result =
            source.try_pushdown_filters(vec![filter], &ConfigOptions::default())?;

        // The filter should not have been pushed
        assert!(matches!(pushdown_result.filters[0], PushedDown::Yes));

        let base_config = FileScanConfigBuilder::new(
            ObjectStoreUrl::parse("s3://in-memory")?,
            table_schema.clone(),
            source.clone(),
        )
        .build();

        let opener = source.create_file_opener(object_store.clone(), &base_config, 0);

        let (file, part_file) = files.get("path/file.vortex");

        let data = opener
            .open(file, part_file)?
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        assert_eq!(data.len(), 1);

        // The opener will return batches that have been adapted with the extra "field3" with
        // nulls added.
        let field3 = create_array!(Utf8, vec![Option::<String>::None; 3]);
        let table_struct = StructArray::new(
            Fields::from(vec![
                Field::new("field1", DataType::Utf8, true),
                Field::new("field2", DataType::Utf8, true),
                Field::new("field3", DataType::Utf8, true),
            ]),
            vec![field1, field2, field3],
            None,
        );

        let batch_with_nulls =
            RecordBatch::try_new(table_schema.clone(), vec![Arc::new(table_struct)])?;

        // The opener returns us a stream where field3 is replaced with nulls
        assert_eq!(data[0], batch_with_nulls);

        Ok(())
    }
}
