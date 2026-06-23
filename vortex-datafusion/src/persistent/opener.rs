// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;
use std::sync::Arc;
use std::sync::Weak;

use arrow_schema::Field;
use arrow_schema::Schema;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::arrow::array::AsArray;
use datafusion_common::arrow::array::RecordBatch;
use datafusion_common::exec_datafusion_err;
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::TableSchema;
use datafusion_datasource::file_stream::FileOpenFuture;
use datafusion_datasource::file_stream::FileOpener;
use datafusion_execution::cache::cache_manager::CachedFileMetadataEntry;
use datafusion_execution::cache::cache_manager::FileMetadataCache;
use datafusion_physical_expr::PhysicalExprRef;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr::simplifier::PhysicalExprSimplifier;
use datafusion_physical_expr::split_conjunction;
use datafusion_physical_expr::utils::reassign_expr_columns;
use datafusion_physical_expr_adapter::PhysicalExprAdapterFactory;
use datafusion_physical_expr_adapter::replace_columns_with_literals;
use datafusion_physical_expr_common::physical_expr::is_dynamic_physical_expr;
use datafusion_physical_plan::metrics::Count;
use datafusion_pruning::FilePruner;
use futures::FutureExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use itertools::Itertools;
use object_store::path::Path;
use tracing::Instrument;
use vortex::array::VortexSessionExecute;
use vortex::array::arrow::ArrowSessionExt;
use vortex::dtype::FieldMask;
use vortex::error::VortexError;
use vortex::error::VortexExpect;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::io::InstrumentedReadAt;
use vortex::io::session::RuntimeSessionExt;
use vortex::layout::LayoutReader;
use vortex::layout::scan::scan_builder::ScanBuilder;
use vortex::layout::scan::split_by::SplitBy;
use vortex::metrics::Label;
use vortex::metrics::MetricsRegistry;
use vortex::scan::ScanRequest;
use vortex::session::VortexSession;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::dash_map::Entry;
use vortex_utils::parallelism::get_available_parallelism;

use crate::VortexAccessPlan;
use crate::convert::exprs::ExpressionConvertor;
use crate::convert::exprs::ProcessedProjection;
use crate::convert::exprs::make_vortex_predicate;
use crate::convert::schema::calculate_physical_schema;
use crate::metrics::PARTITION_LABEL;
use crate::metrics::PATH_LABEL;
use crate::persistent::cache::CachedVortexMetadata;
use crate::persistent::reader::VortexReaderFactory;
use crate::persistent::stream::PrunableStream;

#[derive(Clone)]
pub(crate) struct VortexOpener {
    /// The partition this opener is assigned to. Only used for labeling metrics.
    pub partition: usize,
    pub session: VortexSession,
    pub vortex_reader_factory: Arc<dyn VortexReaderFactory>,
    pub scan_v2: bool,
    /// Optional table schema projection. The indices are w.r.t. the `table_schema`, which is
    /// all fields in the final scan result not including the partition columns.
    pub projection: ProjectionExprs,
    /// Filter expression optimized for pushdown into Vortex scan operations.
    /// This may be a subset of file_pruning_predicate containing only expressions
    /// that Vortex can efficiently evaluate.
    pub filter: Option<PhysicalExprRef>,
    /// Filter expression used by DataFusion's FilePruner to eliminate files based on
    /// statistics and partition values without opening them.
    pub file_pruning_predicate: Option<PhysicalExprRef>,
    pub expr_adapter_factory: Arc<dyn PhysicalExprAdapterFactory>,
    /// This is the table's schema without partition columns. It may contain fields which do
    /// not exist in the file, and are supplied by the `schema_adapter_factory`.
    pub table_schema: TableSchema,
    /// A hint for the desired row count of record batches returned from the scan.
    pub batch_size: usize,
    /// If provided, the scan will not return more than this many rows.
    pub limit: Option<u64>,
    /// A metrics object for tracking performance of the scan.
    pub metrics_registry: Arc<dyn MetricsRegistry>,
    /// A shared cache of file readers.
    ///
    /// To save on the overhead of reparsing FlatBuffers and rebuilding the layout tree, we cache
    /// a file reader the first time we read a file.
    pub layout_readers: Arc<DashMap<Path, Weak<dyn LayoutReader>>>,
    /// Shared full-file natural split ranges keyed by file path.
    pub natural_split_ranges: Arc<DashMap<Path, Arc<[Range<u64>]>>>,
    /// Shared V2 file handles keyed by file path.
    pub vortex_files: Arc<DashMap<Path, Arc<VortexFile>>>,
    /// Whether the query has output ordering specified
    pub has_output_ordering: bool,

    pub expression_convertor: Arc<dyn ExpressionConvertor>,
    pub file_metadata_cache: Option<Arc<dyn FileMetadataCache>>,
    /// Whether to enable expression pushdown into the underlying Vortex scan.
    pub projection_pushdown: bool,
    pub scan_concurrency: Option<usize>,
}

impl FileOpener for VortexOpener {
    fn open(&self, file: PartitionedFile) -> DFResult<FileOpenFuture> {
        let session = self.session.clone();
        let metrics_registry = Arc::clone(&self.metrics_registry);
        let labels = vec![
            Label::new(PATH_LABEL, file.path().to_string()),
            Label::new(PARTITION_LABEL, self.partition.to_string()),
        ];

        let mut projection = self.projection.clone();
        let mut filter = self.filter.clone();

        let reader = self.vortex_reader_factory.create_reader(&file, &session)?;

        let reader =
            InstrumentedReadAt::new_with_labels(reader, metrics_registry.as_ref(), labels.clone());

        let file_pruning_predicate = self.file_pruning_predicate.clone();
        let expr_adapter_factory = Arc::clone(&self.expr_adapter_factory);
        let file_metadata_cache = self.file_metadata_cache.clone();

        let unified_file_schema = Arc::clone(self.table_schema.file_schema());
        let batch_size = self.batch_size;
        let limit = self.limit;
        let layout_readers = Arc::clone(&self.layout_readers);
        let natural_split_ranges = Arc::clone(&self.natural_split_ranges);
        let vortex_files = Arc::clone(&self.vortex_files);
        let has_output_ordering = self.has_output_ordering;
        let scan_concurrency = self.scan_concurrency;

        let expr_convertor = Arc::clone(&self.expression_convertor);
        let projection_pushdown = self.projection_pushdown;
        let scan_v2 = self.scan_v2;

        // Replace column access for partition columns with literals
        #[expect(clippy::disallowed_types)]
        let literal_value_cols = self
            .table_schema
            .table_partition_cols()
            .iter()
            .map(|f| f.name())
            .cloned()
            .zip(file.partition_values.clone())
            .collect::<std::collections::HashMap<String, ScalarValue>>();

        if !literal_value_cols.is_empty() {
            projection = projection.try_map_exprs(|expr| {
                replace_columns_with_literals(Arc::clone(&expr), &literal_value_cols)
            })?;
            filter = filter
                .map(|p| replace_columns_with_literals(p, &literal_value_cols))
                .transpose()?;
        }

        Ok(async move {
            // Create FilePruner when we have a predicate and either dynamic expressions
            // or file statistics available. The pruner can eliminate files without
            // opening them based on File-level statistics (min/max values per column)
            let mut file_pruner = file_pruning_predicate
                .filter(|p| {
                    // Only create pruner if we have dynamic expressions or file statistics
                    // to work with. Static predicates without stats won't benefit from pruning.
                    is_dynamic_physical_expr(p) || file.has_statistics()
                })
                .and_then(|predicate| {
                    FilePruner::try_new(
                        Arc::clone(&predicate),
                        &unified_file_schema,
                        &file,
                        Count::default(),
                    )
                });

            // Check if this file should be pruned based on statistics/partition values.
            // Returns empty stream if file can be skipped entirely.
            if let Some(file_pruner) = file_pruner.as_mut()
                && file_pruner.should_prune()?
            {
                return Ok(stream::empty().boxed());
            }

            let mut open_opts = session
                .open_options()
                .with_file_size(file.object_meta.size)
                .with_metrics_registry(Arc::clone(&metrics_registry))
                .with_labels(labels);

            let cached_footer = file_metadata_cache
                .as_ref()
                .and_then(|cache| cache.get(file.path()))
                .filter(|entry| entry.is_valid_for(&file.object_meta))
                .and_then(|entry| {
                    entry
                        .file_metadata
                        .as_any()
                        .downcast_ref::<CachedVortexMetadata>()
                        .map(|vortex_metadata| vortex_metadata.footer().clone())
                });
            let footer_cache_hit = cached_footer.is_some();

            if let Some(footer) = cached_footer {
                open_opts = open_opts.with_footer(footer);
            }

            let vxf = if let Some(hit) = vortex_files.get(&file.object_meta.location) {
                Arc::clone(hit.value())
            } else {
                let opened = Arc::new(
                    open_opts
                        .open_read(reader)
                        .await
                        .map_err(|e| exec_datafusion_err!("Failed to open Vortex file {e}"))?,
                );

                match vortex_files.entry(file.object_meta.location.clone()) {
                    Entry::Occupied(entry) => Arc::clone(entry.get()),
                    Entry::Vacant(entry) => {
                        entry.insert(Arc::clone(&opened));
                        opened
                    }
                }
            };

            // On a miss, cache the parsed footer so other partitions and later executions
            // skip the footer fetch and parse. `infer_schema`/`infer_stats` also populate
            // this cache, but only when planning goes through `VortexFormat`.
            if !footer_cache_hit && let Some(cache) = &file_metadata_cache {
                cache.put(
                    file.path(),
                    CachedFileMetadataEntry::new(
                        file.object_meta.clone(),
                        Arc::new(CachedVortexMetadata::new(&vxf)),
                    ),
                );
            }

            // Check if there are rows in this file. If not, we can save
            // ourselves some work and return an empty stream.
            if vxf.row_count() == 0 {
                return Ok(stream::empty().boxed());
            }

            // This is the expected arrow types of the actual columns in the file, which might have different types
            // from the unified logical schema or miss
            let this_file_schema = Arc::new(calculate_physical_schema(
                vxf.dtype(),
                &unified_file_schema,
                session.arrow(),
            )?);

            let projected_physical_schema = projection.project_schema(&unified_file_schema)?;

            let expr_adapter = expr_adapter_factory.create(
                Arc::clone(&unified_file_schema),
                Arc::clone(&this_file_schema),
            )?;

            let simplifier = PhysicalExprSimplifier::new(&this_file_schema);

            // The adapter rewrites the expressions to the local file schema, allowing
            // for schema evolution and divergence between the table's schema and individual files.
            let filter = filter
                .map(|filter| {
                    // Expression might now reference columns that don't exist in the file, so we can give it
                    // another simplification pass.
                    simplifier.simplify(expr_adapter.rewrite(filter)?)
                })
                .transpose()?;
            let projection =
                projection.try_map_exprs(|p| simplifier.simplify(expr_adapter.rewrite(p)?))?;

            let ProcessedProjection {
                scan_projection,
                leftover_projection,
            } = if projection_pushdown {
                expr_convertor.split_projection(
                    projection.clone(),
                    &this_file_schema,
                    &projected_physical_schema,
                )?
            } else {
                // When projection pushdown is disabled, read only the required columns
                // and apply the full projection after the scan.
                expr_convertor.no_pushdown_projection(projection.clone(), &this_file_schema)?
            };

            // The schema of the stream returned from the vortex scan.
            // We use a reference schema for types that don't roundtrip (Dictionary, Utf8, etc.).
            let scan_dtype = scan_projection.return_dtype(vxf.dtype()).map_err(|_e| {
                exec_datafusion_err!("Couldn't get the dtype for the underlying Vortex scan")
            })?;

            // When projection pushdown is enabled, the scan outputs the projected columns.
            // When disabled, the scan outputs raw columns and the projection is applied after.
            let scan_reference_schema = if projection_pushdown {
                projected_physical_schema
            } else {
                // Build schema from the raw columns being read
                let column_indices = projection.column_indices();
                let fields: Vec<_> = column_indices
                    .into_iter()
                    .map(|idx| this_file_schema.field(idx).clone())
                    .collect();
                Schema::new(fields)
            };
            let stream_schema =
                calculate_physical_schema(&scan_dtype, &scan_reference_schema, session.arrow())?;

            let leftover_projection = leftover_projection
                .try_map_exprs(|expr| reassign_expr_columns(expr, &stream_schema))?;
            let projector = leftover_projection.make_projector(&stream_schema)?;

            let filter = filter
                .and_then(|f| {
                    // Verify that all filters we've accepted from DataFusion get pushed down.
                    // This will only fail if the user has not configured a suitable
                    // PhysicalExprAdapterFactory on the file source to handle rewriting the
                    // expression to handle missing/reordered columns in the Vortex file.
                    let (pushed, unpushed): (Vec<PhysicalExprRef>, Vec<PhysicalExprRef>) =
                        split_conjunction(&f)
                            .into_iter()
                            .cloned()
                            .partition(|expr| {
                                expr_convertor.can_be_pushed_down(expr, &this_file_schema)
                            });

                    if !unpushed.is_empty() {
                        return Some(Err(exec_datafusion_err!(
                            r#"VortexSource accepted but failed to push {} filters.
                            This should never happen if you have a properly configured
                            PhysicalExprAdapterFactory configured on the source.

                            Failed filters:

                            {unpushed:#?}
                            "#,
                            unpushed.len()
                        )));
                    }

                    make_vortex_predicate(expr_convertor.as_ref(), &pushed).transpose()
                })
                .transpose()?;

            if scan_v2 {
                let row_range = if let Some(file_range) = file.range {
                    let byte_range = Range {
                        start: u64::try_from(file_range.start).map_err(|_| {
                            exec_datafusion_err!("Vortex file range start is negative")
                        })?,
                        end: u64::try_from(file_range.end).map_err(|_| {
                            exec_datafusion_err!("Vortex file range end is negative")
                        })?,
                    };
                    if byte_range.start == 0 && byte_range.end == file.object_meta.size {
                        None
                    } else {
                        // DataFusion partitions a single file by byte ranges. V2 may expose only
                        // coarse top-level split hints, so assigning whole natural splits here can
                        // collapse many byte ranges into a few row ranges. Slice proportionally by
                        // row count; the V2 scan plan will still split the resulting row range into
                        // layout-aware morsels during preparation.
                        let Some(row_range) = byte_range_to_row_range(
                            byte_range,
                            file.object_meta.size,
                            vxf.row_count(),
                        ) else {
                            return Ok(stream::empty().boxed());
                        };
                        Some(row_range)
                    }
                } else {
                    None
                };

                let selection = file
                    .extensions
                    .get::<VortexAccessPlan>()
                    .and_then(|vortex_plan| vortex_plan.selection().cloned())
                    .unwrap_or_default();
                let stream_target_field =
                    Field::new_struct("", stream_schema.fields().clone(), false);
                let file_location = file.object_meta.location.clone();
                let array_stream = vxf
                    .scan_plan_stream(ScanRequest {
                        projection: scan_projection,
                        filter,
                        row_range,
                        selection,
                        ordered: has_output_ordering,
                        limit,
                        ..Default::default()
                    })
                    .map_err(|e| {
                        exec_datafusion_err!("Failed to create Vortex scan2 stream: {e}")
                    })?;
                // The Vortex->Arrow conversion (decode + canonicalize) is CPU-bound, so spawn each
                // chunk's conversion onto the runtime's CPU pool and buffer them. This fans the
                // decode out within a single partition instead of running serially on the consumer's
                // poll thread, which matters for scans with few partitions (e.g. small tables).
                // `buffered` preserves order for ordered consumers.
                let handle = session.handle();
                let decode_concurrency = 4 * get_available_parallelism().unwrap_or(1);
                let converted = array_stream.map(move |chunk| {
                    let session = session.clone();
                    let stream_target_field = stream_target_field.clone();
                    handle.spawn_cpu(move || {
                        let chunk = chunk?;
                        let mut ctx = session.create_execution_ctx();
                        let arrow_session = ctx.session().clone();
                        let arrow = arrow_session.arrow().execute_arrow(
                            chunk,
                            Some(&stream_target_field),
                            &mut ctx,
                        )?;
                        Ok(RecordBatch::from(arrow.as_struct().clone()))
                    })
                });
                let stream = if has_output_ordering {
                    converted.buffered(decode_concurrency).boxed()
                } else {
                    converted.buffer_unordered(decode_concurrency).boxed()
                }
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
                    DataFusionError::External(Box::new(
                        e.with_context(format!("Failed to read Vortex file: {file_location}")),
                    ))
                })
                .try_flatten()
                .map(move |batch| {
                    if projector.projection().as_ref().is_empty() {
                        batch
                    } else {
                        batch.and_then(|b| projector.project_batch(&b))
                    }
                })
                .boxed();

                return if let Some(file_pruner) = file_pruner {
                    Ok(PrunableStream::new(file_pruner, stream).boxed())
                } else {
                    Ok(stream)
                };
            }

            // We share our layout readers with others partitions in the scan, so we can only need to read each layout in each file once.
            let layout_reader =
                layout_reader_for_file(layout_readers.as_ref(), &file.object_meta.location, &vxf)?;

            let mut scan_builder = ScanBuilder::new(session.clone(), Arc::clone(&layout_reader));

            if let Some(vortex_plan) = file.extensions.get::<VortexAccessPlan>() {
                scan_builder = vortex_plan.apply_to_builder(scan_builder);
            }

            if let Some(file_range) = file.range {
                let byte_range = Range {
                    start: u64::try_from(file_range.start)
                        .map_err(|_| exec_datafusion_err!("Vortex file range start is negative"))?,
                    end: u64::try_from(file_range.end)
                        .map_err(|_| exec_datafusion_err!("Vortex file range end is negative"))?,
                };
                if byte_range.start != 0 || byte_range.end != file.object_meta.size {
                    // Full-file scans already cover every natural split. Only translate the
                    // byte range back into row boundaries when DataFusion has trimmed the file.
                    let natural_split_ranges = natural_split_ranges_for_file(
                        natural_split_ranges.as_ref(),
                        &file.object_meta.location,
                        &layout_reader,
                    )?;

                    let Some(row_range) = split_aligned_row_range(
                        byte_range,
                        file.object_meta.size,
                        natural_split_ranges.as_ref(),
                    ) else {
                        return Ok(stream::empty().boxed());
                    };

                    scan_builder = scan_builder.with_row_range(row_range);
                }
            }

            if let Some(limit) = limit
                && filter.is_none()
            {
                scan_builder = scan_builder.with_limit(limit);
            }

            if let Some(concurrency) = scan_concurrency {
                scan_builder = scan_builder.with_concurrency(concurrency);
            }

            let stream_target_field = Field::new_struct("", stream_schema.fields().clone(), false);
            let stream = scan_builder
                .with_metrics_registry(metrics_registry)
                .with_projection(scan_projection)
                .with_some_filter(filter)
                .with_ordered(has_output_ordering)
                .map(move |chunk| {
                    let mut ctx = session.create_execution_ctx();
                    let arrow_session = ctx.session().clone();
                    let arrow = arrow_session.arrow().execute_arrow(
                        chunk,
                        Some(&stream_target_field),
                        &mut ctx,
                    )?;
                    Ok(RecordBatch::from(arrow.as_struct().clone()))
                })
                .into_stream()
                .map_err(|e| exec_datafusion_err!("Failed to create Vortex stream: {e}"))?
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
                    DataFusionError::External(Box::new(e.with_context(format!(
                        "Failed to read Vortex file: {}",
                        file.object_meta.location
                    ))))
                })
                .try_flatten()
                .map(move |batch| {
                    if projector.projection().as_ref().is_empty() {
                        batch
                    } else {
                        batch.and_then(|b| projector.project_batch(&b))
                    }
                })
                .boxed();

            if let Some(file_pruner) = file_pruner {
                Ok(PrunableStream::new(file_pruner, stream).boxed())
            } else {
                Ok(stream)
            }
        }
        .in_current_span()
        .boxed())
    }
}

/// Get or create a shared layout reader for a file. Layout readers are cached (weakly) per path so
/// each file's layout is parsed only once across all partitions of a scan.
fn layout_reader_for_file(
    layout_readers: &DashMap<Path, Weak<dyn LayoutReader>>,
    path: &Path,
    vxf: &VortexFile,
) -> DFResult<Arc<dyn LayoutReader>> {
    let create = || {
        vxf.layout_reader()
            .map_err(|e| DataFusionError::Execution(format!("Failed to create layout reader: {e}")))
    };

    match layout_readers.entry(path.clone()) {
        Entry::Occupied(mut occupied_entry) => {
            if let Some(reader) = occupied_entry.get().upgrade() {
                Ok(reader)
            } else {
                let reader = create()?;
                occupied_entry.insert(Arc::downgrade(&reader));
                Ok(reader)
            }
        }
        Entry::Vacant(vacant_entry) => {
            let reader = create()?;
            vacant_entry.insert(Arc::downgrade(&reader));
            Ok(reader)
        }
    }
}

fn natural_split_ranges_for_file(
    natural_split_ranges: &DashMap<Path, Arc<[Range<u64>]>>,
    path: &Path,
    layout_reader: &Arc<dyn LayoutReader>,
) -> DFResult<Arc<[Range<u64>]>> {
    if let Some(split_ranges) = natural_split_ranges.get(path) {
        return Ok(Arc::clone(split_ranges.value()));
    }

    let split_ranges = compute_natural_split_ranges(layout_reader.as_ref())?;

    match natural_split_ranges.entry(path.clone()) {
        Entry::Occupied(entry) => Ok(Arc::clone(entry.get())),
        Entry::Vacant(entry) => {
            entry.insert(Arc::clone(&split_ranges));
            Ok(split_ranges)
        }
    }
}

fn compute_natural_split_ranges(layout_reader: &dyn LayoutReader) -> DFResult<Arc<[Range<u64>]>> {
    let row_count = layout_reader.row_count();
    let row_range = 0..row_count;
    let split_points: Vec<_> = SplitBy::Layout
        .splits(layout_reader, &row_range, &[FieldMask::All])
        .map_err(|e| exec_datafusion_err!("Failed to compute Vortex natural splits: {e}"))?
        .into_iter()
        .tuple_windows()
        .map(|(s, e)| s..e)
        .collect::<Vec<_>>();

    Ok(split_points.into())
}

fn byte_range_to_row_range(
    byte_range: Range<u64>,
    total_size: u64,
    row_count: u64,
) -> Option<Range<u64>> {
    if byte_range.start >= byte_range.end || total_size == 0 || row_count == 0 {
        return None;
    }

    let start_byte = byte_range.start.min(total_size);
    let end_byte = byte_range.end.min(total_size);
    if start_byte >= end_byte {
        return None;
    }

    let start = byte_to_row(start_byte, total_size, row_count);
    let end = if end_byte == total_size {
        row_count
    } else {
        byte_to_row(end_byte, total_size, row_count)
    };

    (start < end).then_some(start..end)
}

fn byte_to_row(byte: u64, total_size: u64, row_count: u64) -> u64 {
    let row = (u128::from(byte) * u128::from(row_count)) / u128::from(total_size);
    u64::try_from(row).vortex_expect("byte-to-row projection should fit into u64")
}

/// Translate a DataFusion byte range to the contiguous natural split ranges it owns.
/// Most splits are assigned by midpoint, but the leading split stays with the range that owns
/// byte 0 so a tiny first byte range still claims the first rows.
fn split_aligned_row_range(
    byte_range: Range<u64>,
    total_size: u64,
    split_ranges: &[Range<u64>],
) -> Option<Range<u64>> {
    if byte_range.start >= byte_range.end {
        return None;
    }

    let row_count = split_ranges.last().map(|split| split.end)?;
    if row_count == 0 {
        return None;
    }

    let mut owned_splits = split_ranges
        .iter()
        .enumerate()
        .filter_map(|(idx, split_range)| {
            let assignment_byte = split_assignment_byte(idx, split_range, row_count, total_size);
            byte_range.contains(&assignment_byte).then_some(split_range)
        });

    let first_split = owned_splits.next()?;
    let mut row_range = first_split.start..first_split.end;
    for split_range in owned_splits {
        row_range.end = split_range.end;
    }

    Some(row_range)
}

fn split_assignment_byte(
    idx: usize,
    split_range: &Range<u64>,
    row_count: u64,
    total_size: u64,
) -> u64 {
    if idx == 0 && split_range.start == 0 {
        // Byte 0 is the only stable representative for the leading split. A midpoint can fall
        // into the next DataFusion byte range and leave the first range with no rows to read.
        0
    } else {
        split_midpoint_to_byte(split_range, row_count, total_size)
    }
}

fn split_midpoint_to_byte(split_range: &Range<u64>, row_count: u64, total_size: u64) -> u64 {
    let midpoint_row = split_range.start + (split_range.end - split_range.start) / 2;
    let midpoint_byte = (u128::from(midpoint_row) * u128::from(total_size)) / u128::from(row_count);

    u64::try_from(midpoint_byte).vortex_expect("midpoint byte projection should fit into u64")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use arrow_schema::Field;
    use arrow_schema::Fields;
    use arrow_schema::SchemaRef;
    use datafusion::arrow::array::DictionaryArray;
    use datafusion::arrow::array::RecordBatch;
    use datafusion::arrow::array::StringArray;
    use datafusion::arrow::array::StructArray;
    use datafusion::arrow::datatypes::DataType;
    use datafusion::arrow::datatypes::Schema;
    use datafusion::arrow::datatypes::UInt32Type;
    use datafusion::arrow::util::display::FormatOptions;
    use datafusion::arrow::util::pretty::pretty_format_batches_with_options;
    use datafusion::common::record_batch;
    use datafusion::logical_expr::col;
    use datafusion::logical_expr::lit;
    use datafusion::physical_expr::planner::logical2physical;
    use datafusion::physical_expr_adapter::DefaultPhysicalExprAdapterFactory;
    use datafusion::scalar::ScalarValue;
    use datafusion_execution::cache::DefaultFilesMetadataCache;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions as df_expr;
    use datafusion_physical_expr::projection::ProjectionExpr;
    use insta::assert_snapshot;
    use itertools::Itertools;
    use object_store::ObjectStore;
    use object_store::memory::InMemory;
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::array::ArrayRef;
    use vortex::array::arrow::FromArrowArray;
    use vortex::buffer::Buffer;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::io::VortexWrite;
    use vortex::io::object_store::ObjectStoreWrite;
    use vortex::metrics::DefaultMetricsRegistry;
    use vortex::scan::selection::Selection;
    use vortex::session::VortexSession;

    use super::*;
    use crate::VortexAccessPlan;
    use crate::convert::exprs::DefaultExpressionConvertor;
    use crate::persistent::reader::DefaultVortexReaderFactory;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::default);

    #[rstest]
    #[case(0..10, 100, 50, Some(0..5))]
    #[case(10..20, 100, 50, Some(5..10))]
    #[case(90..100, 100, 50, Some(45..50))]
    #[case(100..110, 100, 50, None)]
    #[case(0..1, 100, 50, None)]
    fn test_byte_range_to_row_range(
        #[case] byte_range: Range<u64>,
        #[case] total_size: u64,
        #[case] row_count: u64,
        #[case] expected: Option<Range<u64>>,
    ) {
        assert_eq!(
            byte_range_to_row_range(byte_range, total_size, row_count),
            expected
        );
    }

    #[test]
    fn test_byte_ranges_cover_rows_exactly_once() {
        let total_size = 179_114_706;
        let row_count = 6_001_215;
        let partitions = 18;
        let byte_ranges = (0..partitions)
            .map(|idx| {
                let start = idx * total_size / partitions;
                let end = (idx + 1) * total_size / partitions;
                start..end
            })
            .collect::<Vec<_>>();

        let row_ranges = byte_ranges
            .into_iter()
            .filter_map(|byte_range| byte_range_to_row_range(byte_range, total_size, row_count))
            .collect::<Vec<_>>();

        assert_eq!(u64::try_from(row_ranges.len()), Ok(partitions));
        assert_eq!(row_ranges.first().map(|range| range.start), Some(0));
        assert_eq!(row_ranges.last().map(|range| range.end), Some(row_count));
        assert_eq!(
            row_ranges
                .iter()
                .map(|range| range.end - range.start)
                .sum::<u64>(),
            row_count
        );
        for (left, right) in row_ranges.iter().tuple_windows() {
            assert_eq!(left.end, right.start);
        }
    }

    #[rstest]
    #[case(0..3, 10, vec![0..2, 2..5, 5..10], Some(0..2))]
    #[case(3..7, 10, vec![0..2, 2..5, 5..10], Some(2..5))]
    #[case(1..8, 10, vec![0..1, 1..9, 9..10], Some(1..9))]
    #[case(1..4, 16, vec![0..1, 1..2, 2..3, 3..4], None)]
    #[case(0..1, 10, vec![0..2, 2..10], Some(0..2))]
    fn test_split_aligned_row_range(
        #[case] byte_range: Range<u64>,
        #[case] total_size: u64,
        #[case] split_ranges: Vec<Range<u64>>,
        #[case] expected: Option<Range<u64>>,
    ) {
        assert_eq!(
            split_aligned_row_range(byte_range, total_size, &split_ranges),
            expected
        );
    }

    #[test]
    fn test_split_aligned_ranges_cover_splits_exactly_once() {
        let split_ranges = vec![0..1, 1..4, 4..10, 10..13];
        let byte_ranges = [0..4, 4..8, 8..12, 12..16];

        let assigned = byte_ranges
            .into_iter()
            .filter_map(|byte_range| split_aligned_row_range(byte_range, 16, &split_ranges))
            .collect::<Vec<_>>();

        assert_eq!(assigned, vec![0..4, 4..10, 10..13]);
        assert_eq!(
            assigned
                .iter()
                .map(|range| range.end - range.start)
                .sum::<u64>(),
            13
        );

        let split_starts = split_ranges
            .iter()
            .map(|range| range.start)
            .collect::<Vec<_>>();
        let split_ends = split_ranges
            .iter()
            .map(|range| range.end)
            .collect::<Vec<_>>();

        for range in &assigned {
            assert!(split_starts.contains(&range.start));
            assert!(split_ends.contains(&range.end));
        }

        for (left, right) in assigned.iter().tuple_windows() {
            assert_eq!(left.end, right.start);
        }
    }

    async fn write_arrow_to_vortex(
        object_store: Arc<dyn ObjectStore>,
        path: &str,
        rb: RecordBatch,
    ) -> anyhow::Result<u64> {
        let array = ArrayRef::from_arrow(rb, false)?;
        let path = Path::parse(path)?;

        let mut write = ObjectStoreWrite::new(object_store, &path).await?;
        let summary = SESSION
            .write_options()
            .write(&mut write, array.to_array_stream())
            .await?;
        write.shutdown().await?;

        Ok(summary.size())
    }

    fn make_opener(
        object_store: Arc<dyn ObjectStore>,
        table_schema: TableSchema,
        filter: Option<PhysicalExprRef>,
    ) -> VortexOpener {
        VortexOpener {
            partition: 1,
            session: SESSION.clone(),
            vortex_reader_factory: Arc::new(DefaultVortexReaderFactory::new(object_store)),
            scan_v2: false,
            projection: ProjectionExprs::from_indices(&[0], table_schema.file_schema()),
            filter,
            file_pruning_predicate: None,
            expr_adapter_factory: Arc::new(DefaultPhysicalExprAdapterFactory),
            table_schema,
            batch_size: 100,
            limit: None,
            metrics_registry: Arc::new(DefaultMetricsRegistry::default()),
            layout_readers: Default::default(),
            natural_split_ranges: Default::default(),
            vortex_files: Default::default(),
            has_output_ordering: false,
            expression_convertor: Arc::new(DefaultExpressionConvertor::default()),
            file_metadata_cache: None,
            projection_pushdown: false,
            scan_concurrency: None,
        }
    }

    #[tokio::test]
    async fn test_open() -> anyhow::Result<()> {
        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let file_path = "part=1/file.vortex";
        let batch = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)])).unwrap();
        let data_size =
            write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch.clone()).await?;

        let file_schema = batch.schema();
        let mut file = PartitionedFile::new(file_path.to_string(), data_size);
        file.partition_values = vec![ScalarValue::Int32(Some(1))];

        let table_schema = TableSchema::new(
            Arc::clone(&file_schema),
            vec![Arc::new(Field::new("part", DataType::Int32, false))],
        );

        // filter matches partition value
        let filter = col("part").eq(lit(1));
        let filter = logical2physical(&filter, table_schema.table_schema());

        let opener = make_opener(
            Arc::clone(&object_store),
            table_schema.clone(),
            Some(filter),
        );
        let stream = opener.open(file.clone()).unwrap().await.unwrap();

        let data = stream.try_collect::<Vec<_>>().await?;
        let num_batches = data.len();
        let num_rows = data.iter().map(|rb| rb.num_rows()).sum::<usize>();

        assert_eq!((num_batches, num_rows), (1, 3));

        // filter doesn't matches partition value
        let filter = col("part").eq(lit(2));
        let filter = logical2physical(&filter, table_schema.table_schema());

        let opener = make_opener(
            Arc::clone(&object_store),
            table_schema.clone(),
            Some(filter),
        );
        let stream = opener.open(file.clone()).unwrap().await.unwrap();

        let data = stream.try_collect::<Vec<_>>().await?;
        let num_batches = data.len();
        let num_rows = data.iter().map(|rb| rb.num_rows()).sum::<usize>();
        assert_eq!((num_batches, num_rows), (0, 0));

        Ok(())
    }

    #[tokio::test]
    async fn test_open_empty_file() -> anyhow::Result<()> {
        use futures::TryStreamExt;

        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let data_batch = record_batch!(("a", Int32, Vec::<i32>::new())).unwrap();
        let file_path = "part=1/empty.vortex";
        let file_size =
            write_arrow_to_vortex(Arc::clone(&object_store), file_path, data_batch.clone()).await?;

        let file_schema = data_batch.schema();
        // Parallel scans may attach a byte range even for empty files; the
        // opener must return early before attempting split-aligned translation.
        let file =
            PartitionedFile::new_with_range(file_path.to_string(), file_size, 0, file_size as i64);

        let table_schema = TableSchema::from_file_schema(Arc::clone(&file_schema));

        let opener = make_opener(object_store, table_schema, None);
        let stream = opener.open(file)?.await?;
        let data = stream.try_collect::<Vec<_>>().await?;

        assert_eq!(data.len(), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_open_scan_v2() -> anyhow::Result<()> {
        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let file_path = "scan2/file.vortex";
        let batch = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)])).unwrap();
        let data_size =
            write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch.clone()).await?;

        let table_schema = TableSchema::from_file_schema(batch.schema());
        let mut opener = make_opener(object_store, table_schema, None);
        opener.scan_v2 = true;

        let stream = opener
            .open(PartitionedFile::new(file_path.to_string(), data_size))?
            .await?;
        let data = stream.try_collect::<Vec<_>>().await?;
        let num_rows = data.iter().map(|rb| rb.num_rows()).sum::<usize>();

        assert_eq!(num_rows, 3);

        Ok(())
    }

    #[tokio::test]
    async fn test_open_populates_file_metadata_cache() -> anyhow::Result<()> {
        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let file_path = "cached/file.vortex";
        let batch = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)])).unwrap();
        let data_size =
            write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch.clone()).await?;

        let file = PartitionedFile::new(file_path.to_string(), data_size);
        let table_schema = TableSchema::from_file_schema(batch.schema());

        let cache: Arc<dyn FileMetadataCache> =
            Arc::new(DefaultFilesMetadataCache::new(64 * 1024 * 1024));
        let mut opener = make_opener(Arc::clone(&object_store), table_schema, None);
        opener.file_metadata_cache = Some(Arc::clone(&cache));

        // The first open misses the cache and must write the parsed footer back.
        let stream = opener.open(file.clone())?.await?;
        stream.try_collect::<Vec<_>>().await?;

        let entry = cache
            .get(file.path())
            .ok_or_else(|| anyhow::anyhow!("footer was not cached after open"))?;
        assert!(entry.is_valid_for(&file.object_meta));
        assert!(
            entry
                .file_metadata
                .as_any()
                .downcast_ref::<CachedVortexMetadata>()
                .is_some()
        );

        // The second open hits the cache and still returns the same data.
        let stream = opener.open(file.clone())?.await?;
        let data = stream.try_collect::<Vec<_>>().await?;
        assert_eq!(data.iter().map(|rb| rb.num_rows()).sum::<usize>(), 3);

        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn test_open_files_different_table_schema() -> anyhow::Result<()> {
        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;

        let file1 = {
            let file1_path = "/path/file1.vortex";
            let batch1 = record_batch!(("a", Int32, vec![Some(1), Some(2), Some(3)])).unwrap();
            let data_size1 =
                write_arrow_to_vortex(Arc::clone(&object_store), file1_path, batch1).await?;
            PartitionedFile::new(file1_path.to_string(), data_size1)
        };

        let file2 = {
            let file2_path = "/path/file2.vortex";
            let batch2 = record_batch!(("a", Int16, vec![Some(-1), Some(-2), Some(-3)])).unwrap();
            let data_size2 =
                write_arrow_to_vortex(Arc::clone(&object_store), file2_path, batch2).await?;
            PartitionedFile::new(file2_path.to_string(), data_size2)
        };

        // Table schema has can accommodate both files
        let table_schema = TableSchema::from_file_schema(Arc::new(Schema::new(vec![Field::new(
            "a",
            DataType::Int32,
            true,
        )])));

        let make_opener = |filter| VortexOpener {
            partition: 1,
            session: SESSION.clone(),
            vortex_reader_factory: Arc::new(DefaultVortexReaderFactory::new(Arc::clone(
                &object_store,
            ))),
            scan_v2: false,
            projection: ProjectionExprs::from_indices(&[0], table_schema.file_schema()),
            filter: Some(filter),
            file_pruning_predicate: None,
            expr_adapter_factory: Arc::new(DefaultPhysicalExprAdapterFactory),
            table_schema: table_schema.clone(),
            batch_size: 100,
            limit: None,
            metrics_registry: Arc::new(DefaultMetricsRegistry::default()),
            layout_readers: Default::default(),
            natural_split_ranges: Default::default(),
            vortex_files: Default::default(),
            has_output_ordering: false,
            expression_convertor: Arc::new(DefaultExpressionConvertor::default()),
            file_metadata_cache: None,
            projection_pushdown: false,
            scan_concurrency: None,
        };

        let filter = col("a").lt(lit(100_i32));
        let filter = logical2physical(&filter, table_schema.table_schema());

        let opener1 = make_opener(Arc::clone(&filter));
        let stream = opener1.open(file1)?.await?;

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

        let opener2 = make_opener(Arc::clone(&filter));
        let stream = opener2.open(file2)?.await?;

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
        let data_size = write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch).await?;
        let file = PartitionedFile::new(file_path.to_string(), data_size);

        // Table schema has columns in different order: a, b, c
        let table_schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
            Field::new("c", DataType::Int32, true),
        ]));

        let opener = VortexOpener {
            partition: 1,
            session: SESSION.clone(),
            vortex_reader_factory: Arc::new(DefaultVortexReaderFactory::new(object_store)),
            scan_v2: false,
            projection: ProjectionExprs::from_indices(&[0, 1, 2], &table_schema),
            filter: None,
            file_pruning_predicate: None,
            expr_adapter_factory: Arc::new(DefaultPhysicalExprAdapterFactory),
            table_schema: TableSchema::from_file_schema(Arc::clone(&table_schema)),
            batch_size: 100,
            limit: None,
            metrics_registry: Arc::new(DefaultMetricsRegistry::default()),
            layout_readers: Default::default(),
            natural_split_ranges: Default::default(),
            vortex_files: Default::default(),
            has_output_ordering: false,
            expression_convertor: Arc::new(DefaultExpressionConvertor::default()),
            file_metadata_cache: None,
            projection_pushdown: false,
            scan_concurrency: None,
        };

        let stream = opener.open(file)?.await?;

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
        let data_size = write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch).await?;

        // Table schema has an extra utf8 field.
        let table_schema = TableSchema::from_file_schema(Arc::new(Schema::new(vec![Field::new(
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
        )])));

        let opener = make_opener(
            Arc::clone(&object_store),
            table_schema.clone(),
            // expression references my_struct column which has different fields in each
            // field.
            Some(logical2physical(
                &col("my_struct").is_not_null(),
                table_schema.table_schema(),
            )),
        );

        // The opener should be able to open the file with a filter on the
        // struct column.
        let data = opener
            .open(PartitionedFile::new(file_path.to_string(), data_size))?
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        assert_eq!(data.len(), 1);
        assert_eq!(data[0].num_rows(), 3);

        Ok(())
    }

    #[tokio::test]
    // Minimal reproducing test for the schema projection bug.
    // Before the fix, this would fail with a cast error when the file schema
    // and table schema have different field orders and we project a subset of columns.
    async fn test_projection_bug_minimal_repro() -> anyhow::Result<()> {
        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let file_path = "/path/file.vortex";

        // File has columns in order: a, b, c with simple types
        let batch = record_batch!(
            ("a", Int32, vec![Some(1)]),
            ("b", Utf8, vec![Some("test")]),
            ("c", Int32, vec![Some(2)])
        )
        .unwrap();
        let data_size = write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch).await?;

        // Table schema has columns in DIFFERENT order: c, a, b
        // and different types that require casting (Utf8 -> Dictionary)
        let table_schema = TableSchema::new(
            Arc::new(Schema::new(vec![
                Field::new("c", DataType::Int32, true),
                Field::new("a", DataType::Int32, true),
                Field::new(
                    "b",
                    DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
                    true,
                ),
            ])),
            vec![],
        );

        // Project columns [0, 2] from table schema, which should give us: c, b
        // Before the fix, the schema adapter would get confused about which fields
        // to select from the file, causing incorrect type mappings.
        let projection = vec![0, 2];

        let opener = VortexOpener {
            partition: 1,
            session: SESSION.clone(),
            vortex_reader_factory: Arc::new(DefaultVortexReaderFactory::new(Arc::clone(
                &object_store,
            ))),
            scan_v2: false,
            projection: ProjectionExprs::from_indices(
                projection.as_ref(),
                table_schema.file_schema(),
            ),
            filter: None,
            file_pruning_predicate: None,
            expr_adapter_factory: Arc::new(DefaultPhysicalExprAdapterFactory),
            table_schema: table_schema.clone(),
            batch_size: 100,
            limit: None,
            metrics_registry: Arc::new(DefaultMetricsRegistry::default()),
            layout_readers: Default::default(),
            natural_split_ranges: Default::default(),
            vortex_files: Default::default(),
            has_output_ordering: false,
            expression_convertor: Arc::new(DefaultExpressionConvertor::default()),
            file_metadata_cache: None,
            projection_pushdown: false,
            scan_concurrency: None,
        };

        // This should succeed and return the correctly projected and cast data
        let data = opener
            .open(PartitionedFile::new(file_path.to_string(), data_size))?
            .await?
            .try_collect::<Vec<_>>()
            .await?;

        // Verify the columns are in the right order and have the right values
        use datafusion::arrow::util::pretty::pretty_format_batches_with_options;
        let format_opts = FormatOptions::new().with_types_info(true);
        assert_snapshot!(pretty_format_batches_with_options(&data, &format_opts)?.to_string(), @r"
        +-------+--------------------------+
        | c     | b                        |
        | Int32 | Dictionary(UInt32, Utf8) |
        +-------+--------------------------+
        | 2     | test                     |
        +-------+--------------------------+
        ");

        Ok(())
    }

    fn make_test_batch_with_10_rows() -> RecordBatch {
        record_batch!(
            ("a", Int32, (0..=9).map(Some).collect::<Vec<_>>()),
            (
                "b",
                Utf8,
                (0..=9).map(|i| Some(format!("r{}", i))).collect::<Vec<_>>()
            )
        )
        .unwrap()
    }

    fn make_test_opener(
        object_store: Arc<dyn ObjectStore>,
        schema: SchemaRef,
        projection: ProjectionExprs,
    ) -> VortexOpener {
        VortexOpener {
            partition: 1,
            session: SESSION.clone(),
            vortex_reader_factory: Arc::new(DefaultVortexReaderFactory::new(object_store)),
            scan_v2: false,
            projection,
            filter: None,
            file_pruning_predicate: None,
            expr_adapter_factory: Arc::new(DefaultPhysicalExprAdapterFactory),
            table_schema: TableSchema::from_file_schema(schema),
            batch_size: 100,
            limit: None,
            metrics_registry: Arc::new(DefaultMetricsRegistry::default()),
            layout_readers: Default::default(),
            natural_split_ranges: Default::default(),
            vortex_files: Default::default(),
            has_output_ordering: false,
            expression_convertor: Arc::new(DefaultExpressionConvertor::default()),
            file_metadata_cache: None,
            projection_pushdown: false,
            scan_concurrency: None,
        }
    }

    #[tokio::test]
    // Test that Selection::IncludeByIndex filters to specific row indices.
    async fn test_selection_include_by_index() -> anyhow::Result<()> {
        use datafusion::arrow::util::pretty::pretty_format_batches_with_options;
        use vortex::buffer::Buffer;
        use vortex::scan::selection::Selection;

        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let file_path = "/path/file.vortex";

        let batch = make_test_batch_with_10_rows();
        let data_size =
            write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch.clone()).await?;

        let schema = batch.schema();
        let mut file = PartitionedFile::new(file_path.to_string(), data_size);
        file.extensions
            .insert(
                VortexAccessPlan::default().with_selection(Selection::IncludeByIndex(
                    Buffer::from_iter(vec![1, 3, 5, 7]),
                )),
            );

        let opener = make_test_opener(
            Arc::clone(&object_store),
            Arc::clone(&schema),
            ProjectionExprs::from_indices(&[0, 1], &schema),
        );

        let stream = opener.open(file)?.await?;
        let data = stream.try_collect::<Vec<_>>().await?;
        let format_opts = FormatOptions::new().with_types_info(true);

        assert_snapshot!(pretty_format_batches_with_options(&data, &format_opts)?.to_string(), @r"
        +-------+------+
        | a     | b    |
        | Int32 | Utf8 |
        +-------+------+
        | 1     | r1   |
        | 3     | r3   |
        | 5     | r5   |
        | 7     | r7   |
        +-------+------+
        ");

        Ok(())
    }

    #[tokio::test]
    // Test that Selection::ExcludeByIndex excludes specific row indices.
    async fn test_selection_exclude_by_index() -> anyhow::Result<()> {
        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let file_path = "/path/file.vortex";

        let batch = make_test_batch_with_10_rows();
        let data_size =
            write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch.clone()).await?;

        let schema = batch.schema();
        let mut file = PartitionedFile::new(file_path.to_string(), data_size);
        file.extensions
            .insert(
                VortexAccessPlan::default().with_selection(Selection::ExcludeByIndex(
                    Buffer::from_iter(vec![0, 2, 4, 6, 8]),
                )),
            );

        let opener = make_test_opener(
            Arc::clone(&object_store),
            Arc::clone(&schema),
            ProjectionExprs::from_indices(&[0, 1], &schema),
        );

        let stream = opener.open(file)?.await?;
        let data = stream.try_collect::<Vec<_>>().await?;
        let format_opts = FormatOptions::new().with_types_info(true);

        assert_snapshot!(pretty_format_batches_with_options(&data, &format_opts)?.to_string(), @r"
        +-------+------+
        | a     | b    |
        | Int32 | Utf8 |
        +-------+------+
        | 1     | r1   |
        | 3     | r3   |
        | 5     | r5   |
        | 7     | r7   |
        | 9     | r9   |
        +-------+------+
        ");

        Ok(())
    }

    #[tokio::test]
    // Test that Selection::All returns all rows.
    async fn test_selection_all() -> anyhow::Result<()> {
        use vortex::scan::selection::Selection;

        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let file_path = "/path/file.vortex";

        let batch = make_test_batch_with_10_rows();
        let data_size =
            write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch.clone()).await?;

        let schema = batch.schema();
        let mut file = PartitionedFile::new(file_path.to_string(), data_size);
        file.extensions
            .insert(VortexAccessPlan::default().with_selection(Selection::All));

        let opener = make_test_opener(
            Arc::clone(&object_store),
            Arc::clone(&schema),
            ProjectionExprs::from_indices(&[0], &schema),
        );

        let stream = opener.open(file)?.await?;
        let data = stream.try_collect::<Vec<_>>().await?;

        let total_rows: usize = data.iter().map(|rb| rb.num_rows()).sum();
        assert_eq!(total_rows, 10);

        Ok(())
    }

    #[tokio::test]
    // Test that when no extensions are provided, all rows are returned (backward compatibility).
    async fn test_selection_no_extensions() -> anyhow::Result<()> {
        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let file_path = "/path/file.vortex";

        let batch = make_test_batch_with_10_rows();
        let data_size =
            write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch.clone()).await?;

        let schema = batch.schema();
        let file = PartitionedFile::new(file_path.to_string(), data_size);
        // file.extensions is None by default

        let opener = make_test_opener(
            Arc::clone(&object_store),
            Arc::clone(&schema),
            ProjectionExprs::from_indices(&[0], &schema),
        );

        let stream = opener.open(file)?.await?;
        let data = stream.try_collect::<Vec<_>>().await?;

        let total_rows: usize = data.iter().map(|rb| rb.num_rows()).sum();
        assert_eq!(total_rows, 10);

        Ok(())
    }

    #[tokio::test]
    async fn test_projection_expr_pushdown() -> anyhow::Result<()> {
        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;
        let file_path = "/path/file.vortex";

        let batch = record_batch!(
            ("a", Int32, vec![Some(1), Some(2), Some(3)]),
            ("b", Int32, vec![Some(10), Some(20), Some(30)])
        )
        .unwrap();
        let data_size =
            write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch.clone()).await?;

        let file_schema = batch.schema();
        let table_schema = TableSchema::from_file_schema(Arc::clone(&file_schema));

        // Create a projection that includes an arithmetic expression: a + b * 2
        let col_a = df_expr::col("a", &file_schema)?;
        let col_b = df_expr::col("b", &file_schema)?;
        let two = df_expr::lit(ScalarValue::Int32(Some(2)));

        // b * 2
        let b_times_2 = df_expr::binary(col_b, Operator::Multiply, two, &file_schema)?;
        // a + (b * 2)
        let a_plus_b_times_2 = df_expr::binary(col_a, Operator::Plus, b_times_2, &file_schema)?;

        let projection = ProjectionExprs::new(vec![ProjectionExpr::new(
            a_plus_b_times_2,
            "result".to_string(),
        )]);

        let opener = VortexOpener {
            partition: 1,
            session: SESSION.clone(),
            vortex_reader_factory: Arc::new(DefaultVortexReaderFactory::new(Arc::clone(
                &object_store,
            ))),
            scan_v2: false,
            projection,
            filter: None,
            file_pruning_predicate: None,
            expr_adapter_factory: Arc::new(DefaultPhysicalExprAdapterFactory),
            table_schema,
            batch_size: 100,
            limit: None,
            metrics_registry: Arc::new(DefaultMetricsRegistry::default()),
            layout_readers: Default::default(),
            natural_split_ranges: Default::default(),
            vortex_files: Default::default(),
            has_output_ordering: false,
            expression_convertor: Arc::new(DefaultExpressionConvertor::default()),
            file_metadata_cache: None,
            projection_pushdown: false,
            scan_concurrency: None,
        };

        let file = PartitionedFile::new(file_path.to_string(), data_size);
        let stream = opener.open(file)?.await?;
        let data = stream.try_collect::<Vec<_>>().await?;

        // Expected: a + b * 2
        // row 0: 1 + 10 * 2 = 21
        // row 1: 2 + 20 * 2 = 42
        // row 2: 3 + 30 * 2 = 63
        assert_snapshot!(pretty_format_batches_with_options(&data, &FormatOptions::new().with_types_info(true))?.to_string(), @r"
        +--------+
        | result |
        | Int32  |
        +--------+
        | 21     |
        | 42     |
        | 63     |
        +--------+
        ");

        Ok(())
    }

    /// When a Struct contains Dictionary fields, writing to vortex and reading back
    /// should preserve the Dictionary type.
    #[tokio::test]
    async fn test_struct_with_dictionary_roundtrip() -> anyhow::Result<()> {
        let object_store = Arc::new(InMemory::new()) as Arc<dyn ObjectStore>;

        let struct_fields = Fields::from(vec![
            Field::new_dictionary("a", DataType::UInt32, DataType::Utf8, true),
            Field::new_dictionary("b", DataType::UInt32, DataType::Utf8, true),
        ]);
        let struct_array = StructArray::new(
            struct_fields.clone(),
            vec![
                Arc::new(DictionaryArray::<UInt32Type>::from_iter(["x", "y", "x"])),
                Arc::new(DictionaryArray::<UInt32Type>::from_iter(["p", "p", "q"])),
            ],
            None,
        );

        let schema = Arc::new(Schema::new(vec![Field::new(
            "labels",
            DataType::Struct(struct_fields.clone()),
            false,
        )]));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(struct_array)])?;

        let file_path = "/test.vortex";
        let data_size = write_arrow_to_vortex(Arc::clone(&object_store), file_path, batch).await?;

        let opener = make_test_opener(
            Arc::clone(&object_store),
            Arc::clone(&schema),
            ProjectionExprs::from_indices(&[0], &schema),
        );
        let data: Vec<_> = opener
            .open(PartitionedFile::new(file_path.to_string(), data_size))?
            .await?
            .try_collect()
            .await?;

        assert_eq!(
            data[0].schema().field(0).data_type(),
            &DataType::Struct(struct_fields),
            "Struct(Dictionary) type should be preserved"
        );
        Ok(())
    }
}
