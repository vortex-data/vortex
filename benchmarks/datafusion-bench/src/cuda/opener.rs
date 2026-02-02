// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA-accelerated file opener for Vortex files.
//!
//! This is a modified version of `vortex_datafusion::persistent::opener::VortexOpener`
//! that uses CUDA execution instead of CPU execution for array processing.

use std::ops::Range;
use std::sync::Arc;
use std::sync::Weak;

use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::ScalarValue;
use datafusion_common::exec_datafusion_err;
use datafusion_datasource::FileRange;
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::TableSchema;
use datafusion_datasource::file_stream::FileOpenFuture;
use datafusion_datasource::file_stream::FileOpener;
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
use object_store::path::Path;
use tracing::Instrument;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrow::ArrowArrayExecutor;
use vortex::error::VortexError;
use vortex::error::vortex_err;
use vortex::file::OpenOptionsSessionExt;
use vortex::io::InstrumentedReadAt;
use vortex::layout::LayoutReader;
use vortex::metrics::VortexMetrics;
use vortex::scan::ScanBuilder;
use vortex::session::VortexSession;
use vortex_cuda::CudaSession;
use vortex_cuda::executor::CudaArrayExt;
use vortex_datafusion::ExpressionConvertor;
use vortex_datafusion::ProcessedProjection;
use vortex_datafusion::VortexAccessPlan;
use vortex_datafusion::VortexReaderFactory;
use vortex_datafusion::calculate_physical_schema;
use vortex_datafusion::make_vortex_predicate;
use vortex_utils::aliases::dash_map::DashMap;
use vortex_utils::aliases::dash_map::Entry;

// Note: We avoid using CachedVortexMetadata and PrunableStream from vortex_datafusion
// as they are not public. Instead, we simplify the implementation.

/// File opener that uses CUDA for array execution.
///
/// This is similar to `VortexOpener` but uses `execute_cuda()` instead of CPU execution.
#[derive(Clone)]
pub(crate) struct CudaVortexOpener {
    pub session: VortexSession,
    pub vortex_reader_factory: Arc<dyn VortexReaderFactory>,
    pub projection: ProjectionExprs,
    pub filter: Option<PhysicalExprRef>,
    pub file_pruning_predicate: Option<PhysicalExprRef>,
    pub expr_adapter_factory: Arc<dyn PhysicalExprAdapterFactory>,
    pub table_schema: TableSchema,
    pub batch_size: usize,
    pub limit: Option<u64>,
    pub metrics: VortexMetrics,
    pub layout_readers: Arc<DashMap<Path, Weak<dyn LayoutReader>>>,
    pub has_output_ordering: bool,
    pub expression_convertor: Arc<dyn ExpressionConvertor>,
    pub file_metadata_cache: Option<Arc<dyn FileMetadataCache>>,
}

impl FileOpener for CudaVortexOpener {
    fn open(&self, file: PartitionedFile) -> DFResult<FileOpenFuture> {
        let session = self.session.clone();
        let metrics = self
            .metrics
            .child_with_tags([("file_path", file.path().to_string())]);

        let mut projection = self.projection.clone();
        let mut filter = self.filter.clone();

        let reader = self
            .vortex_reader_factory
            .create_reader(file.path().as_ref(), &session)?;

        let reader = InstrumentedReadAt::new(reader, &metrics);

        let file_pruning_predicate = self.file_pruning_predicate.clone();
        let expr_adapter_factory = self.expr_adapter_factory.clone();
        let file_metadata_cache = self.file_metadata_cache.clone();

        let unified_file_schema = self.table_schema.file_schema().clone();
        let batch_size = self.batch_size;
        let limit = self.limit;
        let layout_reader = self.layout_readers.clone();
        let has_output_ordering = self.has_output_ordering;

        let expr_convertor = self.expression_convertor.clone();

        // Replace column access for partition columns with literals
        #[allow(clippy::disallowed_types)]
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
            // or file statistics available.
            let mut file_pruner = file_pruning_predicate
                .filter(|p| is_dynamic_physical_expr(p) || file.has_statistics())
                .and_then(|predicate| {
                    FilePruner::try_new(
                        predicate.clone(),
                        &unified_file_schema,
                        &file,
                        Count::default(),
                    )
                });

            // Check if this file should be pruned based on statistics/partition values.
            if let Some(file_pruner) = file_pruner.as_mut()
                && file_pruner.should_prune()?
            {
                return Ok(stream::empty().boxed());
            }

            let open_opts = session
                .open_options()
                .with_file_size(file.object_meta.size)
                .with_metrics(metrics.clone());

            // Note: We skip using CachedVortexMetadata here to avoid depending on
            // private vortex-datafusion internals. The footer will be re-read from the file.
            drop(file_metadata_cache);

            let vxf = open_opts
                .open_read(reader)
                .await
                .map_err(|e| exec_datafusion_err!("Failed to open Vortex file {e}"))?;

            let this_file_schema = Arc::new(calculate_physical_schema(
                vxf.dtype(),
                &unified_file_schema,
            )?);

            let projected_physical_schema = projection.project_schema(&unified_file_schema)?;

            let expr_adapter = expr_adapter_factory.create(
                Arc::clone(&unified_file_schema),
                Arc::clone(&this_file_schema),
            );

            let simplifier = PhysicalExprSimplifier::new(&this_file_schema);

            let filter = filter
                .map(|filter| simplifier.simplify(expr_adapter.rewrite(filter)?))
                .transpose()?;
            let projection =
                projection.try_map_exprs(|p| simplifier.simplify(expr_adapter.rewrite(p)?))?;

            let ProcessedProjection {
                scan_projection,
                leftover_projection,
            } = expr_convertor.split_projection(
                projection,
                &this_file_schema,
                &projected_physical_schema,
            )?;

            let scan_dtype = scan_projection.return_dtype(vxf.dtype()).map_err(|_e| {
                exec_datafusion_err!("Couldn't get the dtype for the underlying Vortex scan")
            })?;
            let stream_schema = calculate_physical_schema(&scan_dtype, &projected_physical_schema)?;

            let leftover_projection = leftover_projection
                .try_map_exprs(|expr| reassign_expr_columns(expr, &stream_schema))?;
            let projector = leftover_projection.make_projector(&stream_schema)?;

            // Share layout readers with other partitions
            let layout_reader = match layout_reader.entry(file.object_meta.location.clone()) {
                Entry::Occupied(mut occupied_entry) => {
                    if let Some(reader) = occupied_entry.get().upgrade() {
                        tracing::trace!("reusing layout reader for {}", occupied_entry.key());
                        reader
                    } else {
                        tracing::trace!("creating layout reader for {}", occupied_entry.key());
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
                    tracing::trace!("creating layout reader for {}", vacant_entry.key());
                    let reader = vxf.layout_reader().map_err(|e| {
                        DataFusionError::Execution(format!("Failed to create layout reader: {e}"))
                    })?;
                    vacant_entry.insert(Arc::downgrade(&reader));

                    reader
                }
            };

            let mut scan_builder = ScanBuilder::new(session.clone(), layout_reader);

            if let Some(extensions) = file.extensions
                && let Some(vortex_plan) = extensions.downcast_ref::<VortexAccessPlan>()
            {
                scan_builder = vortex_plan.apply_to_builder(scan_builder);
            }

            if let Some(file_range) = file.range {
                scan_builder = apply_byte_range(
                    file_range,
                    file.object_meta.size,
                    vxf.row_count(),
                    scan_builder,
                );
            }

            let filter = filter
                .and_then(|f| {
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

            if let Some(limit) = limit
                && filter.is_none()
            {
                scan_builder = scan_builder.with_limit(limit);
            }

            // CUDA EXECUTION PATH
            // This is the key difference from the standard VortexOpener:
            // We use execute_cuda() instead of execute_record_batch()
            let stream = scan_builder
                .with_metrics(metrics)
                .with_projection(scan_projection)
                .with_some_filter(filter)
                .with_ordered(has_output_ordering)
                .into_stream()
                .map_err(|e| exec_datafusion_err!("Failed to create Vortex stream: {e}"))?
                .then(move |chunk_result| {
                    let session = session.clone();
                    let stream_schema = stream_schema.clone();
                    async move {
                        let chunk = chunk_result?;

                        // Execute on CUDA - this is the main difference from CPU path
                        let mut cuda_ctx =
                            CudaSession::create_execution_ctx(&session).map_err(|e| {
                                vortex_err!("Failed to create CUDA execution context: {e}")
                            })?;

                        tracing::debug!("Executing array on CUDA device");
                        let canonical = chunk.execute_cuda(&mut cuda_ctx).await?;

                        // Convert canonical result to ArrayRef and then to RecordBatch
                        let array: ArrayRef = canonical.into_array();
                        let mut cpu_ctx = session.create_execution_ctx();
                        array.execute_record_batch(&stream_schema, &mut cpu_ctx)
                    }
                })
                .map_ok(move |rb| {
                    // Slice the stream into batches respecting datafusion's configured batch size
                    stream::iter(
                        (0..rb.num_rows().div_ceil(batch_size * 2))
                            .flat_map(move |block_idx| {
                                let offset = block_idx * batch_size * 2;

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

            // Note: We skip using PrunableStream here to avoid depending on private
            // vortex-datafusion internals. File pruning still happens at the scan level.
            drop(file_pruner);
            Ok(stream)
        }
        .in_current_span()
        .boxed())
    }
}

/// If the file has a [`FileRange`], we translate it into a row range in the file for the scan.
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

    start_row..u64::min(row_count, end_row)
}
