// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Use [`VortexDataSource`] to adapt an existing Vortex [`DataSourceRef`] into
//! a DataFusion [`DataSource`] without going through file discovery.
//!
//! [`VortexDataSource`] is responsible for:
//!
//! - exposing an Arrow schema and output statistics to DataFusion,
//! - translating DataFusion projection, filter, and limit pushdown into a
//!   Vortex [`ScanRequest`],
//! - executing the Vortex scan and converting the results into Arrow
//!   `RecordBatch` values.
//!
//! # Example: Create a `DataSourceExec`
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use arrow_schema::Schema;
//! use datafusion_datasource::source::DataSourceExec;
//! use vortex::VortexSessionDefault;
//! use vortex::scan::DataSourceRef;
//! use vortex::session::VortexSession;
//! use vortex_datafusion::v2::VortexDataSource;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let data_source: DataSourceRef = todo!();
//! let data_source = VortexDataSource::builder(data_source, VortexSession::default())
//!     .with_arrow_schema(Arc::new(Schema::empty()))
//!     .build()
//!     .await?;
//!
//! let exec = DataSourceExec::from_data_source(data_source);
//! # let _ = exec;
//! # Ok(())
//! # }
//! ```
//!
//! # Execution Flow
//!
//! ```text
//!             ▲
//!             │  RecordBatch stream
//!             │
//! ┌───────────────────────┐
//! │     DataSourceExec    │
//! └───────────────────────┘
//!             ▲
//!             │  DataFusion pushdown
//!             │  (projection/filter/limit)
//! ┌───────────────────────┐
//! │   VortexDataSource    │
//! └───────────────────────┘
//!             ▲
//!             │  final ScanRequest
//! ┌───────────────────────┐
//! │    DataSourceRef      │
//! └───────────────────────┘
//! ```
//!
//! Compared with [`crate::VortexSource`], this path starts from an existing
//! Vortex source rather than from DataFusion-managed file discovery.
//!
//! [`DataSource`]: datafusion_datasource::source::DataSource
//! [`DataSourceRef`]: vortex::scan::DataSourceRef
//! [`ScanRequest`]: vortex::scan::ScanRequest

use std::fmt;
use std::fmt::Formatter;
use std::sync::Arc;

use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use datafusion_common::ColumnStatistics;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::Statistics;
use datafusion_common::arrow::array::AsArray;
use datafusion_common::arrow::array::RecordBatch;
use datafusion_common::stats::Precision as DFPrecision;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_datasource::source::DataSource;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_expr::Partitioning;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr::utils::reassign_expr_columns;
use datafusion_physical_expr_common::sort_expr::LexOrdering;
use datafusion_physical_plan::DisplayFormatType;
use datafusion_physical_plan::filter_pushdown::FilterPushdownPropagation;
use datafusion_physical_plan::filter_pushdown::PushedDown;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::try_join_all;
use vortex::array::VortexSessionExecute;
use vortex::dtype::DType;
use vortex::dtype::FieldPath;
use vortex::dtype::Nullability;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::expr::Expression;
use vortex::expr::and as vx_and;
use vortex::expr::get_item;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::expr::stats::Precision;
use vortex::expr::transform::replace;
use vortex::io::session::RuntimeSessionExt;
use vortex::scan::DataSourceRef;
use vortex::scan::ScanRequest;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;
use vortex_utils::parallelism::get_available_parallelism;

use crate::convert::exprs::DefaultExpressionConvertor;
use crate::convert::exprs::ExpressionConvertor;
use crate::convert::exprs::ProcessedProjection;
use crate::convert::exprs::make_vortex_predicate;
use crate::convert::stats::stats_set_to_df;

/// Builder for [`VortexDataSource`].
///
/// Use the builder to declare how an existing Vortex
/// [`DataSourceRef`] should appear to DataFusion.
/// In particular, it lets you choose:
///
/// - the Arrow schema DataFusion should see,
/// - an initial top-level projection if the embedding system already knows
///   which columns are needed.
///
/// The resulting [`VortexDataSource`] is ready to plug into
/// [`DataSourceExec`] or other DataFusion physical planning code.
///
/// # Example
///
/// ```no_run
/// use std::sync::Arc;
///
/// use arrow_schema::Schema;
/// use vortex::VortexSessionDefault;
/// use vortex::scan::DataSourceRef;
/// use vortex::session::VortexSession;
/// use vortex_datafusion::v2::VortexDataSource;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let data_source: DataSourceRef = todo!();
/// let data_source = VortexDataSource::builder(data_source, VortexSession::default())
///     .with_arrow_schema(Arc::new(Schema::empty()))
///     .with_projection(vec![0])
///     .build()
///     .await?;
/// # let _ = data_source;
/// # Ok(())
/// # }
/// ```
///
/// [`DataSourceRef`]: vortex::scan::DataSourceRef
/// [`DataSourceExec`]: datafusion_datasource::source::DataSourceExec
pub struct VortexDataSourceBuilder {
    data_source: DataSourceRef,
    session: VortexSession,

    arrow_schema: Option<SchemaRef>,
    projection: Option<Vec<usize>>,
}

impl VortexDataSourceBuilder {
    /// Sets the Arrow schema exposed to DataFusion.
    ///
    /// If not specified, the builder derives an Arrow schema from the Vortex
    /// dtype.
    ///
    /// Note that this schema is not validated against the Vortex DType so any errors will be
    /// deferred until read time.
    pub fn with_arrow_schema(mut self, arrow_schema: SchemaRef) -> Self {
        self.arrow_schema = Some(arrow_schema);
        self
    }

    /// Configures an initial top-level projection.
    ///
    /// This is useful when the embedding system already knows which columns are
    /// needed before DataFusion applies its own optimizer pushdown.
    pub fn with_projection(mut self, indices: Vec<usize>) -> Self {
        self.projection = Some(indices);
        self
    }

    /// Like [`Self::with_projection`], but accepts an optional projection.
    pub fn with_some_projection(mut self, indices: Option<Vec<usize>>) -> Self {
        self.projection = indices;
        self
    }

    /// Builds the [`VortexDataSource`].
    ///
    /// The builder eagerly resolves statistics for the initial projection
    /// columns because DataFusion expects the `DataSource` to report output
    /// statistics before execution begins.
    pub async fn build(self) -> VortexResult<VortexDataSource> {
        // The projection expression
        let mut projection = root();

        // Resolve the Arrow schema
        let mut arrow_schema = match self.arrow_schema {
            Some(schema) => schema,
            None => Arc::new(
                self.session
                    .arrow()
                    .to_arrow_schema(self.data_source.dtype())?,
            ),
        };

        // Apply any selection and create a projection expression.
        if let Some(indices) = self.projection {
            let fields = indices.iter().map(|&i| {
                let name = arrow_schema.field(i).name().clone();
                let expr = get_item(name.as_str(), root());
                (name, expr)
            });

            // Update the projection expression
            projection = pack(fields, Nullability::NonNullable);

            // Update the arrow schema
            arrow_schema = Arc::new(Schema::new(
                indices
                    .iter()
                    .map(|&i| arrow_schema.field(i).clone())
                    .collect::<Vec<_>>(),
            ));
        }

        let DType::Struct(fields, ..) = projection.return_dtype(self.data_source.dtype())? else {
            vortex_bail!("Projection does not evaluate to a struct");
        };

        // We now compute initial statistics.
        let field_paths: Vec<_> = fields
            .names()
            .iter()
            .cloned()
            .map(FieldPath::from_name)
            .collect();
        let statistics = try_join_all(
            field_paths
                .iter()
                .map(|path| self.data_source.field_statistics(path)),
        )
        .await?
        .iter()
        .zip(fields.fields())
        .map(|(stats, dtype)| stats_set_to_df(stats, &dtype))
        .collect::<VortexResult<Vec<_>>>()?;

        Ok(VortexDataSource {
            data_source: self.data_source,
            session: self.session,
            initial_schema: Arc::clone(&arrow_schema),
            initial_projection: projection.clone(),
            initial_statistics: statistics.clone(),
            projected_projection: projection.clone(),
            projected_schema: Arc::clone(&arrow_schema),
            projected_statistics: statistics.clone(),
            leftover_projection: None,
            leftover_schema: arrow_schema,
            leftover_statistics: statistics,
            filter: None,
            limit: None,
            ordered: false,
            num_partitions: get_available_parallelism().unwrap_or(1),
        })
    }
}

impl VortexDataSource {
    /// Create a builder for a [`VortexDataSource`].
    pub fn builder(data_source: DataSourceRef, session: VortexSession) -> VortexDataSourceBuilder {
        VortexDataSourceBuilder {
            data_source,
            session,
            arrow_schema: None,
            projection: None,
        }
    }
}

/// DataFusion [`DataSource`] backed by a Vortex [`DataSourceRef`].
///
/// `VortexDataSource` is the core execution adapter for the `v2` integration.
/// It presents DataFusion with a scanable Arrow data source while preserving the
/// underlying Vortex source until execution time.
///
/// During planning, it reports the current output schema and column statistics.
/// During execution, it builds the final Vortex [`ScanRequest`] from the
/// current projection, pushed filters, ordering hints, and row limit.
///
/// This integration intentionally reports a single DataFusion output partition.
/// Vortex then handles split-level concurrency internally: when ordering is not
/// requested it polls multiple split streams concurrently, and when ordering is
/// requested it emits split output in partition order. In both cases it enforces
/// the fetch limit globally across the combined output of all splits, so sources
/// do not need their own cross-partition coordinator.
///
/// Use [`crate::VortexSource`] instead when DataFusion should discover and plan
/// `.vortex` files on its own.
#[derive(Clone)]
pub struct VortexDataSource {
    /// The Vortex data source.
    data_source: DataSourceRef,
    /// Vortex session handle.
    session: VortexSession,

    // --- Phase 1: Initial (from the builder, before any optimizer pushdown) ---
    /// The Arrow schema of the data source before any DataFusion projection pushdown.
    initial_schema: SchemaRef,
    /// The initial Vortex projection expression (e.g. column selection from the builder).
    initial_projection: Expression,
    /// Column statistics for the initial projection columns.
    #[expect(dead_code)]
    initial_statistics: Vec<ColumnStatistics>,

    // --- Phase 2: Projected (pushed into the Vortex scan) ---
    /// The Vortex projection expression sent in the [`ScanRequest`].
    /// Composed with `initial_projection` so it operates on the original source columns.
    projected_projection: Expression,
    /// The Arrow schema of the Vortex scan output (before any leftover projection).
    projected_schema: SchemaRef,
    /// Column statistics for the projected (scan output) columns.
    projected_statistics: Vec<ColumnStatistics>,

    // --- Phase 3: Leftover (applied by DataFusion after the scan) ---
    /// DataFusion projection expressions that could not be pushed into the Vortex scan.
    /// Applied after converting arrays to record batches in [`DataSource::open`].
    /// `None` when all projection expressions were successfully pushed down.
    leftover_projection: Option<ProjectionExprs>,
    /// The Arrow schema after applying the leftover projection.
    /// This is the output schema seen by DataFusion.
    leftover_schema: SchemaRef,
    /// Column statistics matching `leftover_schema`.
    leftover_statistics: Vec<ColumnStatistics>,

    /// An optional filter expression.
    /// Populated by [`DataSource::try_pushdown_filters`] when DataFusion pushes filters down.
    filter: Option<Expression>,
    /// An optional row limit populated by [`DataSource::with_fetch`].
    limit: Option<usize>,
    /// Whether to preserve the order of the output rows.
    ordered: bool,

    /// The requested partition count from DataFusion, populated by [`DataSource::repartitioned`].
    /// We use this as a hint for how many splits to execute concurrently in `open()`, but we
    /// always declare to DataFusion that we only have a single partition so that we can
    /// internally manage concurrency and fix the problem of partition skew.
    num_partitions: usize,
}

impl fmt::Debug for VortexDataSource {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("VortexScanSource")
            .field("schema", &self.leftover_schema)
            .field("projection", &format!("{}", self.projected_projection))
            .field("filter", &self.filter.as_ref().map(|e| format!("{}", e)))
            .field("limit", &self.limit)
            .finish()
    }
}

impl DataSource for VortexDataSource {
    fn open(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        // VortexScanSource always uses a single partition since Vortex handles parallelism
        // and concurrency internally.
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "VortexScanSource: expected partition 0, got {partition}"
            )));
        }

        // Build the scan request with pushed-down projection, filter, and limit.
        // The projection is included so the scan can prune columns at the I/O level.
        let scan_request = ScanRequest {
            projection: self.projected_projection.clone(),
            filter: self.filter.clone(),
            limit: self.limit.map(|l| u64::try_from(l).unwrap_or(u64::MAX)),
            ordered: self.ordered,
            ..Default::default()
        };

        let data_source = Arc::clone(&self.data_source);
        let projected_schema = Arc::clone(&self.projected_schema);
        let projected_target_field = Arc::new(Field::new_struct(
            "",
            projected_schema.fields().clone(),
            false,
        ));
        let session = self.session.clone();
        let num_partitions = self.num_partitions;
        let ordered = self.ordered;
        // The `ScanRequest.limit` we push down is only a per-partition upper bound: each
        // partition may independently return up to `limit` rows. We are the single consumer of
        // all partitions, so we own enforcing the limit globally across their combined output.
        let limit = self.limit;

        // Pre-build the leftover projector (if any) so we can apply it after batch conversion.
        let leftover_projector = self
            .leftover_projection
            .as_ref()
            .map(|proj| proj.make_projector(&self.projected_schema))
            .transpose()?;

        // Defer the async DataSource::scan() call to the first poll of the stream.
        let stream = futures::stream::once(async move {
            let scan = data_source
                .scan(scan_request)
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

            // Each split.execute() returns a lazy stream whose early polls do preparation
            // work (expression resolution, layout traversal, first I/O spawns).
            let scan_streams = scan.partitions().map(|split_result| {
                let split = split_result?;
                split.execute()
            });

            // When ordering is requested we must emit partition output in partition order, so we
            // flatten the split streams sequentially. Otherwise we use try_flatten_unordered to
            // poll multiple split streams concurrently so that the next split is already warm when
            // the current one finishes.
            let flattened = if ordered {
                scan_streams.try_flatten().left_stream()
            } else {
                scan_streams
                    .try_flatten_unordered(Some(num_partitions * 2))
                    .right_stream()
            };

            let handle = session.handle();
            let stream = flattened
                .map(move |result| {
                    let session = session.clone();
                    let target_field = Arc::clone(&projected_target_field);
                    handle.spawn_cpu(move || {
                        let mut ctx = session.create_execution_ctx();
                        result.and_then(|chunk| {
                            let arrow = session.arrow().execute_arrow(
                                chunk,
                                Some(target_field.as_ref()),
                                &mut ctx,
                            )?;
                            Ok(RecordBatch::from(arrow.as_struct().clone()))
                        })
                    })
                })
                .buffered(num_partitions)
                .map(|result| result.map_err(|e| DataFusionError::External(Box::new(e))));

            // Apply leftover projection (expressions that couldn't be pushed into Vortex).
            let stream = if let Some(projector) = leftover_projector {
                stream
                    .map(move |batch_result| {
                        batch_result.and_then(|batch| projector.project_batch(&batch))
                    })
                    .boxed()
            } else {
                stream.boxed()
            };

            // Enforce the fetch limit across the combined output of all partitions. We track how
            // many rows we have emitted, slice the batch that crosses the limit, and stop pulling
            // from the partition streams entirely once the limit is met (the `done` flag returns
            // without polling upstream again).
            let stream = match limit {
                Some(limit) => futures::stream::unfold(
                    (stream, 0usize, false),
                    move |(mut stream, emitted, done)| async move {
                        if done || emitted >= limit {
                            return None;
                        }
                        match stream.next().await {
                            Some(Ok(batch)) => {
                                let remaining = limit - emitted;
                                let batch = if batch.num_rows() > remaining {
                                    batch.slice(0, remaining)
                                } else {
                                    batch
                                };
                                let emitted = emitted + batch.num_rows();
                                Some((Ok(batch), (stream, emitted, emitted >= limit)))
                            }
                            Some(Err(e)) => Some((Err(e), (stream, emitted, false))),
                            None => None,
                        }
                    },
                )
                .boxed(),
                None => stream,
            };

            Ok::<_, DataFusionError>(stream)
        })
        .try_flatten();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.leftover_schema),
            stream,
        )))
    }

    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(
            f,
            "VortexScanSource: projection={}",
            self.projected_projection
        )?;
        if let Some(filter) = &self.filter {
            write!(f, ", filter={filter}")?;
        }
        if let Some(limit) = self.limit {
            write!(f, ", limit={limit}")?;
        }
        Ok(())
    }

    fn repartitioned(
        &self,
        target_partitions: usize,
        _repartition_file_min_size: usize,
        output_ordering: Option<LexOrdering>,
    ) -> DFResult<Option<Arc<dyn DataSource>>> {
        // Vortex handles parallelism internally — always use a single partition.
        let mut this = self.clone();
        this.num_partitions = target_partitions;
        this.ordered |= output_ordering.is_some();
        Ok(Some(Arc::new(this)))
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        EquivalenceProperties::new(Arc::clone(&self.leftover_schema))
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> DFResult<Arc<Statistics>> {
        // FIXME(ngates): this should be adjusted based on filters. See DuckDB for heuristics,
        //  and in the future, store the selectivity stats in the session.
        let num_rows = estimate_to_df_precision(&self.data_source.row_count());

        // FIXME(ngates): byte size should be adjusted for the initial projection...
        let total_byte_size = estimate_to_df_precision(&self.data_source.byte_size());

        // Column statistics must match the output schema (leftover_schema), which may differ
        // from the initial schema after try_swapping_with_projection adds computed columns.
        let column_statistics = self.leftover_statistics.clone();

        Ok(Arc::new(Statistics {
            num_rows,
            total_byte_size,
            column_statistics,
        }))
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn DataSource>> {
        let mut this = self.clone();
        this.limit = limit;
        Some(Arc::new(this))
    }

    fn fetch(&self) -> Option<usize> {
        self.limit
    }

    // Note that we're explicitly "swapping" the projection. That means everything we do must
    // be computed over the original input schema, rather than the projected output schema.
    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExprs,
    ) -> DFResult<Option<Arc<dyn DataSource>>> {
        tracing::debug!(
            "VortexScanSource: trying to swap with projection: {}",
            projection
        );

        let convertor = DefaultExpressionConvertor::default();
        let input_schema = self.initial_schema.as_ref();
        let projected_schema = projection.project_schema(input_schema)?;

        // Use the shared ExpressionConvertor to split the projection into a Vortex
        // scan_projection and a leftover DataFusion projection for expressions that
        // can't be pushed down (e.g., unsupported scalar functions, decimal binary).
        let ProcessedProjection {
            scan_projection,
            leftover_projection,
        } = convertor.split_projection(projection.clone(), input_schema, &projected_schema)?;

        // Compose with the initial projection so the scan operates on the original
        // source columns, not the initial projection's output columns.
        let scan_projection = replace(scan_projection, &root(), self.initial_projection.clone());

        // Compute the scan output schema from the Vortex expression's return dtype.
        let scan_dtype = scan_projection
            .return_dtype(self.data_source.dtype())
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let scan_output_schema = Arc::new(
            self.session
                .arrow()
                .to_arrow_schema(&scan_dtype)
                .map_err(|e| DataFusionError::External(Box::new(e)))?,
        );

        // Remap the leftover column references to match the scan output schema.
        let leftover_projection = leftover_projection
            .try_map_exprs(|expr| reassign_expr_columns(expr, &scan_output_schema))?;

        let final_schema = Arc::new(projected_schema);

        let mut this = self.clone();
        this.projected_projection = scan_projection;
        this.projected_schema = Arc::clone(&scan_output_schema);
        this.projected_statistics =
            vec![ColumnStatistics::new_unknown(); scan_output_schema.fields().len()];
        this.leftover_projection = Some(leftover_projection);
        this.leftover_schema = Arc::clone(&final_schema);
        this.leftover_statistics =
            vec![ColumnStatistics::new_unknown(); final_schema.fields().len()];

        Ok(Some(Arc::new(this)))
    }

    fn try_pushdown_filters(
        &self,
        filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &datafusion_common::config::ConfigOptions,
    ) -> DFResult<FilterPushdownPropagation<Arc<dyn DataSource>>> {
        if filters.is_empty() {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                vec![],
            ));
        }

        let convertor = DefaultExpressionConvertor::default();
        let input_schema = self.initial_schema.as_ref();

        // Classify each filter: pushable filters are passed into the ScanRequest in open(),
        // so we can safely claim PushedDown::Yes for them.
        let pushdown_results: Vec<PushedDown> = filters
            .iter()
            .map(|expr| {
                if convertor.can_be_pushed_down(expr, input_schema) {
                    PushedDown::Yes
                } else {
                    PushedDown::No
                }
            })
            .collect();

        // If nothing can be pushed down, return early.
        if pushdown_results.iter().all(|p| matches!(p, PushedDown::No)) {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                pushdown_results,
            ));
        }

        // Collect the pushable filter expressions.
        let pushable: Vec<Arc<dyn PhysicalExpr>> = filters
            .iter()
            .zip(pushdown_results.iter())
            .filter_map(|(expr, pushed)| match pushed {
                PushedDown::Yes => Some(Arc::clone(expr)),
                PushedDown::No => None,
            })
            .collect();

        // Convert to Vortex conjunction.
        let vortex_pred = make_vortex_predicate(&convertor, &pushable)?;

        // Combine with existing filter.
        let new_filter = match (&self.filter, vortex_pred) {
            (Some(existing), Some(new_pred)) => Some(vx_and(existing.clone(), new_pred)),
            (Some(existing), None) => Some(existing.clone()),
            (None, Some(new_pred)) => Some(new_pred),
            (None, None) => None,
        };

        let mut this = self.clone();
        this.filter = new_filter;
        Ok(
            FilterPushdownPropagation::with_parent_pushdown_result(pushdown_results)
                .with_updated_node(Arc::new(this) as _),
        )
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> DFResult<TreeNodeRecursion>,
    ) -> DFResult<TreeNodeRecursion> {
        datafusion_physical_plan::apply_expression_roots(
            self.leftover_projection
                .iter()
                .flat_map(|projection| projection.iter())
                .map(|expr| &expr.expr),
            f,
        )
    }
}

/// Convert a Vortex [`Option<Precision>`] to a DataFusion
/// [`DataFusionPrecision`].
///
/// [`DataFusionPrecision`]: datafusion_common::stats::Precision
fn estimate_to_df_precision(est: &Precision<u64>) -> DFPrecision<usize> {
    match est {
        Precision::Exact(v) => DFPrecision::Exact(usize::try_from(*v).unwrap_or(usize::MAX)),
        Precision::Inexact(v) => DFPrecision::Inexact(usize::try_from(*v).unwrap_or(usize::MAX)),
        Precision::Absent => DFPrecision::Absent,
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::time::Duration;

    use arrow_schema::DataType;
    use async_trait::async_trait;
    use futures::TryStreamExt;
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::stats::StatsSet;
    use vortex::array::stream::ArrayStreamAdapter;
    use vortex::array::stream::ArrayStreamExt;
    use vortex::array::stream::SendableArrayStream;
    use vortex::array::validity::Validity;
    use vortex::buffer::buffer;
    use vortex::dtype::FieldPath;
    use vortex::dtype::PType;
    use vortex::dtype::StructFields;
    // Aliased so it does not shadow the DataFusion `DataSource` trait brought in via `super::*`,
    // which provides the `open` method exercised below.
    use vortex::scan::DataSource as VortexScanDataSource;
    use vortex::scan::DataSourceScan;
    use vortex::scan::DataSourceScanRef;
    use vortex::scan::Partition;
    use vortex::scan::PartitionRef;
    use vortex::scan::PartitionStream;

    use super::*;

    /// A single-column `{ x: i32 }` struct dtype used by the test data source.
    fn test_dtype() -> DType {
        DType::Struct(
            StructFields::from_iter([(
                "x",
                DType::Primitive(PType::I32, Nullability::NonNullable),
            )]),
            Nullability::NonNullable,
        )
    }

    /// A partition that emits a single-row `{ x: value }` batch, optionally after a delay.
    struct TestPartition {
        index: usize,
        value: i32,
        delay: Option<Duration>,
    }

    impl Partition for TestPartition {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn index(&self) -> usize {
            self.index
        }

        fn row_count(&self) -> Precision<u64> {
            Precision::exact(1)
        }

        fn byte_size(&self) -> Precision<u64> {
            Precision::Absent
        }

        fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
            let dtype = test_dtype();
            let value = self.value;
            let delay = self.delay;
            let fut = async move {
                if let Some(delay) = delay {
                    tokio::time::sleep(delay).await;
                }
                let array = StructArray::from_fields(&[(
                    "x",
                    PrimitiveArray::new(buffer![value], Validity::NonNullable).into_array(),
                )])?
                .into_array();
                Ok(array)
            };
            Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
                dtype,
                futures::stream::once(fut),
            )))
        }
    }

    /// A scan that produces two partitions: partition 0 is delayed, partition 1 is immediate.
    struct TestScan {
        dtype: DType,
    }

    impl DataSourceScan for TestScan {
        fn dtype(&self) -> &DType {
            &self.dtype
        }

        fn partition_count(&self) -> Precision<usize> {
            Precision::exact(2)
        }

        fn partitions(self: Box<Self>) -> PartitionStream {
            let partitions: Vec<VortexResult<PartitionRef>> = vec![
                Ok(Box::new(TestPartition {
                    index: 0,
                    value: 0,
                    delay: Some(Duration::from_millis(100)),
                }) as PartitionRef),
                Ok(Box::new(TestPartition {
                    index: 1,
                    value: 100,
                    delay: None,
                }) as PartitionRef),
            ];
            futures::stream::iter(partitions).boxed()
        }
    }

    struct TestDataSource {
        dtype: DType,
    }

    #[async_trait]
    impl VortexScanDataSource for TestDataSource {
        fn dtype(&self) -> &DType {
            &self.dtype
        }

        async fn scan(&self, _scan_request: ScanRequest) -> VortexResult<DataSourceScanRef> {
            Ok(Box::new(TestScan {
                dtype: self.dtype.clone(),
            }))
        }

        async fn field_statistics(&self, _field_path: &FieldPath) -> VortexResult<StatsSet> {
            Ok(StatsSet::default())
        }
    }

    /// With ordering requested and a global fetch of 1, the output must come from the first
    /// partition even though it is delayed and the second partition is immediately ready. This
    /// exercises both ordered split emission and cross-partition limit enforcement.
    #[tokio::test(flavor = "multi_thread")]
    async fn ordered_scan_enforces_global_fetch() -> VortexResult<()> {
        let arrow_schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let data_source: DataSourceRef = Arc::new(TestDataSource {
            dtype: test_dtype(),
        });

        let mut source = VortexDataSource::builder(data_source, VortexSession::default())
            .with_arrow_schema(arrow_schema)
            .build()
            .await?;
        source.ordered = true;
        source.limit = Some(1);

        let stream = source
            .open(0, Arc::new(TaskContext::default()))
            .map_err(|e| vortex::error::vortex_err!("open failed: {e}"))?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| vortex::error::vortex_err!("collect failed: {e}"))?;

        let total_rows: usize = batches.iter().map(RecordBatch::num_rows).sum();
        assert_eq!(total_rows, 1, "global fetch = 1 should yield exactly one row");

        let column = batches[0]
            .column(0)
            .as_primitive::<datafusion_common::arrow::datatypes::Int32Type>();
        assert_eq!(
            column.value(0),
            0,
            "ordered scan must emit the first partition's row, not the faster second partition"
        );

        Ok(())
    }
}
