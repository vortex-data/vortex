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
use std::num::NonZeroUsize;
use std::sync::Arc;

use arrow_schema::DataType;
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
use futures::stream;
use futures::stream::BoxStream;
use tokio::sync::OnceCell;
use vortex::array::ArrayRef;
use vortex::array::VortexSessionExecute;
use vortex::array::arrow::ArrowSessionExt;
use vortex::dtype::DType;
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
use vortex::metrics::MetricsRegistry;
use vortex::scan::DataSourceRef;
use vortex::scan::DataSourceScanRef;
use vortex::scan::ScanPartitioning;
use vortex::scan::ScanRequest;
use vortex::session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

use crate::convert::exprs::DefaultExpressionConvertor;
use crate::convert::exprs::ExpressionConvertor;
use crate::convert::exprs::ProcessedProjection;
use crate::convert::exprs::make_vortex_predicate;
use crate::convert::stats::aggregate_stats_to_df;
use crate::convert::stats::column_statistics_aggregate_fns;

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
    metrics_registry: Option<Arc<dyn MetricsRegistry>>,
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

    /// Attaches a Vortex metrics registry populated by the underlying data source.
    ///
    /// The V2 adapter does not open files itself, so callers that want Vortex read metrics must
    /// also configure the wrapped source to write to this same registry.
    pub fn with_metrics_registry(mut self, metrics_registry: Arc<dyn MetricsRegistry>) -> Self {
        self.metrics_registry = Some(metrics_registry);
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
        let statistics_exprs: Vec<_> = fields
            .names()
            .iter()
            .cloned()
            .map(|name| get_item(name, root()))
            .collect();
        let statistics_funcs = column_statistics_aggregate_fns();
        let statistics = try_join_all(
            statistics_exprs
                .iter()
                .map(|expr| self.data_source.statistics(expr, &statistics_funcs)),
        )
        .await?
        .iter()
        .map(|stats| aggregate_stats_to_df(stats))
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
            metrics_registry: self.metrics_registry,
            scan: Arc::new(OnceCell::new()),
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
            metrics_registry: None,
        }
    }

    fn scan_partition_count(&self) -> usize {
        if self.should_target_partitioning() {
            self.num_partitions.max(1)
        } else {
            1
        }
    }

    fn should_target_partitioning(&self) -> bool {
        !self.ordered && self.limit.is_none()
    }

    fn reset_scan(&mut self) {
        self.scan = Arc::new(OnceCell::new());
    }

    /// Returns the metrics registry attached to this source, if one was configured.
    pub fn metrics_registry(&self) -> Option<&Arc<dyn MetricsRegistry>> {
        self.metrics_registry.as_ref()
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
/// For unordered scans without a limit, this integration passes DataFusion's
/// requested partition count to the Vortex scan request. Ordered and limited scans use one output
/// partition so the source can preserve semantics.
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
    /// When target partitioning is enabled, this is also the count we report back to DataFusion.
    num_partitions: usize,

    /// Optional Vortex metrics registry populated by the wrapped source.
    metrics_registry: Option<Arc<dyn MetricsRegistry>>,

    /// Shared Vortex scan used by all DataFusion output partitions.
    scan: Arc<OnceCell<DataSourceScanRef>>,
}

impl fmt::Debug for VortexDataSource {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("VortexScanSource")
            .field("schema", &self.leftover_schema)
            .field("projection", &format!("{}", &self.projected_projection))
            .field("filter", &self.filter.as_ref().map(|e| format!("{}", e)))
            .field("limit", &self.limit)
            .finish()
    }
}

fn known_partition_count(scan: &DataSourceScanRef) -> Option<usize> {
    match scan.partition_count() {
        Precision::Exact(count) | Precision::Inexact(count) => Some(count),
        Precision::Absent => None,
    }
}

fn scan_partition_to_array_stream(
    scan: DataSourceScanRef,
    partition: usize,
) -> DFResult<BoxStream<'static, VortexResult<ArrayRef>>> {
    if let Some(partition_count) = known_partition_count(&scan)
        && partition >= partition_count
    {
        return Ok(stream::empty().boxed());
    }

    let Some(partition) = scan
        .partition(partition)
        .map_err(|e| DataFusionError::External(Box::new(e)))?
    else {
        return Ok(stream::empty().boxed());
    };
    Ok(partition
        .execute()
        .map_err(|e| DataFusionError::External(Box::new(e)))?
        .boxed())
}

fn scan_partitions_to_array_stream(
    scan: DataSourceScanRef,
    ordered: bool,
    num_partitions: usize,
) -> DFResult<BoxStream<'static, VortexResult<ArrayRef>>> {
    // Each split.execute() returns a lazy stream whose early polls do preparation
    // work (expression resolution, layout traversal, first I/O spawns). Unordered
    // scans can poll multiple split streams concurrently so the next split is
    // already warm when the current one finishes; ordered scans must preserve
    // partition order.
    let scan_streams = scan.partitions().map(|split_result| {
        let split = split_result?;
        split.execute()
    });

    if ordered {
        Ok(scan_streams.try_flatten().boxed())
    } else {
        Ok(scan_streams
            .try_flatten_unordered(Some(num_partitions * 2))
            .boxed())
    }
}

impl DataSource for VortexDataSource {
    fn open(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        let scan_partition_count = self.scan_partition_count();
        if partition >= scan_partition_count {
            return Err(DataFusionError::Internal(format!(
                "VortexScanSource: expected partition in 0..{scan_partition_count}, got {partition}"
            )));
        }

        let use_target_partitioning = self.should_target_partitioning();
        let partitioning = if use_target_partitioning {
            ScanPartitioning::Target(
                NonZeroUsize::new(scan_partition_count)
                    .expect("scan partition count is always non-zero"),
            )
        } else {
            ScanPartitioning::SourceDefault
        };

        // Build the scan request with pushed-down projection, filter, limit, and physical
        // partitioning preference.
        let scan_request = ScanRequest {
            projection: self.projected_projection.clone(),
            filter: self.filter.clone(),
            limit: self.limit.map(|l| u64::try_from(l).unwrap_or(u64::MAX)),
            ordered: self.ordered,
            partitioning,
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
        let num_partitions = self.num_partitions.max(1);
        let ordered = self.ordered;
        let scan = Arc::clone(&self.scan);

        // Pre-build the leftover projector (if any) so we can apply it after batch conversion.
        let leftover_projector = self
            .leftover_projection
            .as_ref()
            .map(|proj| proj.make_projector(&self.projected_schema))
            .transpose()?;

        // Defer the async DataSource work to the first poll of the stream.
        let stream = stream::once(async move {
            let scan = scan
                .get_or_try_init(|| {
                    let data_source = Arc::clone(&data_source);
                    let scan_request = scan_request.clone();
                    async move {
                        data_source
                            .scan(scan_request)
                            .await
                            .map_err(|e| DataFusionError::External(Box::new(e)))
                    }
                })
                .await?;

            let array_stream: BoxStream<'static, VortexResult<ArrayRef>> =
                if use_target_partitioning {
                    scan_partition_to_array_stream(Arc::clone(scan), partition)?
                } else if partition == 0 {
                    scan_partitions_to_array_stream(Arc::clone(scan), ordered, num_partitions)?
                } else {
                    stream::empty().boxed()
                };

            let handle = session.handle();
            let stream = array_stream
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
        let mut this = self.clone();
        this.num_partitions = target_partitions;
        this.ordered |= output_ordering.is_some();
        this.reset_scan();
        Ok(Some(Arc::new(this)))
    }

    fn output_partitioning(&self) -> Partitioning {
        // Report the engine-requested partition count. The Vortex scan request carries the same
        // target, and any source-level fallback maps surplus partitions to empty streams.
        Partitioning::UnknownPartitioning(self.scan_partition_count())
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        EquivalenceProperties::new(Arc::clone(&self.leftover_schema))
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DFResult<Arc<Statistics>> {
        // FIXME(ngates): this should be adjusted based on filters. See DuckDB for heuristics,
        //  and in the future, store the selectivity stats in the session.
        let mut num_rows = estimate_to_df_precision(&self.data_source.row_count());

        // FIXME(ngates): byte size should be adjusted for the initial projection...
        let mut total_byte_size = estimate_to_df_precision(&self.data_source.byte_size());

        if partition.is_some() {
            let partition_count = self.scan_partition_count();
            num_rows = divide_df_precision(num_rows, partition_count);
            total_byte_size = divide_df_precision(total_byte_size, partition_count);
        }

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
        this.reset_scan();
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
        let scan_projection = replace(scan_projection, &root(), self.initial_projection.clone())
            .optimize_recursive(self.data_source.dtype())
            .map_err(|e| DataFusionError::External(Box::new(e)))?;

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
        this.reset_scan();

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
                let is_boolean = matches!(expr.data_type(input_schema), Ok(DataType::Boolean));
                if is_boolean && convertor.can_be_pushed_down(expr, input_schema) {
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
        this.reset_scan();
        Ok(
            FilterPushdownPropagation::with_parent_pushdown_result(pushdown_results)
                .with_updated_node(Arc::new(this) as _),
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

fn divide_df_precision(est: DFPrecision<usize>, divisor: usize) -> DFPrecision<usize> {
    let divisor = divisor.max(1);
    match est {
        DFPrecision::Exact(v) => DFPrecision::Exact(v.div_ceil(divisor)),
        DFPrecision::Inexact(v) => DFPrecision::Inexact(v.div_ceil(divisor)),
        DFPrecision::Absent => DFPrecision::Absent,
    }
}
