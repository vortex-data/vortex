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

use std::any::Any;
use std::fmt;
use std::fmt::Formatter;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use datafusion_common::ColumnStatistics;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::Statistics;
use datafusion_common::arrow::array::AsArray;
use datafusion_common::arrow::array::RecordBatch;
use datafusion_common::config::ConfigOptions;
use datafusion_common::stats::Precision as DFPrecision;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_datasource::source::DataSource;
use datafusion_datasource::source::OpenArgs;
use datafusion_execution::RecordBatchStream;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::DynamicFilterTracking;
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_expr::Partitioning;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr::utils::reassign_expr_columns;
use datafusion_physical_expr_common::sort_expr::LexOrdering;
use datafusion_physical_expr_common::sort_expr::PhysicalSortExpr;
use datafusion_physical_plan::DisplayFormatType;
use datafusion_physical_plan::filter_pushdown::FilterPushdownPropagation;
use datafusion_physical_plan::filter_pushdown::PushedDown;
use datafusion_physical_plan::metrics::BaselineMetrics;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::Stream;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::try_join_all;
use futures::lock::Mutex;
use futures::stream;
use futures::stream::BoxStream;
use futures::task::AtomicWaker;
use tokio::sync::OnceCell;
use vortex::array::ArrayRef;
use vortex::array::VortexSessionExecute;
use vortex::array::stream::SendableArrayStream;
use vortex::dtype::DType;
use vortex::dtype::FieldPath;
use vortex::dtype::Nullability;
use vortex::error::VortexError;
use vortex::error::VortexExpect as _;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::get_item;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::expr::stats::Precision;
use vortex::expr::transform::replace;
use vortex::io::session::RuntimeSessionExt;
use vortex::scan::DataSourceRef;
use vortex::scan::ScanRequest;
use vortex::scan::driver::ScanDriver;
use vortex::session::VortexSession;
use vortex_arrow::ArrowSessionExt;
use vortex_utils::parallelism::get_available_parallelism;

use super::dynamic_filters::convert_dynamic_filter;
use crate::convert::exprs::DefaultExpressionConverter;
use crate::convert::exprs::ExpressionConverter;
use crate::convert::exprs::ProcessedProjection;
use crate::convert::schema::calculate_physical_schema;
use crate::convert::stats::stats_set_to_df;

// Bound active Scan API partition streams independently of the DataFusion output count.
const SCAN_PARTITION_READ_AHEAD: usize = 2;

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
    expression_converter: Arc<dyn ExpressionConverter>,
    output_ordering: Option<LexOrdering>,
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

    /// Sets the converter used for filter and projection pushdown.
    ///
    /// Defaults to [`DefaultExpressionConverter`] using the builder's session.
    /// Unsupported expressions remain in DataFusion. Custom converters must preserve
    /// DataFusion values, nulls, and evaluation errors for expressions they accept.
    /// A producer-guaranteed fixed comparison may be converted to
    /// [`DynamicComparison`](vortex::scalar_fn::fns::dynamic::DynamicComparison).
    pub fn with_expression_converter(
        mut self,
        expression_converter: Arc<dyn ExpressionConverter>,
    ) -> Self {
        self.expression_converter = expression_converter;
        self
    }

    /// Declares a proven global ordering of the builder's Arrow output schema.
    ///
    /// The caller must guarantee this key order when source partitions are concatenated
    /// in partition order. Independently sorted, overlapping partitions are insufficient.
    /// The adapter preserves this sequence using one output partition; it does not sort rows.
    pub fn with_output_ordering(mut self, ordering: LexOrdering) -> Self {
        self.output_ordering = Some(ordering);
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

        if let Some(ordering) = &self.output_ordering {
            for key in ordering {
                key.expr
                    .return_field(&arrow_schema)
                    .map_err(VortexError::external)?;
            }
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
            expression_converter: self.expression_converter,
            initial_ordering: self.output_ordering.clone(),
            output_ordering: self.output_ordering.clone(),
            metrics: ExecutionPlanMetricsSet::new(),
            output_partitions: 1,
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
            dynamic_filters: Vec::new(),
            limit: None,
            ordered: self.output_ordering.is_some(),
            target_parallelism: get_available_parallelism().unwrap_or(1),
        })
    }
}

impl VortexDataSource {
    /// Create a builder for a [`VortexDataSource`].
    pub fn builder(data_source: DataSourceRef, session: VortexSession) -> VortexDataSourceBuilder {
        VortexDataSourceBuilder {
            data_source,
            expression_converter: Arc::new(DefaultExpressionConverter::new(session.clone())),
            session,
            arrow_schema: None,
            projection: None,
            output_ordering: None,
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
/// Unordered scans expose the requested DataFusion output partitions, sharing one lazy scan
/// per execution through [`DataSource::create_sibling_state`]. Ordered and fetch-limited scans
/// use one output partition to preserve partition sequence and a global fetch limit. Direct
/// [`DataSource::open`] calls support only a single output partition; multiple output partitions
/// must be opened through [`DataSourceExec`] or shared [`OpenArgs`]. Dropping an output partition
/// before completion cancels its execution; other output partitions report an error.
/// Scan initialization attempts are serialized; cancellation may drop a pending attempt,
/// allowing another output partition to retry it.
///
/// [`DataSourceExec`]: datafusion_datasource::source::DataSourceExec
///
/// Use [`crate::VortexSource`] instead when DataFusion should discover and plan
/// `.vortex` files on its own.
#[derive(Clone)]
pub struct VortexDataSource {
    /// The Vortex data source.
    data_source: DataSourceRef,
    /// Vortex session handle.
    session: VortexSession,
    expression_converter: Arc<dyn ExpressionConverter>,
    initial_ordering: Option<LexOrdering>,
    output_ordering: Option<LexOrdering>,
    metrics: ExecutionPlanMetricsSet,
    output_partitions: usize,

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
    dynamic_filters: Vec<Arc<dyn PhysicalExpr>>,
    /// An optional row limit populated by [`DataSource::with_fetch`].
    limit: Option<usize>,
    /// Whether to preserve the order of the output rows.
    ordered: bool,

    /// Target Arrow conversion concurrency when only one output partition is exposed.
    /// Each unordered output partition opens at most two Scan API partitions concurrently.
    target_parallelism: usize,
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

impl VortexDataSource {
    fn output_partition_count(&self) -> usize {
        if self.ordered || self.limit.is_some() {
            1
        } else {
            self.output_partitions
        }
    }

    fn open_partition(
        &self,
        partition: usize,
        execution: Arc<ScanExecution>,
    ) -> DFResult<SendableRecordBatchStream> {
        if execution.cancelled.load(Ordering::Acquire) {
            return Err(DataFusionError::Execution(
                "scan execution cancelled".into(),
            ));
        }
        let output = execution.output_partitions.get(partition).ok_or_else(|| {
            DataFusionError::Execution(format!("invalid scan output partition {partition}"))
        })?;
        if output.claimed.swap(true, Ordering::AcqRel) {
            return Err(DataFusionError::Execution(
                "scan output partition already opened; reset the execution plan before replay"
                    .into(),
            ));
        }
        let metrics = BaselineMetrics::new(&self.metrics, partition);
        let stream = if self.limit == Some(0) {
            Box::pin(RecordBatchStreamAdapter::new(
                Arc::clone(&self.leftover_schema),
                stream::empty(),
            )) as SendableRecordBatchStream
        } else {
            let scan_request = ScanRequest {
                projection: self.projected_projection.clone(),
                filter: self.filter.clone(),
                limit: self.limit.map(|l| u64::try_from(l).unwrap_or(u64::MAX)),
                ordered: self.ordered,
                ..Default::default()
            };
            let data_source = Arc::clone(&self.data_source);
            let output_count = NonZeroUsize::new(self.output_partition_count())
                .vortex_expect("positive scan output count");
            let arrays = stream::once({
                let execution = Arc::clone(&execution);
                async move {
                    // Cancellation may retry initialization, but never runs attempts concurrently.
                    let streams = execution
                        .scan_streams
                        .get_or_init(|| async {
                            let scan = data_source.scan(scan_request.clone()).await?;
                            Ok(Mutex::new(
                                ScanDriver::new(scan, &scan_request)
                                    .into_streams(
                                        output_count,
                                        NonZeroUsize::new(SCAN_PARTITION_READ_AHEAD)
                                            .vortex_expect("positive scan read-ahead"),
                                    )
                                    .into_iter()
                                    .map(Some)
                                    .collect(),
                            ))
                        })
                        .await
                        .as_ref()
                        .map_err(Clone::clone)?;
                    streams.lock().await[partition]
                        .take()
                        .ok_or_else(|| vortex_err!("scan output partition already opened"))
                }
            })
            .try_flatten()
            .boxed();
            self.batch_stream(arrays, metrics.clone())?
        };
        Ok(Box::pin(OutputPartitionStream {
            partition,
            schema: Arc::clone(&self.leftover_schema),
            stream: Some(stream),
            execution,
            metrics,
        }))
    }

    fn batch_stream(
        &self,
        arrays: BoxStream<'static, VortexResult<ArrayRef>>,
        metrics: BaselineMetrics,
    ) -> DFResult<SendableRecordBatchStream> {
        let projected_schema = Arc::clone(&self.projected_schema);
        let projected_target_field = Arc::new(Field::new_struct(
            "",
            projected_schema.fields().clone(),
            false,
        ));
        let session = self.session.clone();
        let conversion_concurrency = if self.output_partition_count() == 1 {
            self.target_parallelism
        } else {
            1
        };

        // Pre-build the leftover projector (if any) so we can apply it after batch conversion.
        let leftover_projector = self
            .leftover_projection
            .as_ref()
            .map(|proj| proj.make_projector(&self.projected_schema))
            .transpose()?;

        // Defer the async DataSource::scan() call to the first poll of the stream.
        let stream = stream::once(async move {
            let handle = session.handle();
            let stream = arrays
                .map(move |result| {
                    let session = session.clone();
                    let target_field = Arc::clone(&projected_target_field);
                    let metrics = metrics.intermediate();
                    handle.spawn_cpu(move || {
                        let _timer = metrics.elapsed_compute().timer();
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
                .buffered(conversion_concurrency)
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
}

#[derive(Default)]
struct OutputPartitionState {
    claimed: AtomicBool,
    waker: AtomicWaker,
}

struct ScanExecution {
    output_partitions: Vec<OutputPartitionState>,
    cancelled: AtomicBool,
    scan_streams: OnceCell<VortexResult<Mutex<Vec<Option<SendableArrayStream>>>>>,
}

impl ScanExecution {
    fn new(output_partitions: usize) -> Self {
        Self {
            output_partitions: (0..output_partitions)
                .map(|_| OutputPartitionState::default())
                .collect(),
            cancelled: AtomicBool::new(false),
            scan_streams: OnceCell::new(),
        }
    }

    fn cancel(&self) {
        if !self.cancelled.swap(true, Ordering::AcqRel) {
            for output in &self.output_partitions {
                output.waker.wake();
            }
        }
    }
}

struct OutputPartitionStream {
    partition: usize,
    schema: SchemaRef,
    stream: Option<SendableRecordBatchStream>,
    execution: Arc<ScanExecution>,
    metrics: BaselineMetrics,
}

impl Stream for OutputPartitionStream {
    type Item = DFResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let metrics = self.metrics.clone();
        let _timer = metrics.elapsed_compute().timer();
        self.execution.output_partitions[self.partition]
            .waker
            .register(cx.waker());
        let mut result = if self.execution.cancelled.load(Ordering::Acquire) {
            Poll::Ready(None)
        } else {
            match self.stream.as_mut() {
                Some(stream) => stream.as_mut().poll_next(cx),
                None => Poll::Ready(None),
            }
        };
        if self.stream.is_some() && self.execution.cancelled.load(Ordering::Acquire) {
            result = Poll::Ready(Some(Err(DataFusionError::Execution(
                "scan execution cancelled".into(),
            ))));
        }
        if matches!(result, Poll::Ready(Some(Err(_)))) {
            self.execution.cancel();
        }
        if matches!(result, Poll::Ready(None | Some(Err(_)))) {
            self.stream = None;
            drop(
                self.execution.output_partitions[self.partition]
                    .waker
                    .take(),
            );
        }
        metrics.record_poll(result)
    }
}

impl RecordBatchStream for OutputPartitionStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl Drop for OutputPartitionStream {
    fn drop(&mut self) {
        if self.stream.is_some() {
            self.execution.cancel();
        }
        drop(self.stream.take());
        drop(
            self.execution.output_partitions[self.partition]
                .waker
                .take(),
        );
        if self.execution.cancelled.load(Ordering::Acquire)
            && let Some(Ok(streams)) = self.execution.scan_streams.get()
            && let Some(mut streams) = streams.try_lock()
        {
            for stream in streams.iter_mut() {
                drop(stream.take());
            }
        }
        self.metrics.done();
    }
}

impl DataSource for VortexDataSource {
    fn open(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        if self.output_partition_count() != 1 {
            return Err(DataFusionError::Execution(
                "multiple scan output partitions require sibling execution state".into(),
            ));
        }
        self.open_partition(partition, Arc::new(ScanExecution::new(1)))
    }

    fn create_sibling_state(&self, _config: &ConfigOptions) -> Option<Arc<dyn Any + Send + Sync>> {
        Some(Arc::new(ScanExecution::new(self.output_partition_count())))
    }

    fn open_with_args(&self, args: OpenArgs) -> DFResult<SendableRecordBatchStream> {
        match args.sibling_state {
            Some(state) => {
                let execution = state.downcast::<ScanExecution>().map_err(|_| {
                    DataFusionError::Execution("incompatible scan execution state".into())
                })?;
                self.open_partition(args.partition, execution)
            }
            None => self.open(args.partition, args.context),
        }
    }

    fn metrics(&self) -> ExecutionPlanMetricsSet {
        self.metrics.clone()
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
        this.target_parallelism = target_partitions.max(1);
        this.output_partitions = this.target_parallelism;
        this.ordered |= output_ordering.is_some();
        Ok(Some(Arc::new(this)))
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(self.output_partition_count())
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        EquivalenceProperties::new_with_orderings(
            Arc::clone(&self.leftover_schema),
            self.output_ordering.clone(),
        )
    }

    fn partition_statistics(&self, partition: Option<usize>) -> DFResult<Arc<Statistics>> {
        if partition.is_some() && self.output_partition_count() > 1 {
            return Ok(Arc::new(Statistics::new_unknown(&self.leftover_schema)));
        }
        let num_rows = estimate_to_df_precision(&self.data_source.row_count());

        // FIXME(ngates): byte size should be adjusted for the initial projection...
        let total_byte_size = estimate_to_df_precision(&self.data_source.byte_size());

        // Column statistics must match the output schema (leftover_schema), which may differ
        // from the initial schema after try_swapping_with_projection adds computed columns.
        let column_statistics = self.leftover_statistics.clone();

        let statistics = Statistics {
            num_rows,
            total_byte_size,
            column_statistics,
        };
        // A filter can remove the rows that supplied an exact count or extremum.
        let statistics = if self.filter.is_some() {
            statistics.to_inexact()
        } else {
            statistics
        };
        // Fetch is accepted only for one output partition, so it applies globally.
        Ok(Arc::new(statistics.with_fetch(self.limit, 0, 1)?))
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn DataSource>> {
        // DataSourceExec retains its partition properties when accepting fetch.
        if limit.is_some()
            && (self.output_partition_count() > 1 || !self.dynamic_filters.is_empty())
        {
            return None;
        }
        let mut this = self.clone();
        this.output_partitions = self.output_partition_count();
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

        let converter = &self.expression_converter;
        let input_schema = self.initial_schema.as_ref();
        let projected_schema = projection.project_schema(input_schema)?;

        // Use the shared ExpressionConverter to split the projection into a Vortex
        // scan_projection and a leftover DataFusion projection for expressions that
        // can't be pushed down (e.g., unsupported scalar functions, decimal binary).
        let ProcessedProjection {
            scan_projection,
            scan_reference_schema,
            leftover_projection,
        } = converter.split_projection(projection.clone(), input_schema, &projected_schema)?;

        // Compose with the initial projection so the scan operates on the original
        // source columns, not the initial projection's output columns.
        let scan_projection = replace(scan_projection, &root(), self.initial_projection.clone());

        // Retain Arrow storage types and metadata required by the leftover expressions.
        let scan_dtype = scan_projection
            .return_dtype(self.data_source.dtype())
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
        let scan_output_schema = Arc::new(calculate_physical_schema(
            &scan_dtype,
            &scan_reference_schema,
            &self.session.arrow(),
        )?);

        // Remap the leftover column references to match the scan output schema.
        let leftover_projection = leftover_projection
            .try_map_exprs(|expr| reassign_expr_columns(expr, &scan_output_schema))?;

        let final_schema = Arc::new(projected_schema);

        let mut this = self.clone();
        this.output_ordering = project_output_ordering(self.initial_ordering.as_ref(), projection);
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
        _config: &ConfigOptions,
    ) -> DFResult<FilterPushdownPropagation<Arc<dyn DataSource>>> {
        // Filters arriving after fetch must remain above that row limit.
        if filters.is_empty() || self.limit.is_some() {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                vec![PushedDown::No; filters.len()],
            ));
        }

        let mut this = self.clone();
        let mut pushdown_results = Vec::with_capacity(filters.len());
        let mut converted = Vec::new();
        for filter in filters {
            let filter = match &self.leftover_projection {
                Some(projection) => projection.unproject_expr(&filter)?,
                None => filter,
            };
            let filter = reassign_expr_columns(filter, &self.projected_schema)?;
            if DynamicFilterTracking::classify(&filter).contains_dynamic_filter() {
                converted.push(convert_dynamic_filter(
                    Arc::clone(&filter),
                    Arc::clone(&self.projected_schema),
                    Arc::clone(&self.expression_converter),
                ));
                this.dynamic_filters.push(filter);
                pushdown_results.push(PushedDown::No);
            } else if let Some(expression) = self
                .expression_converter
                .try_convert(&filter, &self.projected_schema)?
            {
                converted.push(expression);
                pushdown_results.push(PushedDown::Yes);
            } else {
                pushdown_results.push(PushedDown::No);
            }
        }

        if converted.is_empty() {
            return Ok(FilterPushdownPropagation::with_parent_pushdown_result(
                pushdown_results,
            ));
        }

        // Rewrite the pushed filters over the scan's source columns and conjoin them with any
        // existing filter.
        let pushed = converted
            .into_iter()
            .map(|expr| replace(expr, &root(), self.projected_projection.clone()));
        this.filter = and_collect(self.filter.iter().cloned().chain(pushed));
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
                .map(|expr| &expr.expr)
                .chain(self.dynamic_filters.iter()),
            f,
        )
    }
}

fn project_output_ordering(
    ordering: Option<&LexOrdering>,
    projection: &ProjectionExprs,
) -> Option<LexOrdering> {
    let mut projected = Vec::new();
    for key in ordering? {
        if !key.expr.is::<Column>() {
            break;
        }
        let Some((index, output)) = projection
            .iter()
            .enumerate()
            .find(|(_, output)| output.expr.eq(&key.expr))
        else {
            break;
        };
        projected.push(PhysicalSortExpr {
            expr: Arc::new(Column::new(&output.alias, index)),
            options: key.options,
        });
    }
    LexOrdering::new(projected)
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
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::task::Poll;

    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    use arrow_schema::DataType;
    use async_trait::async_trait;
    use datafusion::prelude::SessionContext;
    use datafusion_datasource::source::DataSourceExec;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::projection::ProjectionExpr;
    use datafusion_physical_plan::ExecutionPlan;
    use datafusion_physical_plan::collect;
    use datafusion_physical_plan::expressions::BinaryExpr;
    use datafusion_physical_plan::expressions::lit;
    use futures::TryStreamExt;
    use futures::task::ArcWake;
    use futures::task::waker_ref;
    use rstest::rstest;
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::array::stats::StatsSet;
    use vortex::array::stream::ArrayStreamAdapter;
    use vortex::array::stream::ArrayStreamExt;
    use vortex::array::stream::SendableArrayStream;
    use vortex::buffer::ByteBufferMut;
    use vortex::buffer::buffer;
    use vortex::dtype::FieldPath;
    use vortex::dtype::PType;
    use vortex::dtype::StructFields;
    use vortex::error::VortexError;
    use vortex::file::OpenOptionsSessionExt;
    use vortex::file::WriteOptionsSessionExt;
    use vortex::layout::scan::layout::LayoutReaderDataSource;
    // Aliased so it does not shadow the DataFusion `DataSource` trait brought in via `super::*`,
    // which provides the `open` method exercised below.
    use vortex::scan::DataSource as VortexScanDataSource;
    use vortex::scan::DataSourceScan;
    use vortex::scan::DataSourceScanRef;
    use vortex::scan::Partition;
    use vortex::scan::PartitionRef;
    use vortex::scan::PartitionStream;
    use vortex::session::VortexSession;
    use vortex_arrow::ArrowSessionExt;

    use super::*;
    use crate::v2::VortexTable;

    #[derive(Default)]
    struct WakeCounter(AtomicUsize);

    impl ArcWake for WakeCounter {
        fn wake_by_ref(arc_self: &Arc<Self>) {
            arc_self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The source enforces a pushed-down `LIMIT` itself: DataFusion removes its limit operator
    /// once the fetch is accepted, so a filtered scan over several partitions must still return
    /// exactly `LIMIT` rows.
    #[tokio::test]
    async fn filtered_limit_returns_exactly_limit_rows() -> anyhow::Result<()> {
        let session = VortexSession::default();
        let array = StructArray::from_fields(&[(
            "a",
            buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].into_array(),
        )])?
        .into_array();
        let mut buf = ByteBufferMut::empty();
        session
            .write_options()
            .write(&mut buf, array.to_array_stream())
            .await?;
        let file = session.open_options().open_buffer(buf)?;
        let data_source = Arc::new(
            LayoutReaderDataSource::new(file.layout_reader()?, session.clone())
                .with_split_max_row_count(3),
        );
        let schema = session.arrow().to_arrow_schema(file.dtype())?;

        let ctx = SessionContext::new();
        ctx.register_table(
            "t",
            Arc::new(VortexTable::new(data_source, session, Arc::new(schema))),
        )?;
        let batches = ctx
            .sql("SELECT a FROM t WHERE a >= 4 LIMIT 5")
            .await?
            .collect()
            .await?;

        let values = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<Int32Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(values.len(), 5);
        assert!(values.iter().all(|value| (4..12).contains(value)));
        let mut unique = values;
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 5);
        assert!(
            batches
                .iter()
                .all(|batch| batch.column(0).null_count() == 0)
        );
        Ok(())
    }

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

    /// One batch, with the first partition yielding once before it is ready.
    struct TestPartition {
        index: usize,
        values: &'static [i32],
        polls: Arc<AtomicUsize>,
    }

    impl Partition for TestPartition {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn index(&self) -> usize {
            self.index
        }

        fn row_count(&self) -> Precision<u64> {
            Precision::exact(self.values.len() as u64)
        }

        fn byte_size(&self) -> Precision<u64> {
            Precision::Absent
        }

        fn execute(self: Box<Self>) -> VortexResult<SendableArrayStream> {
            let mut yield_once = self.index == 0;
            let mut values = Some(self.values);
            let polls = self.polls;
            let stream = stream::poll_fn(move |cx| {
                polls.fetch_add(1, Ordering::Relaxed);
                if yield_once {
                    yield_once = false;
                    cx.waker().wake_by_ref();
                    return Poll::Pending;
                }
                Poll::Ready(values.take().map(|values| {
                    StructArray::from_fields(&[(
                        "x",
                        PrimitiveArray::from_iter(values.iter().copied()).into_array(),
                    )])
                    .map(IntoArray::into_array)
                }))
            });
            Ok(ArrayStreamExt::boxed(ArrayStreamAdapter::new(
                test_dtype(),
                stream,
            )))
        }
    }

    struct TestScan {
        dtype: DType,
        limit: Option<u64>,
        polls: Arc<AtomicUsize>,
    }

    impl DataSourceScan for TestScan {
        fn dtype(&self) -> &DType {
            &self.dtype
        }

        fn partition_count(&self) -> Precision<usize> {
            Precision::Exact(2)
        }

        fn partitions(self: Box<Self>) -> PartitionStream {
            let limit = self
                .limit
                .map(|limit| usize::try_from(limit).unwrap_or(usize::MAX));
            let partitions: Vec<VortexResult<PartitionRef>> = [&[0, 1][..], &[100, 101][..]]
                .into_iter()
                .enumerate()
                .map(|(index, values)| {
                    // Ordered partitions enforce their own limit; the adapter must enforce the
                    // global limit when it concatenates their outputs.
                    let values = &values[..limit.unwrap_or(values.len()).min(values.len())];
                    Ok(Box::new(TestPartition {
                        index,
                        values,
                        polls: Arc::clone(&self.polls),
                    }) as PartitionRef)
                })
                .collect();
            stream::iter(partitions).boxed()
        }
    }

    struct TestDataSource {
        dtype: DType,
        polls: Arc<AtomicUsize>,
        scans: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl VortexScanDataSource for TestDataSource {
        fn dtype(&self) -> &DType {
            &self.dtype
        }

        async fn scan(&self, request: ScanRequest) -> VortexResult<DataSourceScanRef> {
            self.scans.fetch_add(1, Ordering::Relaxed);
            Ok(Box::new(TestScan {
                dtype: self.dtype.clone(),
                limit: request.limit,
                polls: Arc::clone(&self.polls),
            }))
        }

        async fn field_statistics(&self, _field_path: &FieldPath) -> VortexResult<StatsSet> {
            Ok(StatsSet::default())
        }
    }

    /// The first partition yields once, so completion order differs from partition order.
    /// Fetch three also requires slicing the batch that crosses the global limit.
    #[rstest]
    #[case::all(None, &[0, 1, 100, 101])]
    #[case::empty(Some(0), &[])]
    #[case::first(Some(1), &[0])]
    #[case::across_partitions(Some(3), &[0, 1, 100])]
    #[tokio::test]
    async fn ordered_scan_enforces_global_fetch(
        #[case] limit: Option<usize>,
        #[case] expected: &[i32],
    ) -> VortexResult<()> {
        let arrow_schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let polls = Arc::new(AtomicUsize::new(0));
        let data_source: DataSourceRef = Arc::new(TestDataSource {
            dtype: test_dtype(),
            polls: Arc::clone(&polls),
            scans: Arc::default(),
        });

        let mut source = VortexDataSource::builder(data_source, VortexSession::default())
            .with_arrow_schema(arrow_schema)
            .build()
            .await?;
        source.ordered = true;
        source.limit = limit;

        let mut stream = source
            .open(0, Arc::new(TaskContext::default()))
            .map_err(VortexError::external)?;
        let batches: Vec<RecordBatch> = stream
            .by_ref()
            .try_collect()
            .await
            .map_err(VortexError::external)?;
        let values = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<Int32Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(values, expected);

        let finished_polls = polls.load(Ordering::Relaxed);
        assert!(stream.next().await.is_none());
        assert_eq!(polls.load(Ordering::Relaxed), finished_polls);
        if limit == Some(0) {
            assert_eq!(finished_polls, 0);
        }
        Ok(())
    }

    #[tokio::test]
    async fn output_partitions_share_one_lazy_scan_and_reset_reopens_it() -> anyhow::Result<()> {
        let scans = Arc::new(AtomicUsize::new(0));
        let data_source: DataSourceRef = Arc::new(TestDataSource {
            dtype: test_dtype(),
            polls: Arc::default(),
            scans: Arc::clone(&scans),
        });
        let source = VortexDataSource::builder(data_source, VortexSession::default())
            .build()
            .await?;
        for requested in [0, 1, 257] {
            let repartitioned = source.repartitioned(requested, 0, None)?.unwrap();
            assert_eq!(
                repartitioned.output_partitioning().partition_count(),
                requested.max(1)
            );
        }
        let source = source
            .repartitioned(4, 0, None)?
            .expect("repartitioning supported");
        assert_eq!(source.output_partitioning().partition_count(), 4);
        assert_eq!(
            source.partition_statistics(Some(0))?.num_rows,
            DFPrecision::Absent
        );
        assert!(source.with_fetch(Some(1)).is_none());
        assert!(source.open(0, Arc::new(TaskContext::default())).is_err());
        let plan = Arc::new(DataSourceExec::new(Arc::clone(&source))) as Arc<dyn ExecutionPlan>;
        assert_eq!(scans.load(Ordering::Relaxed), 0);
        let context = Arc::new(TaskContext::default());
        let values = |batches: Vec<RecordBatch>| {
            let mut values = batches
                .iter()
                .flat_map(|batch| {
                    batch
                        .column(0)
                        .as_primitive::<Int32Type>()
                        .values()
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>();
            values.sort_unstable();
            values
        };
        assert_eq!(
            values(collect(Arc::clone(&plan), Arc::clone(&context)).await?),
            [0, 1, 100, 101]
        );
        assert_eq!(scans.load(Ordering::Relaxed), 1);
        assert_eq!(plan.metrics().expect("scan metrics").output_rows(), Some(4));
        assert!(plan.execute(0, Arc::clone(&context)).is_err());
        let reset = Arc::clone(&plan).reset_state()?;
        let independent = Arc::new(DataSourceExec::new(source)) as Arc<dyn ExecutionPlan>;
        let (reset, independent) = futures::try_join!(
            collect(reset, Arc::clone(&context)),
            collect(independent, context)
        )?;
        assert_eq!(values(reset), [0, 1, 100, 101]);
        assert_eq!(values(independent), [0, 1, 100, 101]);
        assert_eq!(scans.load(Ordering::Relaxed), 3);
        let cancelled = Arc::clone(&plan).reset_state()?;
        let mut output = cancelled.execute(0, Arc::new(TaskContext::default()))?;
        assert!(output.try_next().await?.is_some());
        drop(output);
        assert!(
            cancelled
                .execute(1, Arc::new(TaskContext::default()))
                .is_err()
        );

        let execution = Arc::new(ScanExecution::new(2));
        let output = |partition| OutputPartitionStream {
            partition,
            schema: plan.schema(),
            stream: Some(Box::pin(RecordBatchStreamAdapter::new(
                plan.schema(),
                stream::pending(),
            ))),
            execution: Arc::clone(&execution),
            metrics: BaselineMetrics::new(&ExecutionPlanMetricsSet::new(), partition),
        };
        let cancelled = output(0);
        let mut pending = Box::pin(output(1));
        let wakes = Arc::new(WakeCounter::default());
        let waker = waker_ref(&wakes);
        let mut cx = Context::from_waker(&waker);
        assert!(pending.as_mut().poll_next(&mut cx).is_pending());
        drop(cancelled);
        assert_eq!(wakes.0.load(Ordering::Relaxed), 1);
        assert!(matches!(
            pending.as_mut().poll_next(&mut cx),
            Poll::Ready(Some(Err(_)))
        ));
        Ok(())
    }

    #[tokio::test]
    async fn proven_ordering_keeps_one_output_partition_and_remaps_identity_outputs()
    -> anyhow::Result<()> {
        let data_source: DataSourceRef = Arc::new(TestDataSource {
            dtype: test_dtype(),
            polls: Arc::default(),
            scans: Arc::default(),
        });
        let key = Arc::new(Column::new("x", 0)) as Arc<dyn PhysicalExpr>;
        let source = VortexDataSource::builder(data_source, VortexSession::default())
            .with_output_ordering(
                LexOrdering::new([PhysicalSortExpr::new_default(Arc::clone(&key))])
                    .expect("one key"),
            )
            .build()
            .await?;
        let source = source
            .repartitioned(4, 0, None)?
            .expect("repartitioning supported");
        assert_eq!(source.output_partitioning().partition_count(), 1);
        let projected = source
            .try_swapping_with_projection(&ProjectionExprs::from(vec![ProjectionExpr::new(
                Arc::clone(&key),
                "renamed",
            )]))?
            .expect("projection supported");
        let properties = projected.eq_properties();
        let ordering = properties
            .output_ordering()
            .expect("identity preserves ordering");
        assert_eq!(
            ordering[0].expr.as_ref(),
            &Column::new("renamed", 0) as &dyn PhysicalExpr
        );
        let computed = source
            .try_swapping_with_projection(&ProjectionExprs::from(vec![ProjectionExpr::new(
                Arc::new(BinaryExpr::new(key, Operator::Plus, lit(1_i32))),
                "incremented",
            )]))?
            .expect("projection supported");
        assert!(computed.eq_properties().output_ordering().is_none());
        Ok(())
    }
}
