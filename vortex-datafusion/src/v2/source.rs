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

use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use datafusion_common::ColumnStatistics;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::Statistics;
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
use datafusion_physical_plan::limit::LimitStream;
use datafusion_physical_plan::metrics::BaselineMetrics;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::Stream;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::try_join_all;
use futures::lock::Mutex as AsyncMutex;
use futures::stream;
use futures::stream::BoxStream;
use futures::task::AtomicWaker;
use tokio::sync::OnceCell;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::array::stream::ArrayStreamExt;
use vortex::array::stream::SendableArrayStream;
use vortex::dtype::DType;
use vortex::dtype::FieldPath;
use vortex::dtype::Nullability;
use vortex::error::VortexError;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::get_item;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::expr::stats::Precision;
use vortex::expr::transform::replace;
use vortex::scan::DataSourceRef;
use vortex::scan::DataSourceScanRef;
use vortex::scan::PartitionRef;
use vortex::scan::PartitionStream;
use vortex::scan::ScanRequest;
use vortex::scan::stream::ScanExec;
use vortex::session::VortexSession;
use vortex_arrow::ArrowArrayStreamExt;
use vortex_arrow::ArrowSessionExt;
use vortex_utils::parallelism::get_available_parallelism;

use super::dynamic_filters::convert_dynamic_filter;
use crate::convert::exprs::DefaultExpressionConverter;
use crate::convert::exprs::ExpressionConverter;
use crate::convert::exprs::ProcessedProjection;
use crate::convert::schema::calculate_physical_schema;
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
    expression_converter: Arc<dyn ExpressionConverter>,
    output_ordering: Option<LexOrdering>,
    split_scan: bool,
}

/// The environment variable that selects how [`VortexDataSource`] reads its scan: `0` reads the
/// scan's shared partition stream, anything else (the default) reads it through
/// [`DataSourceScan::open`](vortex::scan::DataSourceScan::open) and
/// [`DataSourceScan::split`](vortex::scan::DataSourceScan::split).
///
/// It exists to compare the two in one binary, and goes away with the partition stream.
pub const SCAN_POOL_ENV: &str = "VORTEX_SCAN_POOL";

fn split_scan_from_env() -> bool {
    std::env::var(SCAN_POOL_ENV).map_or(true, |value| value != "0")
}

impl VortexDataSourceBuilder {
    /// Sets the Arrow schema exposed to DataFusion.
    ///
    /// Defaults to the schema inferred by [`ArrowSession::to_arrow_schema`].
    ///
    /// [`Self::build`] checks extension identities and supported Arrow layouts before applying the
    /// initial projection. A schema cannot label a plain column as an extension or erase an existing
    /// extension's identity.
    ///
    /// [`ArrowSession::to_arrow_schema`]: vortex_arrow::ArrowSession::to_arrow_schema
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
    /// [`DynamicComparison`](vortex::functions::dynamic::DynamicComparison).
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

    /// Selects whether output partitions read the scan through its `open` and `split` (`true`) or
    /// take partitions from the scan's shared partition stream (`false`). Defaults to
    /// [`SCAN_POOL_ENV`], read when the builder is created.
    pub fn with_split_scan(mut self, split_scan: bool) -> Self {
        self.split_scan = split_scan;
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
            Some(schema) => {
                // Validate before projection can hide an incompatible column. Keep the supplied
                // schema because each scan derives its own physical schema after projection.
                calculate_physical_schema(self.data_source.dtype(), &schema, &self.session.arrow())
                    .map_err(VortexError::external)?;

                schema
            }
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
            preserve_order: false,
            target_parallelism: get_available_parallelism().unwrap_or(1),
            split_scan: self.split_scan,
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
            split_scan: split_scan_from_env(),
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
/// A scan exposes the requested DataFusion output partitions. With a declared ordering, each output
/// partition keeps that ordering, as a sorted file group does in DataFusion's own file scans, and
/// DataFusion merges the partitions where it needs one global order.
///
/// Each execution scans once, through [`DataSource::create_sibling_state`], and the first output
/// partition polled starts the scan. Like DataFusion's own file scans, the output partitions share
/// the work unless each must keep its rows in order:
///
/// - An unordered scan is [opened] once, and every output partition reads its own stream of it,
///   so partitions that finish early keep taking work.
/// - A scan with a declared ordering, or whose output partitions DataFusion asks to keep their
///   order, is [split] into one fixed piece per output partition, and output partition `i` reads
///   piece `i` in order. The pieces are contiguous, so each partition keeps the ordering. Partitions beyond the
///   piece count return no rows.
///
/// Setting [`SCAN_POOL_ENV`] to `0` instead has the output partitions take partitions from one
/// shared partition stream, for comparison.
///
/// In every mode an output partition holds work that no other partition reads: the work its stream
/// has taken, its piece, or the partitions it has taken. Dropping an output partition before it finishes therefore cancels its
/// execution, and the other output partitions report an error rather than finish without those
/// rows.
///
/// [opened]: vortex::scan::DataSourceScan::open
/// [split]: vortex::scan::DataSourceScan::split
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
    /// Whether to preserve a declared ordering in each output partition.
    ordered: bool,
    /// Whether each output partition must return its rows in order, as DataFusion requests
    /// through [`DataSource::with_preserve_order`].
    preserve_order: bool,

    /// Target Arrow conversion concurrency when only one output partition is exposed.
    target_parallelism: usize,
    /// Whether output partitions read the scan through its `open` and `split` rather than taking
    /// partitions from the scan's shared partition stream, which keeps one output partition for a
    /// limited scan.
    split_scan: bool,
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
        // The shared partition stream keeps an ordering only in one output partition.
        if !self.split_scan && (self.ordered || self.limit.is_some()) {
            1
        } else {
            self.output_partitions
        }
    }

    fn scan_request(&self) -> ScanRequest {
        ScanRequest {
            projection: self.projected_projection.clone(),
            filter: self.filter.clone(),
            limit: self.limit.map(|l| u64::try_from(l).unwrap_or(u64::MAX)),
            ordered: self.ordered || self.preserve_order,
            ..Default::default()
        }
    }

    /// Output partition `partition` of the execution `execution`.
    fn open_partition(
        &self,
        partition: usize,
        execution: Arc<ScanExecution>,
    ) -> DFResult<SendableRecordBatchStream> {
        if execution.cancelled.load(Ordering::Acquire) {
            return Err(cancelled());
        }
        let output = execution
            .output_partitions
            .get(partition)
            .ok_or_else(|| invalid_partition(partition))?;
        if output.claimed.swap(true, Ordering::AcqRel) {
            return Err(DataFusionError::Execution(
                "scan output partition already opened; reset the execution plan before replay"
                    .into(),
            ));
        }
        let data_source = Arc::clone(&self.data_source);
        let request = self.scan_request();
        let session = self.session.clone();
        let schema = Arc::clone(&self.projected_schema);
        let concurrency = self.conversion_concurrency();
        let ordered = self.ordered;
        let shared = Arc::clone(&execution);
        let batches = stream::once(async move {
            let work = execution
                .work
                .get_or_init(|| execution.start(data_source, request))
                .await
                .as_ref()
                .map_err(Clone::clone)?;
            let arrays = match work {
                ScanWork::Partitions(partitions) => Arc::clone(partitions).arrays(ordered),
                ScanWork::Stream(exec) => ArrayStreamExt::boxed(exec.stream()),
                ScanWork::Pieces(pieces) => {
                    let piece = match pieces.get(partition) {
                        Some(piece) => piece.lock().await.take(),
                        None => None,
                    };
                    // Output partitions beyond the scan's pieces have nothing to read.
                    let Some(piece) = piece else {
                        return VortexResult::Ok(stream::empty().boxed());
                    };
                    ArrayStreamExt::boxed(piece.open()?.stream())
                }
            };
            Ok(arrays.to_arrow(&session, Some(schema), concurrency))
        })
        .try_flatten()
        .boxed();
        self.output_stream(batches, partition, shared)
    }

    /// Applies the limit and leftover projection to an output partition's batches and records its
    /// metrics. A zero limit reads nothing.
    fn output_stream(
        &self,
        batches: BoxStream<'static, VortexResult<RecordBatch>>,
        partition: usize,
        execution: Arc<ScanExecution>,
    ) -> DFResult<SendableRecordBatchStream> {
        let metrics = BaselineMetrics::new(&self.metrics, partition);
        let stream = if self.limit == Some(0) {
            Box::pin(RecordBatchStreamAdapter::new(
                Arc::clone(&self.leftover_schema),
                stream::empty(),
            )) as SendableRecordBatchStream
        } else {
            self.batch_stream(batches, partition)?
        };
        Ok(Box::pin(OutputPartitionStream {
            partition,
            schema: Arc::clone(&self.leftover_schema),
            stream: Some(stream),
            execution,
            metrics,
        }))
    }

    /// How many arrays each output partition converts to Arrow at once: one when several
    /// partitions already convert in parallel, or the target parallelism for a single one.
    fn conversion_concurrency(&self) -> NonZeroUsize {
        let concurrency = if self.output_partition_count() == 1 {
            self.target_parallelism
        } else {
            1
        };
        NonZeroUsize::new(concurrency).unwrap_or(NonZeroUsize::MIN)
    }

    fn batch_stream(
        &self,
        batches: BoxStream<'static, VortexResult<RecordBatch>>,
        partition: usize,
    ) -> DFResult<SendableRecordBatchStream> {
        let projected_schema = Arc::clone(&self.projected_schema);
        let limit = self.limit;

        // Pre-build the leftover projector (if any) so we can apply it after batch conversion.
        let leftover_projector = self
            .leftover_projection
            .as_ref()
            .map(|proj| proj.make_projector(&self.projected_schema))
            .transpose()?;

        let stream =
            batches.map(|result| result.map_err(|e| DataFusionError::External(Box::new(e))));

        // With one output partition DataFusion drops its own limit operator once `with_fetch`
        // accepts the limit, so the fetch must be exact here. With several, the fetch bounds each
        // partition and DataFusion keeps a global limit above.
        let stream = match limit {
            Some(limit) => LimitStream::new(
                Box::pin(RecordBatchStreamAdapter::new(projected_schema, stream)),
                0,
                Some(limit),
                BaselineMetrics::new(&ExecutionPlanMetricsSet::new(), partition),
            )
            .boxed(),
            None => stream.boxed(),
        };

        // Apply leftover projection (expressions that couldn't be pushed into Vortex).
        let stream = if let Some(projector) = leftover_projector {
            stream
                .map(move |batch_result| {
                    batch_result.and_then(|batch| projector.project_batch(&batch))
                })
                .boxed()
        } else {
            stream
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            Arc::clone(&self.leftover_schema),
            stream,
        )))
    }
}

fn invalid_partition(partition: usize) -> DataFusionError {
    DataFusionError::Execution(format!("invalid scan output partition {partition}"))
}

/// The scan that every output partition of one execution reads.
///
/// The first output partition polled starts the scan; every other partition reads the same one.
struct ScanExecution {
    split: bool,
    output_partitions: Vec<OutputPartitionState>,
    cancelled: AtomicBool,
    work: OnceCell<VortexResult<ScanWork>>,
}

#[derive(Default)]
struct OutputPartitionState {
    claimed: AtomicBool,
    waker: AtomicWaker,
}

enum ScanWork {
    /// Every output partition takes partitions from this shared stream.
    Partitions(Arc<SharedPartitions>),
    /// Every output partition reads its own stream of the opened scan.
    Stream(ScanExec),
    /// Output partition `i` takes piece `i`, if there is one.
    Pieces(Vec<AsyncMutex<Option<DataSourceScanRef>>>),
}

impl ScanExecution {
    fn new(output_partitions: usize, split: bool) -> Self {
        Self {
            split,
            output_partitions: (0..output_partitions)
                .map(|_| OutputPartitionState::default())
                .collect(),
            cancelled: AtomicBool::new(false),
            work: OnceCell::new(),
        }
    }

    /// Cancels the execution and wakes every output partition, so each reports the error.
    fn cancel(&self) {
        if !self.cancelled.swap(true, Ordering::AcqRel) {
            for output in &self.output_partitions {
                output.waker.wake();
            }
        }
    }

    /// Drops the scan work that no output partition holds, once the execution is cancelled.
    fn release(&self) {
        match self.work.get() {
            Some(Ok(ScanWork::Partitions(shared))) => {
                if let Some(mut partitions) = shared.partitions.try_lock() {
                    drop(partitions.take());
                }
            }
            Some(Ok(ScanWork::Pieces(pieces))) => {
                for piece in pieces {
                    if let Some(mut piece) = piece.try_lock() {
                        drop(piece.take());
                    }
                }
            }
            // The streams that output partitions hold end with the execution's error.
            Some(Ok(ScanWork::Stream(_)) | Err(_)) | None => {}
        }
    }

    async fn start(
        &self,
        data_source: DataSourceRef,
        request: ScanRequest,
    ) -> VortexResult<ScanWork> {
        let ordered = request.ordered;
        let scan = data_source.scan(request).await?;
        if !self.split {
            return Ok(ScanWork::Partitions(Arc::new(SharedPartitions {
                dtype: scan.dtype().clone(),
                partitions: AsyncMutex::new(Some(scan.partitions())),
            })));
        }
        if !ordered {
            return Ok(ScanWork::Stream(scan.open()?));
        }
        let pieces = NonZeroUsize::new(self.output_partitions.len()).unwrap_or(NonZeroUsize::MIN);
        Ok(ScanWork::Pieces(
            scan.split(pieces)?
                .into_iter()
                .map(|piece| AsyncMutex::new(Some(piece)))
                .collect(),
        ))
    }
}

/// Scan API partitions shared by every output partition, each taking the next one it needs.
struct SharedPartitions {
    dtype: DType,
    partitions: AsyncMutex<Option<PartitionStream>>,
}

/// The most partitions each output partition reads at once from [`SharedPartitions`].
const SCAN_PARTITION_READ_AHEAD: usize = 2;

impl SharedPartitions {
    /// Takes the next partition, dropping the stream once it ends.
    async fn next(&self) -> Option<VortexResult<PartitionRef>> {
        let mut partitions = self.partitions.lock().await;
        let next = partitions.as_mut()?.next().await;
        if next.is_none() {
            drop(partitions.take());
        }
        next
    }

    /// The arrays of the partitions this output partition takes, in partition order when
    /// `ordered`, or reading a few partitions at once otherwise.
    fn arrays(self: Arc<Self>, ordered: bool) -> SendableArrayStream {
        let dtype = self.dtype.clone();
        let partitions = stream::unfold(self, |shared| async move {
            let next = shared.next().await?;
            Some((next, shared))
        })
        .map(|partition| partition.and_then(|partition| partition.execute()));
        let arrays = if ordered {
            partitions.try_flatten().boxed()
        } else {
            partitions
                .try_flatten_unordered(Some(SCAN_PARTITION_READ_AHEAD))
                .boxed()
        };
        ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, arrays))
    }
}

/// One output partition of a [`ScanExecution`], which cancels the execution when it fails or is
/// dropped before it finishes.
struct OutputPartitionStream {
    partition: usize,
    schema: SchemaRef,
    /// The partition's batches, until it finishes.
    stream: Option<SendableRecordBatchStream>,
    execution: Arc<ScanExecution>,
    metrics: BaselineMetrics,
}

impl OutputPartitionStream {
    fn finish(&mut self) {
        self.stream = None;
        drop(
            self.execution.output_partitions[self.partition]
                .waker
                .take(),
        );
    }
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
            result = Poll::Ready(Some(Err(cancelled())));
        }
        if matches!(result, Poll::Ready(Some(Err(_)))) {
            self.execution.cancel();
        }
        if matches!(result, Poll::Ready(None | Some(Err(_)))) {
            self.finish();
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
        self.finish();
        if self.execution.cancelled.load(Ordering::Acquire) {
            self.execution.release();
        }
        self.metrics.done();
    }
}

fn cancelled() -> DataFusionError {
    DataFusionError::Execution("scan execution cancelled".into())
}

impl DataSource for VortexDataSource {
    fn open(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DFResult<SendableRecordBatchStream> {
        // Several partitions read one scan only through sibling state.
        if self.output_partition_count() > 1 {
            return Err(DataFusionError::Execution(
                "multiple scan output partitions require sibling execution state".into(),
            ));
        }
        self.open_partition(partition, Arc::new(ScanExecution::new(1, self.split_scan)))
    }

    fn create_sibling_state(&self, _config: &ConfigOptions) -> Option<Arc<dyn Any + Send + Sync>> {
        Some(Arc::new(ScanExecution::new(
            self.output_partition_count(),
            self.split_scan,
        )))
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
        // The fetch bounds each output partition.
        Ok(Arc::new(statistics.with_fetch(
            self.limit,
            0,
            self.output_partition_count(),
        )?))
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn DataSource>> {
        // DataSourceExec retains its partition properties when accepting fetch, so the fetch
        // bounds each output partition and DataFusion keeps a global limit above several.
        if limit.is_some()
            && (!self.dynamic_filters.is_empty()
                || (!self.split_scan && self.output_partition_count() > 1))
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

    fn with_preserve_order(&self, preserve_order: bool) -> Option<Arc<dyn DataSource>> {
        if !self.split_scan {
            return None;
        }
        let mut this = self.clone();
        this.preserve_order = preserve_order;
        Some(Arc::new(this))
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
    use std::num::NonZeroUsize;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::task::Poll;

    use arrow_array::cast::AsArray;
    use arrow_array::types::Int32Type;
    use arrow_schema::DataType;
    use arrow_schema::Field;
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

    /// The values of each test partition.
    const TEST_PARTITIONS: [&[i32]; 2] = [&[0, 1], &[100, 101]];

    /// A scan of some of [`TEST_PARTITIONS`], which splits into one piece per partition.
    struct TestScan {
        dtype: DType,
        limit: Option<u64>,
        polls: Arc<AtomicUsize>,
        partitions: Vec<usize>,
    }

    impl DataSourceScan for TestScan {
        fn dtype(&self) -> &DType {
            &self.dtype
        }

        fn open(self: Box<Self>) -> VortexResult<ScanExec> {
            Ok(ScanExec::from_partitions(self))
        }

        fn partition_count(&self) -> Precision<usize> {
            Precision::Exact(self.partitions.len())
        }

        fn split(self: Box<Self>, n: NonZeroUsize) -> VortexResult<Vec<DataSourceScanRef>> {
            if n.get() < self.partitions.len() {
                return Ok(vec![self]);
            }
            Ok(self
                .partitions
                .iter()
                .map(|&partition| {
                    Box::new(TestScan {
                        dtype: self.dtype.clone(),
                        limit: self.limit,
                        polls: Arc::clone(&self.polls),
                        partitions: vec![partition],
                    }) as DataSourceScanRef
                })
                .collect())
        }

        fn partitions(self: Box<Self>) -> PartitionStream {
            let limit = self
                .limit
                .map(|limit| usize::try_from(limit).unwrap_or(usize::MAX));
            let partitions: Vec<VortexResult<PartitionRef>> = self
                .partitions
                .iter()
                .map(|&index| {
                    let values = TEST_PARTITIONS[index];
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
                partitions: vec![0, 1],
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

    /// One execution scans once, lazily, and a reset execution scans again. Unordered output
    /// partitions share the opened scan's work; output partitions that keep their order read piece
    /// `i` of one split, and partitions beyond the piece count return no rows.
    #[rstest]
    #[case::shared_work(false)]
    #[case::ordered_pieces(true)]
    #[tokio::test]
    async fn output_partitions_read_one_scan_per_execution(
        #[case] preserve_order: bool,
    ) -> anyhow::Result<()> {
        let scans = Arc::new(AtomicUsize::new(0));
        let data_source: DataSourceRef = Arc::new(TestDataSource {
            dtype: test_dtype(),
            polls: Arc::default(),
            scans: Arc::clone(&scans),
        });
        let source = VortexDataSource::builder(data_source, VortexSession::default())
            .with_split_scan(true)
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
        let fetched = source
            .with_fetch(Some(1))
            .expect("fetch bounds each partition");
        assert_eq!(fetched.output_partitioning().partition_count(), 4);
        let source = if preserve_order {
            source
                .with_preserve_order(true)
                .expect("order preserved per partition")
        } else {
            source
        };

        let plan = Arc::new(DataSourceExec::new(Arc::clone(&source))) as Arc<dyn ExecutionPlan>;
        let context = Arc::new(TaskContext::default());
        let streams = (0..4)
            .map(|partition| plan.execute(partition, Arc::clone(&context)))
            .collect::<DFResult<Vec<_>>>()?;
        assert_eq!(scans.load(Ordering::Relaxed), 0);
        let mut partitions = Vec::new();
        for stream in streams {
            let batches: Vec<RecordBatch> = stream.try_collect().await?;
            partitions.push(partition_values(&batches));
        }
        if preserve_order {
            assert_eq!(partitions, [vec![0, 1], vec![100, 101], vec![], vec![]]);
        } else {
            let mut values = partitions.concat();
            values.sort_unstable();
            assert_eq!(values, [0, 1, 100, 101]);
        }
        assert_eq!(scans.load(Ordering::Relaxed), 1);
        assert_eq!(plan.metrics().expect("scan metrics").output_rows(), Some(4));
        assert!(plan.execute(0, Arc::clone(&context)).is_err());
        assert!(plan.execute(4, Arc::clone(&context)).is_err());

        let reset = Arc::clone(&plan).reset_state()?;
        let mut values = partition_values(&collect(reset, Arc::clone(&context)).await?);
        values.sort_unstable();
        assert_eq!(values, [0, 1, 100, 101]);
        assert_eq!(scans.load(Ordering::Relaxed), 2);

        // Without sibling state, several partitions cannot share one scan.
        assert!(source.open(0, context).is_err());
        Ok(())
    }

    /// A file of 12 rows in splits of 3, scanned through the v2 table.
    async fn split_file_source() -> VortexResult<(DataSourceRef, VortexSession, SchemaRef)> {
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
        let data_source: DataSourceRef = Arc::new(
            LayoutReaderDataSource::new(file.layout_reader()?, session.clone())
                .with_split_max_row_count(3),
        );
        let schema = Arc::new(session.arrow().to_arrow_schema(file.dtype())?);
        Ok((data_source, session, schema))
    }

    fn partition_values(batches: &[RecordBatch]) -> Vec<i32> {
        batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<Int32Type>()
                    .values()
                    .iter()
                    .copied()
            })
            .collect()
    }

    /// Without `open` and `split`, output partitions share the scan's partition stream: a
    /// fetch keeps one partition, and direct opens need a single partition.
    #[tokio::test]
    async fn without_split_scan_partitions_share_the_partition_stream() -> anyhow::Result<()> {
        let data_source: DataSourceRef = Arc::new(TestDataSource {
            dtype: test_dtype(),
            polls: Arc::default(),
            scans: Arc::default(),
        });
        let source = VortexDataSource::builder(data_source, VortexSession::default())
            .with_split_scan(false)
            .build()
            .await?
            .repartitioned(4, 0, None)?
            .expect("repartitioning supported");
        assert!(source.with_fetch(Some(1)).is_none());
        assert!(source.with_preserve_order(true).is_none());
        assert!(source.open(0, Arc::new(TaskContext::default())).is_err());

        let plan = Arc::new(DataSourceExec::new(Arc::clone(&source))) as Arc<dyn ExecutionPlan>;
        let mut values = partition_values(&collect(plan, Arc::new(TaskContext::default())).await?);
        values.sort_unstable();
        assert_eq!(values, [0, 1, 100, 101]);
        Ok(())
    }

    #[derive(Default)]
    struct WakeCounter(AtomicUsize);

    impl ArcWake for WakeCounter {
        fn wake_by_ref(arc_self: &Arc<Self>) {
            arc_self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Dropping an output partition before it finishes cancels the execution, so the other
    /// partitions report an error rather than finish without the rows it held.
    #[rstest]
    #[case::split(true)]
    #[case::shared_partitions(false)]
    #[tokio::test]
    async fn dropping_an_unfinished_partition_cancels_the_execution(
        #[case] split: bool,
    ) -> anyhow::Result<()> {
        let data_source: DataSourceRef = Arc::new(TestDataSource {
            dtype: test_dtype(),
            polls: Arc::default(),
            scans: Arc::default(),
        });
        let source = VortexDataSource::builder(data_source, VortexSession::default())
            .with_split_scan(split)
            .build()
            .await?
            .repartitioned(2, 0, None)?
            .expect("repartitioning supported");
        let plan = Arc::new(DataSourceExec::new(source)) as Arc<dyn ExecutionPlan>;
        let mut output = plan.execute(0, Arc::new(TaskContext::default()))?;
        assert!(output.try_next().await?.is_some());
        drop(output);
        assert!(plan.execute(1, Arc::new(TaskContext::default())).is_err());

        // A partition already waiting is woken and reports the cancellation.
        let execution = Arc::new(ScanExecution::new(2, split));
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
        let dropped = output(0);
        let mut pending = Box::pin(output(1));
        let wakes = Arc::new(WakeCounter::default());
        let waker = waker_ref(&wakes);
        let mut cx = Context::from_waker(&waker);
        assert!(pending.as_mut().poll_next(&mut cx).is_pending());
        drop(dropped);
        assert_eq!(wakes.0.load(Ordering::Relaxed), 1);
        assert!(matches!(
            pending.as_mut().poll_next(&mut cx),
            Poll::Ready(Some(Err(_)))
        ));
        Ok(())
    }

    /// A fetch bounds each of several output partitions, which share the opened scan's work.
    #[tokio::test]
    async fn fetch_bounds_each_of_several_partitions() -> anyhow::Result<()> {
        let (data_source, session, schema) = split_file_source().await?;
        let source = VortexDataSource::builder(data_source, session)
            .with_arrow_schema(schema)
            .with_split_scan(true)
            .build()
            .await?
            .repartitioned(4, 0, None)?
            .expect("repartitioning supported")
            .with_fetch(Some(5))
            .expect("fetch supported");
        let plan = Arc::new(DataSourceExec::new(source)) as Arc<dyn ExecutionPlan>;
        assert_eq!(plan.properties().partitioning.partition_count(), 4);
        let context = Arc::new(TaskContext::default());

        let partitions = try_join_all((0..4).map(|partition| {
            let stream = plan.execute(partition, Arc::clone(&context));
            async move {
                let batches: Vec<RecordBatch> = stream?.try_collect().await?;
                Ok::<_, DataFusionError>(partition_values(&batches))
            }
        }))
        .await?;
        assert!(partitions.iter().all(|values| values.len() <= 5));
        let mut values = partitions.concat();
        assert!(values.len() >= 5, "{partitions:?}");
        values.sort_unstable();
        values.dedup();
        assert_eq!(values.len(), partitions.iter().map(Vec::len).sum::<usize>());
        Ok(())
    }

    #[tokio::test]
    async fn proven_ordering_holds_in_each_output_partition_and_remaps_identity_outputs()
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
            .with_split_scan(true)
            .build()
            .await?;
        let mut shared_partitions = source.clone();
        shared_partitions.split_scan = false;
        let shared_partitions = shared_partitions
            .repartitioned(4, 0, None)?
            .expect("repartitioning supported");
        assert_eq!(shared_partitions.output_partitioning().partition_count(), 1);

        // Output partition `i` reads piece `i` of the split, in order.
        let source = source
            .repartitioned(4, 0, None)?
            .expect("repartitioning supported");
        assert_eq!(source.output_partitioning().partition_count(), 4);
        let plan = Arc::new(DataSourceExec::new(Arc::clone(&source))) as Arc<dyn ExecutionPlan>;
        let context = Arc::new(TaskContext::default());
        let mut partitions = Vec::new();
        for partition in 0..4 {
            let batches: Vec<RecordBatch> = plan
                .execute(partition, Arc::clone(&context))?
                .try_collect()
                .await?;
            partitions.push(partition_values(&batches));
        }
        assert_eq!(partitions, [vec![0, 1], vec![100, 101], vec![], vec![]]);

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
