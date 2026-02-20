// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`VortexDataSource`] implements DataFusion's [`DataSource`] trait, deferring scan construction
//! to [`DataSource::open`] so that pushed-down filters and limits are included in the
//! [`ScanRequest`]. A single DataFusion partition is used; Vortex handles internal parallelism
//! by driving splits concurrently via [`TryStreamExt::try_flatten_unordered`].

use std::any::Any;
use std::fmt;
use std::fmt::Formatter;
use std::num::NonZero;
use std::num::NonZeroUsize;
use std::sync::Arc;

use arrow_schema::DataType;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use datafusion_common::ColumnStatistics;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use datafusion_common::Statistics;
use datafusion_common::stats::Precision as DFPrecision;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_datasource::source::DataSource;
use datafusion_execution::SendableRecordBatchStream;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_expr::Partitioning;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr_common::sort_expr::LexOrdering;
use datafusion_physical_plan::DisplayFormatType;
use datafusion_physical_plan::expressions as df_expr;
use datafusion_physical_plan::filter_pushdown::FilterPushdownPropagation;
use datafusion_physical_plan::filter_pushdown::PushedDown;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::try_join_all;
use vortex::array::VortexSessionExecute;
use vortex::array::arrow::ArrowArrayExecutor;
use vortex::dtype::DType;
use vortex::dtype::FieldPath;
use vortex::dtype::Nullability;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::and as vx_and;
use vortex::expr::get_item;
use vortex::expr::pack;
use vortex::expr::root;
use vortex::expr::stats::Precision;
use vortex::expr::transform::replace;
use vortex::io::session::RuntimeSessionExt;
use vortex::scan::api::DataSourceRef;
use vortex::scan::api::ScanRequest;
use vortex::session::VortexSession;

use crate::convert::exprs::DefaultExpressionConvertor;
use crate::convert::exprs::ExpressionConvertor;
use crate::convert::exprs::make_vortex_predicate;
use crate::convert::stats::stats_set_to_df;

/// A builder for a [`VortexDataSource`].
pub struct VortexDataSourceBuilder {
    data_source: DataSourceRef,
    session: VortexSession,

    arrow_schema: Option<SchemaRef>,
    projection: Option<Vec<usize>>,
}

impl VortexDataSourceBuilder {
    /// Manually configure an Arrow schema to use when reading from the Vortex source.
    /// If not specified, the data source will infer an Arrow schema from the Vortex DType.
    ///
    /// Note that this schema is not validated against the Vortex DType so any errors will be
    /// deferred until read time.
    pub fn with_arrow_schema(mut self, arrow_schema: SchemaRef) -> Self {
        self.arrow_schema = Some(arrow_schema);
        self
    }

    /// Configure an initial projection using top-level field indices.
    pub fn with_projection(mut self, indices: Vec<usize>) -> Self {
        self.projection = Some(indices);
        self
    }

    /// Configure an initial projection using top-level field indices.
    pub fn with_some_projection(mut self, indices: Option<Vec<usize>>) -> Self {
        self.projection = indices;
        self
    }

    /// Build the [`VortexDataSource`].
    ///
    /// FIXME(ngates): Note that due to the DataFusion API, this function eagerly resolves
    ///   statistics for all projected columns. That said.. we only need to do this for aggregation
    ///   reductions. Any stats used for pruning are handled internally. We could possibly look
    ///   at the plan ourselves and decide whether there is any need for the stats?
    pub async fn build(self) -> VortexResult<VortexDataSource> {
        // The projection expression
        let mut projection = root();

        // Resolve the Arrow schema
        let mut arrow_schema = match self.arrow_schema {
            Some(schema) => schema,
            None => {
                let data_type = self.data_source.dtype().to_arrow_dtype()?;
                let DataType::Struct(fields) = data_type else {
                    vortex_bail!("Expected a struct-like DataType, found {}", data_type);
                };
                Arc::new(Schema::new(fields))
            }
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
                .map(|path| self.data_source.field_statistics(&path)),
        )
        .await?
        .iter()
        .zip(fields.fields())
        .map(|(stats, dtype)| stats_set_to_df(stats, &dtype))
        .collect::<VortexResult<Vec<_>>>()?;

        Ok(VortexDataSource {
            data_source: self.data_source,
            session: self.session,
            initial_schema: arrow_schema.clone(),
            initial_projection: projection.clone(),
            initial_statistics: statistics.clone(),
            final_projection: projection,
            final_schema: arrow_schema,
            final_statistics: statistics,
            filter: None,
            limit: None,
            num_partitions: std::thread::available_parallelism()
                .unwrap_or(unsafe { NonZero::new_unchecked(1) }),
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

/// A DataFusion [`DataSource`] that defers Vortex scan construction to [`open`](DataSource::open).
///
/// Holds a [`DataSourceRef`] rather than pre-collected splits, so that filters and limits pushed
/// down by DataFusion's optimizer are included in the [`ScanRequest`]. A single DataFusion
/// partition is exposed; Vortex drives splits concurrently via
/// [`TryStreamExt::try_flatten_unordered`].
#[derive(Clone)]
pub struct VortexDataSource {
    /// The Vortex data source.
    data_source: DataSourceRef,
    /// Vortex session handle.
    session: VortexSession,

    /// The initial Arrow schema of the scan.
    initial_schema: SchemaRef,
    initial_projection: Expression,
    #[allow(dead_code)]
    initial_statistics: Vec<ColumnStatistics>,

    /// The projection expression pushed down by [`DataSource::try_swapping_with_projection`]
    /// This projection has already been evaluated against the initial_projection.
    final_projection: Expression,
    final_schema: SchemaRef,
    final_statistics: Vec<ColumnStatistics>,

    /// An optional filter expression.
    /// Populated by [`DataSource::try_pushdown_filters`] when DataFusion pushes filters down.
    filter: Option<Expression>,
    /// An optional row limit populated by [`DataSource::with_fetch`].
    limit: Option<usize>,

    /// The requested partition count from DataFusion, populated by [`DataSource::repartitioned`].
    /// We use this as a hint for how many splits to execute concurrently in `open()`, but we
    /// always declare to DataFusion that we only have a single partition so that we can
    /// internally manage concurrency and fix the problem of partition skew.
    num_partitions: NonZeroUsize,
}

impl fmt::Debug for VortexDataSource {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("VortexScanSource")
            .field("schema", &self.final_schema)
            .field("projection", &format!("{}", &self.final_projection))
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
            projection: Some(self.final_projection.clone()),
            filter: self.filter.clone(),
            limit: self.limit.map(|l| u64::try_from(l).unwrap_or(u64::MAX)),
            ..Default::default()
        };

        let data_source = self.data_source.clone();
        let output_schema = self.final_schema.clone();
        let session = self.session.clone();
        let num_partitions = self.num_partitions;

        // Defer the async DataSource::scan() call to the first poll of the stream.
        let stream = futures::stream::once(async move {
            let scan = data_source
                .scan(scan_request)
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

            // Each split.execute() returns a lazy stream whose early polls do preparation
            // work (expression resolution, layout traversal, first I/O spawns). We use
            // try_flatten_unordered to poll multiple split streams concurrently so that
            // the next split is already warm when the current one finishes.
            let scan_streams = scan.splits().map(|split_result| {
                let split = split_result?;
                split.execute()
            });

            let handle = session.handle();
            let stream = scan_streams
                .try_flatten_unordered(Some(num_partitions.get() * 2))
                .map(move |result| {
                    let session = session.clone();
                    let schema = output_schema.clone();
                    handle.spawn_cpu(move || {
                        let mut ctx = session.create_execution_ctx();
                        result.and_then(|chunk| chunk.execute_record_batch(&schema, &mut ctx))
                    })
                })
                .buffered(num_partitions.get())
                .map(|result| result.map_err(|e| DataFusionError::External(Box::new(e))))
                .boxed();

            Ok::<_, DataFusionError>(stream)
        })
        .try_flatten();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.final_schema.clone(),
            stream,
        )))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> fmt::Result {
        write!(f, "VortexScanSource: projection={}", self.final_projection)?;
        if let Some(ref filter) = self.filter {
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
        _output_ordering: Option<LexOrdering>,
    ) -> DFResult<Option<Arc<dyn DataSource>>> {
        // Vortex handles parallelism internally — always use a single partition.
        let mut this = self.clone();
        this.num_partitions = NonZero::new(target_partitions)
            .ok_or_else(|| DataFusionError::Internal("non-zero partitions".to_string()))?;
        Ok(Some(Arc::new(this)))
    }

    fn output_partitioning(&self) -> Partitioning {
        Partitioning::UnknownPartitioning(1)
    }

    fn eq_properties(&self) -> EquivalenceProperties {
        EquivalenceProperties::new(self.final_schema.clone())
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> DFResult<Statistics> {
        // FIXME(ngates): this should be adjusted based on filters. See DuckDB for heuristics,
        //  and in the future, store the selectivity stats in the session.
        let num_rows = estimate_to_df_precision(&self.data_source.row_count_estimate());

        // FIXME(ngates): byte size should be adjusted for the initial projection...
        let total_byte_size = estimate_to_df_precision(&self.data_source.byte_size_estimate());

        // Column statistics must match the output schema (final_schema), which may differ
        // from the initial schema after try_swapping_with_projection adds computed columns.
        let column_statistics = self.final_statistics.clone();

        Ok(Statistics {
            num_rows,
            total_byte_size,
            column_statistics,
        })
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

        // Check if all expressions can be pushed down. If any cannot, bail out entirely
        // since DataSource::try_swapping_with_projection replaces the ProjectionExec,
        // requiring the output schema to match the projection output exactly.
        for proj_expr in projection {
            if !convertor.can_be_pushed_down(&proj_expr.expr, input_schema)
                || has_decimal_binary(&proj_expr.expr, input_schema)
            {
                return Ok(None);
            }
        }

        tracing::debug!("Swapping DataFusion projection {:?}", projection);

        // Convert all projection expressions to Vortex.
        let mut scan_columns: Vec<(String, Expression)> = Vec::new();
        let mut scan_fields: Vec<arrow_schema::Field> = Vec::new();

        for proj_expr in projection {
            // We convert the expression, and then swap out the root node
            // for the initial_projection.
            let vx_expr = convertor.convert(proj_expr.expr.as_ref())?;
            let vx_expr = replace(vx_expr, &root(), self.initial_projection.clone());

            let dt = proj_expr.expr.data_type(input_schema)?;
            let nullable = proj_expr.expr.nullable(input_schema)?;

            scan_fields.push(arrow_schema::Field::new(&proj_expr.alias, dt, nullable));
            scan_columns.push((proj_expr.alias.clone(), vx_expr));
        }

        let scan_projection = pack(scan_columns, Nullability::NonNullable);
        let scan_output_schema = Arc::new(Schema::new(scan_fields));

        // TODO(ngates): we need a way to evaluate an expression over a stats set.
        let scan_statistics =
            vec![ColumnStatistics::new_unknown(); scan_output_schema.fields().len()];

        let mut this = self.clone();
        this.final_projection = scan_projection;
        this.final_schema = scan_output_schema;
        this.final_statistics = scan_statistics;

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
                if convertor.can_be_pushed_down(expr, input_schema)
                    && !has_decimal_binary(expr, input_schema)
                {
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
                PushedDown::Yes => Some(expr.clone()),
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
}

/// Check if an expression tree contains decimal binary arithmetic that Vortex cannot handle.
///
/// DataFusion assumes different decimal types can be coerced, but Vortex expects exact type
/// matches for binary operations. We avoid pushing these down.
fn has_decimal_binary(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
    use datafusion_common::tree_node::TreeNode;

    let mut found = false;
    expr.apply(|node| {
        if let Some(binary) = node.as_any().downcast_ref::<df_expr::BinaryExpr>()
            && binary.op().is_numerical_operators()
            && let (Ok(l), Ok(r)) = (
                binary.left().data_type(schema),
                binary.right().data_type(schema),
            )
            && is_decimal(&l)
            && is_decimal(&r)
        {
            found = true;
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .map_err(|_| vortex_err!("Impossible traversal error"))
    .vortex_expect("Impossible traversal error");
    found
}

fn is_decimal(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Decimal32(_, _)
            | DataType::Decimal64(_, _)
            | DataType::Decimal128(_, _)
            | DataType::Decimal256(_, _)
    )
}

/// Convert a Vortex [`Option<Precision>`] to a DataFusion [`Precision`](DFPrecision).
fn estimate_to_df_precision(est: &Option<Precision<u64>>) -> DFPrecision<usize> {
    match est {
        Some(Precision::Exact(v)) => DFPrecision::Exact(usize::try_from(*v).unwrap_or(usize::MAX)),
        Some(Precision::Inexact(v)) => {
            DFPrecision::Inexact(usize::try_from(*v).unwrap_or(usize::MAX))
        }
        None => DFPrecision::Absent,
    }
}
