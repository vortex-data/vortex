// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::max;
use std::fmt::Formatter;
use std::fmt::{self};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;

use custom_labels::CURRENT_LABELSET;
use futures::FutureExt;
use futures::Stream;
use futures::StreamExt;
use futures::future::BoxFuture;
use itertools::Itertools;
use num_traits::AsPrimitive;
use parking_lot::Mutex;
use static_assertions::assert_impl_all;
use tracing::debug;
use vortex::aggregate_fn::DynAccumulator;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::VortexSessionExecute as _;
use vortex::array::arrays::ScalarFn;
use vortex::array::arrays::Struct;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::optimizer::ArrayOptimizer;
use vortex::dtype::DType;
use vortex::dtype::PType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::expr::Expression;
use vortex::expr::stats::Precision;
use vortex::file::v2::FileStatsLayoutReader;
use vortex::io::kanal_ext::KanalExt as _;
use vortex::io::runtime::BlockingRuntime as _;
use vortex::io::runtime::current::ThreadSafeIterator;
use vortex::layout::scan::multi::MultiLayoutChild;
use vortex::layout::scan::multi::MultiLayoutDataSource;
use vortex::metrics::tracing::get_global_labels;
use vortex::scalar::Scalar;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::scalar_fn::fns::pack::Pack;
use vortex::scan::DataSource;
use vortex::scan::ScanRequest;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::parallelism::get_available_parallelism;

use crate::RUNTIME;
use crate::SESSION;
use crate::column_statistics::ColumnStatistics;
use crate::column_statistics::ColumnStatisticsAggregate;
use crate::convert::PushedAggregate;
use crate::convert::try_from_bound_expression;
use crate::convert::try_from_projection_aggregate;
use crate::convert::try_from_projection_expression;
use crate::cpp::DUCKDB_TYPE;
use crate::duckdb::AggregateExpression;
use crate::duckdb::AggregatePushdownInputRef;
use crate::duckdb::BindInputRef;
use crate::duckdb::BindResultRef;
use crate::duckdb::DataChunkRef;
use crate::duckdb::DuckdbStringMapRef;
use crate::duckdb::ExpressionRef;
use crate::duckdb::LogicalTypeRef;
use crate::duckdb::TableInitInput;
use crate::duckdb::Value;
use crate::exporter::ArrayExporter;
use crate::exporter::ConversionCache;
use crate::multi_file::bind_multi_file_scan;
use crate::projection::DuckdbField;
use crate::projection::Filter;
use crate::projection::Projection;
use crate::projection::extract_schema_from_dtype;

// Aggregate projection index for count(*). See cpp/aggregate_fn_pushdown.cpp
pub const COUNT_STAR_PROJ_IDX: u64 = u64::MAX;

pub struct TableFunctionBind {
    data_source: Arc<MultiLayoutDataSource>,
    filter_exprs: Vec<Expression>,
    column_fields: Vec<DuckdbField>,
    // There exists at least one non-optional table filter or at least one
    // complex filter is pushed down.
    has_non_optional_filter: AtomicBool,
    // Non-empty iff this scan is aggregate
    aggregates: Vec<ColumnAggregate>,
}
assert_impl_all!(TableFunctionBind: Send, Clone);

impl Clone for TableFunctionBind {
    fn clone(&self) -> Self {
        Self {
            data_source: Arc::clone(&self.data_source),
            // filter_exprs are consumed once in `init_global`.
            filter_exprs: vec![],
            column_fields: self.column_fields.clone(),
            has_non_optional_filter: AtomicBool::new(
                self.has_non_optional_filter.load(Ordering::Relaxed),
            ),
            aggregates: self.aggregates.clone(),
        }
    }
}

impl fmt::Debug for TableFunctionBind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("DataSourceBindData")
            .field("column_fields", &self.column_fields)
            .field(
                "filter_exprs",
                &self
                    .filter_exprs
                    .iter()
                    .map(|e| e.to_string())
                    .collect::<Vec<String>>(),
            )
            .finish()
    }
}

impl<'a> TableInitInput<'a> {
    pub fn bind_data(&self) -> &TableFunctionBind {
        unsafe { &*self.input.bind_data.cast::<TableFunctionBind>() }
    }
}

type ScanItem = VortexResult<(ArrayRef, Arc<ConversionCache>)>;
type DataSourceIterator = ThreadSafeIterator<ScanItem>;

pub struct TableFunctionGlobal {
    iterator: DataSourceIterator,
    batch_id: AtomicU64,
    bytes_total: Arc<AtomicU64>,
    bytes_read: AtomicU64,
    file_index_column_pos: Option<usize>,
    file_row_number_column_pos: Option<usize>,

    // Following 4 fields are used only in aggregate scans.
    /// ArrayRef's scanned but not aggregated in "partials".
    /// 0 means all arrays have been aggregated but output is not written.
    /// u64::MAX means arrays have been aggregated and we've written output row
    pending: Arc<AtomicU64>,
    aggregates: Vec<ColumnAggregate>,
    // Accumulated partials
    partials: Mutex<Vec<Box<dyn DynAccumulator>>>,
    row_count: AtomicU64,
}
assert_impl_all!(TableFunctionGlobal: Send, Sync);

/// Per-thread scan state
pub struct TableFunctionLocal {
    iterator: DataSourceIterator,
    exporter: Option<ArrayExporter>,
    partition_index: u64,
    file_index: usize,
    // Aggregate scan accumulated partials. Empty for non-aggregate scan
    partials: Vec<Box<dyn DynAccumulator>>,
}

pub struct PartitionData {
    pub partition_index: u64,
    pub file_index_column_pos: Option<usize>,
    pub file_index: usize,
}

#[derive(Clone)]
pub(crate) enum ColumnAggregate {
    Real {
        projection_id: u64,
        aggregate: PushedAggregate,
    },
    CountStar,
}

#[derive(Debug)]
pub enum Cardinality {
    /// Unknown number of rows
    Unknown,
    /// The exact number of rows.
    Exact(u64),
    /// An estimate of the number of rows.
    Estimate(u64),
}

// Called for every new query. For example, if there is a VIEW over *.vortex,
// and after a query another file is added matching the glob, for second query
// bind() will be called again.
pub fn bind(input: &BindInputRef, result: &mut BindResultRef) -> VortexResult<TableFunctionBind> {
    let data_source = bind_multi_file_scan(input)?;
    let column_fields = extract_schema_from_dtype(data_source.dtype())?;
    for fields in &column_fields {
        result.add_result_column(&fields.name, &fields.logical_type);
    }
    Ok(TableFunctionBind {
        data_source: Arc::new(data_source),
        filter_exprs: vec![],
        column_fields,
        has_non_optional_filter: AtomicBool::new(false),
        aggregates: vec![],
    })
}

pub fn init_global(init_input: &TableInitInput) -> VortexResult<TableFunctionGlobal> {
    debug!(input=?init_input, "table function global input");

    let bind_data = init_input.bind_data();
    let column_ids = init_input.column_ids();
    let projection_ids = init_input.projection_ids();

    let Projection {
        projection,
        file_index_column_pos,
        file_row_number_column_pos,
    } = if bind_data.aggregates.is_empty() {
        Projection::new(projection_ids, column_ids, &bind_data.column_fields)
    } else {
        Projection::new_aggregate(&bind_data.aggregates, &bind_data.column_fields)
    };

    let Filter {
        filter,
        row_selection,
        row_range,
        file_selection,
        file_range,
        has_non_optional_filter,
    } = Filter::new(
        init_input.table_filter_set(),
        column_ids,
        &bind_data.column_fields,
        &bind_data.filter_exprs,
        bind_data.data_source.dtype(),
    )?;

    if has_non_optional_filter {
        init_input
            .bind_data()
            .has_non_optional_filter
            .store(true, Ordering::Relaxed);
    }

    debug!(
        %projection,
        filter = filter
            .as_ref()
            .map_or_else(|| "true".to_string(), |f| f.to_string()),
        ?row_selection,
        ?row_range,
        ?file_selection,
        ?file_range,
        "table function scan input"
    );

    let request = ScanRequest {
        projection,
        filter,
        ordered: file_row_number_column_pos.is_some(),
        selection: row_selection,
        row_range,
        partition_selection: file_selection,
        partition_range: file_range,
        limit: None,
    };

    let scan = RUNTIME.block_on(bind_data.data_source.scan(request))?;

    let num_workers = get_available_parallelism().unwrap_or(1);

    // We create an async bounded channel so that all thread-local workers can pull the next
    // available array chunk regardless of which partition it came from.
    let (tx, rx) = kanal::bounded_async(num_workers * 2);

    let pending = Arc::new(AtomicU64::new(0));
    let pending_producer = Arc::clone(&pending);

    // We drive one partition per worker thread. Each partition is driven as a spawned task
    // that pushes array chunks into the shared channel as they are produced. This spawning
    // allows all worker threads to drive the polling of all partitions, and then return the
    // first available array chunk.
    let stream = scan
        .partitions()
        .map(move |partition| {
            let tx = tx.clone();
            let pending = Arc::clone(&pending_producer);
            RUNTIME.handle().spawn(async move {
                let partition = match partition {
                    Ok(partition) => partition,
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                };

                let cache = Arc::new(ConversionCache {
                    file_index: partition.index(),
                    ..Default::default()
                });

                let mut stream = match partition.execute() {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                };
                while let Some(item) = stream.next().await {
                    pending.fetch_add(1, Ordering::Relaxed);
                    if tx
                        .send(item.map(|a| (a, Arc::clone(&cache))))
                        .await
                        .is_err()
                    {
                        // Exit early if the receiver has been dropped, which happens when the
                        // scan is complete or if an error has occurred in another partition.
                        return;
                    }
                }
            })
        })
        .buffer_unordered(num_workers);

    let iterator = RUNTIME.block_on_stream_thread_safe(|_handle| scan_driver_stream(stream, rx));

    let aggregates = bind_data.aggregates.clone();
    let partials = build_partials(
        &aggregates,
        &bind_data.column_fields,
        bind_data.data_source.dtype(),
    )?;

    Ok(TableFunctionGlobal {
        iterator,
        batch_id: AtomicU64::new(0),
        bytes_total: Arc::new(AtomicU64::new(0)),
        bytes_read: AtomicU64::new(0),
        file_index_column_pos,
        file_row_number_column_pos,
        pending,
        aggregates,
        partials: Mutex::new(partials),
        row_count: AtomicU64::new(0),
    })
}

fn scan_driver_stream<S>(stream: S, rx: kanal::AsyncReceiver<ScanItem>) -> ScanDriverStream
where
    S: Stream<Item = ()> + Send + 'static,
{
    ScanDriverStream {
        driver: Some(stream.collect::<()>().boxed()),
        rx: rx.into_stream().boxed(),
    }
}

struct ScanDriverStream {
    driver: Option<BoxFuture<'static, ()>>,
    rx: futures::stream::BoxStream<'static, ScanItem>,
}

impl Stream for ScanDriverStream {
    type Item = ScanItem;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if let Some(driver) = this.driver.as_mut()
            && driver.as_mut().poll(cx).is_ready()
        {
            this.driver = None;
        }

        match this.rx.as_mut().poll_next(cx) {
            Poll::Ready(None) if this.driver.is_some() => Poll::Pending,
            poll => poll,
        }
    }
}

/// Dtype over which we accumulate
fn aggregate_input_dtype(field: &DuckdbField, scope: &DType) -> VortexResult<DType> {
    match &field.projection_expr {
        None => Ok(field.dtype.clone()),
        Some(expr) => expr.return_dtype(scope),
    }
}

fn build_partials(
    aggregates: &[ColumnAggregate],
    fields: &[DuckdbField],
    scope: &DType,
) -> VortexResult<Vec<Box<dyn DynAccumulator>>> {
    aggregates
        .iter()
        .filter_map(|spec| match spec {
            ColumnAggregate::Real {
                projection_id,
                aggregate,
            } => {
                let projection_id: usize = projection_id.as_();
                Some(
                    aggregate_input_dtype(&fields[projection_id], scope)
                        .and_then(|dtype| aggregate.build(dtype)),
                )
            }
            ColumnAggregate::CountStar => None,
        })
        .collect()
}

pub fn init_local(
    bind_data: &TableFunctionBind,
    global: &TableFunctionGlobal,
) -> TableFunctionLocal {
    unsafe {
        use custom_labels::sys;

        if sys::current().is_null() {
            let ls = sys::new(0);
            sys::replace(ls);
        };
    }

    let global_labels = get_global_labels();

    for (key, value) in global_labels {
        CURRENT_LABELSET.set(key, value);
    }

    let partials = build_partials(
        &global.aggregates,
        &bind_data.column_fields,
        bind_data.data_source.dtype(),
    )
    // if aggregate initialization produced an error, it would error in
    // init_global, see "partials" initialization there
    .vortex_expect("local state aggregate initialization failed");

    TableFunctionLocal {
        iterator: global.iterator.clone(),
        exporter: None,
        partition_index: 0,
        file_index: 0,
        partials,
    }
}

fn convert_result(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<StructArray> {
    let array_result = array.optimize_recursive(ctx.session())?;
    Ok(if let Some(array) = array_result.as_opt::<Struct>() {
        array.into_owned()
    } else if let Some(array) = array_result.as_opt::<ScalarFn>()
        && let Some(pack_options) = array.scalar_fn().as_opt::<Pack>()
    {
        StructArray::new(
            pack_options.names.clone(),
            array.children(),
            array.len(),
            pack_options.nullability.into(),
        )
    } else {
        array_result.execute::<Canonical>(ctx)?.into_struct()
    })
}

fn aggregate_output_value(scalar: Scalar, expected: &LogicalTypeRef) -> VortexResult<Value> {
    let primitive_i128 = |scalar: &Scalar| -> VortexResult<Option<i128>> {
        let primitive = scalar.as_primitive();
        Ok(match primitive.ptype() {
            PType::I64 => primitive.typed_value::<i64>().map(i128::from),
            PType::U64 => primitive.typed_value::<u64>().map(i128::from),
            other => vortex_bail!("expected {expected:?} output type, got {other}"),
        })
    };
    match expected.as_type_id() {
        DUCKDB_TYPE::DUCKDB_TYPE_HUGEINT => Ok(match primitive_i128(&scalar)? {
            Some(value) => Value::new_hugeint(value),
            None => Value::null(expected),
        }),
        DUCKDB_TYPE::DUCKDB_TYPE_BIGINT if scalar.dtype().is_unsigned_int() => {
            Ok(match primitive_i128(&scalar)? {
                Some(value) => Value::from(i64::try_from(value)?),
                None => Value::null(expected),
            })
        }
        _ => Value::try_from(scalar),
    }
}

fn scan_aggregate(
    local_state: &mut TableFunctionLocal,
    global_state: &TableFunctionGlobal,
    chunk: &mut DataChunkRef,
) -> VortexResult<()> {
    let aggregates_len = global_state.aggregates.len();
    // seen[k] = output column for requested column k.
    // If min(x), max(x), avg(y) are requested, seen = { 0: 0, 1: 1}
    let mut seen: HashMap<u64, usize> = HashMap::with_capacity(aggregates_len);
    // positions[k] = column id for accumulator k
    // If min(x), max(x), avg(y) are requested, positions = [0, 0, 1]
    let mut positions: Vec<usize> = Vec::with_capacity(aggregates_len);

    for aggregate in &global_state.aggregates {
        let ColumnAggregate::Real { projection_id, .. } = aggregate else {
            continue;
        };
        let len = seen.len();
        let pos = seen.entry_ref(projection_id).or_insert(len);
        positions.push(*pos);
    }
    let has_count_star = local_state.partials.len() < aggregates_len;

    let mut ctx = SESSION.create_execution_ctx();
    loop {
        let Some(result) = local_state.iterator.next() else {
            // 0 means we're the last thread, u64::MAX means output is written.
            // is_err() means CAS didn't succeed
            if global_state
                .pending
                .compare_exchange(0, u64::MAX, Ordering::AcqRel, Ordering::Relaxed)
                .is_err()
            {
                return Ok(());
            }

            let mut accumulators = global_state.partials.lock();
            let row_count = global_state.row_count.load(Ordering::Acquire) as i64;
            let mut accum_iter = accumulators.iter_mut();
            for (idx, aggregate) in global_state.aggregates.iter().enumerate() {
                let value = match aggregate {
                    ColumnAggregate::Real { .. } => {
                        let accum = accum_iter.next().vortex_expect("partial for real agg");
                        let expected = chunk.get_vector_mut(idx).logical_type();
                        aggregate_output_value(accum.finish()?, &expected)?
                    }
                    ColumnAggregate::CountStar => Value::from(row_count),
                };
                chunk.get_vector_mut(idx).reference_value(&value);
            }
            chunk.set_len(1);
            return Ok(());
        };
        let array = convert_result(result?.0, &mut ctx)?;

        for (i, partial) in positions.iter().zip(local_state.partials.iter_mut()) {
            partial.accumulate(array.unmasked_field(*i), &mut ctx)?;
        }

        {
            let mut partials = global_state.partials.lock();
            for (global, local) in partials.iter_mut().zip(&mut local_state.partials) {
                global.combine_partials(local.flush()?)?;
            }
        }

        if has_count_star {
            global_state
                .row_count
                .fetch_add(array.len() as u64, Ordering::Relaxed);
        }
        global_state.pending.fetch_sub(1, Ordering::Release);
    }
}

pub fn scan(
    local_state: &mut TableFunctionLocal,
    global_state: &TableFunctionGlobal,
    chunk: &mut DataChunkRef,
) -> VortexResult<()> {
    if !local_state.partials.is_empty() {
        return scan_aggregate(local_state, global_state, chunk);
    }

    loop {
        if local_state.exporter.is_none() {
            let mut ctx = SESSION.create_execution_ctx();
            let Some(result) = local_state.iterator.next() else {
                return Ok(());
            };
            let (array_result, conversion_cache) = result?;
            local_state.file_index = conversion_cache.file_index;
            let array_result = convert_result(array_result, &mut ctx)?;

            local_state.exporter = Some(ArrayExporter::try_new(
                &array_result,
                &conversion_cache,
                ctx,
            )?);
            // Relaxed since there is no intra-instruction ordering required.
            local_state.partition_index = global_state.batch_id.fetch_add(1, Ordering::Relaxed);
        }

        let exporter = local_state
            .exporter
            .as_mut()
            .vortex_expect("error: exporter missing");
        let has_more_data = exporter.export(
            chunk,
            global_state.file_index_column_pos,
            global_state.file_row_number_column_pos,
        )?;

        global_state
            .bytes_read
            .fetch_add(chunk.len(), Ordering::Relaxed);

        if !has_more_data {
            // This exporter is fully consumed.
            local_state.exporter = None;
            local_state.partition_index = 0;
        } else {
            break;
        }
    }

    assert!(!chunk.is_empty());

    if let Some(pos) = global_state.file_index_column_pos {
        chunk
            .get_vector_mut(pos)
            .reference_value(&Value::from(local_state.file_index as u64));
    }

    Ok(())
}

/// Scan progress as a percentage (0.0–100.0).
pub fn table_scan_progress(global_state: &TableFunctionGlobal) -> f64 {
    progress(&global_state.bytes_read, &global_state.bytes_total)
}

/// Table filter pushdown is used for two tasks in duckdb:
///
/// 1. Prune files based on filename or hive partitioning, see Parquet
///    filter pushdown. We don't use this because we do own file-level pruning
///    in FileStatsLayoutReader, and we don't support hive partitioning yet.
/// 2. Avoid reading unused file data. Filter expressions are pushed to Vortex,
///    converted to Vortex expressions and used during the scan.
///    Duckdb pushes a subset of expressions i.e. equality operators, and also
///    expressions which return true in pushdown_expression.
pub fn pushdown_complex_filter(
    bind_data: &mut TableFunctionBind,
    expr: &ExpressionRef,
) -> VortexResult<bool> {
    debug!(%expr, "pushing down expression");

    let Some(expr) = try_from_bound_expression(expr, &bind_data.column_fields)? else {
        debug!(%expr, "failed to push down expression");
        return Ok(false);
    };

    // Duckdb calls pushdown_complex_filter during planning phase.
    // If all filters are pushed down, duckdb enables a LEFT_DELIM_JOIN ->
    // COMPARISON_JOIN (HASH_JOIN) optimization:
    // duckdb/src/optimizer/deliminator.cpp: Deliminator::HasSelection,
    // Deliminator::Optimize.
    //
    // This leads to a massive regression on tpch sf=10 q17 and other
    // benchmarks.
    //
    // This bug is reported to Duckdb
    // https://github.com/duckdb/duckdb/issues/22669
    //
    // As a hack, report equality filters as not pushed.
    // We can also report only the first filter as not pushed, but this
    // has a negative performance impact.
    let report_pushed = !expr
        .as_opt::<Binary>()
        .map(|op| *op == Operator::Eq)
        .unwrap_or(false);

    // Only table filters may be optional, any complex filter is
    // non-optional by definition.
    bind_data
        .has_non_optional_filter
        .store(true, Ordering::Relaxed);

    debug!(%expr, report_pushed, "pushed down expression");
    bind_data.filter_exprs.push(expr);
    Ok(report_pushed)
}

pub fn pushdown_projection_expression(
    bind_data: &mut TableFunctionBind,
    expr: &ExpressionRef,
    projection_id: usize,
) -> VortexResult<bool> {
    let field = &bind_data.column_fields[projection_id];
    debug!(%expr, %projection_id, col_name=field.name, "pushing down projection expression");
    match try_from_projection_expression(expr, field)? {
        None => {
            debug!(%expr, "failed to push down expression");
            Ok(false)
        }
        Some(vx_expr) => {
            debug!(%expr, "pushed down expression");
            bind_data.column_fields[projection_id].projection_expr = Some(vx_expr);
            Ok(true)
        }
    }
}

/// Turn a scan into an aggregate scan. Input is N aggregations, possibly over
/// same columns. If we return true, optimized pass expands output to N columns,
/// e.g. min(x), max(x) turns into min(x0), max(x1), 2 columns in output.
pub fn pushdown_projection_aggregates(
    bind_data: &mut TableFunctionBind,
    input: &AggregatePushdownInputRef,
) -> VortexResult<bool> {
    let len = input.len();
    let mut aggregates = Vec::with_capacity(len);
    let mut has_non_count_star = false;

    debug!(%len, "pushing down projection aggregates");
    for i in 0..len {
        let AggregateExpression {
            expr,
            projection_id,
        } = input.get(i);
        if projection_id == COUNT_STAR_PROJ_IDX {
            aggregates.push(ColumnAggregate::CountStar);
            continue;
        }
        let Some(aggregate) = try_from_projection_aggregate(expr)? else {
            debug!(%expr, %i, "failed to push down projection aggregate");
            return Ok(false);
        };

        let projection_id_usize: usize = projection_id.as_();
        let field = &bind_data.column_fields[projection_id_usize];
        let Ok(dtype) = aggregate_input_dtype(field, bind_data.data_source.dtype()) else {
            return Ok(false);
        };

        // duckdb's min() returns nan only when every value is nan.
        // vortex's min() either ignores or counts nans.
        // See slt/duckdb/nan_aggregates.slt.
        if aggregate == PushedAggregate::Min && dtype.is_float() {
            return Ok(false);
        }

        let mean_or_sum = matches!(aggregate, PushedAggregate::Sum | PushedAggregate::Mean);

        // duckdb's sum() and avg() on i64/u64 accumulate in i128/u128, vortex
        // sum()/avg() work on i64/u64 max and overflow to null
        if mean_or_sum
            && dtype.is_primitive()
            && matches!(dtype.as_ptype(), PType::I64 | PType::U64)
        {
            return Ok(false);
        }

        // duckdb decimal sum() overflows to error, vortex sum overlows to NULL
        // duckdb's avg() on decimals returns a double, vortex's mean stays
        // decimal.
        if mean_or_sum && matches!(dtype, DType::Decimal(..)) {
            return Ok(false);
        }

        // vortex doesn't have list comparison
        if matches!(aggregate, PushedAggregate::Min | PushedAggregate::Max)
            && matches!(dtype, DType::List(..) | DType::FixedSizeList(..))
        {
            return Ok(false);
        }

        if aggregate.build(dtype).is_err() {
            return Ok(false);
        }

        debug!(%expr, %projection_id, %i, "pushed down projection aggregate");
        aggregates.push(ColumnAggregate::Real {
            projection_id,
            aggregate,
        });
        has_non_count_star = true;
    }
    // DuckDB computes just count(*) faster than us
    if !has_non_count_star {
        return Ok(false);
    }
    bind_data.aggregates = aggregates;
    Ok(true)
}

/// Get column-wise statistics. Available only if we're reading a single file.
pub fn statistics(bind_data: &TableFunctionBind, column_index: usize) -> Option<ColumnStatistics> {
    // Aggregate output columns hold data we don't have in statistics
    if !bind_data.aggregates.is_empty() {
        return None;
    }
    let children = bind_data.data_source.children();
    // Otherwise we'd have to open all files eagerly which is a performance
    // regression. Duckdb's Parquet reader only gets metadata for multiple
    // files with a UNION BY NAME and we don't support it (yet)
    // See duckdb/common/multi_file/multi_file_function.hpp#L691
    if children.len() != 1 {
        return None;
    }
    let MultiLayoutChild::Opened { reader, .. } = &children[0] else {
        return None;
    };
    let stats_sets = match reader.as_any().downcast_ref::<FileStatsLayoutReader>() {
        Some(inner) => inner.file_stats().stats_sets(),
        None => return None,
    };
    // Columns with pushed projection expression output expression results,
    // and not column values
    if bind_data.column_fields[column_index]
        .projection_expr
        .is_some()
    {
        return None;
    }
    let dtype = bind_data.column_fields[column_index].dtype.clone();
    let stats_aggregate = ColumnStatisticsAggregate::new(&stats_sets[column_index]);
    Some(ColumnStatistics::from(&stats_aggregate, dtype))
}

/// Duckdb requires post-filter cardinality estimates, otherwise join planner
/// may flip join sides which is a huge regression for some queries i.e. 1000x
/// for tpcds 85.
///
/// See duckdb/src/optimizer/join_order/relation_statistics_helper.cpp
///
/// As we don't report distinct values (same as Parquet), the only heuristic
/// duckdb uses is a 0.2 filter if there is any non-optional filter. We mimic it
/// here.
const DEFAULT_SELECTIVITY: f64 = 0.2;
pub fn cardinality(bind_data: &TableFunctionBind) -> Cardinality {
    // If we're doing an aggregate scan, we don't change output cardinality to
    // 1 as we want duckdb to do our aggregation in parallel. That may look
    // counterintuitive in the plan, though.
    let has_non_optional_filter = bind_data.has_non_optional_filter.load(Ordering::Relaxed);
    match bind_data.data_source.row_count() {
        Precision::Exact(v) => {
            if !has_non_optional_filter {
                return Cardinality::Exact(v);
            }
            let post_cardinality = v as f64 * DEFAULT_SELECTIVITY;
            let post_cardinality: u64 = post_cardinality.as_();
            Cardinality::Estimate(max(1, post_cardinality))
        }
        Precision::Inexact(v) => {
            if !has_non_optional_filter {
                return Cardinality::Estimate(v);
            }
            let post_cardinality = v as f64 * DEFAULT_SELECTIVITY;
            let post_cardinality: u64 = post_cardinality.as_();
            Cardinality::Estimate(max(1, post_cardinality))
        }
        Precision::Absent => Cardinality::Unknown,
    }
}

/// Duckdb requests this function after exporting the chunk. We answer with
/// partition_index we have exported as well as information about constant
/// columns in this partition. As data is partitioned by array exporters, in
/// each partition ~ exported array file_index is constant.
pub fn get_partition_data(
    global_init_data: &TableFunctionGlobal,
    local_init_data: &mut TableFunctionLocal,
) -> PartitionData {
    PartitionData {
        partition_index: local_init_data.partition_index,
        file_index_column_pos: global_init_data.file_index_column_pos,
        file_index: local_init_data.file_index,
    }
}

pub fn to_string(bind_data: &TableFunctionBind, map: &mut DuckdbStringMapRef) {
    map.push("Function", "Vortex Scan");
    if !bind_data.filter_exprs.is_empty() {
        let mut filters = bind_data.filter_exprs.iter().map(|f| format!("{f}"));
        map.push("Filters", &filters.join("\n"));
    }

    if !bind_data.aggregates.is_empty() {
        let aggregations = bind_data
            .aggregates
            .iter()
            .map(|agg| match agg {
                ColumnAggregate::Real {
                    projection_id,
                    aggregate,
                } => {
                    let projection_id: usize = projection_id.as_();
                    format!(
                        "{aggregate}({})",
                        bind_data.column_fields[projection_id].name
                    )
                }
                ColumnAggregate::CountStar => "count(*)".to_string(),
            })
            .join("\n");
        if !aggregations.is_empty() {
            map.push("Aggregations", &aggregations);
        }
        return;
    }

    let projections = bind_data
        .column_fields
        .iter()
        .filter_map(|field| {
            field
                .projection_expr
                .as_ref()
                .map(|expr| format!("{}: {expr}", field.name))
        })
        .join("\n");
    if !projections.is_empty() {
        map.push("SELECT projections", &projections);
    }
}

fn progress(bytes_read: &AtomicU64, bytes_total: &AtomicU64) -> f64 {
    let read = bytes_read.load(Ordering::Relaxed);
    let mut total = bytes_total.load(Ordering::Relaxed);
    total += (total == 0) as u64;
    read as f64 / total as f64 * 100.
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering::Relaxed;
    use std::task::Poll;

    use crate::RUNTIME;
    use crate::table_function::progress;
    use crate::table_function::scan_driver_stream;

    #[test]
    fn test_table_scan_progress() {
        let bytes_total = AtomicU64::new(100);
        let bytes_read = AtomicU64::new(0);

        assert_eq!(progress(&bytes_read, &bytes_total), 0.0);

        bytes_read.fetch_add(100, Relaxed);
        assert_eq!(progress(&bytes_read, &bytes_total), 100.);

        bytes_total.fetch_add(100, Relaxed);
        assert!((progress(&bytes_read, &bytes_total) - 50.).abs() < f64::EPSILON);
    }

    #[test]
    fn scan_driver_panic_propagates_through_iterator() {
        let (tx, rx) = kanal::bounded_async(1);
        let _tx = tx;
        let stream = futures::stream::poll_fn(|_| -> Poll<Option<()>> {
            panic!("duckdb scan driver panic");
        });

        let mut iter =
            RUNTIME.block_on_stream_thread_safe(|_handle| scan_driver_stream(stream, rx));
        let panic = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| iter.next())) {
            Ok(_) => panic!("driver panic must propagate through iterator"),
            Err(panic) => panic,
        };
        let message = panic
            .downcast_ref::<&'static str>()
            .copied()
            .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("<unknown panic>");
        assert!(message.contains("duckdb scan driver panic"));
    }
}
