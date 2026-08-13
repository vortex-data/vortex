// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::max;
use std::fmt::Formatter;
use std::fmt::{self};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use custom_labels::CURRENT_LABELSET;
use itertools::Itertools;
use num_traits::AsPrimitive;
use parking_lot::Mutex;
use static_assertions::assert_impl_all;
use tracing::debug;
use vortex::aggregate_fn::DynAccumulator;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::arrays::ScalarFn;
use vortex::array::arrays::Struct;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex::array::optimizer::ArrayOptimizer;
use vortex::dtype::DType;
use vortex::dtype::PType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::expr::BoundExpression;
use vortex::expr::Expression;
use vortex::metrics::tracing::get_global_labels;
use vortex::scalar::Scalar;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::operators::Operator;
use vortex::scalar_fn::fns::pack::Pack;
use vortex_utils::aliases::hash_map::HashMap;

use crate::convert::PushedAggregate;
use crate::convert::try_from_bound_expression;
use crate::convert::try_from_projection_aggregate;
use crate::convert::try_from_projection_expression;
use crate::cpp::DUCKDB_TYPE;
use crate::duckdb::AggregateExpression;
use crate::duckdb::AggregatePushdownInputRef;
use crate::duckdb::BindResultRef;
use crate::duckdb::DataChunkRef;
use crate::duckdb::DuckdbStringMapRef;
use crate::duckdb::ExpressionRef;
use crate::duckdb::LogicalType;
use crate::duckdb::LogicalTypeRef;
use crate::duckdb::TableInitInput;
use crate::duckdb::Value;
use crate::exporter::ArrayExporter;
use crate::file_reader::File;
use crate::file_reader::ScanPruning;
use crate::projection::DuckdbField;
use crate::projection::Projection;
use crate::projection::extract_schema_from_dtype;

// Aggregate projection index for count(*). See cpp/aggregate_fn_pushdown.cpp
pub const COUNT_STAR_PROJ_IDX: u64 = u64::MAX;

pub struct TableFunctionBind {
    pub(crate) dtype: DType,
    first_file_row_count: u64,
    pub(crate) filter_exprs: Vec<Expression>,
    pub(crate) column_fields: Vec<DuckdbField>,
    // There exists at least one non-optional table filter or at least one
    // complex filter is pushed down.
    pub(crate) has_non_optional_filter: AtomicBool,
    // Non-empty iff this scan is aggregate
    aggregates: Vec<ColumnAggregate>,
    aggregate_outputs: Vec<(String, LogicalType)>,
}
assert_impl_all!(TableFunctionBind: Send, Clone);

impl Clone for TableFunctionBind {
    fn clone(&self) -> Self {
        Self {
            dtype: self.dtype.clone(),
            first_file_row_count: self.first_file_row_count,
            // Cloning happens only for late materialization refetch
            filter_exprs: vec![],
            column_fields: self.column_fields.clone(),
            has_non_optional_filter: AtomicBool::new(
                self.has_non_optional_filter.load(Ordering::Relaxed),
            ),
            aggregates: self.aggregates.clone(),
            aggregate_outputs: self.aggregate_outputs.clone(),
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

pub struct TableFunctionGlobal {
    pub(crate) pruning: Option<ScanPruning>,
    pub(crate) bound_projection: BoundExpression,
    // Following fields are used only in aggregate scans.
    /// Splits that are not merged into global partials
    /// 0 means everything started is merged.
    /// u64::MAX means output row is written.
    pub(crate) pending: Arc<AtomicU64>,
    pub(crate) aggregates: Vec<ColumnAggregate>,
    pub(crate) aggregate_positions: Vec<usize>,
    // Accumulated partials
    pub(crate) partials: Mutex<Vec<Box<dyn DynAccumulator>>>,
    pub(crate) row_count: AtomicU64,
}
assert_impl_all!(TableFunctionGlobal: Send, Sync);

/// Per-thread scan state
pub struct TableFunctionLocal {
    pub(crate) exporter: Option<ArrayExporter>,
    // Aggregate scan accumulated partials. Empty for non-aggregate scan
    pub(crate) partials: Vec<Box<dyn DynAccumulator>>,
    // Current file processed is read till end
    pub(crate) exhausted: bool,
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
    /// The exact number of rows.
    Exact(u64),
    /// An estimate of the number of rows.
    Estimate(u64),
}

// Called for every new query. For example, if there is a VIEW over *.vortex,
// and after a query another file is added matching the glob, for second query
// bind() will be called again.
pub fn bind(first_file: &File) -> VortexResult<TableFunctionBind> {
    let dtype = first_file.dtype.clone();
    let column_fields = extract_schema_from_dtype(&dtype)?;
    Ok(TableFunctionBind {
        dtype,
        first_file_row_count: first_file.row_count,
        filter_exprs: vec![],
        column_fields,
        has_non_optional_filter: AtomicBool::new(false),
        aggregates: vec![],
        aggregate_outputs: vec![],
    })
}

pub fn bind_schema(bind_data: &TableFunctionBind, result: &mut BindResultRef) {
    if !bind_data.aggregate_outputs.is_empty() {
        for (name, logical_type) in &bind_data.aggregate_outputs {
            result.add_result_column(name, logical_type);
        }
        return;
    }
    for field in &bind_data.column_fields {
        result.add_result_column(&field.name, &field.logical_type);
    }
}

pub fn finalize_scan(global: &TableFunctionGlobal, chunk: &mut DataChunkRef) -> VortexResult<bool> {
    if global.aggregates.is_empty() {
        return Ok(false);
    }
    // 0 means every produced array has been accumulated, u64::MAX means output is
    // written. is_err() covers "still accumulating" and "already emitted"
    if global
        .pending
        .compare_exchange(0, u64::MAX, Ordering::AcqRel, Ordering::Relaxed)
        .is_err()
    {
        return Ok(false);
    }

    let mut accumulators = global.partials.lock();
    let row_count = global.row_count.load(Ordering::Acquire) as i64;
    let mut accum_iter = accumulators.iter_mut();
    for (idx, aggregate) in global.aggregates.iter().enumerate() {
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
    Ok(true)
}

pub fn init_global(init_input: &TableInitInput) -> VortexResult<TableFunctionGlobal> {
    let bind_data = init_input.bind_data();

    let partials = build_partials(
        &bind_data.aggregates,
        &bind_data.column_fields,
        &bind_data.dtype,
    )?;

    let pruning = ScanPruning::new(bind_data, init_input.column_ids(), init_input.input.filters)?;

    let mut seen = HashMap::with_capacity(bind_data.aggregates.len());
    let mut aggregate_positions = Vec::with_capacity(bind_data.aggregates.len());
    for aggregate in &bind_data.aggregates {
        let ColumnAggregate::Real { projection_id, .. } = aggregate else {
            continue;
        };
        let len = seen.len();
        let pos = *seen.entry(*projection_id).or_insert(len);
        aggregate_positions.push(pos);
    }

    let Projection(projection) = if bind_data.aggregates.is_empty() {
        Projection::new(init_input.column_ids(), &bind_data.column_fields)
    } else {
        Projection::new_aggregate(&bind_data.aggregates, &bind_data.column_fields)
    };
    let bound_projection = optimize_and_bind(projection, &bind_data.dtype)?;

    Ok(TableFunctionGlobal {
        pruning,
        aggregate_positions,
        bound_projection,
        pending: Arc::new(AtomicU64::new(0)),
        aggregates: bind_data.aggregates.clone(),
        partials: Mutex::new(partials),
        row_count: AtomicU64::new(0),
    })
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
        &bind_data.dtype,
    )
    // if aggregate initialization produced an error, it would error in
    // init_global, see "partials" initialization there
    .vortex_expect("local state aggregate initialization failed");

    TableFunctionLocal {
        exporter: None,
        partials,
        exhausted: false,
    }
}

pub(crate) fn optimize_and_bind(expr: Expression, dtype: &DType) -> VortexResult<BoundExpression> {
    expr.optimize_recursive(dtype)?.bind(dtype)
}

pub(crate) fn convert_result(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<StructArray> {
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
            let Ok(out_dtype) = vx_expr.return_dtype(&bind_data.dtype) else {
                return Ok(false);
            };
            let field = &mut bind_data.column_fields[projection_id];
            field.logical_type = expr.return_type().to_owned();
            field.dtype = out_dtype;
            field.projection_expr = Some(vx_expr);
            Ok(true)
        }
    }
}

fn can_push_projection_aggregate(
    aggregate: &PushedAggregate,
    bind_data: &TableFunctionBind,
    projection_id: u64,
) -> bool {
    let projection_id_usize: usize = projection_id.as_();
    let field = &bind_data.column_fields[projection_id_usize];
    let Ok(dtype) = aggregate_input_dtype(field, &bind_data.dtype) else {
        return false;
    };

    // duckdb's min() returns nan only when every value is nan.
    // vortex's min() either ignores or counts nans.
    // See slt/duckdb/nan_aggregates.slt.
    if *aggregate == PushedAggregate::Min && dtype.is_float() {
        return false;
    }

    let mean_or_sum = matches!(aggregate, PushedAggregate::Sum | PushedAggregate::Mean);

    // duckdb's sum() and avg() on i64/u64 accumulate in i128/u128, vortex
    // sum()/avg() work on i64/u64 max and overflow to null
    if mean_or_sum && dtype.is_primitive() && matches!(dtype.as_ptype(), PType::I64 | PType::U64) {
        return false;
    }

    // duckdb decimal sum() overflows to error, vortex sum overflows to NULL
    // duckdb's avg() on decimals returns a double, vortex's mean stays
    // decimal.
    if mean_or_sum && matches!(dtype, DType::Decimal(..)) {
        return false;
    }

    // vortex doesn't have list comparison
    if matches!(aggregate, PushedAggregate::Min | PushedAggregate::Max)
        && matches!(dtype, DType::List(..) | DType::FixedSizeList(..))
    {
        return false;
    }

    // UUID is backed by FixedSizeList which aggregations can't compute over
    if dtype.as_extension_opt().is_some_and(|ext| ext.is::<Uuid>())
        && !matches!(aggregate, PushedAggregate::Count)
    {
        return false;
    }

    if aggregate.build(dtype).is_err() {
        return false;
    }

    true
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
    let mut outputs = Vec::with_capacity(len);
    let mut has_non_count_star = false;

    debug!(%len, "pushing down projection aggregates");
    for i in 0..len {
        let expression = input.get(i);
        let output_type = expression.expr.return_type().to_owned();
        let Some(aggregate) = try_push_projection_aggregate(bind_data, expression, i)? else {
            return Ok(false);
        };
        let name = match &aggregate {
            ColumnAggregate::CountStar => "count_star()".to_string(),
            ColumnAggregate::Real { projection_id, .. } => {
                let id: usize = projection_id.as_();
                bind_data.column_fields[id].name.clone()
            }
        };
        outputs.push((name, output_type));
        has_non_count_star |= matches!(aggregate, ColumnAggregate::Real { .. });
        aggregates.push(aggregate);
    }
    // DuckDB computes just count(*) faster than us
    if !has_non_count_star {
        return Ok(false);
    }
    bind_data.aggregates = aggregates;
    bind_data.aggregate_outputs = outputs;
    Ok(true)
}

fn try_push_projection_aggregate(
    bind_data: &TableFunctionBind,
    aggregate: AggregateExpression<'_>,
    i: usize,
) -> VortexResult<Option<ColumnAggregate>> {
    let AggregateExpression {
        expr,
        projection_id,
    } = aggregate;
    if projection_id == COUNT_STAR_PROJ_IDX {
        return Ok(Some(ColumnAggregate::CountStar));
    }
    let Some(aggregate) = try_from_projection_aggregate(expr)? else {
        debug!(%expr, %i, "failed to push down projection aggregate");
        return Ok(None);
    };
    if !can_push_projection_aggregate(&aggregate, bind_data, projection_id) {
        debug!(%expr, %i, "failed to push down projection aggregate");
        return Ok(None);
    }
    debug!(%expr, %projection_id, %i, "pushed down projection aggregate");
    Ok(Some(ColumnAggregate::Real {
        projection_id,
        aggregate,
    }))
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
pub fn cardinality(bind_data: &TableFunctionBind, file_count: u64) -> Cardinality {
    // If we're doing an aggregate scan, we don't change output cardinality to
    // 1 as we want duckdb to do our aggregation in parallel. That may look
    // counterintuitive in the plan, though.
    let has_non_optional_filter = bind_data.has_non_optional_filter.load(Ordering::Relaxed);
    let total = bind_data
        .first_file_row_count
        .saturating_mul(max(file_count, 1));
    if !has_non_optional_filter {
        return if file_count <= 1 {
            Cardinality::Exact(total)
        } else {
            Cardinality::Estimate(total)
        };
    }
    let post_cardinality = total as f64 * DEFAULT_SELECTIVITY;
    let post_cardinality: u64 = post_cardinality.as_();
    Cardinality::Estimate(max(1, post_cardinality))
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
