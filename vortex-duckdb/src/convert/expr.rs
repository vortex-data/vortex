// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result;
use std::sync::Arc;

use tracing::debug;
use vortex::aggregate_fn::Accumulator;
use vortex::aggregate_fn::DynAccumulator;
use vortex::aggregate_fn::EmptyOptions as AggregateEmptyOptions;
use vortex::aggregate_fn::NumericalAggregateOpts;
use vortex::aggregate_fn::combined::PairOptions;
use vortex::aggregate_fn::fns::count::Count;
use vortex::aggregate_fn::fns::first::First;
use vortex::aggregate_fn::fns::max::Max;
use vortex::aggregate_fn::fns::mean::Mean;
use vortex::aggregate_fn::fns::min::Min;
use vortex::aggregate_fn::fns::sum_v2::SumV2;
use vortex::arrow::ArrowSessionExt;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexError;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::expr::Expression;
use vortex::expr::and_collect;
use vortex::expr::byte_length;
use vortex::expr::cast;
use vortex::expr::col;
use vortex::expr::get_item;
use vortex::expr::is_not_null;
use vortex::expr::is_null;
use vortex::expr::list_contains;
use vortex::expr::list_length;
use vortex::expr::lit;
use vortex::expr::not;
use vortex::expr::or_collect;
use vortex::expr::root;
use vortex::extension::datetime::Date;
use vortex::extension::datetime::TimeUnit;
use vortex::layout::layouts::row_idx::row_idx;
use vortex::scalar::Scalar;
use vortex::scalar::ScalarValue;
use vortex::scalar_fn::EmptyOptions as ScalarEmptyOptions;
use vortex::scalar_fn::ScalarFnVTableExt;
use vortex::scalar_fn::fns::between::Between;
use vortex::scalar_fn::fns::between::BetweenOptions;
use vortex::scalar_fn::fns::between::StrictComparison;
use vortex::scalar_fn::fns::binary::Binary;
use vortex::scalar_fn::fns::like::Like;
use vortex::scalar_fn::fns::like::LikeOptions;
use vortex::scalar_fn::fns::literal::Literal;
use vortex::scalar_fn::fns::operators::Operator;
use vortex_spatial::extension::LineString;
use vortex_spatial::extension::MultiLineString;
use vortex_spatial::extension::MultiPoint;
use vortex_spatial::extension::MultiPolygon;
use vortex_spatial::extension::Point;
use vortex_spatial::extension::Polygon;
use vortex_spatial::extension::WellKnownBinary;
use vortex_spatial::extension::native_geometry_scalar_from_wkb;
use vortex_spatial::scalar_fn::contains::SpatialContains;
use vortex_spatial::scalar_fn::distance::SpatialDistance;
use vortex_spatial::scalar_fn::intersects::SpatialIntersects;

use crate::SESSION;
use crate::convert::dtype::FromLogicalType;
use crate::cpp::DUCKDB_TYPE;
use crate::cpp::DUCKDB_VX_EXPR_TYPE;
use crate::duckdb;
use crate::duckdb::BoundFunction;
use crate::duckdb::BoundOperator;
use crate::duckdb::ExpressionClass;
use crate::duckdb::ExpressionClass::BoundBetween;
use crate::duckdb::ExpressionClass::BoundCast;
use crate::duckdb::ExpressionClass::BoundColumnRef;
use crate::duckdb::ExpressionClass::BoundComparison;
use crate::duckdb::ExpressionClass::BoundConjunction;
use crate::duckdb::ExpressionClass::BoundConstant;
use crate::duckdb::ExpressionClass::BoundRef;
use crate::duckdb::ExtractedValue;
use crate::projection::DuckdbField;

fn from_bound_str(value: &duckdb::ExpressionRef) -> VortexResult<String> {
    match value.as_class().vortex_expect("unknown class") {
        BoundConstant(constant) => Ok(constant.value.as_string().as_str().to_owned()),
        _ => vortex_bail!("Expected string expression, got {:?}", value.as_class_id()),
    }
}

/// Whether the expression's return type is a `LIST` or fixed-size `ARRAY`.
fn returns_a_list(expr: &duckdb::ExpressionRef) -> bool {
    matches!(
        expr.return_type().as_type_id(),
        DUCKDB_TYPE::DUCKDB_TYPE_LIST | DUCKDB_TYPE::DUCKDB_TYPE_ARRAY
    )
}

/// Wrap `expr` in `list_length`. Since vortex `list_length` returns u64 but duckdb equivalents
/// return i64, we must cast as well.
fn build_list_length(expr: Expression, nullability: Nullability) -> Expression {
    cast(list_length(expr), DType::Primitive(PType::I64, nullability))
}

/// Read an `f64` from a constant expression (the `ST_DWithin` radius); `None` for non-constants.
fn from_bound_f64(value: &duckdb::ExpressionRef) -> VortexResult<Option<f64>> {
    match value.as_class().vortex_expect("unknown class") {
        BoundConstant(constant) => Ok(Some(f64::try_from(&Scalar::try_from(constant.value)?)?)),
        _ => Ok(None),
    }
}

/// Context threaded through expression conversion.
#[derive(Clone, Copy)]
struct ConvertCtx<'a> {
    /// Substituted for `BoundRef` references when converting scan-scoped table filters.
    col_sub: Option<&'a Expression>,
    /// The scan's fields, when known.
    fields: Option<&'a [DuckdbField]>,
}

/// Whether `name` is a non-nullable native geometry column of the scan. The pushed spatial kernels
/// reject nullable operands and cannot evaluate `vortex.st.wkb` columns, which also surface to
/// DuckDB as `GEOMETRY`.
fn is_native_spatial_column(fields: Option<&[DuckdbField]>, name: &str) -> bool {
    fields
        .into_iter()
        .flatten()
        .filter(|field| field.name == name && !field.dtype.is_nullable())
        .any(|field| match field.dtype.as_extension_opt() {
            Some(ext) => {
                ext.is::<Point>()
                    || ext.is::<LineString>()
                    || ext.is::<MultiPoint>()
                    || ext.is::<Polygon>()
                    || ext.is::<MultiLineString>()
                    || ext.is::<MultiPolygon>()
            }
            None => false,
        })
}

/// Lower a spatial operand: a `GEOMETRY` literal arrives as WKB, decoded once to its native type so the
/// pushed `SpatialDistance` stays native; a column must be native geometry. `None` skips the push.
fn spatial_operand(
    value: &duckdb::ExpressionRef,
    ctx: ConvertCtx<'_>,
) -> VortexResult<Option<Expression>> {
    match value.as_class() {
        Some(BoundConstant(constant)) => {
            let scalar = Scalar::try_from(constant.value)?;
            let DType::Extension(ext_dtype) = scalar.dtype() else {
                return Ok(None);
            };
            if !ext_dtype.is::<WellKnownBinary>() {
                return Ok(None);
            }
            let storage = scalar.as_extension().to_storage_scalar();
            let Some(buf) = storage.as_binary_opt().and_then(|b| b.value()) else {
                return Ok(None);
            };
            Ok(native_geometry_scalar_from_wkb(buf.as_slice(), &SESSION.arrow())?.map(lit))
        }
        Some(BoundColumnRef(col_ref))
            if is_native_spatial_column(ctx.fields, col_ref.name.as_ref()) =>
        {
            try_from_expression_inner(value, ctx)
        }
        _ => Ok(None),
    }
}

/// Lower all geometry operands of a spatial function. Returns `None`, skipping the push, when any
/// operand is neither a constant geometry nor a native geometry column.
fn spatial_operands(
    children: &[&duckdb::ExpressionRef],
    ctx: ConvertCtx<'_>,
) -> VortexResult<Option<Vec<Expression>>> {
    children
        .iter()
        .map(|child| spatial_operand(child, ctx))
        .collect()
}

/// Lower spatial UDFs to native Vortex spatial operations so the work runs in the scan. `None` otherwise.
fn try_from_spatial_function(
    name: &str,
    func: &BoundFunction,
    ctx: ConvertCtx<'_>,
) -> VortexResult<Option<Expression>> {
    let children: Vec<_> = func.children().collect();
    let expr = match name.to_ascii_lowercase().as_str() {
        // DuckDB's spatial extension folds the radius of `ST_DWithin` into bind data; the override
        // (cpp/spatial_overrides.cpp) keeps it visible here as `children[2]`.
        "st_dwithin" => {
            if children.len() != 3 {
                return Ok(None);
            }
            let Some(operands) = spatial_operands(&children[..2], ctx)? else {
                return Ok(None);
            };
            // A non-constant radius is left for DuckDB to evaluate.
            let Some(distance) = from_bound_f64(children[2])? else {
                return Ok(None);
            };
            let spatial_distance = SpatialDistance.new_expr(ScalarEmptyOptions, operands);
            Binary.new_expr(Operator::Lte, [spatial_distance, lit(distance)])
        }
        "st_distance" => {
            if children.len() != 2 {
                return Ok(None);
            }
            let Some(operands) = spatial_operands(&children, ctx)? else {
                return Ok(None);
            };
            SpatialDistance.new_expr(ScalarEmptyOptions, operands)
        }
        "st_intersects" => {
            if children.len() != 2 {
                return Ok(None);
            }
            let Some(operands) = spatial_operands(&children, ctx)? else {
                return Ok(None);
            };
            SpatialIntersects.new_expr(ScalarEmptyOptions, operands)
        }
        containment @ ("st_contains" | "st_within") => {
            if children.len() != 2 {
                return Ok(None);
            }
            let Some(mut operands) = spatial_operands(&children, ctx)? else {
                return Ok(None);
            };
            // `st_within(a, b)` is `st_contains(b, a)`; both lower to the contains kernel.
            if containment == "st_within" {
                operands.swap(0, 1);
            }
            SpatialContains.new_expr(ScalarEmptyOptions, operands)
        }
        _ => return Ok(None),
    };

    Ok(Some(expr))
}

fn try_from_bound_function(
    func: &BoundFunction,
    ctx: ConvertCtx<'_>,
) -> VortexResult<Option<Expression>> {
    let expr = match func.scalar_function.name() {
        "strlen" => {
            let children: Vec<_> = func.children().collect();
            vortex_ensure!(children.len() == 1);
            let Some(col) = try_from_expression_inner(children[0], ctx)? else {
                return Ok(None);
            };
            let col = byte_length(col);
            // byte_length returns u64, strlen expects i64.
            // At this point we don't know column's dtype so we ultimately
            // set it to be nullable.
            let dtype = DType::Primitive(PType::I64, Nullability::Nullable);
            cast(col, dtype)
        }
        "struct_extract" => {
            let children: Vec<_> = func.children().collect();
            vortex_ensure!(children.len() == 2);
            let Some(child) = try_from_expression_inner(children[0], ctx)? else {
                return Ok(None);
            };
            let field = from_bound_str(children[1])?;
            get_item(field, child)
        }
        like @ ("~~" | "!~~") => {
            let children: Vec<_> = func.children().collect();
            vortex_ensure!(children.len() == 2);
            let Some(string) = try_from_expression_inner(children[0], ctx)? else {
                return Ok(None);
            };
            let Some(target) = try_from_expression_inner(children[1], ctx)? else {
                return Ok(None);
            };
            let opts = LikeOptions {
                negated: like == "!~~",
                case_insensitive: false,
            };
            Like.new_expr(opts, [string, target])
        }
        matchers @ ("contains" | "prefix" | "suffix") => {
            let children: Vec<_> = func.children().collect();
            vortex_ensure!(children.len() == 2);
            let Some(value) = try_from_expression_inner(children[0], ctx)? else {
                return Ok(None);
            };
            let pattern = from_bound_str(children[1])?;
            let pattern = match matchers {
                "contains" => format!("%{pattern}%"),
                "prefix" => format!("{pattern}%"),
                "suffix" => format!("%{pattern}"),
                _ => unreachable!(),
            };
            Like.new_expr(LikeOptions::default(), [value, lit(pattern)])
        }
        "array_length" => {
            let children = func.children().collect::<Vec<_>>();
            // Only accept array_length(expr) rather than array_length(expr, dim).
            if children.len() != 1 {
                return Ok(None);
            }
            let Some(col) = try_from_expression_inner(children[0], ctx)? else {
                return Ok(None);
            };

            // We don't know the column's nullability here
            build_list_length(col, Nullability::Nullable)
        }
        // len/length semantics depend on the return type of underlying expr.
        "len" | "length" => {
            let children: Vec<_> = func.children().collect();
            vortex_ensure!(children.len() == 1);
            let child = children[0];

            if returns_a_list(child) {
                let Some(col) = try_from_expression_inner(child, ctx)? else {
                    return Ok(None);
                };

                // We don't know the column's nullability here
                let list_len_expr = build_list_length(col, Nullability::Nullable);
                return Ok(Some(list_len_expr));
            } else {
                return Ok(None);
            }
        }
        // Spatial UDFs are handled here; non-spatial names return `None` inside.
        name => return try_from_spatial_function(name, func, ctx),
    };

    Ok(Some(expr))
}

pub fn try_from_bound_expression(
    value: &duckdb::ExpressionRef,
    fields: &[DuckdbField],
) -> VortexResult<Option<Expression>> {
    try_from_expression_inner(
        value,
        ConvertCtx {
            col_sub: None,
            fields: Some(fields),
        },
    )
}

pub(super) fn try_from_bound_expression_with_col_sub(
    value: &duckdb::ExpressionRef,
    col_sub: &Expression,
) -> VortexResult<Option<Expression>> {
    // No fields: scan-time table filters never carry spatial functions, because
    // `can_push_expression` refuses them.
    try_from_expression_inner(
        value,
        ConvertCtx {
            col_sub: Some(col_sub),
            fields: None,
        },
    )
}

fn is_supported_length_alias(func: &BoundFunction) -> bool {
    let children: Vec<_> = func.children().collect();
    children.len() == 1 && returns_a_list(children[0])
}

/// The `DATE` operand of a `CAST(<date> AS TIMESTAMP)`, or `None` for anything else.
///
/// `TIMESTAMP WITH TIME ZONE` is excluded on purpose: that cast depends on the session
/// timezone, so it has no fixed `DATE` equivalent.
fn date_to_timestamp_cast_child(expr: &duckdb::ExpressionRef) -> Option<&duckdb::ExpressionRef> {
    let BoundCast(cast) = expr.as_class()? else {
        return None;
    };
    // TRY_CAST yields NULL where CAST errors, which the fold below would not reproduce.
    if cast.is_try || cast.child.return_type().as_type_id() != DUCKDB_TYPE::DUCKDB_TYPE_DATE {
        return None;
    }
    matches!(
        expr.return_type().as_type_id(),
        DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP
            | DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_S
            | DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_MS
            | DUCKDB_TYPE::DUCKDB_TYPE_TIMESTAMP_NS
    )
    .then_some(cast.child)
}

/// A timezone-naive timestamp constant, as `(ticks since epoch, ticks per day)`.
fn timestamp_constant(expr: &duckdb::ExpressionRef) -> Option<(i64, i64)> {
    let BoundConstant(constant) = expr.as_class()? else {
        return None;
    };
    Some(match constant.value.extract() {
        ExtractedValue::TimestampS(ticks) => (ticks, 86_400),
        ExtractedValue::TimestampMs(ticks) => (ticks, 86_400_000),
        ExtractedValue::Timestamp(ticks) => (ticks, 86_400_000_000),
        ExtractedValue::TimestampNs(ticks) => (ticks, 86_400_000_000_000),
        _ => return None,
    })
}

/// The operator that holds once the operands are exchanged: `a < b` becomes `b > a`.
fn reverse_operator(op: Operator) -> Option<Operator> {
    Some(match op {
        Operator::Eq => Operator::Eq,
        Operator::NotEq => Operator::NotEq,
        Operator::Lt => Operator::Gt,
        Operator::Lte => Operator::Gte,
        Operator::Gt => Operator::Lt,
        Operator::Gte => Operator::Lte,
        _ => return None,
    })
}

/// Rewrites `<date> op <timestamp>` into an equivalent `<date> op' <day>`.
///
/// The cast from `DATE` to `TIMESTAMP` is strictly increasing, so a timestamp bound always has
/// an exact bound on whole days. When the timestamp lands exactly on midnight the operator is
/// unchanged; otherwise it falls strictly inside `days`, which every date compares against the
/// same way it compares against the end of that day.
fn fold_timestamp_bound(op: Operator, ticks: i64, ticks_per_day: i64) -> Option<(Operator, i32)> {
    // Euclidean division so that pre-epoch timestamps still floor towards the earlier day.
    let days = i32::try_from(ticks.div_euclid(ticks_per_day)).ok()?;
    let time_of_day = ticks.rem_euclid(ticks_per_day);

    if time_of_day == 0 {
        return Some((op, days));
    }
    Some(match op {
        Operator::Lt | Operator::Lte => (Operator::Lte, days),
        Operator::Gt | Operator::Gte => (Operator::Gt, days),
        // `= t` is unsatisfiable and `!= t` a tautology, but only for non-null rows. Leave
        // both to DuckDB rather than folding away the null cases.
        _ => return None,
    })
}

/// Recognizes a comparison that DuckDB widened to `TIMESTAMP` only to line a `DATE` column up
/// with a timestamp literal, and rewrites it back to a `DATE` comparison.
///
/// `o_orderdate < date '1993-07-01' + interval '3' month` binds as
/// `CAST(o_orderdate AS TIMESTAMP) < TIMESTAMP '1993-10-01 00:00:00'`, because `date + interval`
/// returns a `TIMESTAMP`. The cast hides the column, so the bound cannot become a table filter
/// and stays in a DuckDB `FILTER` above the scan (TPC-H q4, q15 and q20 all lose their upper
/// date bound this way, while the matching lower bound pushes normally).
///
/// Folding the cast into the literal rather than evaluating it keeps the predicate in
/// `column <op> literal` form, which is what lets it prune with statistics and fuse into a
/// range filter. Evaluating the cast per batch would recover the rows but neither of those.
///
/// Returns the `DATE` operand together with the rewritten operator and literal.
fn date_timestamp_comparison<'a>(
    compare: &duckdb::BoundComparison<'a>,
) -> Option<(&'a duckdb::ExpressionRef, Operator, Scalar)> {
    let op: Operator = compare.op.try_into().ok()?;

    let (date, op, (ticks, ticks_per_day)) =
        if let Some(date) = date_to_timestamp_cast_child(compare.left) {
            (date, op, timestamp_constant(compare.right)?)
        } else {
            let date = date_to_timestamp_cast_child(compare.right)?;
            (
                date,
                reverse_operator(op)?,
                timestamp_constant(compare.left)?,
            )
        };

    let (op, days) = fold_timestamp_bound(op, ticks, ticks_per_day)?;
    let literal = Scalar::extension::<Date>(
        TimeUnit::Days,
        Scalar::try_new(
            DType::Primitive(PType::I32, Nullability::Nullable),
            Some(ScalarValue::from(days)),
        )
        .ok()?,
    );
    Some((date, op, literal))
}

/// Whether `value` is a comparison that [`date_timestamp_comparison`] rewrites into a `DATE`
/// bound.
///
/// `pushdown_complex_filter` needs this to decide what to report back to DuckDB; see the
/// Deliminator note there.
pub fn is_folded_date_comparison(value: &duckdb::ExpressionRef) -> bool {
    matches!(
        value.as_class(),
        Some(BoundComparison(compare)) if date_timestamp_comparison(&compare).is_some()
    )
}

// We limit casting to Primitive types, because some conversions yield an error
// like vortex.date[days](i32) -> vortex.timestamp[µs](i64?). However, when we
// push down the cast, we don't have access to column's dtype, so we need to
// be overly restrictive.
// TODO(myrrc) change after https://github.com/vortex-data/vortex/issues/8570
// is resolved
//
// We also don't push floats and doubles because Vortex truncates to zero and
// Duckdb rounds the result
fn can_push_cast(cast: &duckdb::BoundCast<'_>, target: &duckdb::LogicalTypeRef) -> bool {
    !cast.is_try && target.is_primitive_integer() && cast.child.return_type().is_primitive_integer()
}

// Called before pushdown_complex_filter or a table filter expression call.
// As we support complex filter pushdown, Duckdb pushes expressions to Vortex.
// However, it doesn't know what type of expressions we can handle. Here we list
// all expressions that are quaranteed to be converted to Vortex expressions.
//
// If we return true here, and expression is in the list for
// pushdown_complex_filter, we must handle it, or query engine will break.
//
// Example: we don't support substr() expression so we tell Duckdb we can't
// push it.
// Example: we support CAST but not TRY_CAST.
// Example: optional filters may fail to parse on our side (we return
// Ok(None)), so we don't allow pushing these.
pub fn can_push_expression(value: &duckdb::ExpressionRef) -> bool {
    let Some(class) = value.as_class() else {
        return false;
    };
    match class {
        BoundColumnRef(_) => true,
        BoundConstant(_) => true,
        BoundCast(cast) => {
            can_push_cast(&cast, value.return_type()) && can_push_expression(cast.child)
        }
        BoundRef => true,
        BoundComparison(comp) => {
            // Handled by `date_timestamp_comparison`, which pushes this shape by rewriting it
            // rather than by pushing the cast itself.
            if let Some((date, ..)) = date_timestamp_comparison(&comp) {
                return can_push_expression(date);
            }
            can_push_expression(comp.left) && can_push_expression(comp.right)
        }
        BoundBetween(between) => {
            can_push_expression(between.input)
                && can_push_expression(between.lower)
                && can_push_expression(between.upper)
        }
        BoundConjunction(conj) => conj.children().all(can_push_expression),
        ExpressionClass::BoundFunction(func) => {
            let name = func.scalar_function.name();
            name == "struct_extract"
                || name == "contains"
                || name == "prefix"
                || name == "suffix"
                || name == "~~"
                || name == "!~~"
                || name == "strlen"
                || name == "array_length"
                || (matches!(name, "len" | "length") && is_supported_length_alias(&func))
            // Spatial functions are absent on purpose: they push only via
            // `pushdown_complex_filter`, which has the scan's fields to verify the geometry
            // columns are native.
        }
        ExpressionClass::BoundOperator(op) => {
            if !matches!(
                op.op,
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_NOT
                    | DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NULL
                    | DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NOT_NULL
                    | DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_IN
                    | DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_NOT_IN
            ) {
                return false;
            }
            op.children().all(can_push_expression)
        }
        ExpressionClass::BoundAggregate(_) => false,
    }
}

/// Applies `list_length` expression to a duckdb field
fn list_length_on_field(field: &DuckdbField) -> Expression {
    let col = get_item(field.name.as_str(), root());

    build_list_length(col, field.dtype.nullability())
}

pub fn try_from_projection_expression(
    value: &duckdb::ExpressionRef,
    field: &DuckdbField,
) -> VortexResult<Option<Expression>> {
    let Some(class) = value.as_class() else {
        return Ok(None);
    };
    Ok(match class {
        ExpressionClass::BoundFunction(func) => {
            match func.scalar_function.name() {
                "strlen" => {
                    let col = byte_length(get_item(field.name.as_str(), root()));
                    // byte_length returns u64, strlen expects i64
                    let dtype = DType::Primitive(PType::I64, field.dtype.nullability());
                    let col = cast(col, dtype);
                    Some(col)
                }
                "array_length" => {
                    // Only accept array_length(expr) rather than array_length(expr, dim).
                    (func.children().count() == 1).then(|| list_length_on_field(field))
                }
                // len/length have different semantics depending on field dtype.
                "len" | "length" => {
                    matches!(field.dtype, DType::List(..) | DType::FixedSizeList(..))
                        .then(|| list_length_on_field(field))
                }
                _ => None,
            }
        }
        BoundCast(c) => {
            let target = value.return_type();
            if !can_push_cast(&c, target) {
                None
            } else {
                let dtype = DType::from_logical_type(target, field.dtype.nullability())?;
                let col = get_item(field.name.as_str(), root());
                Some(cast(col, dtype))
            }
        }
        _ => None,
    })
}

/// Aggregations we have pushed down in Vortex
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PushedAggregate {
    Min,
    Max,
    Sum,
    Mean,
    // Also used for ANY_VALUE() which is allowed by definition
    First,
    // Valid values in column
    Count,
}

impl Display for PushedAggregate {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            PushedAggregate::Min => f.write_str("min"),
            PushedAggregate::Max => f.write_str("max"),
            PushedAggregate::Sum => f.write_str("sum"),
            PushedAggregate::Mean => f.write_str("mean"),
            PushedAggregate::First => f.write_str("first"),
            PushedAggregate::Count => f.write_str("count"),
        }
    }
}

impl PushedAggregate {
    pub fn build(self, dtype: DType) -> VortexResult<Box<dyn DynAccumulator>> {
        let opts = if dtype.is_float() {
            // duckdb treats nan as a real value, vortex defaults skip nans
            NumericalAggregateOpts::include_nans()
        } else {
            NumericalAggregateOpts::default()
        };
        Ok(match self {
            Self::Min => Box::new(Accumulator::try_new(Min, opts, dtype)?),
            Self::Max => Box::new(Accumulator::try_new(Max, opts, dtype)?),
            Self::Sum => Box::new(Accumulator::try_new(SumV2, opts, dtype)?),
            Self::Mean => Box::new(Accumulator::try_new(
                Mean::combined(),
                PairOptions(opts, opts),
                dtype,
            )?),
            Self::First => Box::new(Accumulator::try_new(First, AggregateEmptyOptions, dtype)?),
            Self::Count => Box::new(Accumulator::try_new(Count, opts, dtype)?),
        })
    }
}

/// Check if this is an aggregate function we can handle in Vortex
pub fn try_from_projection_aggregate(
    expr: &duckdb::ExpressionRef,
) -> VortexResult<Option<PushedAggregate>> {
    let Some(expr) = expr.as_class() else {
        return Ok(None);
    };
    let ExpressionClass::BoundAggregate(agg) = expr else {
        return Ok(None);
    };
    Ok(Some(match agg.aggregate_function.name() {
        "min" => PushedAggregate::Min,
        "max" => PushedAggregate::Max,
        "sum" | "sum_no_overflow" => PushedAggregate::Sum,
        "avg" | "mean" => PushedAggregate::Mean,
        "first" | "any_value" => PushedAggregate::First,
        "count" => PushedAggregate::Count,
        _ => return Ok(None),
    }))
}

// If you want to add support for other expressions, also change
// can_push_expression
/// Converts a comparison, first trying the `DATE`/`TIMESTAMP` rewrite that
/// [`date_timestamp_comparison`] recognizes.
fn try_from_comparison(
    compare: &duckdb::BoundComparison<'_>,
    ctx: ConvertCtx<'_>,
) -> VortexResult<Option<Expression>> {
    if let Some((date, operator, literal)) = date_timestamp_comparison(compare) {
        let Some(date) = try_from_expression_inner(date, ctx)? else {
            return Ok(None);
        };
        return Ok(Some(Binary.new_expr(operator, [date, lit(literal)])));
    }

    let operator: Operator = compare.op.try_into()?;
    let Some(left) = try_from_expression_inner(compare.left, ctx)? else {
        return Ok(None);
    };
    let Some(right) = try_from_expression_inner(compare.right, ctx)? else {
        return Ok(None);
    };
    Ok(Some(Binary.new_expr(operator, [left, right])))
}

fn try_from_expression_inner(
    value: &duckdb::ExpressionRef,
    ctx: ConvertCtx<'_>,
) -> VortexResult<Option<Expression>> {
    let Some(class) = value.as_class() else {
        debug!(
            class_id = ?value.as_class_id(),
            "unknown expression class id"
        );
        return Ok(None);
    };
    Ok(Some(match class {
        BoundRef => {
            let Some(col) = ctx.col_sub else {
                vortex_bail!("BoundRef requested but no column supplied");
            };
            col.clone()
        }
        BoundColumnRef(col_ref) => {
            let name = col_ref.name.as_ref();
            if name == "file_row_number" {
                return Ok(Some(row_idx()));
            }

            // Duckdb generates some columns (e.g. hive partitions) after we
            // load file data, so filters on these columns can't be evaluated
            if ctx
                .fields
                .is_some_and(|fields| !fields.iter().any(|field| field.name == name))
            {
                return Ok(None);
            }
            col(name)
        }
        BoundConstant(const_) => lit(Scalar::try_from(const_.value)?),
        BoundComparison(compare) => return try_from_comparison(&compare, ctx),
        BoundBetween(between) => {
            let Some(array) = try_from_expression_inner(between.input, ctx)? else {
                return Ok(None);
            };
            let Some(lower) = try_from_expression_inner(between.lower, ctx)? else {
                return Ok(None);
            };
            let Some(upper) = try_from_expression_inner(between.upper, ctx)? else {
                return Ok(None);
            };
            Between.new_expr(
                BetweenOptions {
                    lower_strict: if between.lower_inclusive {
                        StrictComparison::NonStrict
                    } else {
                        StrictComparison::Strict
                    },
                    upper_strict: if between.upper_inclusive {
                        StrictComparison::NonStrict
                    } else {
                        StrictComparison::Strict
                    },
                },
                [array, lower, upper],
            )
        }
        ExpressionClass::BoundOperator(operator) => match operator.op {
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_NOT
            | DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NULL
            | DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NOT_NULL => {
                let children: Vec<_> = operator.children().collect();
                vortex_ensure!(children.len() == 1);
                let Some(child) = try_from_expression_inner(children[0], ctx)? else {
                    return Ok(None);
                };
                match operator.op {
                    DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_NOT => not(child),
                    DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NULL => is_null(child),
                    DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_OPERATOR_IS_NOT_NULL => {
                        is_not_null(child)
                    }
                    _ => unreachable!(),
                }
            }
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_IN => {
                return try_from_compare_in(operator, ctx, false);
            }
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_NOT_IN => {
                return try_from_compare_in(operator, ctx, true);
            }
            _ => {
                debug!(op=?operator.op, "cannot push down operator");
                return Ok(None);
            }
        },
        ExpressionClass::BoundFunction(func) => {
            return try_from_bound_function(&func, ctx);
        }
        BoundCast(cast_inner) => {
            let target = value.return_type();
            if !can_push_cast(&cast_inner, target) {
                return Ok(None);
            }
            let Some(child) = try_from_expression_inner(cast_inner.child, ctx)? else {
                return Ok(None);
            };
            // We don't know the column's nullability here
            let dtype = DType::from_logical_type(target, Nullability::Nullable)?;
            cast(child, dtype)
        }
        BoundConjunction(conj) => {
            let Some(children) = conj
                .children()
                .map(|c| try_from_expression_inner(c, ctx))
                .collect::<VortexResult<Option<Vec<_>>>>()?
            else {
                return Ok(None);
            };
            match conj.op {
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_CONJUNCTION_AND => {
                    and_collect(children).vortex_expect("cannot be empty")
                }
                DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_CONJUNCTION_OR => {
                    or_collect(children).vortex_expect("cannot be empty")
                }
                _ => vortex_bail!("unexpected operator {:?} in bound conjunction", conj.op),
            }
        }
        ExpressionClass::BoundAggregate(_) => return Ok(None),
    }))
}

fn try_from_compare_in(
    operator: BoundOperator,
    ctx: ConvertCtx<'_>,
    not_in: bool,
) -> VortexResult<Option<Expression>> {
    // First child is element, rest form the list.
    let children: Vec<_> = operator.children().collect();
    assert!(children.len() >= 2);
    let Some(element) = try_from_expression_inner(children[0], ctx)? else {
        return Ok(None);
    };

    let Some(list_elements) = children
        .iter()
        .skip(1)
        .map(|c| {
            let Some(value) = try_from_expression_inner(c, ctx)? else {
                return Ok(None);
            };
            Ok(Some(
                value
                    .as_opt::<Literal>()
                    .ok_or_else(|| vortex_err!("cannot have a non literal in a in_list"))?
                    .clone(),
            ))
        })
        .collect::<VortexResult<Option<Vec<_>>>>()?
    else {
        return Ok(None);
    };
    let list = Scalar::list(
        Arc::new(list_elements[0].dtype().clone()),
        list_elements,
        Nullability::Nullable,
    );

    let expr = list_contains(lit(list), element);
    Ok(Some(if not_in { not(expr) } else { expr }))
}

impl TryFrom<DUCKDB_VX_EXPR_TYPE> for Operator {
    type Error = VortexError;

    fn try_from(value: DUCKDB_VX_EXPR_TYPE) -> VortexResult<Self> {
        Ok(match value {
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_INVALID => vortex_bail!("invalid expression"),
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_EQUAL => Operator::Eq,
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_NOTEQUAL => Operator::NotEq,
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHAN => Operator::Lt,
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHAN => Operator::Gt,
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_LESSTHANOREQUALTO => Operator::Lte,
            DUCKDB_VX_EXPR_TYPE::DUCKDB_VX_EXPR_TYPE_COMPARE_GREATERTHANOREQUALTO => Operator::Gte,
            _ => vortex_bail!("cannot convert {:?}", value),
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const US_PER_DAY: i64 = 86_400_000_000;
    /// An arbitrary day well inside the range both `DATE` and `TIMESTAMP` can hold.
    const DAY: i32 = 8674;

    /// A timestamp exactly at midnight names a whole day, so the operator carries over as-is.
    #[rstest]
    #[case(Operator::Lt)]
    #[case(Operator::Lte)]
    #[case(Operator::Gt)]
    #[case(Operator::Gte)]
    #[case(Operator::Eq)]
    #[case(Operator::NotEq)]
    fn fold_at_midnight_keeps_the_operator(#[case] op: Operator) {
        assert_eq!(
            fold_timestamp_bound(op, i64::from(DAY) * US_PER_DAY, US_PER_DAY),
            Some((op, DAY))
        );
    }

    /// A timestamp strictly inside a day sits between that date and the next, so both `<` and
    /// `<=` admit the day itself and both `>` and `>=` exclude it.
    #[rstest]
    #[case(Operator::Lt, Some(Operator::Lte))]
    #[case(Operator::Lte, Some(Operator::Lte))]
    #[case(Operator::Gt, Some(Operator::Gt))]
    #[case(Operator::Gte, Some(Operator::Gt))]
    #[case(Operator::Eq, None)]
    #[case(Operator::NotEq, None)]
    fn fold_inside_a_day_rounds_to_the_day(
        #[case] op: Operator,
        #[case] expected: Option<Operator>,
    ) {
        let noon = i64::from(DAY) * US_PER_DAY + US_PER_DAY / 2;
        assert_eq!(
            fold_timestamp_bound(op, noon, US_PER_DAY),
            expected.map(|op| (op, DAY))
        );
    }

    /// Pre-epoch timestamps floor towards the earlier day, not towards zero.
    #[rstest]
    #[case(-US_PER_DAY, Operator::Lt, -1)]
    #[case(-1, Operator::Lte, -1)]
    fn fold_before_the_epoch(
        #[case] ticks: i64,
        #[case] expected_op: Operator,
        #[case] expected_day: i32,
    ) {
        assert_eq!(
            fold_timestamp_bound(Operator::Lt, ticks, US_PER_DAY),
            Some((expected_op, expected_day))
        );
    }

    /// A day count past `DATE`'s storage has no equivalent literal, so the fold declines.
    #[test]
    fn fold_rejects_days_beyond_i32() {
        assert_eq!(fold_timestamp_bound(Operator::Lt, i64::MAX, 1), None);
    }
}
