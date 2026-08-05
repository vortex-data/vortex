// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Chunk pruning for spatial filters, using the per-chunk [`GeometryAabb`] axis-aligned
//! bounding box (AABB).
//!
//! One module per predicate. To prune a new one: recover the geometry column and the constant
//! (symmetric predicates can use `geometry_and_constant`), reduce the constant to its box with
//! `query_aabb`, build the proof over `aabb_stat` from the box-relation builders (`min_dist_sq`,
//! `max_dist_sq`), and register the rule in [`crate::initialize`]. No new statistic or
//! file-format change is needed.

mod distance;
mod intersects;
#[cfg(test)]
mod test_harness;

pub use distance::GeoDistancePrune;
use geo::BoundingRect;
use geo::Rect as GeoRect;
pub use intersects::GeoIntersectsPrune;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::dtype::FieldName;
use vortex_array::expr::BoundExpression as BoundExpr;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::EmptyOptions as ScalarEmptyOptions;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::case_when::CaseWhen;
use vortex_array::scalar_fn::fns::case_when::CaseWhenOptions;
use vortex_array::scalar_fn::fns::ext_storage::ExtStorage;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::scalar_fn::fns::stat::StatFn;
use vortex_array::scalar_fn::fns::stat::StatOptions;
use vortex_array::stats::rewrite::StatsRewriteCtx;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::aggregate_fn::GeometryAabb;
use crate::extension::is_native_geometry;
use crate::extension::single_geometry;

/// Splits a symmetric two-operand geo predicate into the scope-rooted geometry column and the
/// constant operand's scalar.
///
/// `None` means the rule must decline: the expression doesn't have the `f(column, constant)`
/// shape (in either operand order), or the column's dtype carries no [`GeometryAabb`] statistic.
/// An asymmetric predicate (e.g. a future contains) must recover which operand is the column
/// itself instead of calling this.
fn geometry_and_constant<'a>(
    expr: &'a BoundExpr,
    ctx: &StatsRewriteCtx<'_>,
) -> VortexResult<Option<(&'a BoundExpr, &'a Scalar)>> {
    // The predicate is symmetric, so the column (scope root) and the constant may be on either
    // side.
    let (lhs, rhs) = (expr.child(0), expr.child(1));
    let (geom, constant) = if lhs.is_root() {
        (lhs, rhs)
    } else if rhs.is_root() {
        (rhs, lhs)
    } else {
        return Ok(None);
    };

    // A `GeometryAabb` stat reference only binds for dtypes it supports; anything else (e.g. a
    // WKB column) must fall through to the scan.
    if !is_native_geometry(&ctx.return_dtype(geom)?) {
        return Ok(None);
    }

    Ok(constant.as_opt::<Literal>().map(|scalar| (geom, scalar)))
}

/// The 2D bounding box of a constant geometry of any type, or `None` for one without an extent
/// (e.g. an empty geometry).
///
/// Prove claims against this box rather than the constant itself: the constant lies inside it,
/// so whatever holds for the box holds for the geometry.
fn query_aabb(constant: &Scalar, ctx: &StatsRewriteCtx<'_>) -> VortexResult<Option<GeoRect<f64>>> {
    // A null geometry literal has no extent to prove against, so it can never prune.
    if constant.is_null() {
        return Ok(None);
    }
    // Decoding the constant into a concrete geometry runs through the compute stack, which needs
    // an execution context.
    let mut exec = ctx.session().create_execution_ctx();
    Ok(single_geometry(constant, &mut exec)?.bounding_rect())
}

/// The chunk's AABB statistic, as the storage struct with `xmin`/`ymin`/`xmax`/`ymax` fields.
///
/// A chunk written without the statistic reads as null here; every proof built on top must let
/// that null propagate to its root, where the zone map keeps the chunk.
fn aabb_stat(geom: &BoundExpr) -> BoundExpr {
    // `ext_storage` unwraps the native `geoarrow.box` stat value to its backing struct, so
    // proofs can `get_item` the coordinate fields.
    ext_storage(stat(geom.clone(), GeometryAabb.bind(EmptyOptions)))
}

fn stat(expr: BoundExpr, aggregate_fn: vortex_array::aggregate_fn::AggregateFnRef) -> BoundExpr {
    StatFn
        .try_new_bound_expr(StatOptions::new(aggregate_fn), [expr])
        .vortex_expect("geometry pruning only requests a supported aggregate")
}

fn ext_storage(input: BoundExpr) -> BoundExpr {
    ExtStorage
        .try_new_bound_expr(ScalarEmptyOptions, [input])
        .vortex_expect("the geometry AABB aggregate returns an extension value")
}

fn get_item(field: impl Into<FieldName>, child: BoundExpr) -> BoundExpr {
    GetItem
        .try_new_bound_expr(field.into(), [child])
        .vortex_expect("geometry AABB fields are fixed by its storage dtype")
}

fn lit(value: impl Into<Scalar>) -> BoundExpr {
    Literal
        .try_new_bound_expr(value.into(), [])
        .vortex_expect("literal expressions are always well-typed")
}

fn case_when(condition: BoundExpr, then_value: BoundExpr, else_value: BoundExpr) -> BoundExpr {
    CaseWhen
        .try_new_bound_expr(
            CaseWhenOptions {
                num_when_then_pairs: 1,
                has_else: true,
            },
            [condition, then_value, else_value],
        )
        .vortex_expect("geometry pruning case expressions have matching branch dtypes")
}

fn gt(lhs: BoundExpr, rhs: BoundExpr) -> BoundExpr {
    binop(Operator::Gt, lhs, rhs)
}

fn gt_eq(lhs: BoundExpr, rhs: BoundExpr) -> BoundExpr {
    binop(Operator::Gte, lhs, rhs)
}

fn lt(lhs: BoundExpr, rhs: BoundExpr) -> BoundExpr {
    binop(Operator::Lt, lhs, rhs)
}

fn lt_eq(lhs: BoundExpr, rhs: BoundExpr) -> BoundExpr {
    binop(Operator::Lte, lhs, rhs)
}

fn checked_add(lhs: BoundExpr, rhs: BoundExpr) -> BoundExpr {
    binop(Operator::Add, lhs, rhs)
}

/// Lower bound on every row's squared distance to the query AABB: zero when the boxes overlap or
/// touch, positive iff they are strictly separated.
///
/// Prunes "near" predicates: `min_dist_sq > r^2` proves every row is farther than `r`.
fn min_dist_sq(aabb: &BoundExpr, query: GeoRect<f64>) -> BoundExpr {
    let field = |name: &str| get_item(name, aabb.clone());
    // Per axis: gap = max(0, q_lo - aabb_hi, aabb_lo - q_hi), positive only when the intervals
    // are separated. The nearest two points of the boxes are one axis-gap apart per axis, so the
    // squared distance is gap_x^2 + gap_y^2 (squared throughout to avoid a sqrt).
    let gap = |q_lo: f64, q_hi: f64, lo: BoundExpr, hi: BoundExpr| {
        maximum(
            lit(0.0),
            maximum(
                binop(Operator::Sub, lit(q_lo), hi),
                binop(Operator::Sub, lo, lit(q_hi)),
            ),
        )
    };
    let dx = gap(query.min().x, query.max().x, field("xmin"), field("xmax"));
    let dy = gap(query.min().y, query.max().y, field("ymin"), field("ymax"));
    checked_add(square(dx), square(dy))
}

/// Upper bound on every row's squared distance to the query AABB.
///
/// Prunes "far" predicates: `max_dist_sq < r^2` proves every row is within `r`.
fn max_dist_sq(aabb: &BoundExpr, query: GeoRect<f64>) -> BoundExpr {
    let field = |name: &str| get_item(name, aabb.clone());
    // Per axis: span = max(q_hi, aabb_hi) - min(q_lo, aabb_lo), the farthest two points of the
    // boxes can be apart. The nullable AABB field is the second `maximum`/`minimum` argument so
    // that `case_when`'s else branch carries the nullability - a missing stat propagates null.
    let span = |q_lo: f64, q_hi: f64, lo: BoundExpr, hi: BoundExpr| {
        binop(
            Operator::Sub,
            maximum(lit(q_hi), hi),
            minimum(lit(q_lo), lo),
        )
    };
    let dx = span(query.min().x, query.max().x, field("xmin"), field("xmax"));
    let dy = span(query.min().y, query.max().y, field("ymin"), field("ymax"));
    checked_add(square(dx), square(dy))
}

/// `a <op> b`.
fn binop(op: Operator, a: BoundExpr, b: BoundExpr) -> BoundExpr {
    Binary
        .try_new_bound_expr(op, [a, b])
        .vortex_expect("binary expression")
}

/// `e * e`.
fn square(e: BoundExpr) -> BoundExpr {
    binop(Operator::Mul, e.clone(), e)
}

/// `max(a, b)`.
fn maximum(a: BoundExpr, b: BoundExpr) -> BoundExpr {
    case_when(gt(a.clone(), b.clone()), a, b)
}

/// `min(a, b)`.
fn minimum(a: BoundExpr, b: BoundExpr) -> BoundExpr {
    case_when(lt(a.clone(), b.clone()), a, b)
}
