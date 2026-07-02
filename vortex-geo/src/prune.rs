// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stats-rewrite pruning for spatial filters, backed by the per-chunk [`GeometryBounds`] MBR.
//!
//! [`GeoDistanceBoundsPrune`] falsifies `ST_Distance(geom, const) <= r` (or `< r`): any row within
//! `r` of the constant must lie inside the constant's bounding box expanded by `r`, so if a chunk's
//! geometry MBR is disjoint from that expanded box, no row in the chunk can match and the chunk is
//! skipped.
//!
//! # Limitations
//!
//! Only the "near" forms `<= r` / `< r` are handled — the predicates a radius/within search uses.
//! They prune via the MBR's *lower* distance bound (the nearest point of the box to `const`). Every
//! other comparison falls through to `None`, leaving the chunk to be scanned — correct, just not
//! pruned:
//!
//! - `> r` / `>= r` are *soundly* prunable via the symmetric upper bound (the farthest corner of the
//!   MBR being `<= r` proves every geometry is within `r`), but are intentionally omitted: "far from"
//!   filters are rare and rarely selective, so the prune would almost never fire.
//! - `== r` would need both bounds at once and is not a realistic query.
//! - `!= r` is unprunable: a bounding box cannot prove every row sits at exactly distance `r`.

use geo::BoundingRect;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::expr::Expression;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::is_root;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::or;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::stats::rewrite::StatsRewriteCtx;
use vortex_array::stats::rewrite::StatsRewriteRule;
use vortex_array::stats::stat;
use vortex_error::VortexResult;

use crate::aggregate_fn::GeometryBounds;
use crate::extension::is_native_geometry;
use crate::extension::single_geometry;
use crate::scalar_fn::distance::GeoDistance;

/// Prunes chunks for `GeoDistance(geom, const) <= r` / `< r` using the chunk's [`GeometryBounds`]
/// MBR. Registered against the comparison's scalar-function id, since the comparison — not
/// `GeoDistance` — is the predicate root.
#[derive(Debug)]
pub struct GeoDistanceBoundsPrune;

impl StatsRewriteRule for GeoDistanceBoundsPrune {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Binary.id()
    }

    fn falsify(
        &self,
        expr: &Expression,
        ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
        // Only the near forms `<= r` / `< r` prune (see the module Limitations).
        match expr.as_::<Binary>() {
            Operator::Lte | Operator::Lt => {}
            _ => return Ok(None),
        }
        let distance = expr.child(0);
        let threshold = expr.child(1);

        // The left operand must be `GeoDistance(geom, const)`.
        if distance.as_opt::<GeoDistance>().is_none() {
            return Ok(None);
        }

        // Distance is symmetric, so the geometry column (scope root) and constant may be on either
        // side.
        let (lhs, rhs) = (distance.child(0), distance.child(1));
        let (geom, constant) = if is_root(lhs) {
            (lhs, rhs)
        } else if is_root(rhs) {
            (rhs, lhs)
        } else {
            return Ok(None);
        };

        // The stat reference below only binds for dtypes `GeometryBounds` supports; anything else
        // (e.g. a WKB column) must fall through to the scan.
        if !is_native_geometry(&ctx.return_dtype(geom)?) {
            return Ok(None);
        }

        let (Some(constant), Some(radius)) =
            (constant.as_opt::<Literal>(), threshold.as_opt::<Literal>())
        else {
            return Ok(None);
        };
        let Ok(radius) = f64::try_from(radius) else {
            return Ok(None);
        };
        // A NaN radius would poison the expanded box into pruning every zone, while the scan's
        // total-order compare treats `distance <= NaN` as true for every row.
        if radius.is_nan() {
            return Ok(None);
        }

        // Bounding box of the constant geometry, expanded by the radius.
        let mut exec = ctx.session().create_execution_ctx();
        let Some(rect) = single_geometry(constant, &mut exec)?.bounding_rect() else {
            return Ok(None);
        };
        let (xmin, xmax) = (rect.min().x - radius, rect.max().x + radius);
        let (ymin, ymax) = (rect.min().y - radius, rect.max().y + radius);

        // Prune when the chunk MBR is disjoint from the expanded box on any axis: no row can match.
        let mbr = stat(geom.clone(), GeometryBounds.bind(EmptyOptions));
        let proof = or(
            or(
                lt(get_item("xmax", mbr.clone()), lit(xmin)),
                gt(get_item("xmin", mbr.clone()), lit(xmax)),
            ),
            or(
                lt(get_item("ymax", mbr.clone()), lit(ymin)),
                gt(get_item("ymin", mbr), lit(ymax)),
            ),
        );
        Ok(Some(proof))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::EmptyOptions as AggregateEmptyOptions;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::Expression;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt_eq;
    use vortex_array::expr::root;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnVTableExt;
    use vortex_array::scalar_fn::fns::binary::Binary;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_array::stats::rewrite::StatsRewriteCtx;
    use vortex_array::stats::rewrite::StatsRewriteRule;
    use vortex_array::validity::Validity;
    use vortex_error::VortexResult;
    use vortex_layout::layouts::zoned::zone_map::ZoneMap;

    use super::GeoDistanceBoundsPrune;
    use crate::aggregate_fn::GeometryBounds;
    use crate::scalar_fn::distance::GeoDistance;
    use crate::test_harness::point_column;

    /// Run the rule against `GeoDistance(root, origin) <operator> radius`, operands swapped when
    /// `geom_first` is false.
    fn falsify_distance(
        operator: Operator,
        geom_first: bool,
        radius: f64,
    ) -> VortexResult<Option<Expression>> {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        let scope = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let origin = point_column(vec![0.0], vec![0.0])?.execute_scalar(0, &mut ctx)?;
        let operands = if geom_first {
            [root(), lit(origin)]
        } else {
            [lit(origin), root()]
        };
        let distance = GeoDistance.new_expr(EmptyOptions, operands);
        let predicate = Binary.new_expr(operator, [distance, lit(radius)]);

        GeoDistanceBoundsPrune.falsify(&predicate, &StatsRewriteCtx::new(&session, &scope))
    }

    /// Only the upper-bounded "near" forms (`<=`/`<`) are pruned; the rest are left to the scan.
    #[rstest]
    #[case(Operator::Lte, true)]
    #[case(Operator::Lt, true)]
    #[case(Operator::Gt, false)]
    #[case(Operator::Gte, false)]
    #[case(Operator::Eq, false)]
    #[case(Operator::NotEq, false)]
    fn prunes_only_near_distance(
        #[case] operator: Operator,
        #[case] prunes: bool,
    ) -> VortexResult<()> {
        assert_eq!(falsify_distance(operator, true, 0.5)?.is_some(), prunes);
        Ok(())
    }

    /// Distance is symmetric: `GeoDistance(const, geom) <= r` falsifies just like the geom-first form.
    #[test]
    fn falsifies_with_constant_as_left_operand() -> VortexResult<()> {
        assert!(falsify_distance(Operator::Lte, false, 0.5)?.is_some());
        Ok(())
    }

    /// A NaN radius must not prune — the scan's total-order compare treats `dist <= NaN` as true.
    #[test]
    fn nan_radius_never_prunes() -> VortexResult<()> {
        assert!(falsify_distance(Operator::Lte, true, f64::NAN)?.is_none());
        Ok(())
    }

    /// A negative radius prunes every zone — vacuously sound: `distance <= r < 0` matches no row.
    #[test]
    fn negative_radius_prunes_vacuously() -> VortexResult<()> {
        assert!(falsify_distance(Operator::Lte, true, -0.5)?.is_some());
        Ok(())
    }

    /// A scope dtype without `GeometryBounds` support gets no proof — the stat reference would
    /// fail to bind at prune time.
    #[test]
    fn unsupported_scope_is_not_pruned() -> VortexResult<()> {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        let scope = DType::Primitive(PType::F64, Nullability::NonNullable);
        let origin = point_column(vec![0.0], vec![0.0])?.execute_scalar(0, &mut ctx)?;
        let distance = GeoDistance.new_expr(EmptyOptions, [root(), lit(origin)]);
        let predicate = lt_eq(distance, lit(0.5f64));

        let ctx = StatsRewriteCtx::new(&session, &scope);
        assert!(GeoDistanceBoundsPrune.falsify(&predicate, &ctx)?.is_none());
        Ok(())
    }

    /// A comparison that does not wrap `GeoDistance` is left untouched.
    #[test]
    fn ignores_non_distance_comparison() -> VortexResult<()> {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        let scope = point_column(vec![0.0], vec![0.0])?.dtype().clone();

        let predicate = lt_eq(lit(1.0f64), lit(2.0f64));
        let ctx = StatsRewriteCtx::new(&session, &scope);
        assert!(GeoDistanceBoundsPrune.falsify(&predicate, &ctx)?.is_none());
        Ok(())
    }

    /// End-to-end over a hand-built zone map: the far chunk is skipped, the near one kept.
    #[test]
    fn prunes_far_chunk_keeps_near() -> VortexResult<()> {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        let point_dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let bounds_fn = GeometryBounds.bind(AggregateEmptyOptions);

        // Two chunks: chunk 0 near the origin (MBR 0,0..1,1), chunk 1 far away (MBR 100,100..101,101).
        let coord =
            |a: f64, b: f64| PrimitiveArray::from_option_iter([Some(a), Some(b)]).into_array();
        let mbrs = StructArray::try_new(
            ["xmin", "ymin", "xmax", "ymax"].into(),
            vec![
                coord(0.0, 100.0),
                coord(0.0, 100.0),
                coord(1.0, 101.0),
                coord(1.0, 101.0),
            ],
            2,
            Validity::AllValid,
        )?
        .into_array();
        let zone_array = StructArray::from_fields(&[(bounds_fn.to_string().as_str(), mbrs)])?;
        let zone_map =
            ZoneMap::try_new(point_dtype.clone(), zone_array, Arc::new([bounds_fn]), 1, 2)?;

        let origin = point_column(vec![0.0], vec![0.0])?.execute_scalar(0, &mut ctx)?;
        let distance = GeoDistance.new_expr(EmptyOptions, [root(), lit(origin)]);
        let predicate = lt_eq(distance, lit(0.5f64));
        let proof = predicate
            .falsify(&point_dtype, &session)?
            .expect("distance filter should be falsifiable");

        // `true` means the zone is pruned: chunk 0 (near origin) is kept, chunk 1 (far) is skipped.
        let mask = zone_map.prune(&proof, &session)?;
        assert_eq!(mask.iter().collect::<Vec<bool>>(), vec![false, true]);
        Ok(())
    }

    /// Backward compat: a zone map written without the `GeometryBounds` stat (an older file) keeps
    /// every zone — the missing stat binds to null and `null_as_false` retains the zone.
    #[test]
    fn missing_bounds_stat_keeps_all_zones() -> VortexResult<()> {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        let point_dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let zone_map = ZoneMap::try_new(
            point_dtype.clone(),
            StructArray::try_new(FieldNames::empty(), vec![], 2, Validity::NonNullable)?,
            Arc::new([]),
            1,
            2,
        )?;

        let origin = point_column(vec![0.0], vec![0.0])?.execute_scalar(0, &mut ctx)?;
        let distance = GeoDistance.new_expr(EmptyOptions, [root(), lit(origin)]);
        let proof = lt_eq(distance, lit(0.5f64))
            .falsify(&point_dtype, &session)?
            .expect("distance filter should be falsifiable");

        let mask = zone_map.prune(&proof, &session)?;
        assert_eq!(mask.iter().collect::<Vec<bool>>(), vec![false, false]);
        Ok(())
    }
}
