// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ST_Distance(geom, const) <op> radius` pruning.

use geo::Rect as SpatialRect;
use vortex_array::expr::BoundExpression;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::stats::rewrite::StatsRewriteCtx;
use vortex_array::stats::rewrite::StatsRewriteRule;
use vortex_error::VortexResult;

use super::aabb_stat;
use super::geometry_and_constant;
use super::gt;
use super::gt_eq;
use super::lit;
use super::lt;
use super::lt_eq;
use super::max_dist_sq;
use super::min_dist_sq;
use super::query_aabb;
use crate::scalar_fn::distance::SpatialDistance;

/// Prunes chunks for `ST_Distance(geom, const) <op> r` filters.
///
/// All four comparisons prune: `<= r` / `< r` skip a chunk whose box is wholly beyond `r` (box
/// min-distance); `>= r` / `> r` skip one wholly within `r` (box max-distance). `==` / `!=` don't
/// prune.
#[derive(Debug)]
pub struct SpatialDistancePrune;

impl StatsRewriteRule for SpatialDistancePrune {
    fn scalar_fn_id(&self) -> ScalarFnId {
        // The predicate root is the comparison, not `SpatialDistance`, so key on `Binary`.
        Binary.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<BoundExpression>> {
        // Only the ordered comparisons prune today. `== r` could prune in the future (a chunk is
        // provably empty when `r` lies outside its box's [min, max] distance interval), it's just
        // not implemented. `!= r` cannot: pruning would need every row's distance to equal `r`,
        // which an AABB can't prove.
        let op = *expr.as_::<Binary>();
        if !matches!(
            op,
            Operator::Lte | Operator::Lt | Operator::Gte | Operator::Gt
        ) {
            return Ok(None);
        }

        // The left operand must be `SpatialDistance(geom, const)`; the right, the radius literal.
        let distance = expr.child(0);
        if distance.as_opt::<SpatialDistance>().is_none() {
            return Ok(None);
        }
        let Some(radius) = expr.child(1).as_opt::<Literal>() else {
            return Ok(None);
        };
        // Casts any primitive radius (filters arrive uncoerced, so integer literals are
        // legitimate); a null or non-primitive (e.g. extension-typed) literal has no value to
        // reason about, decline and scan, never error.
        let Ok(radius) = f64::try_from(radius) else {
            return Ok(None);
        };
        // A NaN radius has no sound proof: `distance <op> NaN` is not a total order, so the chunk
        // must be scanned, not pruned.
        if radius.is_nan() {
            return Ok(None);
        }

        let Some((geom, constant)) = geometry_and_constant(distance, ctx)? else {
            return Ok(None);
        };
        let Some(query) = query_aabb(constant, ctx)? else {
            return Ok(None);
        };
        Ok(distance_prune_proof(geom, query, op, radius))
    }
}

/// Build the prune proof for `ST_Distance(geom, const) <op> radius`, or `None` when this
/// operator/radius cannot prune. The box-to-box distance bounds every row's true distance for any
/// geometry types: `<=` / `<` prune a chunk whose box is wholly beyond `radius` (min
/// box-distance); `>=` / `>` prune one wholly within it (max box-distance).
///
/// A distance is always `>= 0`, which decides the degenerate radii up front.
fn distance_prune_proof(
    geom: &BoundExpression,
    query: SpatialRect<f64>,
    op: Operator,
    radius: f64,
) -> Option<BoundExpression> {
    // A distance is always non-negative, so degenerate radii resolve without touching the box.
    match op {
        // `<= r` / `< r` with a negative radius (or zero, for `<`) match nothing: prune every chunk.
        Operator::Lte if radius < 0.0 => return Some(lit(true)),
        Operator::Lt if radius <= 0.0 => return Some(lit(true)),
        // `>= r` / `> r` with a negative radius (or zero, for `>=`) match all rows: never prune.
        Operator::Gte if radius <= 0.0 => return None,
        Operator::Gt if radius < 0.0 => return None,
        _ => {}
    }
    // Compared squared to avoid a `sqrt`; all operands are `>= 0`.
    let aabb = aabb_stat(geom);
    let r2 = lit(radius * radius);
    Some(match op {
        // Beyond the threshold: even the nearest the box can be exceeds `r`.
        Operator::Lte => gt(min_dist_sq(&aabb, query), r2),
        Operator::Lt => gt_eq(min_dist_sq(&aabb, query), r2),
        // Within the threshold: even the farthest the box can be is below `r`.
        Operator::Gte => lt(max_dist_sq(&aabb, query), r2),
        Operator::Gt => lt_eq(max_dist_sq(&aabb, query), r2),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::BoundExpression;
    use vortex_array::expr::gt_eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt_eq;
    use vortex_array::expr::root;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnVTableExt;
    use vortex_array::scalar_fn::fns::binary::Binary;
    use vortex_array::scalar_fn::fns::operators::Operator;
    use vortex_array::stats::rewrite::StatsRewriteCtx;
    use vortex_array::stats::rewrite::StatsRewriteRule;
    use vortex_error::VortexResult;

    use super::SpatialDistancePrune;
    use crate::prune::test_harness::aabb_zone_map;
    use crate::prune::test_harness::empty_zone_map;
    use crate::scalar_fn::distance::SpatialDistance;
    use crate::test_harness::point_column;
    use crate::test_harness::spatial_session;

    /// Run the rule against `SpatialDistance(root, origin) <operator> radius`, operands swapped when
    /// `geom_first` is false. The radius is any literal scalar, matching the uncoerced filter
    /// expressions the rule sees in production.
    fn falsify_distance(
        operator: Operator,
        geom_first: bool,
        radius: impl Into<Scalar>,
    ) -> VortexResult<Option<BoundExpression>> {
        let session = spatial_session();
        let mut ctx = session.create_execution_ctx();

        let scope = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let origin = point_column(vec![0.0], vec![0.0])?.execute_scalar(0, &mut ctx)?;
        let operands = if geom_first {
            [root(), lit(origin)]
        } else {
            [lit(origin), root()]
        };
        let distance = SpatialDistance.new_expr(EmptyOptions, operands);
        let predicate = Binary
            .new_expr(operator, [distance, lit(radius.into())])
            .bind(&scope)?;

        SpatialDistancePrune.falsify(&predicate, &StatsRewriteCtx::new(&session))
    }

    /// A null geometry literal (`ST_Distance(geom, NULL) <= r`) declines cleanly instead of
    /// erroring in the stats rewrite: the all-null predicate can never prune.
    #[test]
    fn null_literal_is_not_pruned() -> VortexResult<()> {
        let session = spatial_session();

        let scope = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let null_query = Scalar::null(scope.as_nullable());
        let distance = SpatialDistance.new_expr(EmptyOptions, [root(), lit(null_query)]);
        let predicate = Binary
            .new_expr(Operator::Lte, [distance, lit(0.5f64)])
            .bind(&scope)?;

        let ctx = StatsRewriteCtx::new(&session);
        assert!(SpatialDistancePrune.falsify(&predicate, &ctx)?.is_none());
        Ok(())
    }

    /// All four distance comparisons prune (`<=`/`<` via min-distance, `>=`/`>` via max-distance);
    /// `==`/`!=` are left to the scan.
    #[rstest]
    #[case(Operator::Lte, true)]
    #[case(Operator::Lt, true)]
    #[case(Operator::Gt, true)]
    #[case(Operator::Gte, true)]
    #[case(Operator::Eq, false)]
    #[case(Operator::NotEq, false)]
    fn prunes_distance_comparisons(
        #[case] operator: Operator,
        #[case] prunes: bool,
    ) -> VortexResult<()> {
        assert_eq!(falsify_distance(operator, true, 0.5)?.is_some(), prunes);
        Ok(())
    }

    /// Distance is symmetric: both operand orders produce a proof.
    #[rstest]
    #[case(true)]
    #[case(false)]
    fn falsifies_either_operand_order(#[case] geom_first: bool) -> VortexResult<()> {
        assert!(falsify_distance(Operator::Lte, geom_first, 0.5)?.is_some());
        Ok(())
    }

    /// A NaN radius must not prune - the scan's total-order compare treats `dist <= NaN` as true.
    #[test]
    fn nan_radius_never_prunes() -> VortexResult<()> {
        assert!(falsify_distance(Operator::Lte, true, f64::NAN)?.is_none());
        Ok(())
    }

    /// A negative radius prunes every zone - vacuously sound: `distance <= r < 0` matches no row.
    #[test]
    fn negative_radius_prunes_vacuously() -> VortexResult<()> {
        assert!(falsify_distance(Operator::Lte, true, -0.5)?.is_some());
        Ok(())
    }

    /// An uncoerced integer radius is rejected while binding the comparison.
    #[test]
    fn uncoerced_integer_radius_fails_to_bind() -> VortexResult<()> {
        assert!(falsify_distance(Operator::Lte, true, 10i64).is_err());
        Ok(())
    }

    /// A null radius has no value to reason about; the rule declines and the chunk is scanned.
    #[test]
    fn null_radius_never_prunes() -> VortexResult<()> {
        let radius = Scalar::null(DType::Primitive(PType::F64, Nullability::Nullable));
        assert!(falsify_distance(Operator::Lte, true, radius)?.is_none());
        Ok(())
    }

    /// An extension-typed radius passes `Binary`'s typecheck (extension operands are exempt) but
    /// has no numeric value - the rule declines rather than erroring, and the chunk is scanned.
    #[test]
    fn extension_radius_never_prunes() -> VortexResult<()> {
        let session = spatial_session();
        let mut ctx = session.create_execution_ctx();

        let geometry = point_column(vec![0.0], vec![0.0])?.execute_scalar(0, &mut ctx)?;
        assert!(falsify_distance(Operator::Lte, true, geometry)?.is_none());
        Ok(())
    }

    /// A non-geometry scope is rejected while binding, before stats rewriting.
    #[test]
    fn unsupported_scope_is_not_pruned() -> VortexResult<()> {
        let session = spatial_session();
        let mut ctx = session.create_execution_ctx();

        let scope = DType::Primitive(PType::F64, Nullability::NonNullable);
        let origin = point_column(vec![0.0], vec![0.0])?.execute_scalar(0, &mut ctx)?;
        let distance = SpatialDistance.new_expr(EmptyOptions, [root(), lit(origin)]);
        let predicate = lt_eq(distance, lit(0.5f64));
        assert!(predicate.bind(&scope).is_err());
        Ok(())
    }

    /// A comparison that does not wrap `SpatialDistance` is left untouched.
    #[test]
    fn ignores_non_distance_comparison() -> VortexResult<()> {
        let session = spatial_session();
        let scope = point_column(vec![0.0], vec![0.0])?.dtype().clone();

        let predicate = lt_eq(lit(1.0f64), lit(2.0f64)).bind(&scope)?;
        let ctx = StatsRewriteCtx::new(&session);
        assert!(SpatialDistancePrune.falsify(&predicate, &ctx)?.is_none());
        Ok(())
    }

    /// End-to-end over a hand-built zone map: the far chunk is skipped, the near one kept.
    #[test]
    fn prunes_far_chunk_keeps_near() -> VortexResult<()> {
        let session = spatial_session();
        let mut ctx = session.create_execution_ctx();

        let point_dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        // Chunk 0 near the origin (0,0..1,1), chunk 1 far away (100,100..101,101).
        let zone_map = aabb_zone_map(
            &point_dtype,
            &[[0.0, 0.0, 1.0, 1.0], [100.0, 100.0, 101.0, 101.0]],
        )?;

        let origin = point_column(vec![0.0], vec![0.0])?.execute_scalar(0, &mut ctx)?;
        let distance = SpatialDistance.new_expr(EmptyOptions, [root(), lit(origin)]);
        let predicate = lt_eq(distance, lit(0.5f64));
        let proof = predicate
            .bind(&point_dtype)?
            .falsify(&session)?
            .expect("distance filter should be falsifiable");

        // `true` means the zone is pruned: chunk 0 (near origin) is kept, chunk 1 (far) is skipped.
        let mask = zone_map.prune(&proof, &session)?;
        assert_eq!(mask.iter().collect::<Vec<bool>>(), vec![false, true]);
        Ok(())
    }

    /// The true-distance prune skips a chunk that is *diagonally* farther than `r`, even though
    /// neither axis alone exceeds `r`, the case a per-axis box-overlap test would wrongly keep.
    #[test]
    fn prunes_diagonally_distant_chunk() -> VortexResult<()> {
        let session = spatial_session();
        let mut ctx = session.create_execution_ctx();

        let point_dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        // One chunk, AABB (0.8,0.8)..(0.9,0.9): each axis is only 0.8 from the origin (<= r = 1), but
        // the near corner is sqrt(0.8^2 + 0.8^2) ~= 1.13 away (> 1), so no point in the box is within 1.
        let zone_map = aabb_zone_map(&point_dtype, &[[0.8, 0.8, 0.9, 0.9]])?;

        let origin = point_column(vec![0.0], vec![0.0])?.execute_scalar(0, &mut ctx)?;
        let distance = SpatialDistance.new_expr(EmptyOptions, [root(), lit(origin)]);
        let predicate = lt_eq(distance, lit(1.0f64));
        let proof = predicate
            .bind(&point_dtype)?
            .falsify(&session)?
            .expect("distance filter should be falsifiable");

        assert_eq!(
            zone_map
                .prune(&proof, &session)?
                .iter()
                .collect::<Vec<bool>>(),
            vec![true],
        );
        Ok(())
    }

    /// A `>= r` filter prunes a chunk lying wholly *within* `r` (every row nearer than `r`, so none
    /// satisfy `>= r`) via the box max-distance, while a chunk beyond `r` is kept.
    #[test]
    fn prunes_within_chunk_for_far_filter() -> VortexResult<()> {
        let session = spatial_session();
        let mut ctx = session.create_execution_ctx();

        let point_dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        // Chunk 0 (AABB 0,0..0.5,0.5, farthest corner ~= 0.707) is entirely within 2 of the origin;
        // chunk 1 (100,100..101,101) is entirely beyond it.
        let zone_map = aabb_zone_map(
            &point_dtype,
            &[[0.0, 0.0, 0.5, 0.5], [100.0, 100.0, 101.0, 101.0]],
        )?;

        let origin = point_column(vec![0.0], vec![0.0])?.execute_scalar(0, &mut ctx)?;
        let distance = SpatialDistance.new_expr(EmptyOptions, [root(), lit(origin)]);
        let proof = gt_eq(distance, lit(2.0f64))
            .bind(&point_dtype)?
            .falsify(&session)?
            .expect("distance filter should be falsifiable");

        // Chunk 0 (within 2) is pruned for `>= 2`; chunk 1 (beyond 2) is kept.
        let mask = zone_map.prune(&proof, &session)?;
        assert_eq!(mask.iter().collect::<Vec<bool>>(), vec![true, false]);
        Ok(())
    }

    /// Backward compat: a zone map written without the `GeometryAabb` stat (an older file) keeps
    /// every zone, the missing stat binds to null and `null_as_false` retains the zone.
    #[test]
    fn missing_aabb_stat_keeps_all_zones() -> VortexResult<()> {
        let session = spatial_session();
        let mut ctx = session.create_execution_ctx();

        let point_dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let zone_map = empty_zone_map(&point_dtype)?;

        let origin = point_column(vec![0.0], vec![0.0])?.execute_scalar(0, &mut ctx)?;
        let distance = SpatialDistance.new_expr(EmptyOptions, [root(), lit(origin)]);
        let proof = lt_eq(distance, lit(0.5f64))
            .bind(&point_dtype)?
            .falsify(&session)?
            .expect("distance filter should be falsifiable");

        let mask = zone_map.prune(&proof, &session)?;
        assert_eq!(mask.iter().collect::<Vec<bool>>(), vec![false, false]);
        Ok(())
    }
}
