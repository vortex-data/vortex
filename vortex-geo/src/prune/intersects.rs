// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ST_Intersects(geom, const)` pruning.

use vortex_array::expr::BoundExpression;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::stats::rewrite::StatsRewriteCtx;
use vortex_array::stats::rewrite::StatsRewriteRule;
use vortex_error::VortexResult;

use super::aabb_stat;
use super::geometry_and_constant;
use super::gt;
use super::lit;
use super::min_dist_sq;
use super::query_aabb;
use crate::scalar_fn::intersects::GeoIntersects;

/// Prunes chunks for `ST_Intersects(geom, const)` filters: a chunk whose box is strictly
/// separated from the constant's bounding box cannot contain an intersecting row.
///
/// Only the positive form prunes. `NOT ST_Intersects` cannot: it would need every row to provably
/// intersect the constant, and box overlap never proves geometry intersection.
#[derive(Debug)]
pub struct GeoIntersectsPrune;

impl StatsRewriteRule for GeoIntersectsPrune {
    fn scalar_fn_id(&self) -> ScalarFnId {
        // Unlike the distance rule, the boolean predicate is itself the expression root.
        GeoIntersects.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<BoundExpression>> {
        let Some((geom, constant)) = geometry_and_constant(expr, ctx)? else {
            return Ok(None);
        };
        let Some(query) = query_aabb(constant, ctx)? else {
            return Ok(None);
        };
        // Disjoint iff the minimum box-to-box distance is positive. Strictly (`gt`, not `gt_eq`):
        // boxes that merely touch must scan, since touching geometries do intersect.
        Ok(Some(gt(min_dist_sq(&aabb_stat(geom), query), lit(0.0))))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::BoundExpression;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnVTableExt;
    use vortex_array::stats::rewrite::StatsRewriteCtx;
    use vortex_array::stats::rewrite::StatsRewriteRule;
    use vortex_error::VortexResult;

    use super::GeoIntersectsPrune;
    use crate::prune::test_harness::aabb_zone_map;
    use crate::prune::test_harness::empty_zone_map;
    use crate::scalar_fn::intersects::GeoIntersects;
    use crate::test_harness::geo_session;
    use crate::test_harness::point_column;

    /// Run the intersects rule against `GeoIntersects(root, point(1.0, 0.5))`, operands swapped
    /// when `geom_first` is false.
    fn falsify_intersects(geom_first: bool) -> VortexResult<Option<BoundExpression>> {
        let session = geo_session();
        let mut ctx = session.create_execution_ctx();

        let scope = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let query = point_column(vec![1.0], vec![0.5])?.execute_scalar(0, &mut ctx)?;
        let operands = if geom_first {
            [root(), lit(query)]
        } else {
            [lit(query), root()]
        };
        let predicate = GeoIntersects
            .new_expr(EmptyOptions, operands)
            .bind(&scope)?;
        GeoIntersectsPrune.falsify(&predicate, &StatsRewriteCtx::new(&session))
    }

    /// Intersects is symmetric: both operand orders produce a proof.
    #[rstest]
    #[case(true)]
    #[case(false)]
    fn falsifies_either_operand_order(#[case] geom_first: bool) -> VortexResult<()> {
        assert!(falsify_intersects(geom_first)?.is_some());
        Ok(())
    }

    /// A non-geometry scope is rejected while binding, before stats rewriting.
    #[test]
    fn unsupported_scope_is_not_pruned() -> VortexResult<()> {
        let session = geo_session();
        let mut ctx = session.create_execution_ctx();

        let scope = DType::Primitive(PType::F64, Nullability::NonNullable);
        let query = point_column(vec![0.0], vec![0.0])?.execute_scalar(0, &mut ctx)?;
        let predicate = GeoIntersects.new_expr(EmptyOptions, [root(), lit(query)]);
        assert!(predicate.bind(&scope).is_err());
        Ok(())
    }

    /// A null geometry literal (`ST_Intersects(geom, NULL)`) declines cleanly instead of erroring
    /// in the stats rewrite: the all-null predicate can never prune.
    #[test]
    fn null_literal_is_not_pruned() -> VortexResult<()> {
        let session = geo_session();

        let scope = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let null_query = Scalar::null(scope.as_nullable());
        let predicate = GeoIntersects
            .new_expr(EmptyOptions, [root(), lit(null_query)])
            .bind(&scope)?;

        let ctx = StatsRewriteCtx::new(&session);
        assert!(GeoIntersectsPrune.falsify(&predicate, &ctx)?.is_none());
        Ok(())
    }

    /// End-to-end: a zone strictly separated from the query is skipped; zones containing or merely
    /// touching the query must scan, touching geometries intersect under OGC semantics.
    #[test]
    fn prunes_disjoint_keeps_touching_and_containing() -> VortexResult<()> {
        let session = geo_session();
        let mut ctx = session.create_execution_ctx();

        let point_dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        // Query point (1.0, 0.5): zone 0 contains it, zone 1 only touches it at `x == 1`, zone 2
        // is strictly separated.
        let zone_map = aabb_zone_map(
            &point_dtype,
            &[
                [0.5, 0.0, 1.5, 1.0],
                [-1.0, 0.0, 1.0, 1.0],
                [3.0, 0.0, 4.0, 1.0],
            ],
        )?;

        let query = point_column(vec![1.0], vec![0.5])?.execute_scalar(0, &mut ctx)?;
        let predicate = GeoIntersects.new_expr(EmptyOptions, [root(), lit(query)]);
        let proof = predicate
            .bind(&point_dtype)?
            .falsify(&session)?
            .expect("intersects filter should be falsifiable");

        let mask = zone_map.prune(&proof, &session)?;
        assert_eq!(mask.iter().collect::<Vec<bool>>(), vec![false, false, true]);
        Ok(())
    }

    /// Backward compat: a zone map written without the `GeometryAabb` stat keeps every zone.
    #[test]
    fn missing_aabb_stat_keeps_all_zones() -> VortexResult<()> {
        let session = geo_session();
        let mut ctx = session.create_execution_ctx();

        let point_dtype = point_column(vec![0.0], vec![0.0])?.dtype().clone();
        let zone_map = empty_zone_map(&point_dtype)?;

        let query = point_column(vec![0.0], vec![0.0])?.execute_scalar(0, &mut ctx)?;
        let proof = GeoIntersects
            .new_expr(EmptyOptions, [root(), lit(query)])
            .bind(&point_dtype)?
            .falsify(&session)?
            .expect("intersects filter should be falsifiable");

        let mask = zone_map.prune(&proof, &session)?;
        assert_eq!(mask.iter().collect::<Vec<bool>>(), vec![false, false]);
        Ok(())
    }
}
