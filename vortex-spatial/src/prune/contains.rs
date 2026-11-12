// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ST_Contains` pruning with a geometry column and a constant in either operand order.

use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound::get_item;
use vortex_array::expr::bound::gt;
use vortex_array::expr::bound::lit;
use vortex_array::expr::bound::lt;
use vortex_array::expr::bound::or;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_error::VortexResult;
use vortex_expr::stats::rewrite::StatsRewriteRule;
use vortex_session::VortexSession;

use super::aabb_stat;
use super::geometry_and_constant;
use super::is_column;
use super::query_aabb;
use crate::scalar_fn::contains::SpatialContains;

/// Prunes files and zones for positive `ST_Contains` filters using their aggregate AABB.
///
/// For `contains(column, constant)`, the aggregate box must cover the constant's box. For
/// `contains(constant, column)` (equivalently `within(column, constant)`), disjoint boxes prove
/// no row can match. Neither box relation proves containment, so this rule supplies no proof
/// for `NOT ST_Contains`.
#[derive(Debug)]
pub struct SpatialContainsPrune;

impl StatsRewriteRule for SpatialContainsPrune {
    fn scalar_fn_id(&self) -> ScalarFnId {
        SpatialContains.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        let Some((geom, constant)) = geometry_and_constant(expr)? else {
            return Ok(None);
        };
        let Some(query) = query_aabb(constant, session)? else {
            return Ok(None);
        };
        let bounds = aabb_stat(geom);
        let field = |name| get_item(name, bounds.clone());
        let outside = if is_column(expr.child(0)) {
            // Every row's box lies inside the aggregate box. If even that union cannot cover
            // the constant, no individual geometry can contain it.
            [
                gt(field("xmin"), lit(query.min().x)),
                gt(field("ymin"), lit(query.min().y)),
                lt(field("xmax"), lit(query.max().x)),
                lt(field("ymax"), lit(query.max().y)),
            ]
        } else {
            // The union may extend outside the constant while some rows remain inside it.
            // Only strict separation proves every row is outside; touching boxes must scan.
            [
                gt(field("xmin"), lit(query.max().x)),
                gt(field("ymin"), lit(query.max().y)),
                lt(field("xmax"), lit(query.min().x)),
                lt(field("ymax"), lit(query.min().y)),
            ]
        };
        let [xmin, ymin, xmax, ymax] = outside;
        Ok(Some(or(or(xmin, ymin), or(xmax, ymax))))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::EmptyOptions as AggregateOptions;
    use vortex_array::arrays::StructArray;
    use vortex_array::expr::Expression;
    use vortex_array::expr::fill_null;
    use vortex_array::expr::get_item;
    use vortex_array::expr::lit;
    use vortex_array::expr::not;
    use vortex_array::expr::root;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnVTableExt;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_edition::EDITION_DECLARATIONS;
    use vortex_edition::EDITION_FAMILIES;
    use vortex_edition::EditionSessionExt;
    use vortex_edition::declarations::core::CORE_2026_08_3;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_expr::stats::rewrite::BoundExpressionPruneExt;
    use vortex_file::OpenOptionsSessionExt;
    use vortex_file::WriteOptionsSessionExt;
    use vortex_io::runtime::Handle;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
    use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
    use vortex_layout::layouts::repartition::RepartitionStrategy;
    use vortex_layout::layouts::repartition::RepartitionWriterOptions;
    use vortex_layout::layouts::zoned::writer::ZonedLayoutOptions;
    use vortex_layout::layouts::zoned::writer::ZonedStrategy;
    use vortex_layout::layouts::zoned::zone_map::ZoneMap;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::aggregate_fn::GeometryAabb;
    use crate::prune::test_harness::aabb_zone_map;
    use crate::prune::test_harness::empty_zone_map;
    use crate::scalar_fn::contains::SpatialContains;
    use crate::scalar_fn::envelope::SpatialEnvelope;
    use crate::test_harness::multipoint_column;
    use crate::test_harness::nullable_point_column;
    use crate::test_harness::nullable_rect_column;
    use crate::test_harness::point_column;
    use crate::test_harness::rect_column;
    use crate::test_harness::spatial_session;

    fn contains(column: Expression, constant: Scalar, column_contains: bool) -> Expression {
        let operands = if column_contains {
            [column, lit(constant)]
        } else {
            [lit(constant), column]
        };
        SpatialContains.new_expr(EmptyOptions, operands)
    }

    #[rstest]
    #[case::covers([-1.0, -1.0, 3.0, 3.0], false, false)]
    #[case::equal([0.0, 0.0, 2.0, 2.0], false, false)]
    #[case::inside([0.5, 0.5, 1.5, 1.5], true, false)]
    #[case::overlaps_right([1.0, 0.0, 3.0, 2.0], true, false)]
    #[case::overlaps_left([-1.0, 0.0, 1.0, 2.0], true, false)]
    #[case::overlaps_above([0.0, 1.0, 2.0, 3.0], true, false)]
    #[case::overlaps_below([0.0, -1.0, 2.0, 1.0], true, false)]
    #[case::disjoint_right([3.0, 0.0, 4.0, 2.0], true, true)]
    #[case::disjoint_left([-2.0, 0.0, -1.0, 2.0], true, true)]
    #[case::disjoint_above([0.0, 3.0, 2.0, 4.0], true, true)]
    #[case::disjoint_below([0.0, -2.0, 2.0, -1.0], true, true)]
    #[case::touches([2.0, 0.0, 3.0, 2.0], true, false)]
    fn box_relations_respect_operand_order(
        #[case] bounds: [f64; 4],
        #[case] prune_column_contains: bool,
        #[case] prune_constant_contains: bool,
        #[values(false, true)] column_contains: bool,
    ) -> VortexResult<()> {
        let session = spatial_session();
        let query = rect_column(vec![(0.0, 0.0, 2.0, 2.0)])?
            .execute_scalar(0, &mut session.create_execution_ctx())?;
        let dtype = query.dtype().clone();
        let proof = contains(root(), query, column_contains)
            .bind(&dtype)?
            .falsify(&session)?
            .ok_or_else(|| vortex_err!("missing contains proof"))?;
        let expected = if column_contains {
            prune_column_contains
        } else {
            prune_constant_contains
        };
        let zones = aabb_zone_map(&dtype, &[bounds])?;
        assert_eq!(zones.prune(&proof, &session)?, Mask::from_iter([expected]));
        Ok(())
    }

    #[rstest]
    fn equal_points_are_not_pruned(
        #[values(false, true)] column_contains: bool,
    ) -> VortexResult<()> {
        let session = spatial_session();
        let query = point_column(vec![0.0], vec![0.0])?
            .execute_scalar(0, &mut session.create_execution_ctx())?;
        let dtype = query.dtype().clone();
        let proof = contains(root(), query, column_contains)
            .bind(&dtype)?
            .falsify(&session)?
            .ok_or_else(|| vortex_err!("missing contains proof"))?;
        let zones = aabb_zone_map(&dtype, &[[0.0; 4]])?;
        assert_eq!(zones.prune(&proof, &session)?, Mask::new_false(1));
        Ok(())
    }

    #[rstest]
    fn missing_and_null_bounds_keep_zones(
        #[values(false, true)] column_contains: bool,
    ) -> VortexResult<()> {
        let session = spatial_session();
        let query = rect_column(vec![(0.0, 0.0, 2.0, 2.0)])?
            .execute_scalar(0, &mut session.create_execution_ctx())?;
        let dtype = query.dtype().clone();
        let proof = contains(root(), query, column_contains)
            .bind(&dtype)?
            .falsify(&session)?
            .ok_or_else(|| vortex_err!("missing contains proof"))?;
        let aabb = GeometryAabb.bind(AggregateOptions);
        let null_bounds = ZoneMap::try_new(
            dtype.clone(),
            StructArray::from_fields(&[(
                aabb.to_string().as_str(),
                nullable_rect_column(vec![None, None])?,
            )])?,
            Arc::new([aabb]),
            1,
            2,
        )?;
        for zones in [empty_zone_map(&dtype)?, null_bounds] {
            assert_eq!(zones.prune(&proof, &session)?, Mask::new_false(2));
        }
        Ok(())
    }

    #[rstest]
    fn null_and_empty_constants_decline(
        #[values(false, true)] column_contains: bool,
    ) -> VortexResult<()> {
        let session = spatial_session();
        let empty = multipoint_column(vec![vec![]])?
            .execute_scalar(0, &mut session.create_execution_ctx())?;
        let dtype = empty.dtype().clone();
        for query in [Scalar::null(dtype.as_nullable()), empty] {
            assert!(
                contains(root(), query, column_contains)
                    .bind(&dtype)?
                    .falsify(&session)?
                    .is_none()
            );
        }
        Ok(())
    }

    #[rstest]
    fn negated_and_computed_predicates_decline(
        #[values(false, true)] column_contains: bool,
    ) -> VortexResult<()> {
        let session = spatial_session();
        let query = rect_column(vec![(0.0, 0.0, 2.0, 2.0)])?
            .execute_scalar(0, &mut session.create_execution_ctx())?;
        let dtype = query.dtype().clone();
        for predicate in [
            not(contains(root(), query.clone(), column_contains)),
            contains(
                SpatialEnvelope.new_expr(EmptyOptions, [root()]),
                query,
                column_contains,
            ),
            SpatialContains.new_expr(EmptyOptions, [root(), root()]),
        ] {
            assert!(predicate.bind(&dtype)?.falsify(&session)?.is_none());
        }
        Ok(())
    }

    fn file_session(handle: Handle) -> VortexResult<VortexSession> {
        let session = VortexSession::empty().with_handle(handle);
        for family in EDITION_FAMILIES {
            session.editions().declare_family(family)?;
        }
        for declaration in EDITION_DECLARATIONS {
            session.register_edition(declaration)?;
        }
        session.enable_edition(CORE_2026_08_3)?;
        vortex_file::register_default_encodings(&session);
        crate::initialize(&session);
        Ok(session)
    }

    #[rstest]
    fn prunes_files_without_zone_maps(
        #[values(false, true)] nested: bool,
        #[values(false, true)] record_bounds: bool,
        #[values(false, true)] column_contains: bool,
    ) -> VortexResult<()> {
        block_on(|handle| async move {
            let session = file_session(handle)?;
            let points = nullable_point_column(vec![Some((0.0, 0.0)), None, Some((1.0, 1.0))])?;
            let batch = StructArray::from_fields(&[("geom", points)])?.into_array();
            let (batch, column) = if nested {
                (
                    StructArray::from_fields(&[("location", batch)])?.into_array(),
                    get_item("geom", get_item("location", root())),
                )
            } else {
                (batch, get_item("geom", root()))
            };
            let mut buffer = ByteBufferMut::empty();
            session
                .write_options()
                .with_strategy(Arc::new(ChunkedLayoutStrategy::new(
                    FlatLayoutStrategy::default(),
                )))
                .include_nested_statistics()
                .with_file_statistics(if record_bounds {
                    vec![GeometryAabb.bind(AggregateOptions)]
                } else {
                    vec![]
                })
                .write(&mut buffer, batch.to_array_stream())
                .await?;
            let file = session
                .open_options()
                .open_buffer(ByteBuffer::from(buffer))?;
            let mut ctx = session.create_execution_ctx();
            let far =
                nullable_point_column(vec![Some((10.0, 10.0))])?.execute_scalar(0, &mut ctx)?;
            let near =
                nullable_point_column(vec![Some((0.0, 0.0))])?.execute_scalar(0, &mut ctx)?;
            let enclosing =
                rect_column(vec![(-1.0, -1.0, 2.0, 2.0)])?.execute_scalar(0, &mut ctx)?;

            for (query, cannot_match) in [(far, true), (near, false), (enclosing, column_contains)]
            {
                let predicate = contains(column.clone(), query, column_contains);
                let matches = batch
                    .clone()
                    .apply(&fill_null(predicate.clone(), lit(false)))?
                    .execute::<Mask>(&mut ctx)?;
                let expected = record_bounds && cannot_match;
                assert_eq!(file.can_prune(&predicate)?, expected, "{predicate}");
                let mask = file
                    .layout_reader()?
                    .pruning_evaluation(&(0..3), &predicate.bind(file.dtype())?, Mask::new_true(3))?
                    .await?;
                assert_eq!(mask, Mask::from_iter([!expected; 3]), "{predicate}");
                assert_eq!(
                    &mask & &matches,
                    matches,
                    "pruning dropped matches for {predicate}"
                );
            }
            Ok(())
        })
    }

    #[rstest]
    fn prunes_zones_without_file_statistics(
        #[values(false, true)] record_bounds: bool,
        #[values(false, true)] column_contains: bool,
    ) -> VortexResult<()> {
        block_on(|handle| async move {
            let session = file_session(handle)?;
            let array = nullable_point_column(vec![
                Some((0.0, 0.0)),
                Some((1.0, 1.0)),
                Some((10.0, 10.0)),
                Some((11.0, 11.0)),
            ])?;
            let aggregates = if record_bounds {
                vec![GeometryAabb.bind(AggregateOptions)]
            } else {
                vec![]
            };
            let strategy = RepartitionStrategy::new(
                ZonedStrategy::new(
                    ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
                    FlatLayoutStrategy::default(),
                    ZonedLayoutOptions {
                        block_size: NonZeroUsize::new(2).vortex_expect("non-zero zone size"),
                        aggregate_fns: Some(aggregates.into()),
                        ..Default::default()
                    },
                ),
                RepartitionWriterOptions {
                    block_size_minimum: 0,
                    block_len_multiple: 2,
                    block_size_target: None,
                    canonicalize: false,
                },
            );
            let mut buffer = ByteBufferMut::empty();
            session
                .write_options()
                .with_strategy(Arc::new(strategy))
                .with_file_statistics(vec![])
                .write(&mut buffer, array.to_array_stream())
                .await?;
            let file = session
                .open_options()
                .open_buffer(ByteBuffer::from(buffer))?;
            assert!(file.file_stats().is_none());
            let mut ctx = session.create_execution_ctx();
            let query =
                nullable_point_column(vec![Some((10.0, 10.0))])?.execute_scalar(0, &mut ctx)?;
            let predicate = contains(root(), query, column_contains);
            let matches = array
                .clone()
                .apply(&fill_null(predicate.clone(), lit(false)))?
                .execute::<Mask>(&mut ctx)?;
            let mask = file
                .layout_reader()?
                .pruning_evaluation(&(0..4), &predicate.bind(file.dtype())?, Mask::new_true(4))?
                .await?;
            assert_eq!(
                mask,
                Mask::from_iter([!record_bounds, !record_bounds, true, true])
            );
            assert_eq!(
                &mask & &matches,
                matches,
                "pruning dropped matches for {predicate}"
            );
            Ok(())
        })
    }
}
