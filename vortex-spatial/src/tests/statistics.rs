// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::expr::get_item;
use vortex_array::expr::gt;
use vortex_array::expr::gt_eq;
use vortex_array::expr::lit;
use vortex_array::expr::lt;
use vortex_array::expr::lt_eq;
use vortex_array::expr::root;
use vortex_array::expr::stats::Precision;
use vortex_array::flatbuffers::WriteFlatBufferExt;
use vortex_array::iter::ArrayIteratorAdapter;
use vortex_array::iter::ArrayIteratorExt;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::EmptyOptions as ScalarFnOptions;
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
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::runtime::Handle;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::repartition::RepartitionStrategy;
use vortex_layout::layouts::repartition::RepartitionWriterOptions;
use vortex_layout::layouts::zoned::Zoned;
use vortex_layout::layouts::zoned::writer::ZonedLayoutOptions;
use vortex_layout::layouts::zoned::writer::ZonedStrategy;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::aggregate_fn::GeometryAabb;
use crate::extension::Rect;
use crate::extension::SpatialMetadata;
use crate::extension::box_storage_dtype;
use crate::extension::coordinate::Dimension;
use crate::scalar_fn::distance::SpatialDistance;
use crate::scalar_fn::intersects::SpatialIntersects;
use crate::test_harness::nullable_point_column;

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
    Ok(session)
}

fn geometry_batch(points: Vec<Option<(f64, f64)>>, nested: bool) -> VortexResult<ArrayRef> {
    let batch = StructArray::from_fields(&[("geom", nullable_point_column(points)?)])?.into_array();
    if nested {
        Ok(StructArray::from_fields(&[("location", batch)])?.into_array())
    } else {
        Ok(batch)
    }
}

#[rstest]
#[case::default(false, [true; 4])]
#[case::explicit(true, [false, false, true, true])]
fn zone_bounds_are_opt_in(
    #[case] record_bounds: bool,
    #[case] expected_mask: [bool; 4],
) -> VortexResult<()> {
    block_on(|handle| async move {
        let session = file_session(handle)?;
        crate::initialize(&session);
        let array = nullable_point_column(vec![
            Some((0.0, 0.0)),
            Some((1.0, 1.0)),
            Some((10.0, 10.0)),
            Some((11.0, 11.0)),
        ])?;
        let strategy = RepartitionStrategy::new(
            ZonedStrategy::new(
                ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
                FlatLayoutStrategy::default(),
                ZonedLayoutOptions {
                    block_size: NonZeroUsize::new(2).vortex_expect("non-zero zone size"),
                    aggregate_fns: record_bounds
                        .then(|| Arc::from([GeometryAabb.bind(EmptyOptions)])),
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
        let zones = file
            .footer()
            .layout()
            .as_opt::<Zoned>()
            .ok_or_else(|| vortex_err!("missing zoned layout"))?;
        assert_eq!(zones.nzones(), 2);
        assert_eq!(
            zones
                .present_aggregates()
                .contains(&GeometryAabb.bind(EmptyOptions).to_string()),
            record_bounds
        );

        let query = nullable_point_column(vec![Some((10.0, 10.0))])?
            .execute_scalar(0, &mut session.create_execution_ctx())?;
        let predicate = SpatialIntersects
            .new_expr(ScalarFnOptions, [root(), lit(query)])
            .bind(file.dtype())?;
        let mask = file
            .layout_reader()?
            .pruning_evaluation(&(0..4), &predicate, Mask::new_true(4))?
            .await?;
        assert_eq!(mask, Mask::from_iter(expected_mask));
        Ok(())
    })
}

#[rstest]
fn explicit_file_bounds_span_batches(
    #[values(false, true)] nested: bool,
    #[values(false, true)] without_zones: bool,
) -> VortexResult<()> {
    block_on(|handle| async move {
        let session = file_session(handle)?;
        crate::initialize(&session);

        let batches = [
            geometry_batch(vec![Some((10.0, 20.0)), None], nested)?,
            geometry_batch(vec![None], nested)?,
            geometry_batch(vec![Some((-10.0, 80.0)), Some((30.0, 40.0))], nested)?,
        ];
        let dtype = batches[0].dtype().clone();
        let stream =
            ArrayIteratorAdapter::new(dtype, batches.into_iter().map(Ok)).into_array_stream();
        let mut options = session
            .write_options()
            .include_nested_statistics()
            .with_file_statistics(vec![
                GeometryAabb.bind(EmptyOptions),
                NullCount.bind(EmptyOptions),
            ]);
        if without_zones {
            options = options.with_strategy(Arc::new(ChunkedLayoutStrategy::new(
                FlatLayoutStrategy::default(),
            )));
        }
        let mut buffer = ByteBufferMut::empty();
        let summary = options.write(&mut buffer, stream).await?;
        let file = session
            .open_options()
            .open_buffer(ByteBuffer::from(buffer))?;
        let path = if nested {
            FieldPath::from_name("location").push("geom")
        } else {
            FieldPath::from_name("geom")
        };
        let expected = Scalar::extension::<Rect>(
            SpatialMetadata::default(),
            Scalar::struct_(
                box_storage_dtype(Dimension::Xy, Nullability::Nullable),
                vec![(-10.0).into(), 20.0.into(), 30.0.into(), 80.0.into()],
            ),
        );
        for stats in [summary.footer().statistics(), file.file_stats()] {
            let (stats, _) = stats
                .and_then(|stats| stats.get_by_path(&path))
                .ok_or_else(|| vortex_err!("missing geometry file statistics"))?;
            assert_eq!(
                stats.get(&GeometryAabb.bind(EmptyOptions)),
                Precision::exact(expected.clone())
            );
            assert_eq!(
                stats.get(&NullCount.bind(EmptyOptions)),
                Precision::exact(Scalar::from(2u64))
            );
        }
        Ok(())
    })
}

#[rstest]
fn file_bounds_enable_spatial_pruning(
    #[values(false, true)] nested: bool,
    #[values(false, true)] record_bounds: bool,
    #[values(false, true)] geometry_first: bool,
) -> VortexResult<()> {
    block_on(|handle| async move {
        let session = file_session(handle)?;
        crate::initialize(&session);
        let batch = geometry_batch(vec![Some((0.0, 0.0)), None, Some((1.0, 1.0))], nested)?;
        let mut buffer = ByteBufferMut::empty();
        session
            .write_options()
            .with_strategy(Arc::new(ChunkedLayoutStrategy::new(
                FlatLayoutStrategy::default(),
            )))
            .include_nested_statistics()
            .with_file_statistics(if record_bounds {
                vec![GeometryAabb.bind(EmptyOptions)]
            } else {
                vec![]
            })
            .write(&mut buffer, batch.to_array_stream())
            .await?;
        let file = session
            .open_options()
            .open_buffer(ByteBuffer::from(buffer))?;
        let column = get_item(
            "geom",
            if nested {
                get_item("location", root())
            } else {
                root()
            },
        );
        let query = nullable_point_column(vec![Some((10.0, 10.0))])?
            .execute_scalar(0, &mut session.create_execution_ctx())?;
        let operands = if geometry_first {
            [column.clone(), lit(query)]
        } else {
            [lit(query), column.clone()]
        };
        let distance = SpatialDistance.new_expr(ScalarFnOptions, operands.clone());
        let nearby_query = nullable_point_column(vec![Some((0.0, 0.0))])?
            .execute_scalar(0, &mut session.create_execution_ctx())?;

        for (predicate, expected) in [
            (
                SpatialIntersects.new_expr(ScalarFnOptions, operands),
                record_bounds,
            ),
            (lt(distance.clone(), lit(5.0)), record_bounds),
            (lt_eq(distance.clone(), lit(5.0)), record_bounds),
            (gt(distance.clone(), lit(20.0)), record_bounds),
            (gt_eq(distance.clone(), lit(20.0)), record_bounds),
            (lt(distance.clone(), lit(20.0)), false),
            (gt(distance, lit(5.0)), false),
            (
                SpatialIntersects.new_expr(ScalarFnOptions, [column, lit(nearby_query)]),
                false,
            ),
        ] {
            assert_eq!(file.can_prune(&predicate)?, expected, "{predicate}");
            let mask = file
                .layout_reader()?
                .pruning_evaluation(&(0..3), &predicate.bind(file.dtype())?, Mask::new_true(3))?
                .await?;
            assert_eq!(mask, Mask::from_iter([!expected; 3]), "{predicate}");
        }
        Ok(())
    })
}

#[rstest]
fn file_statistics_without_geometry_bounds(
    #[values(false, true)] disabled: bool,
) -> VortexResult<()> {
    block_on(|handle| async move {
        let session = file_session(handle)?;
        crate::initialize(&session);
        let batch = geometry_batch(vec![Some((10.0, 20.0)), None], false)?;
        let requested = if disabled {
            vec![]
        } else {
            vec![NullCount.bind(EmptyOptions)]
        };
        let mut buffer = ByteBufferMut::empty();
        session
            .write_options()
            .include_nested_statistics()
            .with_file_statistics(requested)
            .write(&mut buffer, batch.to_array_stream())
            .await?;
        let file = session
            .open_options()
            .open_buffer(ByteBuffer::from(buffer))?;
        if disabled {
            assert!(file.file_stats().is_none());
        } else {
            let (stats, _) = file
                .file_stats()
                .and_then(|stats| stats.get_by_path(&FieldPath::from_name("geom")))
                .ok_or_else(|| vortex_err!("missing geometry file statistics"))?;
            assert!(stats.get(&GeometryAabb.bind(EmptyOptions)).is_absent());
            assert_eq!(
                stats.get(&NullCount.bind(EmptyOptions)),
                Precision::exact(Scalar::from(1u64))
            );
        }
        Ok(())
    })
}

#[rstest]
fn default_file_statistics_do_not_record_geometry_bounds(
    #[values(false, true)] nested_stats: bool,
) -> VortexResult<()> {
    block_on(|handle| async move {
        let session = file_session(handle)?;
        crate::initialize(&session);
        let batch = geometry_batch(vec![Some((10.0, 20.0)), None], false)?;
        let mut buffer = ByteBufferMut::empty();
        let options = session.write_options();
        let options = if nested_stats {
            options.include_nested_statistics()
        } else {
            options
        };
        let summary = options.write(&mut buffer, batch.to_array_stream()).await?;
        let file = session
            .open_options()
            .open_buffer(ByteBuffer::from(buffer))?;
        for stats in [summary.footer().statistics(), file.file_stats()] {
            let (stats, _) = stats
                .and_then(|stats| stats.get_by_path(&FieldPath::from_name("geom")))
                .ok_or_else(|| vortex_err!("missing geometry file statistics"))?;
            assert!(stats.get(&GeometryAabb.bind(EmptyOptions)).is_absent());
            assert_eq!(
                stats.get(&NullCount.bind(EmptyOptions)),
                Precision::exact(Scalar::from(1u64))
            );
        }
        Ok(())
    })
}

#[rstest]
fn spatial_initialization_preserves_ordinary_file_statistics(
    #[values(false, true)] nested_stats: bool,
) -> VortexResult<()> {
    block_on(|handle| async move {
        let session = file_session(handle)?;
        let batch = StructArray::from_fields(&[(
            "value",
            PrimitiveArray::from_option_iter([Some(3i32), None, Some(-2)]).into_array(),
        )])?
        .into_array();
        let mut statistics = Vec::new();
        for spatial in [false, true] {
            if spatial {
                crate::initialize(&session);
            }
            let mut buffer = ByteBufferMut::empty();
            let options = session.write_options();
            let options = if nested_stats {
                options.include_nested_statistics()
            } else {
                options
            };
            options.write(&mut buffer, batch.to_array_stream()).await?;
            let file = session
                .open_options()
                .open_buffer(ByteBuffer::from(buffer))?;
            statistics.push(
                file.file_stats()
                    .ok_or_else(|| vortex_err!("missing file statistics"))?
                    .write_flatbuffer_bytes()?,
            );
        }
        assert_eq!(statistics[0], statistics[1]);
        Ok(())
    })
}
