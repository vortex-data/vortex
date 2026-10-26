// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use rstest::rstest;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::expr::stats::Precision;
use vortex_array::flatbuffers::WriteFlatBufferExt;
use vortex_array::iter::ArrayIteratorAdapter;
use vortex_array::iter::ArrayIteratorExt;
use vortex_array::scalar::Scalar;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_edition::EDITION_DECLARATIONS;
use vortex_edition::EDITION_FAMILIES;
use vortex_edition::EditionSessionExt;
use vortex_edition::declarations::core::CORE_2026_08_3;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_io::runtime::Handle;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSessionExt;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_session::VortexSession;

use crate::aggregate_fn::GeometryAabb;
use crate::extension::Rect;
use crate::extension::SpatialMetadata;
use crate::extension::box_storage_dtype;
use crate::extension::coordinate::Dimension;
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
fn default_file_bbox_spans_batches(
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
        let mut options = session.write_options();
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
fn explicit_file_statistics_override_spatial_default(
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

#[test]
fn spatial_initialization_preserves_ordinary_file_statistics() -> VortexResult<()> {
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
            session
                .write_options()
                .write(&mut buffer, batch.to_array_stream())
                .await?;
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
