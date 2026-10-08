// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::TryStreamExt;
use futures::stream;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::DeserializeMetadata;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::expr::gt;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::expr::stats::Stat;
use vortex_array::stats::as_stat_bitset_bytes;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use super::scan_file;
use crate::LayoutBuildContext;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::OwnedLayoutChildren;
use crate::VTable;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::layouts::zoned::LegacyStats;
use crate::layouts::zoned::LegacyStatsMetadata;
use crate::scan::v2::ScanBuilder;
use crate::segments::SegmentSource;
use crate::segments::TestSegments;
use crate::sequence::SequenceId;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;
use crate::test::new_session;

async fn flat(
    array: ArrayRef,
    segments: Arc<TestSegments>,
    session: &VortexSession,
) -> VortexResult<LayoutRef> {
    let (mut sequence, eof) = SequenceId::root().split();
    FlatLayoutStrategy::default()
        .write_stream(
            ArrayContext::empty().into(),
            segments,
            SequentialStreamAdapter::new(
                array.dtype().clone(),
                stream::iter([Ok((sequence.advance(), array))]),
            )
            .sendable(),
            eof,
            session,
        )
        .await
}

#[tokio::test]
async fn legacy_string_zones_scan() -> VortexResult<()> {
    let session = new_session().with_tokio();
    let segments = Arc::new(TestSegments::default());
    let data = VarBinViewArray::from_iter_str(["a", "b", "y", "z"]).into_array();
    let zones = StructArray::from_fields(&[
        (
            "max",
            VarBinViewArray::from_iter([Some("b"), Some("z")], data.dtype().as_nullable())
                .into_array(),
        ),
        (
            "max_is_truncated",
            BoolArray::from_iter([false, false]).into_array(),
        ),
    ])?
    .into_array();
    let children = OwnedLayoutChildren::layout_children(vec![
        flat(data.clone(), segments.clone(), &session).await?,
        flat(zones, segments.clone(), &session).await?,
    ]);
    let read_ctx = ReadContext::new([]);
    let mut metadata = 2u32.to_le_bytes().to_vec();
    metadata.extend(as_stat_bitset_bytes(&[Stat::Max]));
    let layout = <LegacyStats as VTable>::build(
        &LegacyStats,
        data.dtype(),
        4,
        &LegacyStatsMetadata::deserialize(&metadata)?,
        vec![],
        children.as_ref(),
        &LayoutBuildContext {
            session: &session,
            array_read_ctx: &read_ctx,
        },
    )?
    .into_layout();
    let segments: Arc<dyn SegmentSource> = segments;
    let reader = layout.new_reader("".into(), segments.clone(), &session, &Default::default())?;
    let output = ScanBuilder::new(session.clone(), reader, scan_file(&segments, &layout)?)
        .with_filter(gt(root(), lit("m")).bind(data.dtype())?)
        .into_stream()?
        .try_collect::<Vec<_>>()
        .await?;
    assert_arrays_eq!(
        ChunkedArray::try_new(output, data.dtype().clone())?,
        VarBinViewArray::from_iter_str(["y", "z"]),
        &mut session.create_execution_ctx()
    );
    Ok(())
}
