// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::TryStreamExt;
use futures::stream;
use rstest::rstest;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::expr::gt;
use vortex_array::expr::list_length;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSessionExt;

use super::scan_file;
use crate::LayoutStrategy;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::list::writer::ListLayoutStrategy;
use crate::scan::scan_builder::ScanBuilder as LegacyScanBuilder;
use crate::scan::v2::ScanBuilder;
use crate::segments::SegmentSource;
use crate::segments::TestSegments;
use crate::sequence::SequenceId;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;
use crate::test::new_session;

#[rstest]
#[case::project(false, false, None)]
#[case::filter(true, false, None)]
#[case::length(false, true, None)]
#[case::filtered_length(true, true, None)]
#[case::limit(false, true, Some(1))]
#[tokio::test]
async fn list_scan(
    #[case] filtered: bool,
    #[case] lengths: bool,
    #[case] limit: Option<u64>,
) -> VortexResult<()> {
    let session = new_session().with_tokio();
    let chunk = ListArray::try_new(
        PrimitiveArray::from_option_iter([Some(0i32), None, Some(2), Some(1), Some(0)])
            .into_array(),
        buffer![0u64, 2, 2, 3, 5].into_array(),
        Validity::from_iter([true, true, false, true]),
    )?
    .into_array();
    let dtype = chunk.dtype().clone();
    let segments = Arc::new(TestSegments::default());
    let (mut sequence, eof) = SequenceId::root().split();
    let chunks = vec![
        Ok((sequence.advance(), chunk.clone())),
        Ok((sequence.advance(), chunk)),
    ];
    let layout = ChunkedLayoutStrategy::new(ListLayoutStrategy::default())
        .write_stream(
            ArrayContext::empty().into(),
            Arc::<TestSegments>::clone(&segments),
            SequentialStreamAdapter::new(dtype.clone(), stream::iter(chunks)).sendable(),
            eof,
            &session,
        )
        .await?;
    let segments: Arc<dyn SegmentSource> = segments;
    let reader = layout.new_reader(
        "".into(),
        Arc::clone(&segments),
        &session,
        &Default::default(),
    )?;
    let projection = if lengths { list_length(root()) } else { root() }.bind(&dtype)?;
    let filter = filtered
        .then(|| gt(list_length(root()), lit(1u64)).bind(&dtype))
        .transpose()?;
    let baseline = LegacyScanBuilder::new(session.clone(), Arc::clone(&reader))
        .with_projection(projection.clone())
        .with_some_filter(filter.clone())
        .with_row_range(1..7)
        .with_some_limit(limit)
        .with_ordered(true);
    let expected = baseline.into_stream()?.try_collect::<Vec<_>>().await?;
    let actual = ScanBuilder::new(session.clone(), reader, scan_file(&segments, &layout)?)
        .with_projection(projection.clone())
        .with_some_filter(filter)
        .with_row_range(1..7)
        .with_some_limit(limit)
        .with_ordered(true)
        .into_stream()?
        .try_collect::<Vec<_>>()
        .await?;
    assert_arrays_eq!(
        ChunkedArray::try_new(actual, projection.dtype().clone())?,
        ChunkedArray::try_new(expected, projection.dtype().clone())?,
        &mut session.create_execution_ctx()
    );
    Ok(())
}
