// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use futures::future;
use futures::stream;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability::NonNullable;
use vortex_array::dtype::PType;
use vortex_array::expr::root;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;
use vortex_scan::planning::morsel::Morsel;

use super::SplitMorsel;
use crate::LayoutStrategy;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::scan::planning::PollingSegmentSource;
use crate::segments::TestSegments;
use crate::sequence::SequenceId;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt as _;
use crate::test::new_session;

#[test]
fn reader_pending_on_a_non_segment_is_an_error() -> VortexResult<()> {
    let dtype = DType::Primitive(PType::I32, NonNullable);
    let segments = Arc::new(TestSegments::default());
    let session = new_session();
    let layout = block_on(|handle| {
        let session = session.clone().with_handle(handle);
        let segments = Arc::clone(&segments);
        let dtype = dtype.clone();
        async move {
            let (mut sequence_id, eof) = SequenceId::root().split();
            let array = buffer![1i32, 2, 3, 4, 5, 6, 7, 8].into_array();
            FlatLayoutStrategy::default()
                .write_stream(
                    ArrayContext::empty().into(),
                    segments,
                    SequentialStreamAdapter::new(
                        dtype,
                        stream::iter([Ok((sequence_id.advance(), array))]),
                    )
                    .sendable(),
                    eof,
                    &session,
                )
                .await
        }
    })?;

    let source = Arc::new(PollingSegmentSource::new(Arc::from([])));
    let reader = layout.new_reader(
        "".into(),
        Arc::clone(&source) as _,
        &session,
        &Default::default(),
    )?;
    let mut morsel = SplitMorsel::new(
        source,
        reader,
        0..8,
        Mask::new_true(8),
        None,
        root().bind(&dtype)?,
    );
    morsel.pending = Some(future::pending().boxed());

    let err = morsel.compute().err().map(|e| e.to_string());
    assert!(
        err.as_deref().is_some_and(|m| m.contains("not a segment")),
        "{err:?}"
    );
    Ok(())
}
