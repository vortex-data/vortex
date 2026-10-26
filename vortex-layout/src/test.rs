// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::expr::root;
use vortex_error::VortexResult;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::segments::SegmentSource;
use crate::segments::TestSegments;
use crate::session::LayoutSession;

pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| new_session().with_tokio());

pub fn new_session() -> VortexSession {
    VortexSession::empty()
        .with::<LayoutSession>()
        .with::<RuntimeSession>()
}

/// A child strategy that fails with [`FailingStrategy::MESSAGE`] without reading its stream, as a
/// writer that rejects its input does.
pub struct FailingStrategy;

impl FailingStrategy {
    pub const MESSAGE: &str = "child writer rejected its input";
}

#[async_trait::async_trait]
impl crate::LayoutStrategy for FailingStrategy {
    async fn write_stream(
        &self,
        _ctx: crate::LayoutWriterContext,
        _segment_sink: crate::segments::SegmentSinkRef,
        _stream: crate::sequence::SendableSequentialStream,
        _eof: crate::sequence::SequencePointer,
        _session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        vortex_error::vortex_bail!("{}", Self::MESSAGE)
    }
}

/// Read the zone map column that a zoned `layout` stores for `aggregate_fn`, one partial per zone.
pub fn read_zone_stats(
    layout: &LayoutRef,
    segments: &Arc<TestSegments>,
    aggregate_fn: &AggregateFnRef,
) -> VortexResult<ArrayRef> {
    let zones = Arc::clone(&layout.children()?[1]);
    let segments: Arc<dyn SegmentSource> = Arc::<TestSegments>::clone(segments);
    let row_count = zones.row_count();
    let table = block_on(|handle| async move {
        let session = new_session().with_handle(handle);
        let reader = zones.new_reader("".into(), segments, &session, &Default::default())?;
        reader
            .projection_evaluation(
                &(0..row_count),
                &root().bind(reader.dtype())?,
                MaskFuture::new_true(usize::try_from(row_count)?),
            )?
            .await
    })?;
    table.get_item(aggregate_fn.to_string())
}
