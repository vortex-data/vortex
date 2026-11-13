// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use vortex_io::session::RuntimeSession;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::session::LayoutSession;

pub static SESSION: LazyLock<VortexSession> = LazyLock::new(|| new_session().with_tokio());

pub fn new_session() -> VortexSession {
    let session = VortexSession::empty()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    vortex_functions::initialize(&session);
    session
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
    ) -> vortex_error::VortexResult<crate::LayoutRef> {
        vortex_error::vortex_bail!("{}", Self::MESSAGE)
    }
}
