// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::segments::EncryptingSegmentSink;
use crate::segments::EncryptionKeyName;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;

/// A layout strategy that encrypts every segment its child writes with a named key.
///
/// This covers all segments of the child's subtree, including statistics and dictionary values.
/// The key name is resolved by the segment sink; the file writer fails if it has no key with that
/// name.
#[derive(Clone)]
pub struct EncryptingStrategy {
    child: Arc<dyn LayoutStrategy>,
    key: EncryptionKeyName,
}

impl EncryptingStrategy {
    /// Encrypt the segments written by `child` with `key`.
    pub fn new(child: Arc<dyn LayoutStrategy>, key: impl Into<EncryptionKeyName>) -> Self {
        Self {
            child,
            key: key.into(),
        }
    }
}

#[async_trait]
impl LayoutStrategy for EncryptingStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let segment_sink = Arc::new(EncryptingSegmentSink::new(segment_sink, self.key.clone()));
        self.child
            .write_stream(ctx, segment_sink, stream, eof, session)
            .await
    }
}
