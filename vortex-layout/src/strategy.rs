// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use vortex_array::ArrayId;
use vortex_array::normalize::NormalizeOptions;
use vortex_array::normalize::Operation;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// A layout strategy wrapper that rejects arrays containing encodings outside an allow-list.
///
/// Canonical encodings are always permitted. Every chunk is recursively validated before it is
/// passed to the wrapped strategy.
#[derive(Clone)]
pub struct LayoutStrategyEncodingValidator {
    child: Arc<dyn LayoutStrategy>,
    allowed_encodings: Arc<HashSet<ArrayId>>,
}

impl LayoutStrategyEncodingValidator {
    /// Creates a validator around `child` using the supplied encoding allow-list.
    pub fn new<S: LayoutStrategy>(child: S, allowed_encodings: HashSet<ArrayId>) -> Self {
        Self {
            child: Arc::new(child),
            allowed_encodings: Arc::new(allowed_encodings),
        }
    }
}

#[async_trait]
impl LayoutStrategy for LayoutStrategyEncodingValidator {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        let allowed_encodings = Arc::clone(&self.allowed_encodings);
        let stream = stream.map(move |chunk| {
            let (sequence_id, chunk) = chunk?;
            let chunk = chunk.normalize(&mut NormalizeOptions {
                allowed: &allowed_encodings,
                operation: Operation::Error,
            })?;
            Ok((sequence_id, chunk))
        });

        self.child
            .write_stream(
                ctx,
                segment_sink,
                SequentialStreamAdapter::new(dtype, stream).sendable(),
                eof,
                session,
            )
            .await
    }
}
