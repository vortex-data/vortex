// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Adapts extension streams to the struct writer without changing their row order.

use async_trait::async_trait;
use futures::StreamExt;
use vortex_array::VortexSessionExecute;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::VecExecutionArgs;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_functions::ext_storage::ExtStorage;
use vortex_session::VortexSession;

use super::ExtensionLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::layouts::struct_::StructStrategy;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Sends extension storage through the dispatcher-selected struct writer.
pub(in crate::layouts) struct ExtensionStrategy {
    storage: StructStrategy,
}

impl ExtensionStrategy {
    pub(in crate::layouts) fn new(storage: StructStrategy) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl LayoutStrategy for ExtensionStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let ext_dtype = stream
            .dtype()
            .as_extension_opt()
            .ok_or_else(|| {
                vortex_err!(
                    "extension writer requires an extension dtype, got {}",
                    stream.dtype()
                )
            })?
            .clone();
        let storage_dtype = ext_dtype.storage_dtype().clone();

        let storage_session = session.clone();
        let storage_chunks = stream.map(move |chunk| {
            let (sequence_id, chunk) = chunk?;
            let len = chunk.len();
            let storage = ExtStorage.execute(
                &EmptyOptions,
                &VecExecutionArgs::new(vec![chunk], len),
                &mut storage_session.create_execution_ctx(),
            )?;
            Ok((sequence_id, storage))
        });
        let storage_stream = SequentialStreamAdapter::new(storage_dtype, storage_chunks).sendable();

        let storage = self
            .storage
            .write_stream(ctx, segment_sink, storage_stream, eof, session)
            .await?;

        Ok(ExtensionLayout::try_new(ext_dtype, storage)?.into_layout())
    }
}
