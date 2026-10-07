// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A writer strategy for shredded Variant arrays.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::try_join3;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::variant::VariantArraySlotsExt;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_io::kanal_ext::KanalExt;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::layouts::variant::VariantLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Writes shredded Variant arrays into a [`VariantLayout`].
///
/// Every chunk is canonicalized to a Variant array. When the chunks carry a shredded tree, the
/// core storage is written through `core` and the shredded tree through `shredded`, which should
/// decompose structs into columns. Unshredded Variant streams are written through `core` alone.
#[derive(Clone)]
pub struct VariantStrategy {
    core: Arc<dyn LayoutStrategy>,
    shredded: Arc<dyn LayoutStrategy>,
}

impl VariantStrategy {
    /// A Variant writer writing core storage through `core` and shredded trees through `shredded`.
    pub fn new(core: Arc<dyn LayoutStrategy>, shredded: Arc<dyn LayoutStrategy>) -> Self {
        Self { core, shredded }
    }
}

#[async_trait]
impl LayoutStrategy for VariantStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        if !dtype.is_variant() {
            vortex_bail!("VariantStrategy can only write Variant streams, got {dtype}");
        }

        let canonical_session = session.clone();
        let mut canonical = Box::pin(
            stream
                .map(move |chunk| {
                    let (sequence_id, chunk) = chunk?;
                    let mut exec_ctx = canonical_session.create_execution_ctx();
                    Ok((sequence_id, chunk.execute::<VariantArray>(&mut exec_ctx)?))
                })
                .peekable(),
        );

        // The first chunk decides the shredded dtype every chunk must share.
        let shredded_dtype = match canonical.as_mut().peek().await {
            Some(Ok((_, first))) => first.shredded().map(|shredded| shredded.dtype().clone()),
            Some(Err(_)) | None => None,
        };
        let Some(shredded_dtype) = shredded_dtype else {
            let unshredded =
                canonical.map_ok(|(sequence_id, chunk)| (sequence_id, chunk.into_array()));
            let stream = SequentialStreamAdapter::new(dtype, unshredded.boxed()).sendable();
            return self
                .core
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        };

        let (core_tx, core_rx) = kanal::bounded_async(1);
        let (shredded_tx, shredded_rx) = kanal::bounded_async(1);
        let expected_shredded = shredded_dtype.clone();
        let fanout = async move {
            while let Some(chunk) = canonical.next().await {
                let (core, shredded) = match chunk.and_then(|(sequence_id, chunk)| {
                    let shredded = chunk.shredded().cloned().ok_or_else(|| {
                        vortex_error::vortex_err!(
                            "Variant chunks disagree on whether they are shredded"
                        )
                    })?;
                    vortex_ensure!(
                        shredded.dtype() == &expected_shredded,
                        "Variant chunks disagree on the shredded dtype: {} vs {}",
                        shredded.dtype(),
                        expected_shredded
                    );
                    let mut pointer = sequence_id.descend();
                    Ok((
                        (pointer.advance(), chunk.core_storage().clone()),
                        (pointer.advance(), shredded),
                    ))
                }) {
                    Ok(columns) => columns,
                    Err(err) => {
                        let err = Arc::new(err);
                        let _ = core_tx.send(Err(VortexError::from(Arc::clone(&err)))).await;
                        let _ = shredded_tx
                            .send(Err(VortexError::from(Arc::clone(&err))))
                            .await;
                        return Err(VortexError::from(err));
                    }
                };
                if core_tx.send(Ok::<_, VortexError>(core)).await.is_err()
                    || shredded_tx.send(Ok(shredded)).await.is_err()
                {
                    vortex_bail!("Variant child writer finished before all chunks were sent");
                }
            }
            Ok(())
        };

        let handle = session.handle();
        let core_stream =
            SequentialStreamAdapter::new(dtype.clone(), core_rx.into_stream().boxed()).sendable();
        let core_writer = Arc::clone(&self.core);
        let core_eof = eof.split_off();
        let core_ctx = ctx.clone();
        let core_sink = Arc::clone(&segment_sink);
        let core_session = session.clone();
        let core_layout = handle.spawn_nested(move |_| async move {
            core_writer
                .write_stream(core_ctx, core_sink, core_stream, core_eof, &core_session)
                .await
        });

        let shredded_stream =
            SequentialStreamAdapter::new(shredded_dtype.clone(), shredded_rx.into_stream().boxed())
                .sendable();
        let shredded_writer = Arc::clone(&self.shredded);
        let shredded_eof = eof.split_off();
        let shredded_session = session.clone();
        let shredded_layout = handle.spawn_nested(move |_| async move {
            shredded_writer
                .write_stream(
                    ctx,
                    segment_sink,
                    shredded_stream,
                    shredded_eof,
                    &shredded_session,
                )
                .await
        });

        let ((), core, shredded) = try_join3(fanout, core_layout, shredded_layout).await?;
        Ok(
            VariantLayout::new(core.row_count(), dtype, shredded_dtype, core, shredded)
                .into_layout(),
        )
    }
}
