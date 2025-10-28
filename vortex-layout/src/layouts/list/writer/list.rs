// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::future::try_join_all;
use futures::pin_mut;
use vortex_array::Array;
use vortex_array::ArrayContext;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::list_from_list_view;
use vortex_array::serde::SerializeOptions;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::kanal_ext::KanalExt;
use vortex_io::runtime::Handle;

use crate::IntoLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::flat::FlatLayout;
use crate::layouts::list::ListLayout;
use crate::layouts::list::ListLayoutInner;
use crate::segments::SegmentId;
use crate::segments::SegmentSink;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStream;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Strategy for writing List and FixedSizeList arrays to a sink.
pub struct ListStrategy {
    offsets_strategy: Arc<dyn LayoutStrategy>,
    elements_strategy: Arc<dyn LayoutStrategy>,
    validity_strategy: Arc<dyn LayoutStrategy>,
}

impl ListStrategy {
    pub fn new(
        offsets_strategy: Arc<dyn LayoutStrategy>,
        elements_strategy: Arc<dyn LayoutStrategy>,
        validity_strategy: Arc<dyn LayoutStrategy>,
    ) -> Self {
        Self {
            offsets_strategy,
            elements_strategy,
            validity_strategy,
        }
    }
}

#[async_trait]
impl LayoutStrategy for ListStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        handle: Handle,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        let elements_dtype = dtype
            .as_list_element_opt()
            .vortex_expect("must be List type")
            .clone();

        // Unfortunately, we need to wait for the first chunk of data to be available before we can
        // find out the DType of the offsets stream. We wait for a chunk and peek it here so we
        // can finish setting up the stream.
        let stream = stream.peekable();
        pin_mut!(stream);

        let offsets_dtype = match stream.as_mut().peek().await {
            None => {
                // If input stream is empty, we emit a single Flat segment with an empty array
                let sequence_id = eof.downgrade();
                let segment_id =
                    write_empty_list(&dtype, sequence_id, &ctx, segment_sink.as_ref()).await?;
                return Ok(FlatLayout::new(0, dtype, segment_id, ctx).into_layout());
            }
            Some(first) => match first {
                Ok((_, chunk)) => {
                    // Figure out the list DType.
                    let chunk = list_from_list_view(chunk.to_listview());
                    chunk.offsets().dtype().clone()
                }
                Err(err) => {
                    vortex_bail!("failed to access first chunk: {err}")
                }
            },
        };

        let offsets_eof = eof.split_off();
        let elements_eof = eof.split_off();

        let (offsets_tx, offsets_rx) = kanal::bounded_async(1);
        let (elements_tx, elements_rx) = kanal::bounded_async(1);
        let (validity_tx, validity_rx) = kanal::bounded_async(1);

        let mut tasks = Vec::new();

        // Spawn offsets writing.
        let offsets_task = handle.spawn_nested(|h| {
            let offsets_strategy = self.offsets_strategy.clone();
            let ctx = ctx.clone();
            let segment_sink = segment_sink.clone();
            let stream =
                SequentialStreamAdapter::new(offsets_dtype, offsets_rx.into_stream().boxed())
                    .sendable();
            async move {
                offsets_strategy
                    .write_stream(ctx, segment_sink, stream, offsets_eof, h)
                    .await
            }
        });

        tasks.push(offsets_task);

        let elements_task = handle.spawn_nested(|h| {
            let elements_strategy = self.elements_strategy.clone();
            let ctx = ctx.clone();
            let segment_sink = segment_sink.clone();
            let stream = SequentialStreamAdapter::new(
                elements_dtype.deref().clone(),
                elements_rx.into_stream().boxed(),
            )
            .sendable();
            async move {
                elements_strategy
                    .write_stream(ctx, segment_sink, stream, elements_eof, h)
                    .await
            }
        });

        tasks.push(elements_task);

        // Push a separate task for writing validity stream
        if dtype.is_nullable() {
            let validity_eof = eof.split_off();

            let validity_task = handle.spawn_nested(|h| {
                let validity_strategy = self.validity_strategy.clone();
                let ctx = ctx.clone();
                let segment_sink = segment_sink.clone();
                let stream = SequentialStreamAdapter::new(
                    DType::Bool(Nullability::NonNullable),
                    validity_rx.into_stream().boxed(),
                )
                .sendable();

                async move {
                    validity_strategy
                        .write_stream(ctx, segment_sink, stream, validity_eof, h)
                        .await
                }
            });

            tasks.push(validity_task);
        }

        // Pump chunks to the output nodes.
        let mut row_count = 0;
        while let Some((sequence_id, chunk)) = stream.next().await.transpose()? {
            row_count += chunk.len() as u64;

            let list = list_from_list_view(chunk.to_listview());
            let offsets = list.offsets().clone();
            let elements = list.elements().clone();

            let mut sequence_pointer = sequence_id.descend();

            drop(
                offsets_tx
                    .send(Ok((sequence_pointer.advance(), offsets)))
                    .await,
            );
            drop(elements_tx.send(Ok((sequence_pointer.advance(), elements))));

            if dtype.is_nullable() {
                let validity = chunk.validity_mask().into_array();
                drop(
                    validity_tx
                        .send(Ok((sequence_pointer.advance(), validity)))
                        .await,
                );
            }
        }

        // Join the offsets and elements tasks
        let mut layouts = try_join_all(tasks).await?;
        let offsets_layout = layouts.remove(0);
        let elements_layout = layouts.remove(0);
        let validity_layout = if dtype.is_nullable() {
            Some(layouts.remove(0))
        } else {
            None
        };

        let elements_count = elements_layout.row_count();

        // Write the list layout back out to disk.
        Ok(ListLayout {
            dtype,
            row_count,
            elements_count,
            inner: Arc::new(ListLayoutInner::List {
                offsets: offsets_layout,
                elements: elements_layout,
                validity: validity_layout,
            }),
        }
        .into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        self.offsets_strategy.buffered_bytes()
            + self.elements_strategy.buffered_bytes()
            + self.validity_strategy.buffered_bytes()
    }
}

async fn write_empty_list(
    dtype: &DType,
    sequence_id: SequenceId,
    ctx: &ArrayContext,
    sink: &dyn SegmentSink,
) -> VortexResult<SegmentId> {
    let empty_list = Canonical::empty(&dtype).into_array();

    let buffers = empty_list.serialize(ctx, &SerializeOptions::default())?;
    sink.write(sequence_id, buffers).await
}

#[cfg(test)]
mod tests {
    use crate::LayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::list::writer::ListStrategy;
    use crate::layouts::struct_::writer::StructStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::{SequenceId, SequentialArrayStreamExt};
    use crate::test::SESSION;
    use std::sync::Arc;
    use vortex_array::arrays::{ListArray, StructArray};
    use vortex_array::validity::Validity;
    use vortex_array::{ArrayContext, IntoArray};
    use vortex_buffer::buffer;
    use vortex_dtype::FieldNames;
    use vortex_io::session::RuntimeSessionExt;

    #[tokio::test(flavor = "multi_thread", worker_threads = 3)]
    async fn test_simple() {
        let elements = StructArray::new(
            FieldNames::from(["a", "b"]),
            vec![
                buffer![0, 1, 2, 3, 4].into_array(),
                buffer![0, 1, 2, 3, 4].into_array(),
            ],
            5,
            Validity::NonNullable,
        )
        .into_array();

        // lengths: 2, 1, 3
        let offsets = buffer![0, 2, 3, 5].into_array();

        let list = ListArray::new(elements, offsets, Validity::NonNullable).into_array();

        // Make a new strategy for writing here.
        let segments = Arc::new(TestSegments::default());
        let writer = Arc::new(ListStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(StructStrategy::new(
                FlatLayoutStrategy::default(),
                FlatLayoutStrategy::default(),
            )),
            Arc::new(FlatLayoutStrategy::default()),
        ));

        let (ptr, eof) = SequenceId::root().split();
        let stream = list.to_array_stream().sequenced(ptr);

        let summary = writer
            .write_stream(
                ArrayContext::empty(),
                segments.clone(),
                stream,
                eof,
                SESSION.handle(),
            )
            .await
            .unwrap();

        // Read the segments
    }
}
