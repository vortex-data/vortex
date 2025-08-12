// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};

use async_stream::try_stream;
use async_trait::async_trait;
use futures::channel::{mpsc, oneshot};
use futures::stream::{BoxStream, once};
use futures::{FutureExt, SinkExt, Stream, StreamExt, pin_mut, try_join};
use vortex_array::{Array, ArrayContext, ArrayRef};
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_dict::DictEncoding;
use vortex_dict::builders::{DictConstraints, DictEncoder, dict_encoder};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap, vortex_bail, vortex_err};

use super::DictLayout;
use crate::layouts::chunked::ChunkedLayout;
use crate::segments::SequenceWriter;
use crate::sequence::{SequenceId, SequencePointer};
use crate::{
    IntoLayout, LayoutRef, LayoutStrategy, OwnedLayoutChildren, SendableSequentialStream,
    SequentialStreamAdapter, SequentialStreamExt, TaskExecutor, TaskExecutorExt as _,
};

#[derive(Clone)]
pub struct DictLayoutOptions {
    pub constraints: DictConstraints,
    /// Max number of encoded chunks to keep in memory.
    pub encoded_buffer_size: usize,
}

impl Default for DictLayoutOptions {
    fn default() -> Self {
        Self {
            constraints: DictConstraints {
                max_bytes: 1024 * 1024,
                max_len: u16::MAX as usize,
            },
            encoded_buffer_size: 8,
        }
    }
}

/// A layout strategy that encodes chunk into values and codes, if found
/// appropriate by the btrblocks compressor. Current implementation only
/// checks the first chunk to decide whether to apply dict layout and
/// encodes chunks into dictionaries. When the dict constraints are hit, a
/// new dictionary is created.
#[derive(Clone)]
pub struct DictStrategy<Codes, Values, Fallback> {
    codes: Codes,
    values: Values,
    fallback: Fallback,
    options: DictLayoutOptions,
    executor: Arc<dyn TaskExecutor>,
}

impl<Codes, Values, Fallback> DictStrategy<Codes, Values, Fallback>
where
    Codes: LayoutStrategy,
    Values: LayoutStrategy,
    Fallback: LayoutStrategy,
{
    pub fn new(
        codes: Codes,
        values: Values,
        fallback: Fallback,
        options: DictLayoutOptions,
        executor: Arc<dyn TaskExecutor>,
    ) -> Self {
        Self {
            codes,
            values,
            fallback,
            options,
            executor,
        }
    }
}

#[async_trait]
impl<Codes, Values, Fallback> LayoutStrategy for DictStrategy<Codes, Values, Fallback>
where
    Codes: LayoutStrategy,
    Values: LayoutStrategy,
    Fallback: LayoutStrategy,
{
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        stream: SendableSequentialStream,
    ) -> VortexResult<LayoutRef> {
        if !dict_layout_supported(stream.dtype()) {
            return self
                .fallback
                .write_stream(ctx, sequence_writer, stream)
                .await;
        }
        let ctx = ctx.clone();
        let options = self.options.clone();
        let dtype = stream.dtype().clone();
        let executor = self.executor.clone();
        // 0. decide if chunks are eligible for dict encoding
        let (stream, first_chunk) = peek_first_chunk(stream).await?;
        let stream = SequentialStreamAdapter::new(dtype.clone(), stream).sendable();

        let should_fallback = match first_chunk {
            None => true, // empty stream
            Some(chunk) => {
                let compressed = BtrBlocksCompressor.compress(&chunk)?;
                !compressed.is_encoding(DictEncoding.id())
            }
        };
        if should_fallback {
            // first chunk did not compress to dict, or did not exist. Skip dict layout
            return self
                .fallback
                .write_stream(&ctx, sequence_writer.clone(), stream)
                .await;
        }

        // 1. from a chunk stream, create a stream that yields codes
        // followed by a single value chunk when dict constraints are hit.
        // (a1, a2) -> (code(c1), code(c2), values(v1), code(c3), ...)
        let mut dict_stream = dict_encode_stream(stream, options.constraints);

        // 2.a spawn encoding codes
        let (mut encoded_tx, encoded_rx) = mpsc::channel(options.encoded_buffer_size);
        let encode_handle = executor.spawn({
            async move {
                while let Some(item) = dict_stream.next().await {
                    encoded_tx
                        .send(item)
                        .await
                        .map_err(|e| vortex_err!("rx dropped: {}", e))?;
                }
                Ok(())
            }
            .boxed()
        });

        // 2.b get contiguous runs of codes from the dict stream and
        // create child dict layouts from them.
        let dtype_clone = dtype.clone();
        let child_layouts_fut = async move {
            let mut children = Vec::new();
            let mut runs = DictEncodedRuns::new(Box::pin(encoded_rx));
            while let Some((codes_stream, values_future)) = runs.next_run().await {
                let (codes_stream, first_chunk) = peek_first_chunk(codes_stream.boxed()).await?;
                let codes_dtype = match first_chunk {
                    // codes_stream is empty, this would happen if the parent stream end coincided with a dict run end
                    None => break,
                    Some(chunk) => chunk.dtype().clone(),
                };
                let codes_layout = self
                    .codes
                    .write_stream(
                        &ctx,
                        sequence_writer.clone(),
                        SequentialStreamAdapter::new(codes_dtype, codes_stream).sendable(),
                    )
                    .await?;
                let values_layout = self
                    .values
                    .write_stream(
                        &ctx,
                        sequence_writer.clone(),
                        SequentialStreamAdapter::new(dtype_clone.clone(), once(values_future))
                            .sendable(),
                    )
                    .await?;
                children.push(DictLayout::new(values_layout, codes_layout).into_layout());
            }
            Ok(children)
        };

        // join dict encoding task
        let (mut children, _) = try_join!(child_layouts_fut, encode_handle)?;

        if children.len() == 1 {
            return Ok(children.remove(0));
        }

        let row_count = children.iter().map(|child| child.row_count()).sum();
        Ok(ChunkedLayout::new(
            row_count,
            dtype,
            OwnedLayoutChildren::layout_children(children),
        )
        .into_layout())
    }
}

enum DictionaryChunk {
    Codes((SequenceId, ArrayRef)),
    Values((SequenceId, ArrayRef)),
}

type DictionaryStream = BoxStream<'static, VortexResult<DictionaryChunk>>;

fn dict_encode_stream(
    input: SendableSequentialStream,
    constraints: DictConstraints,
) -> DictionaryStream {
    Box::pin(try_stream! {
        let mut state = DictStreamState {
            encoder: None,
            constraints,
        };
        let input = input.peekable();
        pin_mut!(input);
        while let Some(item) = input.as_mut().next().await {
            let (sequence_id, chunk) = item?;
            // labeler potentially creates sub sequences, we must
            // create it on both arms to avoid having a SequencePointer
            // between await points
            match input.as_mut().peek().await {
                Some(_) => {
                    let mut labeler = DictChunkLabeler::new(sequence_id);
                    let chunks = state.encode(&mut labeler, chunk);
                    drop(labeler);
                    for dict_chunk in chunks {
                        yield dict_chunk?;
                    }
                }
                None => {
                    // this is the last element, encode and drain chunks
                    let mut labeler = DictChunkLabeler::new(sequence_id);
                    let encoded = state.encode(&mut labeler, chunk);
                    let drained = state.drain_values(&mut labeler);
                    drop(labeler);
                    for dict_chunk in encoded.into_iter().chain(drained.into_iter()) {
                        yield dict_chunk?;
                    }
                }
            }
        }
    })
}

struct DictStreamState {
    encoder: Option<Box<dyn DictEncoder>>,
    constraints: DictConstraints,
}

impl DictStreamState {
    fn encode(
        &mut self,
        labeler: &mut DictChunkLabeler,
        chunk: ArrayRef,
    ) -> Vec<VortexResult<DictionaryChunk>> {
        self.try_encode(labeler, chunk)
            .unwrap_or_else(|e| vec![Err(e)])
    }

    fn try_encode(
        &mut self,
        labeler: &mut DictChunkLabeler,
        chunk: ArrayRef,
    ) -> VortexResult<Vec<VortexResult<DictionaryChunk>>> {
        let mut res = Vec::new();
        let mut to_be_encoded = Some(chunk);
        while let Some(remaining) = to_be_encoded.take() {
            match self.encoder.take() {
                None => match start_encoding(&self.constraints, &remaining)? {
                    EncodingState::Continue((encoder, encoded)) => {
                        res.push(Ok(labeler.codes(encoded)));
                        self.encoder = Some(encoder);
                    }
                    EncodingState::Done((values, encoded, unencoded)) => {
                        res.push(Ok(labeler.codes(encoded)));
                        res.push(Ok(labeler.values(values)));
                        to_be_encoded = Some(unencoded);
                    }
                },
                Some(encoder) => match encode_chunk(encoder, &remaining)? {
                    EncodingState::Continue((encoder, encoded)) => {
                        res.push(Ok(labeler.codes(encoded)));
                        self.encoder = Some(encoder);
                    }
                    EncodingState::Done((values, encoded, unencoded)) => {
                        res.push(Ok(labeler.codes(encoded)));
                        res.push(Ok(labeler.values(values)));
                        to_be_encoded = Some(unencoded);
                    }
                },
            }
        }
        Ok(res)
    }

    fn drain_values(
        &mut self,
        labeler: &mut DictChunkLabeler,
    ) -> Vec<VortexResult<DictionaryChunk>> {
        match self.encoder.as_mut() {
            None => Vec::new(),
            Some(encoder) => vec![encoder.values().map(|val| labeler.values(val))],
        }
    }
}

struct DictChunkLabeler {
    sequence_pointer: SequencePointer,
}

impl DictChunkLabeler {
    fn new(starting_id: SequenceId) -> Self {
        let sequence_pointer = starting_id.descend();
        Self { sequence_pointer }
    }

    fn codes(&mut self, chunk: ArrayRef) -> DictionaryChunk {
        DictionaryChunk::Codes((self.sequence_pointer.advance(), chunk))
    }

    fn values(&mut self, chunk: ArrayRef) -> DictionaryChunk {
        DictionaryChunk::Values((self.sequence_pointer.advance(), chunk))
    }
}

type SequencedChunk = VortexResult<(SequenceId, ArrayRef)>;

struct DictEncodedRuns {
    input: Option<oneshot::Receiver<Option<DictionaryStream>>>,
}

impl DictEncodedRuns {
    fn new(input: DictionaryStream) -> Self {
        let (tx, rx) = oneshot::channel();
        tx.send(Some(input))
            .map_err(|_input| vortex_err!("just created rx"))
            .vortex_unwrap();
        Self { input: Some(rx) }
    }

    async fn next_run(
        &mut self,
    ) -> Option<(
        DictEncodedRunStream,
        impl Future<Output = SequencedChunk> + use<>,
    )> {
        // get input to send to the run stream.
        let Ok(Some(input)) = self.input.take()?.await else {
            // input exhausted
            return None;
        };
        let (input_tx, input_rx) = oneshot::channel();
        self.input = Some(input_rx);

        let (values_tx, values_rx) = oneshot::channel();
        let values_future = async {
            values_rx
                .await
                .unwrap_or_else(|_| vortex_bail!("sender dropped"))
        };

        let codes_stream = DictEncodedRunStream {
            input: Some(input),
            input_tx: Some(input_tx),
            values_tx: Some(values_tx),
        };

        Some((codes_stream, values_future))
    }
}

struct DictEncodedRunStream {
    input: Option<DictionaryStream>,
    input_tx: Option<oneshot::Sender<Option<DictionaryStream>>>,
    values_tx: Option<oneshot::Sender<SequencedChunk>>,
}

impl Stream for DictEncodedRunStream {
    type Item = SequencedChunk;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let poll_result = {
            let Some(stream) = self.input.as_mut() else {
                return Poll::Ready(None);
            };
            ready!(stream.poll_next_unpin(cx))
        };

        match poll_result {
            Some(Ok(DictionaryChunk::Codes(item))) => Poll::Ready(Some(Ok(item))),
            Some(Ok(DictionaryChunk::Values(item))) => {
                self.send_values(item);
                self.send_back_input_stream();
                Poll::Ready(None)
            }
            Some(Err(e)) => Poll::Ready(Some(Err(e))),
            None => {
                self.send_back_input_stream();
                Poll::Ready(None)
            }
        }
    }
}

impl DictEncodedRunStream {
    fn send_values(&mut self, item: (SequenceId, ArrayRef)) {
        // ignore receiver drops
        let _ = self
            .values_tx
            .take()
            .vortex_expect("must not be polled after returning None")
            .send(Ok(item));
    }

    fn send_back_input_stream(&mut self) {
        // ignore receiver drops
        let _ = self
            .input_tx
            .take()
            .vortex_expect("input already sent")
            .send(self.input.take());
    }
}

impl Drop for DictEncodedRunStream {
    fn drop(&mut self) {
        if let Some(tx) = self.input_tx.take() {
            let _ = tx.send(self.input.take());
        }
    }
}

async fn peek_first_chunk(
    mut stream: BoxStream<'static, SequencedChunk>,
) -> VortexResult<(BoxStream<'static, SequencedChunk>, Option<ArrayRef>)> {
    match stream.next().await {
        None => Ok((stream.boxed(), None)),
        Some(Err(e)) => Err(e),
        Some(Ok((sequence_id, chunk))) => {
            let chunk_clone = chunk.clone();
            let reconstructed_stream =
                once(async move { Ok((sequence_id, chunk_clone)) }).chain(stream);
            Ok((reconstructed_stream.boxed(), Some(chunk)))
        }
    }
}

pub fn dict_layout_supported(dtype: &DType) -> bool {
    matches!(
        dtype,
        DType::Primitive(..) | DType::Utf8(_) | DType::Binary(_)
    )
}

#[derive(prost::Message)]
pub struct DictLayoutMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    // i32 is required for proto, use the generated getter to read this field.
    codes_ptype: i32,
}

impl DictLayoutMetadata {
    pub fn new(codes_ptype: PType) -> Self {
        let mut metadata = Self::default();
        metadata.set_codes_ptype(codes_ptype);
        metadata
    }
}

enum EncodingState {
    Continue((Box<dyn DictEncoder>, ArrayRef)),
    // (values, encoded, unencoded)
    Done((ArrayRef, ArrayRef, ArrayRef)),
}

fn start_encoding(constraints: &DictConstraints, chunk: &dyn Array) -> VortexResult<EncodingState> {
    let encoder = dict_encoder(chunk, constraints)?;
    encode_chunk(encoder, chunk)
}

fn encode_chunk(
    mut encoder: Box<dyn DictEncoder>,
    chunk: &dyn Array,
) -> VortexResult<EncodingState> {
    let encoded = encoder.encode(chunk)?;
    Ok(match remainder(chunk, encoded.len())? {
        None => EncodingState::Continue((encoder, encoded)),
        Some(unencoded) => EncodingState::Done((encoder.values()?, encoded, unencoded)),
    })
}

fn remainder(array: &dyn Array, encoded_len: usize) -> VortexResult<Option<ArrayRef>> {
    (encoded_len < array.len())
        .then(|| array.slice(encoded_len, array.len()))
        .transpose()
}
