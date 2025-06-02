use std::pin::Pin;
use std::task::{Context, Poll, ready};

use arcref::ArcRef;
use async_stream::try_stream;
use futures::channel::oneshot;
use futures::stream::once;
use futures::{Stream, StreamExt, pin_mut};
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
    IntoLayout, LayoutStrategy, OwnedLayoutChildren, SendableLayoutWriter,
    SendableSequentialStream, SequentialStreamAdapter, SequentialStreamExt,
};

#[derive(Clone)]
pub struct DictLayoutOptions {
    pub constraints: DictConstraints,
}

impl Default for DictLayoutOptions {
    fn default() -> Self {
        Self {
            constraints: DictConstraints {
                max_bytes: 1024 * 1024,
                max_len: u16::MAX as usize,
            },
        }
    }
}

/// A layout strategy that encodes chunk into values and codes, if found
/// appropriate by the btrblocks compressor. Current implementation only
/// checks the first chunk to decide whether to apply dict layout and
/// encodes chunks into dictionaries. When the dict constraints are hit, a
/// new dictionary is created.
pub struct DictStrategy {
    codes: ArcRef<dyn LayoutStrategy>,
    values: ArcRef<dyn LayoutStrategy>,
    fallback: ArcRef<dyn LayoutStrategy>,
    options: DictLayoutOptions,
}

impl DictStrategy {
    pub fn new(
        codes: ArcRef<dyn LayoutStrategy>,
        values: ArcRef<dyn LayoutStrategy>,
        fallback: ArcRef<dyn LayoutStrategy>,
        options: DictLayoutOptions,
    ) -> Self {
        Self {
            codes,
            values,
            fallback,
            options,
        }
    }
}

impl LayoutStrategy for DictStrategy {
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        stream: SendableSequentialStream,
    ) -> SendableLayoutWriter {
        if !dict_layout_supported(stream.dtype()) {
            return self.fallback.write_stream(ctx, sequence_writer, stream);
        }
        let codes = self.codes.clone();
        let values = self.values.clone();
        let fallback = self.fallback.clone();
        let ctx = ctx.clone();
        let constraints = self.options.constraints.clone();
        let dtype = stream.dtype().clone();
        Box::pin(async move {
            // 0. decide if chunks are eligible for dict encoding
            let (stream, is_dict_encoding) = call_for_first_item(stream, |chunk| {
                let compressed = BtrBlocksCompressor.compress(chunk)?;
                Ok(compressed.is_encoding(DictEncoding.id()))
            })
            .await;
            let stream = SequentialStreamAdapter::new(dtype.clone(), stream).sendable();
            if !is_dict_encoding? {
                // first chunk did not compress to dict, skip dict layout
                return fallback
                    .write_stream(&ctx, sequence_writer.clone(), stream)
                    .await;
            }

            // 1. from a chunk stream, create a stream that yields codes
            // followed by a single value chunk when dict constraints are hit.
            // (a1, a2) -> (code(c1), code(c2), values(v1), code(c3), ...)
            let dict_stream = dict_encode_stream(stream, constraints);

            // 2. get contiguous runs of codes from the dict stream and
            // create child dict layouts from them.
            let mut children = Vec::new();
            let mut runs = DictEncodedRuns::new(dict_stream);
            while let Some((codes_stream, values_future)) = runs.next_run().await {
                let (codes_stream, codes_dtype) =
                    call_for_first_item(codes_stream.boxed(), |chunk| Ok(chunk.dtype().clone()))
                        .await;
                let Ok(codes_dtype) = codes_dtype else {
                    // codes_stream is empty, this would happen if the parent stream end coincided with a dict run end
                    break;
                };
                let codes_layout = codes
                    .write_stream(
                        &ctx,
                        sequence_writer.clone(),
                        SequentialStreamAdapter::new(codes_dtype, codes_stream).sendable(),
                    )
                    .await?;
                let values_layout = values
                    .write_stream(
                        &ctx,
                        sequence_writer.clone(),
                        SequentialStreamAdapter::new(dtype.clone(), once(values_future)).sendable(),
                    )
                    .await?;
                children.push(DictLayout::new(values_layout, codes_layout).into_layout());
            }
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
        })
    }
}

type DictionaryStream = Pin<Box<dyn Stream<Item = VortexResult<DictionaryChunk>> + Send>>;

enum DictionaryChunk {
    Codes((SequenceId, ArrayRef)),
    Values((SequenceId, ArrayRef)),
}

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
        impl Future<Output = VortexResult<(SequenceId, ArrayRef)>> + use<>,
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
    values_tx: Option<oneshot::Sender<VortexResult<(SequenceId, ArrayRef)>>>,
}

impl Stream for DictEncodedRunStream {
    type Item = VortexResult<(SequenceId, ArrayRef)>;

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

type Sequenced = Pin<Box<dyn Stream<Item = VortexResult<(SequenceId, ArrayRef)>> + Send>>;
async fn call_for_first_item<T>(
    mut stream: Sequenced,
    func: impl Fn(&ArrayRef) -> VortexResult<T>,
) -> (Sequenced, VortexResult<T>) {
    let Some(Ok((sequence_id, first_chunk))) = stream.next().await else {
        return (stream.boxed(), Err(vortex_err!("empty stream")));
    };
    let res = func(&first_chunk);
    // reconstruct the stream
    let stream = once(async { Ok((sequence_id, first_chunk)) }).chain(stream);
    (stream.boxed(), res)
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
