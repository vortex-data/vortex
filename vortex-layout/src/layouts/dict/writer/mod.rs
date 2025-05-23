use std::future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, ready};

use bytes::Bytes;
use futures::channel::oneshot;
use futures::stream::{iter, once};
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use vortex_array::vtable::EncodingVTable as _;
use vortex_array::{Array, ArrayContext, ArrayRef, ProstMetadata, SerializeMetadata};
use arcref::ArcRef;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_dict::DictEncoding;
use vortex_dict::builders::{DictConstraints, DictEncoder, dict_encoder};
use vortex_dtype::{DType, PType};
use vortex_error::{VortexExpect, VortexResult, vortex_err};

use crate::segments::SegmentWriter;
use crate::sequence::{SequenceId, SequencePointer};
use crate::{LayoutRef, LayoutStrategy, LayoutWriter, SequentialArrayStream};

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
    options: DictLayoutOptions,
    codes: ArcRef<dyn LayoutStrategy>,
    values: ArcRef<dyn LayoutStrategy>,
    fallback: ArcRef<dyn LayoutStrategy>,
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
        dtype: &DType,
        segment_writer: Arc<dyn SegmentWriter>,
        stream: SequentialArrayStream,
    ) -> Pin<Box<dyn LayoutWriter>> {
        if !dict_layout_supported(dtype) {
            return self
                .fallback
                .write_stream(ctx, dtype, segment_writer, stream);
        }
        let codes = self.codes.clone();
        let values = self.values.clone();
        let fallback = self.fallback.clone();
        let ctx = ctx.clone();
        let dtype = dtype.clone();
        let constraints = self.options.constraints.clone();
        Box::pin(async move {
            // 0. decide if chunks are eligible for dict encoding
            let (stream, is_dict_encoding) = call_for_first_item(stream, |chunk| {
                let compressed = BtrBlocksCompressor.compress(chunk)?;
                Ok(compressed.is_encoding(DictEncoding.id()))
            })
            .await;
            if !is_dict_encoding? {
                // first chunk did not compress to dict, skip dict layout
                return fallback
                    .write_stream(&ctx, &dtype, segment_writer.clone(), stream)
                    .await;
            }

            // 1. from a chunk stream, create a stream that yields codes
            // followed by a single value chunk when dict constraints are hit.
            // (a1, a2) -> (code(c1), code(c2), values(v1), code(c3), ...)
            let dict_stream = dict_encode_stream(stream, constraints);

            // 2. get contiguous runs of codes from the dict stream and
            // create child dict layouts from them.
            let mut children = Vec::new();
            let mut runs = DictEncodedRuns {
                input: Arc::new(Mutex::new(dict_stream)),
                exhausted: Default::default(),
            };
            while let Some((codes_stream, values_future)) = runs.next_run() {
                let (codes_stream, codes_dtype) =
                    call_for_first_item(codes_stream, |chunk| Ok(chunk.dtype().clone())).await;
                let Ok(codes_dtype) = codes_dtype else {
                    // codes_stream is empty, this would happen if the parent stream end coincided with a dict run end
                    break;
                };
                let codes_layout = codes
                    .write_stream(&ctx, &codes_dtype, segment_writer.clone(), codes_stream)
                    .await?;
                let values_layout = values
                    .write_stream(
                        &ctx,
                        &dtype,
                        segment_writer.clone(),
                        Box::pin(once(values_future)),
                    )
                    .await?;
                children.push(dict_layout(values_layout, codes_layout)?);
            }
            if children.len() == 1 {
                return Ok(children.remove(0));
            }

            let row_count = children.iter().map(|child| child.row_count()).sum();
            Ok(chunked_layout(dtype.clone(), row_count, children))
        })
    }
}

type DictionaryStream = Pin<Box<dyn Stream<Item = VortexResult<DictionaryChunk>> + Send>>;

enum DictionaryChunk {
    Codes((SequenceId, ArrayRef)),
    Values((SequenceId, ArrayRef)),
}

fn dict_encode_stream(
    input: SequentialArrayStream,
    constraints: DictConstraints,
) -> DictionaryStream {
    let state = Arc::new(Mutex::new(DictStreamState {
        encoder: None,
        labeler: None,
        constraints,
    }));
    Box::pin(
        input
            .scan(state.clone(), |state, item| {
                future::ready(Some(state.lock().encode(item)))
            })
            .chain(once(async move { state.lock().drain_values() }))
            .flat_map(|chunks| iter(chunks.into_iter())),
    )
}

struct DictStreamState {
    encoder: Option<Box<dyn DictEncoder>>,
    labeler: Option<DictChunkLabeler>,
    constraints: DictConstraints,
}

impl DictStreamState {
    fn encode(
        &mut self,
        item: VortexResult<(SequenceId, ArrayRef)>,
    ) -> Vec<VortexResult<DictionaryChunk>> {
        match self.try_encode(item) {
            Ok(chunks) => chunks,
            Err(e) => vec![Err(e)],
        }
    }

    fn try_encode(
        &mut self,
        item: VortexResult<(SequenceId, ArrayRef)>,
    ) -> VortexResult<Vec<VortexResult<DictionaryChunk>>> {
        let (sequence_id, chunk) = item?;
        let mut labeler = DictChunkLabeler::new(sequence_id);
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
        self.labeler = Some(labeler);
        Ok(res)
    }

    fn drain_values(&mut self) -> Vec<VortexResult<DictionaryChunk>> {
        match (self.encoder.as_mut(), self.labeler.as_mut()) {
            (None, _) => Vec::new(),
            (Some(_), None) => vec![Err(vortex_err!(
                "invalid state, if encoded we must have a labeler"
            ))],
            (Some(encoder), Some(labeler)) => vec![encoder.values().map(|val| labeler.values(val))],
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
    input: Arc<Mutex<DictionaryStream>>,
    exhausted: Arc<AtomicBool>,
}

impl DictEncodedRuns {
    fn next_run(
        &mut self,
    ) -> Option<(
        SequentialArrayStream,
        impl Future<Output = VortexResult<(SequenceId, ArrayRef)>> + use<>,
    )> {
        if self.exhausted.load(Ordering::SeqCst) {
            return None;
        }

        let (values_tx, values_rx) = oneshot::channel();
        let values_future = async {
            match values_rx.await {
                Ok(values) => values,
                Err(_) => Err(vortex_err!("sender dropped")),
            }
        };
        let codes_stream = Box::pin(DictEncodedRunStream {
            input: self.input.clone(),
            values_tx: Some(values_tx),
            exhausted: self.exhausted.clone(),
        });

        Some((codes_stream, values_future))
    }
}

struct DictEncodedRunStream {
    input: Arc<Mutex<DictionaryStream>>,
    values_tx: Option<oneshot::Sender<VortexResult<(SequenceId, ArrayRef)>>>,
    exhausted: Arc<AtomicBool>,
}

impl Stream for DictEncodedRunStream {
    type Item = VortexResult<(SequenceId, ArrayRef)>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let poll_result = ready!(self.input.lock().as_mut().poll_next(cx));
        match poll_result {
            Some(Ok(DictionaryChunk::Codes(item))) => Poll::Ready(Some(Ok(item))),
            Some(Ok(DictionaryChunk::Values(item))) => {
                // ignore receiver drops
                let _ = self
                    .values_tx
                    .take()
                    .vortex_expect("must not be polled after returning None")
                    .send(Ok(item));
                Poll::Ready(None)
            }
            Some(Err(e)) => Poll::Ready(Some(Err(e))),
            None => {
                self.exhausted.store(true, Ordering::SeqCst);
                Poll::Ready(None)
            }
        }
    }
}

async fn call_for_first_item<T>(
    mut stream: SequentialArrayStream,
    func: impl Fn(&ArrayRef) -> VortexResult<T>,
) -> (SequentialArrayStream, VortexResult<T>) {
    let Some(Ok((sequence_id, first_chunk))) = stream.next().await else {
        return (stream, Err(vortex_err!("empty stream")));
    };
    let res = func(&first_chunk);
    // reconstruct the stream
    let stream = Box::pin(once(async { Ok((sequence_id, first_chunk)) }).chain(stream));
    (stream, res)
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

fn dict_layout(values: Layout, codes: Layout) -> VortexResult<Layout> {
    let metadata = Bytes::from(
        ProstMetadata(DictLayoutMetadata::new(codes.dtype().try_into()?))
            .serialize()
            .ok_or_else(|| vortex_err!("could not serialize dict layout metadata"))?,
    );
    Ok(Layout::new_owned(
        "dict".into(),
        LayoutVTableRef::new_ref(&DictLayout),
        values.dtype().clone(),
        codes.row_count(),
        vec![],
        vec![values, codes],
        Some(metadata),
    ))
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
