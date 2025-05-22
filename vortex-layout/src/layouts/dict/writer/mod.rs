use std::future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};

use async_trait::async_trait;
use bytes::Bytes;
use futures::future::BoxFuture;
use futures::stream::{self, once};
use futures::{Stream, StreamExt};
use pin_project_lite::pin_project;
use vortex_array::vtable::EncodingVTable as _;
use vortex_array::{Array, ArrayContext, ArrayRef, ProstMetadata, SerializeMetadata};
use arcref::ArcRef;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_dict::DictEncoding;
use vortex_dict::builders::{DictConstraints, DictEncoder, dict_encoder};
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

mod repeating;

use crate::segments::NewSegmentWriter;
use crate::sequence::{SequenceId, SequencePointer};
use crate::{
    LayoutRef, LayoutStrategy, LayoutWriter, LayoutWriterExt, NewLayoutStrategy,
    NewLayoutWriter, SequentialArrayStream,
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

pub struct NewDictStrategy {
    options: DictLayoutOptions,
    codes: ArcRef<dyn NewLayoutStrategy>,
    values: ArcRef<dyn NewLayoutStrategy>,
    fallback: ArcRef<dyn NewLayoutStrategy>,
}

impl NewLayoutStrategy for NewDictStrategy {
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        dtype: &DType,
        segment_writer: Arc<dyn NewSegmentWriter>,
        mut stream: SequentialArrayStream,
    ) -> Pin<Box<dyn NewLayoutWriter>> {
        if !dict_layout_supported(dtype) {
            return self
                .fallback
                .write_stream(ctx, dtype, segment_writer, stream);
        }
        let fallback = self.fallback.clone();
        let ctx = ctx.clone();
        let dtype = dtype.clone();
        let constraints = self.options.constraints.clone();
        Box::pin(async move {
            let Some(Ok((sequence_id, first_chunk))) = stream.next().await else {
                vortex_bail!("need at least one chunk")
            };
            let compressed = BtrBlocksCompressor.compress(&first_chunk)?;
            // reconstruct the stream
            let stream = Box::pin(once(async { Ok((sequence_id, first_chunk)) }).chain(stream));

            if !compressed.is_encoding(DictEncoding.id()) {
                // first chunk did not compress to dict, skip dict layout
                return fallback
                    .write_stream(&ctx, &dtype, segment_writer, stream)
                    .await;
            }
            let dict_stream = dict_encode_stream(stream, constraints);
            // TODO(os): split dict_stream into runs
        })
    }
}

fn dict_encode_stream(
    input: SequentialArrayStream,
    constraints: DictConstraints,
) -> impl Stream<Item = VortexResult<DictionaryChunk>> {
    let state = DictStreamState {
        encoder: None,
        constraints,
    };
    input
        .scan(state, |state, item| future::ready(Some(state.encode(item))))
        .flat_map(|chunks| stream::iter(chunks.into_iter()))
    // TODO(os): flat map input or use try_stream!{}
}

struct DictStreamState {
    encoder: Option<Box<dyn DictEncoder>>,
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
        let mut labeler = DictChunks::new(sequence_id);
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
}

enum DictionaryChunk {
    Codes((SequenceId, ArrayRef)),
    Values((SequenceId, ArrayRef)),
}

struct DictChunks {
    sequence_pointer: SequencePointer,
}

impl DictChunks {
    fn new(starting_id: SequenceId) -> Self {
        let (_, sequence_pointer) = starting_id.descend();
        Self { sequence_pointer }
    }

    fn codes(&mut self, chunk: ArrayRef) -> DictionaryChunk {
        DictionaryChunk::Codes((self.sequence_pointer.advance(), chunk))
    }

    fn values(&mut self, chunk: ArrayRef) -> DictionaryChunk {
        DictionaryChunk::Values((self.sequence_pointer.advance(), chunk))
    }
}

/// A layout strategy that encodes chunk into values and codes, if found
/// appropriate by the btrblocks compressor. Current implementation only
/// checks the first chunk to decide whether to apply dict layout and
/// encodes chunks into dictionaries. When the dict constraints are hit, a
/// new dictionary is created.
#[derive(Clone)]
pub struct DictStrategy {
    pub options: DictLayoutOptions,
    pub codes: ArcRef<dyn LayoutStrategy>,
    pub values: ArcRef<dyn LayoutStrategy>,
    pub fallback: ArcRef<dyn LayoutStrategy>,
}

impl LayoutStrategy for DictStrategy {
    fn new_writer(&self, ctx: &ArrayContext, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        if !dict_layout_supported(dtype) {
            return self.fallback.new_writer(ctx, dtype);
        }
        Ok(DelegatingDictLayoutWriter {
            ctx: ctx.clone(),
            strategy: self.clone(),
            dtype: dtype.clone(),
            writer: None,
        }
        .boxed())
    }
}

pub fn dict_layout_supported(dtype: &DType) -> bool {
    matches!(
        dtype,
        DType::Primitive(..) | DType::Utf8(_) | DType::Binary(_)
    )
}

struct DelegatingDictLayoutWriter {
    ctx: ArrayContext,
    strategy: DictStrategy,
    dtype: DType,
    writer: Option<Box<dyn LayoutWriter>>,
}

#[async_trait]
impl LayoutWriter for DelegatingDictLayoutWriter {
    async fn push_chunk(
        &mut self,
        segment_writer: &mut dyn crate::segments::ConcurrentSegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        assert_eq!(
            chunk.dtype(),
            &self.dtype,
            "Can't push chunks of the wrong dtype into a LayoutWriter. Pushed {} but expected {}.",
            chunk.dtype(),
            self.dtype
        );
        match self.writer.as_mut() {
            Some(writer) => writer.push_chunk(segment_writer, chunk).await,
            None => {
                let compressed = BtrBlocksCompressor.compress(&chunk)?;
                let mut writer = if !compressed.is_encoding(DictEncoding.id()) {
                    self.strategy.fallback.new_writer(&self.ctx, &self.dtype)?
                } else {
                    repeating::DictLayoutWriter::new(
                        self.ctx.clone(),
                        &self.dtype,
                        self.strategy.clone(),
                    )
                    .boxed()
                };
                writer.push_chunk(segment_writer, chunk).await?;
                self.writer = Some(writer);
                Ok(())
            }
        }
    }

    async fn flush(
        &mut self,
        segment_writer: &mut dyn crate::segments::ConcurrentSegmentWriter,
    ) -> VortexResult<()> {
        match self.writer.as_mut() {
            None => vortex_bail!("flush called before push_chunk"),
            Some(writer) => writer.flush(segment_writer).await,
        }
    }

    async fn finish(
        &mut self,
        segment_writer: &mut dyn crate::segments::ConcurrentSegmentWriter,
    ) -> VortexResult<LayoutRef> {
        match self.writer.as_mut() {
            None => vortex_bail!("finish called before push_chunk"),
            Some(writer) => writer.finish(segment_writer).await,
        }
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
