// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use async_stream::stream;
use async_stream::try_stream;
use async_trait::async_trait;
use futures::FutureExt;
use futures::Stream;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use futures::pin_mut;
use futures::stream::BoxStream;
use futures::stream::once;
use futures::try_join;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::arrays::Dict;
use vortex_array::builders::dict::DictConstraints;
use vortex_array::builders::dict::DictEncoder;
use vortex_array::builders::dict::dict_encoder;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::kanal_ext::KanalExt;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::IntoLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::dict::DictLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStream;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Constraints for dictionary layout encoding.
///
/// Note that [`max_len`](Self::max_len) is limited to `u16` (65,535 entries) by design. Since
/// layout chunks are typically ~8k elements, having more than 64k unique values in a dictionary
/// means dictionary encoding provides little compression benefit. If a column has very high
/// cardinality, the fallback encoding strategy should be used instead.
#[derive(Clone)]
pub struct DictLayoutConstraints {
    /// Maximum size of the dictionary in bytes.
    pub max_bytes: usize,
    /// Maximum dictionary length. Limited to `u16` because dictionaries with more than 64k unique
    /// values provide diminishing compression returns given typical chunk sizes (~8k elements).
    ///
    /// The codes dtype is determined upfront from this constraint:
    /// - [`PType::U8`] when max_len <= 255
    /// - [`PType::U16`] when max_len > 255
    ///
    /// Vortex encoders must always produce unsigned integer codes; signed codes are only accepted for external compatibility.
    pub max_len: u16,
}

impl From<DictLayoutConstraints> for DictConstraints {
    fn from(value: DictLayoutConstraints) -> Self {
        DictConstraints {
            max_bytes: value.max_bytes,
            max_len: value.max_len as usize,
        }
    }
}

impl Default for DictLayoutConstraints {
    fn default() -> Self {
        Self {
            max_bytes: 1024 * 1024,
            max_len: u16::MAX,
        }
    }
}

#[derive(Clone, Default)]
pub struct DictLayoutOptions {
    pub constraints: DictLayoutConstraints,
}

/// A layout strategy that encodes chunk into values and codes, if found
/// appropriate by the btrblocks compressor. Current implementation only
/// checks the first chunk to decide whether to apply dict layout and
/// encodes chunks into dictionaries. When the dict constraints are hit, a
/// new dictionary is created.
#[derive(Clone)]
pub struct DictStrategy {
    codes: Arc<dyn LayoutStrategy>,
    values: Arc<dyn LayoutStrategy>,
    fallback: Arc<dyn LayoutStrategy>,
    options: DictLayoutOptions,
}

impl DictStrategy {
    pub fn new<Codes: LayoutStrategy, Values: LayoutStrategy, Fallback: LayoutStrategy>(
        codes: Codes,
        values: Values,
        fallback: Fallback,
        options: DictLayoutOptions,
    ) -> Self {
        Self {
            codes: Arc::new(codes),
            values: Arc::new(values),
            fallback: Arc::new(fallback),
            options,
        }
    }
}

#[async_trait]
impl LayoutStrategy for DictStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        // Fallback if dtype is not supported
        if !dict_layout_supported(stream.dtype()) {
            return self
                .fallback
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }

        let options = self.options.clone();
        let dtype = stream.dtype().clone();

        // 0. decide if chunks are eligible for dict encoding
        let (stream, first_chunk) = peek_first_chunk(stream).await?;
        let stream = SequentialStreamAdapter::new(dtype.clone(), stream).sendable();

        let should_fallback = match first_chunk {
            None => true, // empty stream
            Some(chunk) => {
                let compressed = BtrBlocksCompressor::default().compress(&chunk)?;
                !compressed.is::<Dict>()
            }
        };
        if should_fallback {
            // first chunk did not compress to dict, or did not exist. Skip dict layout
            return self
                .fallback
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }

        // 1. from a chunk stream, create a stream that yields codes
        // followed by a single value chunk when dict constraints are hit.
        // (a1, a2) -> (code(c1), code(c2), values(v1), code(c3), ...)
        let dict_stream = dict_encode_stream(stream, options.constraints.into());

        // Wrap up the dict stream to yield pairs of (codes_stream, values_future).
        // Each of these pairs becomes a child dict layout.
        let runs = DictionaryTransformer::new(dict_stream);

        let handle = session.handle();
        let dtype2 = dtype.clone();
        let child_layouts = stream! {
            pin_mut!(runs);

            while let Some((codes_stream, values_fut)) = runs.next().await {
                let codes = Arc::clone(&self.codes);
                let codes_eof = eof.split_off();
                let ctx2 = ctx.clone();
                let segment_sink2 = Arc::clone(&segment_sink);
                let session2 = session.clone();
                let codes_fut = handle.spawn_nested(move |h| async move {
                    let session2 = session2.with_handle(h);
                    codes.write_stream(
                        ctx2,
                        segment_sink2,
                        codes_stream.sendable(),
                        codes_eof,
                        &session2,
                    ).await
                });

                let values = Arc::clone(&self.values);
                let values_eof = eof.split_off();
                let ctx2 = ctx.clone();
                let segment_sink2 = Arc::clone(&segment_sink);
                let dtype2 = dtype2.clone();
                let session2 = session.clone();
                let values_layout = handle.spawn_nested(move |h| async move {
                    let session2 = session2.with_handle(h);
                    values.write_stream(
                        ctx2,
                        segment_sink2,
                        SequentialStreamAdapter::new(dtype2, once(values_fut)).sendable(),
                        values_eof,
                        &session2,
                    ).await
                });

                yield async move {
                    try_join!(codes_fut, values_layout)
                }.boxed();
            }
        };

        let mut child_layouts = child_layouts
            .buffered(usize::MAX)
            .map(|result| {
                let (codes_layout, values_layout) = result?;
                // All values are referenced when created via dictionary encoding
                Ok::<_, VortexError>(DictLayout::new(values_layout, codes_layout).into_layout())
            })
            .try_collect::<Vec<_>>()
            .await?;

        if child_layouts.len() == 1 {
            return Ok(child_layouts.remove(0));
        }

        let row_count = child_layouts.iter().map(|child| child.row_count()).sum();
        Ok(ChunkedLayout::new(
            row_count,
            dtype,
            OwnedLayoutChildren::layout_children(child_layouts),
        )
        .into_layout())
    }

    fn buffered_bytes(&self) -> u64 {
        self.codes.buffered_bytes() + self.values.buffered_bytes() + self.fallback.buffered_bytes()
    }
}

enum DictionaryChunk {
    Codes {
        seq_id: SequenceId,
        codes: ArrayRef,
        codes_ptype: PType,
    },
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

        while let Some(item) = input.next().await {
            let (sequence_id, chunk) = item?;

            // labeler potentially creates sub sequences, we must
            // create it on both arms to avoid having a SequencePointer
            // between await points
            match input.as_mut().peek().await {
                Some(_) => {
                    let mut labeler = DictChunkLabeler::new(sequence_id);
                    let chunks = state.encode(&mut labeler, chunk)?;
                    drop(labeler);
                    for dict_chunk in chunks {
                        yield dict_chunk;
                    }
                }
                None => {
                    // this is the last element, encode and drain chunks
                    let mut labeler = DictChunkLabeler::new(sequence_id);
                    let encoded = state.encode(&mut labeler, chunk)?;
                    let drained = state.drain_values(&mut labeler);
                    drop(labeler);
                    for dict_chunk in encoded.into_iter().chain(drained.into_iter()) {
                        yield dict_chunk;
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
    ) -> VortexResult<Vec<DictionaryChunk>> {
        let mut res = Vec::new();
        let mut to_be_encoded = Some(chunk);
        while let Some(remaining) = to_be_encoded.take() {
            match self.encoder.take() {
                None => match start_encoding(&self.constraints, &remaining)? {
                    EncodingState::Continue((encoder, encoded)) => {
                        let ptype = encoder.codes_ptype();
                        res.push(labeler.codes(encoded, ptype));
                        self.encoder = Some(encoder);
                    }
                    EncodingState::Done((values, encoded, unencoded)) => {
                        // Encoder was created and consumed within start_encoding
                        let ptype = PType::try_from(encoded.dtype())
                            .vortex_expect("codes should be primitive");
                        res.push(labeler.codes(encoded, ptype));
                        res.push(labeler.values(values));
                        to_be_encoded = Some(unencoded);
                    }
                },
                Some(encoder) => {
                    let ptype = encoder.codes_ptype();
                    match encode_chunk(encoder, &remaining)? {
                        EncodingState::Continue((encoder, encoded)) => {
                            res.push(labeler.codes(encoded, ptype));
                            self.encoder = Some(encoder);
                        }
                        EncodingState::Done((values, encoded, unencoded)) => {
                            res.push(labeler.codes(encoded, ptype));
                            res.push(labeler.values(values));
                            to_be_encoded = Some(unencoded);
                        }
                    }
                }
            }
        }
        Ok(res)
    }

    fn drain_values(&mut self, labeler: &mut DictChunkLabeler) -> Vec<DictionaryChunk> {
        match self.encoder.as_mut() {
            None => Vec::new(),
            Some(encoder) => vec![labeler.values(encoder.reset())],
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

    fn codes(&mut self, chunk: ArrayRef, ptype: PType) -> DictionaryChunk {
        DictionaryChunk::Codes {
            seq_id: self.sequence_pointer.advance(),
            codes: chunk,
            codes_ptype: ptype,
        }
    }

    fn values(&mut self, chunk: ArrayRef) -> DictionaryChunk {
        DictionaryChunk::Values((self.sequence_pointer.advance(), chunk))
    }
}

type SequencedChunk = VortexResult<(SequenceId, ArrayRef)>;

struct DictionaryTransformer {
    input: DictionaryStream,
    active_codes_tx: Option<kanal::AsyncSender<SequencedChunk>>,
    active_values_tx: Option<oneshot::Sender<SequencedChunk>>,
    pending_send: Option<BoxFuture<'static, Result<(), kanal::SendError>>>,
}

impl DictionaryTransformer {
    fn new(input: DictionaryStream) -> Self {
        Self {
            input,
            active_codes_tx: None,
            active_values_tx: None,
            pending_send: None,
        }
    }
}

impl Stream for DictionaryTransformer {
    type Item = (SendableSequentialStream, BoxFuture<'static, SequencedChunk>);

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            // First, try to complete any pending send
            if let Some(mut send_fut) = self.pending_send.take() {
                match send_fut.poll_unpin(cx) {
                    Poll::Ready(Ok(())) => {
                        // Send completed, continue processing
                    }
                    Poll::Ready(Err(_)) => {
                        // Receiver dropped, close this group
                        self.active_codes_tx = None;
                        if let Some(values_tx) = self.active_values_tx.take() {
                            drop(values_tx.send(Err(vortex_err!("values receiver dropped"))));
                        }
                    }
                    Poll::Pending => {
                        // Still pending, save it and return
                        self.pending_send = Some(send_fut);
                        return Poll::Pending;
                    }
                }
            }

            match self.input.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(DictionaryChunk::Codes {
                    seq_id,
                    codes,
                    codes_ptype,
                }))) => {
                    if self.active_codes_tx.is_none() {
                        // Start a new group
                        let (codes_tx, codes_rx) = kanal::bounded_async::<SequencedChunk>(1);
                        let (values_tx, values_rx) = oneshot::channel();

                        self.active_codes_tx = Some(codes_tx.clone());
                        self.active_values_tx = Some(values_tx);

                        // Use passed codes_ptype instead of getting from array
                        let codes_dtype = DType::Primitive(codes_ptype, Nullability::NonNullable);

                        // Send first codes.
                        self.pending_send =
                            Some(Box::pin(
                                async move { codes_tx.send(Ok((seq_id, codes))).await },
                            ));

                        // Create output streams.
                        let codes_stream = SequentialStreamAdapter::new(
                            codes_dtype,
                            codes_rx.into_stream().boxed(),
                        )
                        .sendable();

                        let values_future = async move {
                            values_rx
                                .await
                                .map_err(|e| vortex_err!("values sender dropped: {}", e))
                                .flatten()
                        }
                        .boxed();

                        return Poll::Ready(Some((codes_stream, values_future)));
                    }

                    // Continue streaming codes to existing group
                    if let Some(tx) = &self.active_codes_tx {
                        let tx = tx.clone();
                        self.pending_send =
                            Some(Box::pin(async move { tx.send(Ok((seq_id, codes))).await }));
                    }
                }
                Poll::Ready(Some(Ok(DictionaryChunk::Values(values)))) => {
                    // Complete the current group
                    if let Some(values_tx) = self.active_values_tx.take() {
                        drop(values_tx.send(Ok(values)));
                    }
                    self.active_codes_tx = None; // Close codes stream
                }
                Poll::Ready(Some(Err(e))) => {
                    // Send error to active channels if any
                    if let Some(values_tx) = self.active_values_tx.take() {
                        drop(values_tx.send(Err(e)));
                    }
                    self.active_codes_tx = None;
                    // And terminate the stream
                    return Poll::Ready(None);
                }
                Poll::Ready(None) => {
                    // Handle any incomplete group
                    if let Some(values_tx) = self.active_values_tx.take() {
                        drop(values_tx.send(Err(vortex_err!("Incomplete dictionary group"))));
                    }
                    self.active_codes_tx = None;
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
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

fn start_encoding(constraints: &DictConstraints, chunk: &ArrayRef) -> VortexResult<EncodingState> {
    let encoder = dict_encoder(chunk, constraints);
    encode_chunk(encoder, chunk)
}

fn encode_chunk(
    mut encoder: Box<dyn DictEncoder>,
    chunk: &ArrayRef,
) -> VortexResult<EncodingState> {
    let encoded = encoder.encode(chunk);
    match remainder(chunk, encoded.len())? {
        None => Ok(EncodingState::Continue((encoder, encoded))),
        Some(unencoded) => Ok(EncodingState::Done((encoder.reset(), encoded, unencoded))),
    }
}

fn remainder(array: &ArrayRef, encoded_len: usize) -> VortexResult<Option<ArrayRef>> {
    if encoded_len < array.len() {
        Ok(Some(array.slice(encoded_len..array.len())?))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;
    use vortex_array::IntoArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::builders::dict::DictConstraints;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;

    use super::DictionaryTransformer;
    use super::dict_encode_stream;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialStream;
    use crate::sequence::SequentialStreamAdapter;
    use crate::sequence::SequentialStreamExt;

    /// Regression test for a bug where the codes stream dtype was hardcoded to U16 instead of
    /// using the actual codes dtype from the array. When `max_len <= 255`, the dict encoder
    /// produces U8 codes, but the stream was incorrectly typed as U16, causing a dtype mismatch
    /// assertion failure in [`SequentialStreamAdapter`].
    #[tokio::test]
    async fn test_dict_transformer_uses_u8_for_small_dictionaries() {
        // Use max_len = 100 to force U8 codes (since 100 <= 255).
        let constraints = DictConstraints {
            max_bytes: 1024 * 1024,
            max_len: 100,
        };

        // Create a simple string array with a few unique values.
        let arr = VarBinArray::from(vec!["hello", "world", "hello", "world"]).into_array();

        // Wrap into a sequential stream.
        let mut pointer = SequenceId::root();
        let input_stream = SequentialStreamAdapter::new(
            arr.dtype().clone(),
            futures::stream::once(async move { Ok((pointer.advance(), arr)) }),
        )
        .sendable();

        // Encode into dict chunks.
        let dict_stream = dict_encode_stream(input_stream, constraints);

        // Transform into codes/values streams.
        let mut transformer = DictionaryTransformer::new(dict_stream);

        // Get the first (and only) run.
        let (codes_stream, _values_fut) = transformer
            .next()
            .await
            .expect("expected at least one dictionary run");

        // The key assertion: codes stream dtype should be U8, not U16.
        assert_eq!(
            codes_stream.dtype(),
            &DType::Primitive(PType::U8, NonNullable),
            "codes stream should use U8 dtype for small dictionaries, not U16"
        );
    }

    /// Test that the codes stream uses U16 dtype when the dictionary has more than 255 entries.
    #[tokio::test]
    async fn test_dict_transformer_uses_u16_for_large_dictionaries() {
        // Use max_len = 1000 to allow U16 codes (since 1000 > 255).
        let constraints = DictConstraints {
            max_bytes: 1024 * 1024,
            max_len: 1000,
        };

        // Create an array with more than 255 distinct values to force U16 codes.
        let values: Vec<String> = (0..300).map(|i| format!("value_{i}")).collect();
        let arr =
            VarBinArray::from(values.iter().map(|s| s.as_str()).collect::<Vec<_>>()).into_array();

        // Wrap into a sequential stream.
        let mut pointer = SequenceId::root();
        let input_stream = SequentialStreamAdapter::new(
            arr.dtype().clone(),
            futures::stream::once(async move { Ok((pointer.advance(), arr)) }),
        )
        .sendable();

        // Encode into dict chunks.
        let dict_stream = dict_encode_stream(input_stream, constraints);

        // Transform into codes/values streams.
        let mut transformer = DictionaryTransformer::new(dict_stream);

        // Get the first (and only) run.
        let (codes_stream, _values_fut) = transformer
            .next()
            .await
            .expect("expected at least one dictionary run");

        // Codes stream dtype should be U16 since we have more than 255 distinct values.
        assert_eq!(
            codes_stream.dtype(),
            &DType::Primitive(PType::U16, NonNullable),
            "codes stream should use U16 dtype for dictionaries with >255 entries"
        );
    }
}
