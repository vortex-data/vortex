// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use async_channel::SendError;
use async_channel::Sender;
use async_channel::bounded;
use async_stream::stream;
use async_stream::try_stream;
use async_trait::async_trait;
use futures::FutureExt;
use futures::Stream;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::channel::oneshot;
use futures::future::BoxFuture;
use futures::pin_mut;
use futures::stream::BoxStream;
use futures::stream::iter;
use futures::stream::once;
use futures::try_join;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Dict;
use vortex_array::builders::dict::DictConstraints;
use vortex_array::builders::dict::DictEncoder;
use vortex_array::builders::dict::dict_encoder_in;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::compressed::CompressorPlugin;
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

#[derive(Clone)]
pub struct DictLayoutOptions {
    pub constraints: DictLayoutConstraints,
    /// Number of leading rows the probe compressor sees when deciding whether to use a dictionary.
    pub probe_len: usize,
}

impl Default for DictLayoutOptions {
    fn default() -> Self {
        Self {
            constraints: DictLayoutConstraints::default(),
            probe_len: 8192,
        }
    }
}

/// A layout strategy that encodes chunk into values and codes, if found
/// appropriate by the btrblocks compressor. Current implementation only
/// checks the first [`probe_len`](DictLayoutOptions::probe_len) rows to decide whether to apply
/// dict layout and encodes chunks into dictionaries. When the dict constraints are hit, a new
/// dictionary is created.
#[derive(Clone)]
pub struct DictStrategy {
    codes: Arc<dyn LayoutStrategy>,
    values: Arc<dyn LayoutStrategy>,
    fallback: Arc<dyn LayoutStrategy>,
    options: DictLayoutOptions,
    probe_compressor: Arc<dyn CompressorPlugin>,
}

impl DictStrategy {
    pub fn new<Codes: LayoutStrategy, Values: LayoutStrategy, Fallback: LayoutStrategy>(
        codes: Codes,
        values: Values,
        fallback: Fallback,
        options: DictLayoutOptions,
        probe_compressor: Arc<dyn CompressorPlugin>,
    ) -> Self {
        Self {
            codes: Arc::new(codes),
            values: Arc::new(values),
            fallback: Arc::new(fallback),
            options,
            probe_compressor,
        }
    }
}

#[async_trait]
impl LayoutStrategy for DictStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
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
        let (stream, sample) = peek_probe_sample(stream, options.probe_len).await?;
        let stream = SequentialStreamAdapter::new(dtype.clone(), stream).sendable();

        let should_fallback = match sample {
            None => true, // empty stream
            Some(sample) => {
                let mut exec_ctx = session.create_execution_ctx();
                let compressed = self
                    .probe_compressor
                    .compress_chunk(&sample, &mut exec_ctx)?;
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
        let dict_stream = dict_encode_stream(
            stream,
            options.constraints.into(),
            session.create_execution_ctx(),
        );

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
                let codes_fut = handle.spawn_nested(move |_| async move {
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
                let values_layout = handle.spawn_nested(move |_| async move {
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
    mut exec_ctx: ExecutionCtx,
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
                    let chunks = state.encode(&mut labeler, chunk, &mut exec_ctx)?;
                    drop(labeler);
                    for dict_chunk in chunks {
                        yield dict_chunk;
                    }
                }
                None => {
                    // this is the last element, encode and drain chunks
                    let mut labeler = DictChunkLabeler::new(sequence_id);
                    let encoded = state.encode(&mut labeler, chunk, &mut exec_ctx)?;
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
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Vec<DictionaryChunk>> {
        let mut res = Vec::new();
        let mut to_be_encoded = Some(chunk);
        while let Some(remaining) = to_be_encoded.take() {
            match self.encoder.take() {
                None => match start_encoding(&self.constraints, &remaining, exec_ctx)? {
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
                    match encode_chunk(encoder, &remaining, exec_ctx)? {
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
    active_codes_tx: Option<Sender<SequencedChunk>>,
    active_values_tx: Option<oneshot::Sender<SequencedChunk>>,
    pending_send: Option<BoxFuture<'static, Result<(), SendError<SequencedChunk>>>>,
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
                        let (codes_tx, codes_rx) = bounded::<SequencedChunk>(1);
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
                        let codes_stream =
                            SequentialStreamAdapter::new(codes_dtype, codes_rx.boxed()).sendable();

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

/// Peeks chunks until they hold `probe_len` rows, returning the stream with those chunks back in
/// front and its first `probe_len` rows, or `None` if the stream is empty.
async fn peek_probe_sample(
    mut stream: BoxStream<'static, SequencedChunk>,
    probe_len: usize,
) -> VortexResult<(BoxStream<'static, SequencedChunk>, Option<ArrayRef>)> {
    let mut peeked = Vec::new();
    let mut peeked_len = 0;
    while peeked_len < probe_len {
        let Some(item) = stream.next().await else {
            break;
        };
        let (sequence_id, chunk) = item?;
        peeked_len += chunk.len();
        peeked.push((sequence_id, chunk));
    }
    let sample_len = peeked_len.min(probe_len);
    let sample = match peeked.as_slice() {
        [] => return Ok((stream, None)),
        [(_, chunk)] => chunk.slice(0..sample_len)?,
        [(_, first), ..] => ChunkedArray::try_new(
            peeked.iter().map(|(_, chunk)| chunk.clone()),
            first.dtype().clone(),
        )?
        .into_array()
        .slice(0..sample_len)?,
    };
    let stream = iter(peeked.into_iter().map(Ok)).chain(stream).boxed();
    Ok((stream, Some(sample)))
}

pub fn dict_layout_supported(dtype: &DType) -> bool {
    matches!(
        dtype,
        DType::Primitive(..) | DType::Utf8(_) | DType::Binary(_)
    )
}

enum EncodingState {
    Continue((Box<dyn DictEncoder>, ArrayRef)),
    // (values, encoded, unencoded)
    Done((ArrayRef, ArrayRef, ArrayRef)),
}

fn start_encoding(
    constraints: &DictConstraints,
    chunk: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<EncodingState> {
    let encoder = dict_encoder_in(chunk, constraints, ctx.allocator().clone());
    encode_chunk(encoder, chunk, ctx)
}

fn encode_chunk(
    mut encoder: Box<dyn DictEncoder>,
    chunk: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<EncodingState> {
    let encoded = encoder.encode(chunk, ctx)?.into_array();
    match remainder(chunk, encoded.len(), ctx.session())? {
        None => Ok(EncodingState::Continue((encoder, encoded))),
        Some(unencoded) => Ok(EncodingState::Done((encoder.reset(), encoded, unencoded))),
    }
}

fn remainder(
    array: &ArrayRef,
    encoded_len: usize,
    session: &VortexSession,
) -> VortexResult<Option<ArrayRef>> {
    if encoded_len < array.len() {
        // The dictionary encoder takes concrete encodings, so optimize the lazy slice into one.
        Ok(Some(
            array.slice(encoded_len..array.len())?.optimize(session)?,
        ))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::StreamExt;
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::Canonical;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::MaskFuture;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builders::dict::DictConstraints;
    use vortex_array::builders::dict::dict_encode;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::dtype::PType;
    use vortex_array::expr::root;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_io::runtime::single::block_on;
    use vortex_io::session::RuntimeSessionExt;
    use vortex_session::VortexSession;

    use super::DictChunkLabeler;
    use super::DictLayoutOptions;
    use super::DictStrategy;
    use super::DictStreamState;
    use super::DictionaryChunk;
    use super::DictionaryTransformer;
    use super::dict_encode_stream;
    use crate::LayoutStrategy;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::dict::Dict;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialStream;
    use crate::sequence::SequentialStreamAdapter;
    use crate::sequence::SequentialStreamExt;
    use crate::test::new_session;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

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
        let dict_stream =
            dict_encode_stream(input_stream, constraints, SESSION.create_execution_ctx());

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
        let dict_stream =
            dict_encode_stream(input_stream, constraints, SESSION.create_execution_ctx());

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

    fn encoded_row_count(chunks: &[DictionaryChunk]) -> usize {
        chunks
            .iter()
            .map(|chunk| match chunk {
                DictionaryChunk::Codes { codes, .. } => codes.len(),
                DictionaryChunk::Values(_) => 0,
            })
            .sum()
    }

    /// A 49-byte value plus its 16-byte `BinaryView` exceeds the 64-byte budget of an empty
    /// dictionary. Encoding must still make progress instead of looping forever.
    #[test]
    fn test_encode_value_larger_than_dictionary_budget_terminates() -> VortexResult<()> {
        let mut state = DictStreamState {
            encoder: None,
            constraints: DictConstraints {
                max_bytes: 64,
                max_len: 100,
            },
        };

        let value = "x".repeat(49);
        let chunk = VarBinArray::from(vec![value.as_str()]).into_array();
        let mut labeler = DictChunkLabeler::new(SequenceId::root().advance());
        let encoded = state.encode(&mut labeler, chunk, &mut SESSION.create_execution_ctx())?;

        assert_eq!(encoded_row_count(&encoded), 1);
        Ok(())
    }

    #[test]
    fn test_encode_splits_oversized_value_into_its_own_dictionary() -> VortexResult<()> {
        let mut state = DictStreamState {
            encoder: None,
            constraints: DictConstraints {
                max_bytes: 64,
                max_len: 100,
            },
        };

        let big = "x".repeat(49);
        let chunk = VarBinArray::from(vec!["a", big.as_str(), "b"]).into_array();
        let mut labeler = DictChunkLabeler::new(SequenceId::root().advance());
        let encoded = state.encode(&mut labeler, chunk, &mut SESSION.create_execution_ctx())?;
        let drained = state.drain_values(&mut labeler);

        assert_eq!(encoded_row_count(&encoded) + encoded_row_count(&drained), 3);
        let dictionaries = encoded
            .iter()
            .chain(&drained)
            .filter_map(|chunk| match chunk {
                DictionaryChunk::Values((_, values)) => Some(values),
                DictionaryChunk::Codes { .. } => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(dictionaries.len(), 3);
        let mut ctx = SESSION.create_execution_ctx();
        for (values, expected) in dictionaries.into_iter().zip(["a", big.as_str(), "b"]) {
            assert_arrays_eq!(values, VarBinArray::from(vec![expected]), &mut ctx);
        }
        Ok(())
    }

    /// The probe sees the first `probe_len` rows however the stream is chunked, and every chunk
    /// it peeked at is still written.
    #[rstest]
    #[case::small_chunks(10, 10, true)]
    #[case::shorter_than_probe(5, 10, false)]
    #[case::one_large_chunk(1, 200, true)]
    fn probe_samples_leading_rows(
        #[case] nchunks: usize,
        #[case] chunk_len: usize,
        #[case] dict: bool,
    ) -> VortexResult<()> {
        const PROBE_LEN: usize = 100;
        let probed_len = Arc::new(AtomicUsize::new(0));
        let probed_len2 = Arc::clone(&probed_len);
        // Dictionary-encodes only a full-size sample, so the decision shows how many rows it saw.
        let probe = move |chunk: &ArrayRef, ctx: &mut ExecutionCtx| -> VortexResult<ArrayRef> {
            probed_len2.store(chunk.len(), Ordering::Relaxed);
            if chunk.len() < PROBE_LEN {
                return Ok(chunk.clone());
            }
            let chunk = chunk.clone().execute::<Canonical>(ctx)?.into_array();
            Ok(dict_encode(&chunk, ctx)?.into_array())
        };
        let strategy = DictStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            FlatLayoutStrategy::default(),
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            DictLayoutOptions {
                probe_len: PROBE_LEN,
                ..Default::default()
            },
            Arc::new(probe),
        );

        let values =
            PrimitiveArray::from_iter((0i32..3).cycle().take(nchunks * chunk_len)).into_array();
        let dtype = values.dtype().clone();
        let (mut sequence_id, eof) = SequenceId::root().split();
        let chunks = (0..nchunks)
            .map(|i| {
                // Writers receive optimized chunks, so slice to the primitive array.
                let chunk = values
                    .slice(i * chunk_len..(i + 1) * chunk_len)?
                    .optimize(&SESSION)?;
                Ok((sequence_id.advance(), chunk))
            })
            .collect::<Vec<VortexResult<_>>>();
        let segments = Arc::new(TestSegments::default());

        let actual = block_on(|handle| async move {
            let session = new_session().with_handle(handle);
            let layout = strategy
                .write_stream(
                    ArrayContext::empty().into(),
                    Arc::<TestSegments>::clone(&segments),
                    SequentialStreamAdapter::new(dtype, futures::stream::iter(chunks)).sendable(),
                    eof,
                    &session,
                )
                .await?;
            assert_eq!(layout.is::<Dict>(), dict);

            let reader = layout.new_reader("".into(), segments, &session, &Default::default())?;
            let row_count = layout.row_count();
            reader
                .projection_evaluation(
                    &(0..row_count),
                    &root().bind(reader.dtype())?,
                    MaskFuture::new_true(usize::try_from(row_count)?),
                )?
                .await
        })?;

        assert_eq!(
            probed_len.load(Ordering::Relaxed),
            PROBE_LEN.min(nchunks * chunk_len)
        );
        assert_arrays_eq!(actual, values, &mut SESSION.create_execution_ctx());
        Ok(())
    }
}
